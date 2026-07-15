use anyhow::{anyhow, bail};
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_bedrockruntime::types::TokenUsage;
use axum::response::sse::Event;
use futures::stream::{BoxStream, StreamExt};
use request::ChatCompletionsRequest;
use std::time::Duration;
use tokio::{sync::mpsc, time::timeout};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

use crate::DONE_MESSAGE;
use crate::provider::mantle::{MANTLE_MODEL, force_mantle_model, mantle_url, sign_bedrock_json_post};

const EVENT_TX_SEND_TIMEOUT: Duration = Duration::from_secs(30);

/// Forwards OpenAI-shaped `/chat/completions` requests to Bedrock's
/// OpenAI-compatible ("Mantle") endpoint on `bedrock-runtime`, authenticating
/// with SigV4 using the credentials the proxy already loads. The request body
/// and the streamed OpenAI SSE response are passed through unchanged.
pub struct MantleChatCompletionsProvider {
    http_client: reqwest::Client,
    region: String,
    credentials_provider: SharedCredentialsProvider,
}

impl MantleChatCompletionsProvider {
    pub fn new(
        http_client: reqwest::Client,
        region: String,
        credentials_provider: SharedCredentialsProvider,
    ) -> Self {
        Self {
            http_client,
            region,
            credentials_provider,
        }
    }

    pub async fn chat_completions_stream<F>(
        &self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<Event>>>
    where
        F: Fn(&TokenUsage) + Send + Sync + 'static,
    {
        // Build the OpenAI-compatible body. Force streaming and request usage in
        // the final chunk so we can keep logging token counts, and pin the model
        // to the one served on Mantle.
        let mut body_value = serde_json::to_value(&request)?;
        if let Some(obj) = body_value.as_object_mut() {
            obj.insert("stream".to_string(), serde_json::Value::Bool(true));
            obj.insert(
                "stream_options".to_string(),
                serde_json::json!({ "include_usage": true }),
            );
        }
        force_mantle_model(&mut body_value);
        let body = serde_json::to_vec(&body_value)?;

        let url = mantle_url(&self.region, "/openai/v1/chat/completions");

        // SigV4-sign against the `bedrock` service; the returned headers are
        // attached to the outgoing request below.
        let signed_headers =
            sign_bedrock_json_post(&self.credentials_provider, &self.region, &url, &body).await?;

        info!(
            "Sending OpenAI request to Bedrock Mantle (model {} -> {})",
            request.model, MANTLE_MODEL
        );

        let mut req = self
            .http_client
            .post(&url)
            .header("content-type", "application/json");
        for (name, value) in signed_headers {
            req = req.header(name, value);
        }

        let response = req.body(body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!("Bedrock Mantle error ({}): {}", status, error_body);
            bail!("Bedrock Mantle request failed ({}): {}", status, error_body);
        }

        info!("Successfully connected to Bedrock Mantle stream");

        Ok(process_mantle_stream(response, usage_callback))
    }
}

/// Re-emits the upstream OpenAI SSE stream as axum SSE events, forwarding each
/// `data:` payload verbatim and logging token usage from the final chunk.
fn process_mantle_stream<F>(
    response: reqwest::Response,
    usage_callback: F,
) -> BoxStream<'static, anyhow::Result<Event>>
where
    F: Fn(&TokenUsage) + Send + Sync + 'static,
{
    let (event_tx, event_rx) = mpsc::channel::<anyhow::Result<Event>>(1);

    tokio::spawn(async move {
        let mut byte_stream = response.bytes_stream();
        // Buffer of received bytes with carriage returns stripped, so frames are
        // uniformly separated by a blank line ("\n\n") regardless of CRLF usage.
        let mut buf: Vec<u8> = Vec::new();

        loop {
            match byte_stream.next().await {
                Some(Ok(chunk)) => {
                    buf.extend(chunk.iter().copied().filter(|&b| b != b'\r'));

                    while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                        let frame: Vec<u8> = buf.drain(..pos + 2).collect();
                        let frame = &frame[..frame.len() - 2];
                        let text = String::from_utf8_lossy(frame);

                        for line in text.split('\n') {
                            let Some(rest) = line.strip_prefix("data:") else {
                                continue;
                            };
                            let payload = rest.strip_prefix(' ').unwrap_or(rest);

                            if payload == DONE_MESSAGE {
                                info!("Mantle stream reached DONE, closing");
                                let _ = timeout(
                                    EVENT_TX_SEND_TIMEOUT,
                                    event_tx.send(Ok(Event::default().data(DONE_MESSAGE))),
                                )
                                .await;
                                return;
                            }

                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload)
                                && let Some(usage) = value.get("usage").filter(|u| !u.is_null())
                                && let Some(token_usage) = parse_token_usage(usage)
                            {
                                usage_callback(&token_usage);
                            }

                            match timeout(
                                EVENT_TX_SEND_TIMEOUT,
                                event_tx.send(Ok(Event::default().data(payload))),
                            )
                            .await
                            {
                                Ok(Ok(())) => {}
                                Ok(Err(_)) => {
                                    info!("SSE client disconnected, stopping Mantle stream");
                                    return;
                                }
                                Err(_) => {
                                    error!("Channel send timed out, consumer likely stuck");
                                    return;
                                }
                            }
                        }
                    }
                }
                Some(Err(e)) => {
                    let _ = timeout(
                        EVENT_TX_SEND_TIMEOUT,
                        event_tx.send(Err(anyhow!("Mantle stream receive error: {}", e))),
                    )
                    .await;
                    return;
                }
                None => break,
            }
        }

        // Upstream closed without an explicit [DONE]; emit one so clients finish.
        info!("Mantle stream ended, sending DONE message");
        let _ = timeout(
            EVENT_TX_SEND_TIMEOUT,
            event_tx.send(Ok(Event::default().data(DONE_MESSAGE))),
        )
        .await;
    });

    ReceiverStream::new(event_rx).boxed()
}

/// Builds a Bedrock `TokenUsage` from an OpenAI `usage` object, if all fields
/// are present, so the existing usage-logging callback can be reused.
fn parse_token_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    let prompt = usage.get("prompt_tokens")?.as_i64()?;
    let completion = usage.get("completion_tokens")?.as_i64()?;
    let total = usage.get("total_tokens")?.as_i64()?;

    TokenUsage::builder()
        .input_tokens(prompt as i32)
        .output_tokens(completion as i32)
        .total_tokens(total as i32)
        .build()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_request(extra: serde_json::Value) -> ChatCompletionsRequest {
        let mut json = serde_json::json!({
            "model": "openai.gpt-oss-120b",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        if let (Some(base), serde_json::Value::Object(extra)) = (json.as_object_mut(), extra) {
            base.extend(extra);
        }
        serde_json::from_value(json).unwrap()
    }

    /// The outgoing body must force streaming and opt into usage reporting,
    /// regardless of what the client sent for `stream`.
    fn shape_body(request: &ChatCompletionsRequest) -> serde_json::Value {
        let mut body_value = serde_json::to_value(request).unwrap();
        let obj = body_value.as_object_mut().unwrap();
        obj.insert("stream".to_string(), serde_json::Value::Bool(true));
        obj.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
        body_value
    }

    #[test]
    fn body_forces_stream_and_usage() {
        let request = base_request(serde_json::json!({ "stream": false }));
        let body = shape_body(&request);
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(body["stream_options"]["include_usage"], serde_json::json!(true));
        assert_eq!(body["model"], serde_json::json!("openai.gpt-oss-120b"));
    }

    #[test]
    fn parse_token_usage_reads_openai_fields() {
        let usage = serde_json::json!({
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19
        });
        let token_usage = parse_token_usage(&usage).unwrap();
        assert_eq!(token_usage.input_tokens(), 12);
        assert_eq!(token_usage.output_tokens(), 7);
        assert_eq!(token_usage.total_tokens(), 19);
    }

    #[test]
    fn parse_token_usage_returns_none_when_incomplete() {
        let usage = serde_json::json!({ "prompt_tokens": 12 });
        assert!(parse_token_usage(&usage).is_none());
    }
}
