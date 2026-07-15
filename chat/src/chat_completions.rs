use anyhow::anyhow;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_bedrockruntime::types::TokenUsage;
use axum::response::sse::Event;
use futures::stream::{BoxStream, StreamExt};
use request::ChatCompletionsRequest;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

use crate::DONE_MESSAGE;
use crate::mantle::{bedrock_runtime_openai_url, sign_bedrock_json_post};

pub struct MantleChatCompletionsClient {
    http_client: reqwest::Client,
    region: String,
    credentials_provider: SharedCredentialsProvider,
}

impl MantleChatCompletionsClient {
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
        let body = serde_json::to_vec(&shape_chat_completions_body(&request)?)?;
        let url = bedrock_runtime_openai_url(&self.region, "/openai/v1/chat/completions");
        let signed_headers =
            sign_bedrock_json_post(&self.credentials_provider, &self.region, &url, &body).await?;

        info!(
            "Sending OpenAI request to Bedrock Mantle for model: {}",
            request.model
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
            anyhow::bail!("Bedrock Mantle request failed ({}): {}", status, error_body);
        }

        info!("Successfully connected to Bedrock Mantle stream");
        Ok(process_chat_completions_stream(response, usage_callback))
    }
}

/// Re-emits the upstream OpenAI SSE stream as axum SSE events, forwarding each
/// `data:` payload verbatim and logging token usage from the final chunk.
fn process_chat_completions_stream<F>(
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
                                let _ =
                                    event_tx.send(Ok(Event::default().data(DONE_MESSAGE))).await;
                                return;
                            }

                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload)
                                && let Some(usage) = value.get("usage").filter(|u| !u.is_null())
                                && let Some(token_usage) = parse_token_usage(usage)
                            {
                                usage_callback(&token_usage);
                            }

                            if event_tx
                                .send(Ok(Event::default().data(payload)))
                                .await
                                .is_err()
                            {
                                info!("SSE client disconnected, stopping Mantle stream");
                                return;
                            }
                        }
                    }
                }
                Some(Err(e)) => {
                    let _ = event_tx
                        .send(Err(anyhow!("Mantle stream receive error: {}", e)))
                        .await;
                    return;
                }
                None => break,
            }
        }

        // Upstream closed without an explicit [DONE]; emit one so clients finish.
        info!("Mantle stream ended, sending DONE message");
        let _ = event_tx.send(Ok(Event::default().data(DONE_MESSAGE))).await;
    });

    ReceiverStream::new(event_rx).boxed()
}

fn shape_chat_completions_body(
    request: &ChatCompletionsRequest,
) -> anyhow::Result<serde_json::Value> {
    let mut body = serde_json::to_value(request)?;
    body["stream"] = true.into();
    body["stream_options"] = serde_json::json!({ "include_usage": true });
    Ok(body)
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

    #[test]
    fn body_forces_stream_and_usage() {
        let request = base_request(serde_json::json!({ "stream": false }));
        let body = shape_chat_completions_body(&request).unwrap();
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(
            body["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
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
