use async_trait::async_trait;
use axum::response::sse::{Event, Sse};
use futures::Stream;
use futures::stream::StreamExt;
use request::ChatCompletionsRequest;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use response::{ChatCompletionsResponse, Usage};
use std::env;
use std::sync::Arc;
use tracing::{debug, error, info, trace};

use crate::providers::ChatCompletionsProvider;

const DONE_MESSAGE: &str = "[DONE]";
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

// OpenAI provider implementation
pub struct OpenAIChatCompletionsProvider {
    client: reqwest::Client,
    api_key: String,
}

impl OpenAIChatCompletionsProvider {
    pub async fn new() -> anyhow::Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;

        let client = reqwest::Client::builder().build()?;

        Ok(Self { client, api_key })
    }

    fn create_sse_event(response: &ChatCompletionsResponse) -> anyhow::Result<Event> {
        match serde_json::to_string(response) {
            Ok(data) => Ok(Event::default().data(data)),
            Err(e) => Err(anyhow::anyhow!("Failed to serialize response: {}", e)),
        }
    }
}

#[async_trait]
impl ChatCompletionsProvider for OpenAIChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<Sse<impl Stream<Item = anyhow::Result<Event>>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        debug!(
            "Processing OpenAI chat completions request for model: {}",
            request.model
        );

        // Since ChatCompletionsRequest is already in OpenAI format, we just need to send it directly
        // with minor adjustments such as ensuring stream is true
        let mut request_data = serde_json::to_value(request)?;

        // Ensure stream is set to true
        if let Some(obj) = request_data.as_object_mut() {
            obj.insert("stream".to_string(), serde_json::Value::Bool(true));
        }

        debug!("Prepared request for OpenAI API");

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|e| anyhow::anyhow!("Invalid API key format: {}", e))?,
        );

        info!("Sending request to OpenAI API");
        let response = self
            .client
            .post(OPENAI_API_URL)
            .headers(headers)
            .json(&request_data)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "OpenAI API error: HTTP {}: {}",
                status,
                error_text
            ));
        }

        let stream = response.bytes_stream();
        let usage_callback = Arc::new(usage_callback);

        let stream = async_stream::stream! {
            let mut line_buffer = String::new();
            let mut bytes_stream = stream;

            while let Some(chunk_result) = bytes_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk = chunk.to_vec(); // Convert Bytes to Vec<u8> to make it Sized
                        // Convert chunk to string and append to buffer
                        if let Ok(chunk_str) = std::str::from_utf8(&chunk) {
                            line_buffer.push_str(chunk_str);

                            // Process complete lines
                            while let Some(pos) = line_buffer.find('\n') {
                                let line = line_buffer[..pos].trim().to_string();
                                let remainder = line_buffer[pos + 1..].to_string();
                                line_buffer = remainder;

                                if line.is_empty() {
                                    continue;
                                }

                                if line == "data: [DONE]" {
                                    debug!("Received DONE from OpenAI");
                                    yield Ok(Event::default().data(DONE_MESSAGE));
                                    continue;
                                }

                                if let Some(data) = line.strip_prefix("data: ") {
                                    // Directly deserialize to ChatCompletionsResponse
                                    match serde_json::from_str::<ChatCompletionsResponse>(data) {
                                        Ok(response) => {
                                            // If usage information is available, call the usage callback
                                            if let Some(usage) = &response.usage {
                                                usage_callback(usage);
                                            }

                                            match Self::create_sse_event(&response) {
                                                Ok(event) => {
                                                    trace!("Created SSE event from OpenAI chunk");
                                                    yield Ok(event);
                                                },
                                                Err(e) => {
                                                    error!("Failed to create SSE event: {}", e);
                                                    yield Err(e);
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            error!("Failed to parse OpenAI chunk: {}", e);
                                            error!("Raw chunk: {}", data);
                                            yield Err(anyhow::anyhow!("Failed to parse OpenAI chunk: {}", e));
                                        }
                                    }
                                }
                            }
                        }
                    },
                    Err(e) => {
                        let err_string = e.to_string();
                        error!("Error receiving from OpenAI stream: {}", err_string);
                        yield Err(anyhow::anyhow!("Stream receive error: {}", err_string));
                    }
                }
            }

            info!("OpenAI stream completed");
        };

        Ok(Sse::new(stream))
    }
}
