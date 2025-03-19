use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use axum::response::sse::Event;
use chrono::offset::Utc;
use futures::stream::{BoxStream, StreamExt};
use request::ChatCompletionsRequest;
use response::{
    ChatCompletionsResponse, Usage, converse_stream_output_to_chat_completions_response_builder,
};
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use tracing::{debug, error, info, trace};
use uuid::Uuid;

use crate::ProcessChatCompletionsRequest;
use crate::bedrock::{
    BedrockChatCompletion, process_chat_completions_request_to_bedrock_chat_completion,
};

const DONE_MESSAGE: &str = "[DONE]";

/// OpenAI API compatible error structure
#[derive(Debug, Serialize)]
struct ApiErrorResponse {
    error: ApiError,
}

#[derive(Debug, Serialize)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    param: Option<String>,
    code: Option<String>,
}

#[async_trait]
pub trait ChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, Result<Event, Infallible>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static;
}

pub struct BedrockChatCompletionsProvider {}

impl BedrockChatCompletionsProvider {
    pub async fn new() -> Self {
        Self {}
    }
}

impl ProcessChatCompletionsRequest<BedrockChatCompletion> for BedrockChatCompletionsProvider {
    fn process_chat_completions_request(
        &self,
        request: &ChatCompletionsRequest,
    ) -> BedrockChatCompletion {
        process_chat_completions_request_to_bedrock_chat_completion(request)
    }
}

fn create_sse_event(response: &ChatCompletionsResponse) -> Result<Event, String> {
    match serde_json::to_string(response) {
        Ok(data) => Ok(Event::default().data(data)),
        Err(e) => Err(format!("Serialization error: {}", e)),
    }
}

#[async_trait]
impl ChatCompletionsProvider for BedrockChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, Result<Event, Infallible>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        debug!(
            "Processing chat completions request for model: {}",
            request.model
        );
        let bedrock_chat_completion = self.process_chat_completions_request(&request);
        info!(
            "Processed request to Bedrock format with {} messages",
            bedrock_chat_completion.messages.len()
        );

        debug!("Loading AWS config");
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending request to Bedrock API for model: {}",
            bedrock_chat_completion.model_id
        );
        let mut stream = client
            .converse_stream()
            .model_id(&bedrock_chat_completion.model_id)
            .set_system(Some(bedrock_chat_completion.system_content_blocks))
            .set_messages(Some(bedrock_chat_completion.messages))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Connection error: {}", e))?
            .stream;
        info!("Successfully connected to Bedrock stream");

        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();
        debug!("Created response with id: {}", id);

        let usage_callback = Arc::new(usage_callback);

        let stream = async_stream::stream! {
            trace!("Starting to process stream");
            loop {
                match stream.recv().await {
                    Ok(Some(output)) => {
                        trace!("Received output from Bedrock stream");
                        let usage_callback = usage_callback.clone();
                        let builder = converse_stream_output_to_chat_completions_response_builder(&output, usage_callback);
                        let response = builder
                            .id(Some(id.clone()))
                            .created(Some(created))
                            .build();

                        // Try to create an SSE event, or fall back to an error event
                        let event = match create_sse_event(&response) {
                            Ok(event) => {
                                trace!("Created SSE event");
                                event
                            },
                            Err(e) => {
                                error!("Failed to create SSE event: {}", e);

                                // Create an error event directly
                                let error_message = e.clone(); // Clone the error message
                                let api_error = ApiErrorResponse {
                                    error: ApiError {
                                        message: error_message,
                                        error_type: "server_error".to_string(),
                                        param: None,
                                        code: Some("serialization_error".to_string()),
                                    },
                                };

                                // Attempt to serialize the error
                                match serde_json::to_string(&api_error) {
                                    Ok(json) => Event::default().data(json),
                                    Err(_) => {
                                        // Fallback if serialization fails
                                        let message = e.replace("\"", "\\\"");
                                        Event::default().data(format!(
                                            "{{\"error\":{{\"message\":\"{message}\",\"type\":\"server_error\",\"code\":\"serialization_error\"}}}}"
                                        ))
                                    }
                                }
                            }
                        };

                        yield Ok::<_, Infallible>(event);
                    }
                    Ok(None) => {
                        debug!("Stream completed");
                        break;
                    }
                    Err(e) => {
                        error!("Error receiving from stream: {}", e);

                        // Create an OpenAI-compatible error response
                        let error_message = format!("Stream receive error: {}", e);
                        let api_error = ApiErrorResponse {
                            error: ApiError {
                                message: error_message,
                                error_type: "server_error".to_string(),
                                param: None,
                                code: Some("stream_receive_error".to_string()),
                            },
                        };

                        // Serialize the error to JSON and yield it as an event
                        if let Ok(json) = serde_json::to_string(&api_error) {
                            yield Ok::<_, Infallible>(Event::default().data(json));
                        } else {
                            // Fallback if serialization fails
                            yield Ok::<_, Infallible>(Event::default().data(
                                format!("{{\"error\":{{\"message\":\"Stream error: {}\",\"type\":\"server_error\",\"code\":\"stream_receive_error\"}}}}",
                                e.to_string().replace("\"", "\\\""))
                            ));
                        }

                        // Break the stream after sending the error
                        break;
                    }
                }
            }

            info!("Stream finished, sending DONE message");
            yield Ok::<_, Infallible>(Event::default().data(DONE_MESSAGE));
        };

        Ok(stream.boxed())
    }
}
