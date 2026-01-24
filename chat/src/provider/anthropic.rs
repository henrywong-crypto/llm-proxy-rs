use anthropic_request::V1MessagesRequest;
use anthropic_response::EventConverter;
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver;
use aws_sdk_bedrockruntime::types::{
    ConverseStreamOutput, TokenUsage, error::ConverseStreamOutputError,
};
use axum::response::sse::Event;
use futures::stream::{BoxStream, StreamExt};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

async fn process_bedrock_stream(
    mut stream: EventReceiver<ConverseStreamOutput, ConverseStreamOutputError>,
    id: String,
    model: String,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
) -> BoxStream<'static, anyhow::Result<Event>> {
    let stream = async_stream::stream! {
        let mut converter = EventConverter::new(id, model, usage_callback);

        loop {
            match stream.recv().await {
                Ok(Some(converse_stream_output)) => {
                    if let Some(events) = converter.convert(&converse_stream_output) {
                        for (event_name, event) in events {
                            match serde_json::to_string(&event) {
                                Ok(json) => {
                                    yield Ok(Event::default().event(event_name).data(json));
                                }
                                Err(e) => {
                                    yield Err(anyhow::anyhow!("Failed to serialize event: {}", e));
                                }
                            }
                        }
                    }
                }
                Ok(None) => {
                    break;
                }
                Err(e) => {
                    yield Err(anyhow::anyhow!("Stream receive error: {}", e));
                }
            }
        }

        info!("Bedrock stream finished");
    };

    stream.boxed()
}

#[async_trait]
pub trait V1MessagesProvider {
    async fn v1_messages_stream<F>(
        self,
        request: V1MessagesRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&TokenUsage) + Send + Sync + 'static;
}

pub struct BedrockV1MessagesProvider {}

impl BedrockV1MessagesProvider {
    pub async fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl V1MessagesProvider for BedrockV1MessagesProvider {
    async fn v1_messages_stream<F>(
        self,
        request: V1MessagesRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&TokenUsage) + Send + Sync + 'static,
    {
        let model = request.model.clone();
        let bedrock_chat_completion = crate::bedrock::BedrockChatCompletion::try_from(&request)?;
        info!(
            "Processed Anthropic request to Bedrock format with {} messages",
            bedrock_chat_completion
                .messages
                .as_ref()
                .map_or(0, |m| m.len())
        );

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending Anthropic request to Bedrock API for model: {}",
            bedrock_chat_completion.model_id
        );

        let converse_builder = client
            .converse_stream()
            .model_id(&bedrock_chat_completion.model_id)
            .set_system(bedrock_chat_completion.system_content_blocks)
            .set_messages(bedrock_chat_completion.messages)
            .set_tool_config(bedrock_chat_completion.tool_config)
            .set_inference_config(Some(bedrock_chat_completion.inference_config))
            .set_additional_model_request_fields(
                bedrock_chat_completion.additional_model_request_fields,
            );

        info!("About to send Anthropic request to Bedrock...");
        let result = converse_builder.send().await;

        let stream = match result {
            Ok(response) => {
                info!("Successfully connected to Bedrock stream for Anthropic format");
                let stream = response.stream;

                let id = format!("msg_{}", Uuid::new_v4());
                let usage_callback = Arc::new(usage_callback);

                process_bedrock_stream(stream, id, model, usage_callback).await
            }
            Err(e) => {
                let error_message = format!("{:?}", e);
                let display_message = format!("{}", e);
                tracing::error!("Bedrock API error: {}", display_message);
                info!("Full error details: {}", error_message);
                
                // Always return SSE error event for any Bedrock error
                info!("Converting Bedrock error to SSE error event");
                
                // Map Bedrock errors to Anthropic error types
                // Extract the actual error message from Bedrock
                let bedrock_msg = if let Some(start) = error_message.find("message: Some(\"") {
                    let start = start + 16;
                    if let Some(end) = error_message[start..].find("\")") {
                        error_message[start..start + end].to_string()
                    } else {
                        display_message.clone()
                    }
                } else {
                    display_message.clone()
                };
                
                // Determine error type based on Bedrock error
                let error_type = if error_message.contains("ValidationException") {
                    info!("Mapped to invalid_request_error");
                    "invalid_request_error"
                } else if error_message.contains("ThrottlingException") {
                    info!("Mapped to rate_limit_error");
                    "rate_limit_error"
                } else if error_message.contains("ServiceUnavailableException") {
                    info!("Mapped to overloaded_error");
                    "overloaded_error"
                } else if error_message.contains("AccessDeniedException") {
                    info!("Mapped to permission_error");
                    "permission_error"
                } else if error_message.contains("ResourceNotFoundException") {
                    info!("Mapped to not_found_error");
                    "not_found_error"
                } else if error_message.contains("InternalServerException") {
                    info!("Mapped to api_error");
                    "api_error"
                } else {
                    info!("Mapped to api_error (default)");
                    "api_error"
                };
                
                // Create error JSON in Anthropic format
                use futures::stream;
                let error_json = serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": bedrock_msg
                    }
                });
                info!("Error JSON: type={}, message={}", error_type, bedrock_msg);
                
                // Create a complete streaming response with error
                // According to Anthropic format, we need to send a proper message sequence
                let error_stream = async_stream::stream! {
                    info!("Creating standard Anthropic error response stream");
                    
                    // 1. Send message_start event
                    let message_id = format!("msg_{}", Uuid::new_v4());
                    let message_start = serde_json::json!({
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model.clone(),
                            "stop_reason": null,
                            "stop_sequence": null,
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0
                            }
                        }
                    });
                    let event = Event::default()
                        .event("message_start")
                        .data(message_start.to_string());
                    yield Ok(event);
                    
                    // 2. Send error event
                    info!("Yielding error event");
                    let error_event = Event::default()
                        .event("error")
                        .data(error_json.to_string());
                    yield Ok(error_event);
                    
                    // 3. Send message_stop event
                    let message_stop = serde_json::json!({
                        "type": "message_stop"
                    });
                    let event = Event::default()
                        .event("message_stop")
                        .data(message_stop.to_string());
                    yield Ok(event);
                    
                    info!("Error response stream completed");
                };
                
                info!("Created SSE error stream, returning");
                error_stream.boxed()
            }
        };

        Ok(stream)
    }
}
