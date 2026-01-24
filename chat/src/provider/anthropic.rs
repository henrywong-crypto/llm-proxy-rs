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
                
                // Check if it's a token limit error
                let (error_type, error_msg) = if error_message.contains("Input is too long")
                    || error_message.contains("ValidationException")
                {
                    info!("Detected token limit error - returning specific message");
                    (
                        "invalid_request_error",
                        "Input is too long for the requested model. Please reduce the message history, system prompt length, or max_tokens parameter."
                    )
                } else {
                    info!("Generic error - returning general message");
                    ("api_error", "An error occurred while processing your request")
                };
                
                // Return a stream with a single error event in Anthropic format
                use futures::stream;
                let error_json = serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": error_msg
                    }
                });
                info!("Error JSON: {}", error_json.to_string());
                
                // Create an async stream that yields the error event
                let error_stream = async_stream::stream! {
                    info!("Yielding error event in stream");
                    let error_event = Event::default().event("error").data(error_json.to_string());
                    yield Ok(error_event);
                    info!("Error event yielded successfully");
                };
                
                info!("Created SSE error stream, returning");
                error_stream.boxed()
            }
        };

        Ok(stream)
    }
}
