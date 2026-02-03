use anthropic_request::{V1MessagesCountTokensRequest, V1MessagesRequest};
use anthropic_response::EventConverter;
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver;
use aws_sdk_bedrockruntime::types::{
    ConverseStreamOutput, ConverseTokensRequest, CountTokensInput, SystemContentBlock, TokenUsage,
    error::ConverseStreamOutputError,
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

    async fn count_tokens(&self, request: &V1MessagesCountTokensRequest) -> anyhow::Result<i32>;
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

        match result {
            Ok(response) => {
                info!("Successfully connected to Bedrock stream for Anthropic format");
                let stream = response.stream;

                let id = format!("msg_{}", Uuid::new_v4());
                let usage_callback = Arc::new(usage_callback);

                Ok(process_bedrock_stream(stream, id, model, usage_callback).await)
            }
            Err(e) => {
                tracing::error!("Bedrock API error: {:?}", e);
                Err(anyhow::anyhow!("Bedrock API error: {}", e))
            }
        }
    }

    async fn count_tokens(&self, request: &V1MessagesCountTokensRequest) -> anyhow::Result<i32> {
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        // Convert messages to Bedrock format
        let messages: Option<Vec<aws_sdk_bedrockruntime::types::Message>> =
            Option::try_from(&request.messages)?;

        // Convert system prompt to Bedrock format
        let system: Option<Vec<SystemContentBlock>> = request
            .system
            .as_ref()
            .map(Vec::<SystemContentBlock>::try_from)
            .transpose()?;

        // Build the converse tokens request
        let converse_request = ConverseTokensRequest::builder()
            .set_messages(messages)
            .set_system(system)
            .build();

        let count_input = CountTokensInput::Converse(converse_request);

        info!(
            "Counting tokens for model: {} via Bedrock API",
            request.model
        );

        let result = client
            .count_tokens()
            .model_id(&request.model)
            .input(count_input)
            .send()
            .await?;

        Ok(result.input_tokens)
    }
}
