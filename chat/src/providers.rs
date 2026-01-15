use crate::bedrock::process_chat_completions_request_to_bedrock_chat_completion;
use crate::{create_anthropic_sse_events, create_sse_event, DONE_MESSAGE};
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver;
use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use aws_sdk_bedrockruntime::Client;
use axum::response::sse::Event;
use chrono::offset::Utc;
use futures::stream::{BoxStream, StreamExt};
use request::ChatCompletionsRequest;
use response::{converse_stream_output_to_chat_completions_response_builder, Usage};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

/// Generic stream processor for OpenAI-format SSE
async fn process_bedrock_stream_openai(
    mut stream: EventReceiver<
        aws_sdk_bedrockruntime::types::ConverseStreamOutput,
        ConverseStreamOutputError,
    >,
    id: String,
    created: i64,
    usage_callback: Arc<dyn Fn(&Usage) + Send + Sync>,
) -> BoxStream<'static, anyhow::Result<Event>> {
    let stream = async_stream::stream! {
        loop {
            match stream.recv().await {
                Ok(Some(output)) => {
                    let usage_callback = usage_callback.clone();
                    if let Some(builder) = converse_stream_output_to_chat_completions_response_builder(&output, usage_callback) {
                        let response = builder
                            .id(Some(id.clone()))
                            .created(Some(created))
                            .build();

                        match create_sse_event(&response) {
                            Ok(event) => {
                                yield Ok(event);
                            },
                            Err(e) => {
                                yield Err(e);
                            }
                        }
                    }
                }
                Ok(None) => {
                    break;
                }
                Err(e) => {
                    yield Err(anyhow::anyhow!(
                        "Stream receive error: {}",
                        e
                    ));
                }
            }
        }

        info!("Stream finished, sending DONE message");
        yield Ok(Event::default().data(DONE_MESSAGE));
    };

    stream.boxed()
}

/// Generic stream processor for Anthropic-format SSE
async fn process_bedrock_stream_anthropic(
    mut stream: EventReceiver<
        aws_sdk_bedrockruntime::types::ConverseStreamOutput,
        ConverseStreamOutputError,
    >,
    id: String,
    created: i64,
    usage_callback: Arc<dyn Fn(&Usage) + Send + Sync>,
) -> BoxStream<'static, anyhow::Result<Event>> {
    let stream = async_stream::stream! {
        loop {
            match stream.recv().await {
                Ok(Some(output)) => {
                    let usage_callback = usage_callback.clone();
                    if let Some(builder) = converse_stream_output_to_chat_completions_response_builder(&output, usage_callback) {
                        let response = builder
                            .id(Some(id.clone()))
                            .created(Some(created))
                            .build();

                        // Convert to Anthropic format and stream events
                        for event_result in create_anthropic_sse_events(&response) {
                            match event_result {
                                Ok((anthropic_event, _)) => {
                                    yield Ok(anthropic_event.event);
                                }
                                Err(e) => {
                                    yield Err(e);
                                }
                            }
                        }
                    }
                }
                Ok(None) => {
                    break;
                }
                Err(e) => {
                    yield Err(anyhow::anyhow!(
                        "Stream receive error: {}",
                        e
                    ));
                }
            }
        }

        info!("Stream finished");
    };

    stream.boxed()
}

// Re-export BedrockRequest from request crate
pub use request::BedrockRequest;

/// Trait for converting requests to Bedrock format
pub trait ToBedrockRequest {
    fn to_bedrock_request(&self) -> anyhow::Result<BedrockRequest>;
}

/// Convert ChatCompletionsRequest to Bedrock format
impl ToBedrockRequest for ChatCompletionsRequest {
    fn to_bedrock_request(&self) -> anyhow::Result<BedrockRequest> {
        let bedrock_chat_completion = process_chat_completions_request_to_bedrock_chat_completion(self)?;

        Ok(BedrockRequest {
            model_id: bedrock_chat_completion.model_id,
            messages: bedrock_chat_completion.messages,
            system: bedrock_chat_completion.system_content_blocks,
            inference_config: bedrock_chat_completion.inference_config,
            tool_config: bedrock_chat_completion.tool_config,
            additional_fields: bedrock_chat_completion.additional_model_request_fields,
        })
    }
}

/// Convert AnthropicRequest to Bedrock format
impl ToBedrockRequest for request::AnthropicRequest {
    fn to_bedrock_request(&self) -> anyhow::Result<BedrockRequest> {
        request::anthropic::convert_to_bedrock(self)
    }
}

#[derive(Default)]
pub struct BedrockChatCompletionsProvider;

impl BedrockChatCompletionsProvider {
    pub fn new() -> Self {
        Self
    }

    /// Generic method to stream from Bedrock - OpenAI format
    pub async fn stream_openai<R, U>(
        request: R,
        usage_callback: U,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<Event>>>
    where
        R: ToBedrockRequest,
        U: Fn(&Usage) + Send + Sync + 'static,
    {
        let bedrock_request = request.to_bedrock_request()?;

        info!(
            "Converted request to Bedrock format with {} messages",
            bedrock_request.messages.len()
        );

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending request to Bedrock API for model: {}",
            bedrock_request.model_id
        );

        let mut converse_builder = client
            .converse_stream()
            .model_id(&bedrock_request.model_id)
            .set_system(Some(bedrock_request.system))
            .set_messages(Some(bedrock_request.messages))
            .set_tool_config(bedrock_request.tool_config)
            .set_inference_config(Some(bedrock_request.inference_config));

        if let Some(additional_fields) = bedrock_request.additional_fields {
            converse_builder =
                converse_builder.set_additional_model_request_fields(Some(additional_fields));
        }

        let stream = converse_builder.send().await?.stream;
        info!("Successfully connected to Bedrock stream");

        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();
        let usage_callback = Arc::new(usage_callback);

        Ok(process_bedrock_stream_openai(stream, id, created, usage_callback).await)
    }

    /// Generic method to stream from Bedrock - Anthropic format
    pub async fn stream_anthropic<R, U>(
        request: R,
        usage_callback: U,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<Event>>>
    where
        R: ToBedrockRequest,
        U: Fn(&Usage) + Send + Sync + 'static,
    {
        let bedrock_request = request.to_bedrock_request()?;

        info!(
            "Converted request to Bedrock format with {} messages",
            bedrock_request.messages.len()
        );

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending request to Bedrock API for model: {}",
            bedrock_request.model_id
        );

        let mut converse_builder = client
            .converse_stream()
            .model_id(&bedrock_request.model_id)
            .set_system(Some(bedrock_request.system))
            .set_messages(Some(bedrock_request.messages))
            .set_tool_config(bedrock_request.tool_config)
            .set_inference_config(Some(bedrock_request.inference_config));

        if let Some(additional_fields) = bedrock_request.additional_fields {
            converse_builder =
                converse_builder.set_additional_model_request_fields(Some(additional_fields));
        }

        let stream = converse_builder.send().await?.stream;
        info!("Successfully connected to Bedrock stream");

        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();
        let usage_callback = Arc::new(usage_callback);

        Ok(process_bedrock_stream_anthropic(stream, id, created, usage_callback).await)
    }
}

