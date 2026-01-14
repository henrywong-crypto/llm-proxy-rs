use crate::{
    DONE_MESSAGE, ProcessChatCompletionsRequest,
    bedrock::{BedrockChatCompletion, process_chat_completions_request_to_bedrock_chat_completion},
    create_anthropic_sse_events, create_sse_event,
};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver;
use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use axum::response::sse::Event;
use chrono::offset::Utc;
use futures::stream::{BoxStream, StreamExt};
use request::ChatCompletionsRequest;
use response::{Usage, converse_stream_output_to_chat_completions_response_builder};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

async fn process_bedrock_stream(
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
        let mut has_sent_content_block_start_for_text = false;

        loop {
            match stream.recv().await {
                Ok(Some(output)) => {
                    eprintln!("DEBUG BEDROCK: Received stream output: {:?}", output);
                    let usage_callback = usage_callback.clone();
                    if let Some(builder) = converse_stream_output_to_chat_completions_response_builder(&output, usage_callback) {
                        let response = builder
                            .id(Some(id.clone()))
                            .created(Some(created))
                            .build();

                        // Convert to Anthropic format and stream events
                        for event_result in create_anthropic_sse_events(&response) {
                            match event_result {
                                Ok((anthropic_event, is_text_delta)) => {
                                    // Bedrock doesn't send ContentBlockStart for text, only for tools
                                    // Inject content_block_start before the first text delta
                                    eprintln!("DEBUG SSE: Event type={}, is_text_delta={}, has_sent={}",
                                        anthropic_event.event_type, is_text_delta, has_sent_content_block_start_for_text);

                                    if anthropic_event.event_type == "content_block_delta"
                                        && is_text_delta
                                        && !has_sent_content_block_start_for_text {
                                        eprintln!("DEBUG SSE: Injecting content_block_start for text");
                                        let start_event = Event::default()
                                            .event("content_block_start")
                                            .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#);
                                        eprintln!("DEBUG SSE: Yielding injected content_block_start");
                                        yield Ok(start_event);
                                        has_sent_content_block_start_for_text = true;
                                    }
                                    eprintln!("DEBUG SSE: Yielding event type={}", anthropic_event.event_type);
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

#[async_trait]
pub trait ChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static;

    async fn anthropic_to_bedrock_stream<F>(
        self,
        request: request::AnthropicRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
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
    ) -> anyhow::Result<BedrockChatCompletion> {
        process_chat_completions_request_to_bedrock_chat_completion(request)
    }
}

#[async_trait]
impl ChatCompletionsProvider for BedrockChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        let bedrock_chat_completion = self.process_chat_completions_request(&request)?;
        info!(
            "Processed request to Bedrock format with {} messages",
            bedrock_chat_completion.messages.len()
        );

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending request to Bedrock API for model: {}",
            bedrock_chat_completion.model_id
        );

        let converse_builder = client
            .converse_stream()
            .model_id(&bedrock_chat_completion.model_id)
            .set_system(Some(bedrock_chat_completion.system_content_blocks))
            .set_messages(Some(bedrock_chat_completion.messages))
            .set_tool_config(bedrock_chat_completion.tool_config)
            .set_inference_config(Some(bedrock_chat_completion.inference_config))
            .set_additional_model_request_fields(
                bedrock_chat_completion.additional_model_request_fields,
            );

        let stream = converse_builder.send().await?.stream;
        info!("Successfully connected to Bedrock stream");

        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();

        let usage_callback = Arc::new(usage_callback);

        Ok(process_bedrock_stream(stream, id, created, usage_callback).await)
    }

    async fn anthropic_to_bedrock_stream<F>(
        self,
        request: request::AnthropicRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        eprintln!("DEBUG: anthropic_to_bedrock_stream called");

        // Direct conversion from Anthropic to Bedrock
        let bedrock_request = match request.to_bedrock_converse() {
            Ok(req) => {
                eprintln!("DEBUG: Successfully converted to Bedrock format");
                req
            }
            Err(e) => {
                eprintln!("DEBUG: ERROR - Failed to convert to Bedrock format: {}", e);
                return Err(e);
            }
        };

        info!(
            "Converted Anthropic request directly to Bedrock format with {} messages",
            bedrock_request.messages.len()
        );

        eprintln!("DEBUG: Loading AWS config");
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending request to Bedrock API for model: {}",
            bedrock_request.model_id
        );
        eprintln!("DEBUG: Building converse_stream request");

        let converse_builder = client
            .converse_stream()
            .model_id(&bedrock_request.model_id)
            .set_system(Some(bedrock_request.system))
            .set_messages(Some(bedrock_request.messages))
            .set_tool_config(bedrock_request.tool_config)
            .set_inference_config(Some(bedrock_request.inference_config));

        eprintln!("DEBUG: Sending request to Bedrock API");
        let response = match converse_builder.send().await {
            Ok(resp) => {
                eprintln!("DEBUG: Successfully received response from Bedrock");
                resp
            }
            Err(e) => {
                eprintln!("DEBUG: ERROR - Bedrock API call failed: {:?}", e);
                eprintln!("DEBUG: ERROR - Error details: {}", e);
                return Err(anyhow::anyhow!("Bedrock API error: {}", e));
            }
        };

        let stream = response.stream;
        info!("Successfully connected to Bedrock stream");

        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();

        let usage_callback = Arc::new(usage_callback);

        Ok(process_bedrock_stream_anthropic(stream, id, created, usage_callback).await)
    }
}
