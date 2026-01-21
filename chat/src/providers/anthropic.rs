use anthropic_request::V1MessagesRequest;
use anthropic_response::{
    ContentBlockStartData, Delta, MessageDeltaData, MessageStartData, StreamEvent,
    Usage as AnthropicUsage,
};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver;
use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockStart, ConverseStreamOutput, ReasoningContentBlockDelta,
    StopReason, TokenUsage,
};
use axum::response::sse::Event;
use futures::stream::{BoxStream, StreamExt};
use serde::Serialize;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

use crate::bedrock::BedrockChatCompletion;

async fn process_anthropic_stream(
    mut stream: EventReceiver<ConverseStreamOutput, ConverseStreamOutputError>,
    id: String,
    model: String,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
) -> BoxStream<'static, anyhow::Result<Event>> {
    let stream = async_stream::stream! {
        // Minimal state - only for usage tracking as per spec
        let mut usage_tracker = AnthropicUsage::default();

        loop {
            match stream.recv().await {
                Ok(Some(output)) => {
                    match output {
                        ConverseStreamOutput::MessageStart(_) => {
                            info!("📨 MessageStart");
                            let message_start = StreamEvent::MessageStart {
                                message: MessageStartData {
                                    id: id.clone(),
                                    message_type: "message".to_string(),
                                    role: "assistant".to_string(),
                                    content: vec![],
                                    model: model.clone(),
                                    stop_reason: None,
                                    stop_sequence: None,
                                    usage: AnthropicUsage::default(),
                                },
                            };

                            match create_anthropic_sse_event("message_start", &message_start) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }
                        }

                        ConverseStreamOutput::ContentBlockStart(event) => {
                            info!("🔵 ContentBlockStart at index {}", event.content_block_index);
                            
                            let content_block = match &event.start {
                                Some(ContentBlockStart::ToolUse(tool_use)) => {
                                    info!("🔧 Tool block: {} (id: {})", 
                                        tool_use.name(), tool_use.tool_use_id());
                                    ContentBlockStartData::ToolUse {
                                        id: tool_use.tool_use_id().to_string(),
                                        name: tool_use.name().to_string(),
                                        input: serde_json::json!({}),
                                    }
                                }
                                _ => {
                                    info!("📝 Text block");
                                    ContentBlockStartData::Text {
                                        text: String::new(),
                                    }
                                }
                            };

                            let event_data = StreamEvent::ContentBlockStart {
                                index: event.content_block_index,
                                content_block,
                            };

                            match create_anthropic_sse_event("content_block_start", &event_data) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }
                        }

                        ConverseStreamOutput::ContentBlockDelta(event) => {
                            // Direct 1:1 mapping - no state needed
                            let delta = match &event.delta {
                                Some(ContentBlockDelta::Text(text)) => {
                                    Some(Delta::TextDelta { text: text.clone() })
                                }
                                Some(ContentBlockDelta::ToolUse(tool_use)) => {
                                    Some(Delta::InputJsonDelta { 
                                        partial_json: tool_use.input.clone() 
                                    })
                                }
                                Some(ContentBlockDelta::ReasoningContent(
                                    ReasoningContentBlockDelta::Text(text)
                                )) => {
                                    info!("💭 Thinking delta at index {}", event.content_block_index);
                                    Some(Delta::ThinkingDelta { 
                                        thinking: text.clone() 
                                    })
                                }
                                Some(ContentBlockDelta::ReasoningContent(
                                    ReasoningContentBlockDelta::Signature(sig)
                                )) => {
                                    info!("✍️ Signature delta at index {}: {}", 
                                        event.content_block_index, sig);
                                    Some(Delta::SignatureDelta { 
                                        signature: sig.clone() 
                                    })
                                }
                                _ => None,
                            };

                            if let Some(delta) = delta {
                                let event_data = StreamEvent::ContentBlockDelta {
                                    index: event.content_block_index,
                                    delta,
                                };

                                match create_anthropic_sse_event("content_block_delta", &event_data) {
                                    Ok(event) => yield Ok(event),
                                    Err(e) => yield Err(e),
                                }
                            }
                        }

                        ConverseStreamOutput::ContentBlockStop(event) => {
                            info!("🔴 ContentBlockStop at index {}", event.content_block_index);
                            
                            let event_data = StreamEvent::ContentBlockStop {
                                index: event.content_block_index,
                            };

                            match create_anthropic_sse_event("content_block_stop", &event_data) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }
                        }

                        ConverseStreamOutput::MessageStop(event) => {
                            let stop_reason = match event.stop_reason {
                                StopReason::EndTurn => "end_turn",
                                StopReason::ToolUse => "tool_use",
                                StopReason::MaxTokens => "max_tokens",
                                StopReason::StopSequence => "stop_sequence",
                                StopReason::ContentFiltered => "content_filtered",
                                _ => "unknown",
                            };

                            info!("🛑 MessageStop: {}", stop_reason);

                            // Send message_delta with usage
                            let message_delta = StreamEvent::MessageDelta {
                                delta: MessageDeltaData {
                                    stop_reason: Some(stop_reason.to_string()),
                                    stop_sequence: None,
                                },
                                usage: usage_tracker.clone(),
                            };

                            match create_anthropic_sse_event("message_delta", &message_delta) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }

                            // Send message_stop
                            let message_stop = StreamEvent::MessageStop;
                            match create_anthropic_sse_event("message_stop", &message_stop) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }
                        }

                        ConverseStreamOutput::Metadata(event) => {
                            if let Some(usage) = &event.usage {
                                info!("📊 Usage: input={}, output={}", 
                                    usage.input_tokens, usage.output_tokens);
                                usage_tracker.input_tokens = usage.input_tokens;
                                usage_tracker.output_tokens = usage.output_tokens;
                                usage_callback(usage);
                            }
                            // No SSE event emitted for Metadata
                        }

                        other => {
                            tracing::warn!("⚠️ Unhandled Bedrock event: {:?}", other);
                        }
                    }
                }
                Ok(None) => {
                    info!("✅ Stream ended normally");
                    break;
                }
                Err(e) => {
                    tracing::error!("❌ Stream error: {}", e);
                    yield Err(anyhow::anyhow!("Bedrock stream error: {}", e));
                    break;
                }
            }
        }
    };

    stream.boxed()
}

fn create_anthropic_sse_event(event_name: &str, data: &impl Serialize) -> anyhow::Result<Event> {
    let json = serde_json::to_string(data)?;
    info!("Creating SSE event '{}' with data: {}", event_name, json);
    Ok(Event::default().event(event_name).data(json))
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
        let bedrock_chat_completion = BedrockChatCompletion::try_from(&request)?;
        info!(
            "Processed Anthropic request to Bedrock format with {} messages",
            bedrock_chat_completion.messages.len()
        );

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending Anthropic request to Bedrock API for model: {}",
            bedrock_chat_completion.model_id
        );
        info!(
            "System blocks count: {}",
            bedrock_chat_completion.system_content_blocks.len()
        );
        info!("Messages count: {}", bedrock_chat_completion.messages.len());
        info!(
            "Inference config max_tokens: {:?}",
            bedrock_chat_completion.inference_config.max_tokens()
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

        info!("About to send request to Bedrock...");
        let result = converse_builder.send().await;

        match result {
            Ok(response) => {
                info!("Successfully connected to Bedrock stream for Anthropic format");
                let stream = response.stream;

                let id = format!("msg_{}", Uuid::new_v4());

                let usage_callback = Arc::new(usage_callback);

                Ok(process_anthropic_stream(stream, id, model, usage_callback).await)
            }
            Err(e) => {
                tracing::error!("Bedrock API error: {:?}", e);
                Err(anyhow::anyhow!("Bedrock API error: {}", e))
            }
        }
    }
}
