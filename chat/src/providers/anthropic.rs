use anthropic_request::V1MessagesRequest;
use anthropic_response::{
    ContentBlockStartData, Delta, MessageDeltaData, MessageStartData, StreamEvent,
    Usage as AnthropicUsage,
};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::config::timeout::TimeoutConfig;
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
        // Track which content blocks have started to synthesize missing ContentBlockStart events
        let mut seen_blocks = std::collections::HashSet::new();
        
        loop {
            match stream.recv().await {
                Ok(Some(output)) => {
                    info!("Received Bedrock event for Anthropic: {:?}", output);
                    // // Log all event types including unknown ones
                    // let event_type = match &output {
                    //     ConverseStreamOutput::MessageStart(_) => "MessageStart",
                    //     ConverseStreamOutput::ContentBlockStart(_) => "ContentBlockStart",
                    //     ConverseStreamOutput::ContentBlockDelta(_) => "ContentBlockDelta",
                    //     ConverseStreamOutput::ContentBlockStop(_) => "ContentBlockStop",
                    //     ConverseStreamOutput::MessageStop(_) => "MessageStop",
                    //     ConverseStreamOutput::Metadata(_) => "Metadata",
                    //     _ => "Unknown",
                    // };
                    // info!("Event type: {}", event_type);
                    match &output {
                        ConverseStreamOutput::MessageStart(_event) => {
                            // info!("Processing MessageStart event");
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
                            // info!("Processing ContentBlockStart event at index {}", event.content_block_index);
                            seen_blocks.insert(event.content_block_index);
                            
                            let content_block = match &event.start {
                                Some(ContentBlockStart::ToolUse(tool_use)) => {
                                    let tool_id = tool_use.tool_use_id().to_string();
                                    let tool_name = tool_use.name().to_string();
                                    // info!("Tool use block: id='{}', name='{}'", tool_id, tool_name);
                                    ContentBlockStartData::ToolUse {
                                        id: tool_id,
                                        name: tool_name,
                                        input: serde_json::json!({}),
                                    }
                                }
                                _ => ContentBlockStartData::Text {
                                    text: String::new(),
                                },
                            };

                            let event_data = StreamEvent::ContentBlockStart {
                                index: event.content_block_index,
                                content_block: content_block.clone(),
                            };

                            // info!("Serialized content_block_start: {:?}", serde_json::to_string(&event_data));

                            match create_anthropic_sse_event("content_block_start", &event_data) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }
                        }

                        ConverseStreamOutput::ContentBlockDelta(event) => {
                            // info!("Processing ContentBlockDelta event at index {}", event.content_block_index);
                            // info!("Delta content: {:?}", event.delta);
                            
                            // Bedrock often omits ContentBlockStart for text/thinking blocks
                            // Synthesize it if we haven't seen this block index yet
                            if !seen_blocks.contains(&event.content_block_index) {
                                seen_blocks.insert(event.content_block_index);
                                
                                // Determine block type from delta
                                let content_block = match &event.delta {
                                    Some(ContentBlockDelta::ReasoningContent(_)) => {
                                        ContentBlockStartData::Thinking {
                                            thinking: String::new(),
                                        }
                                    }
                                    _ => {
                                        ContentBlockStartData::Text {
                                            text: String::new(),
                                        }
                                    }
                                };
                                
                                let start_event = StreamEvent::ContentBlockStart {
                                    index: event.content_block_index,
                                    content_block,
                                };
                                
                                match create_anthropic_sse_event("content_block_start", &start_event) {
                                    Ok(evt) => yield Ok(evt),
                                    Err(e) => yield Err(e),
                                }
                            }

                            let delta = match &event.delta {
                                Some(ContentBlockDelta::Text(text)) => {
                                    // info!("Text delta content: '{}'", text);
                                    Some(Delta::TextDelta {
                                        text: text.clone(),
                                    })
                                },
                                Some(ContentBlockDelta::ToolUse(tool_use)) => {
                                    Some(Delta::InputJsonDelta {
                                        partial_json: tool_use.input.clone(),
                                    })
                                }
                                Some(ContentBlockDelta::ReasoningContent(
                                    ReasoningContentBlockDelta::Text(text),
                                )) => {
                                    // info!("⚠️ THINKING DELTA RECEIVED - content: '{}'", text);
                                    Some(Delta::ThinkingDelta {
                                        thinking: text.clone(),
                                    })
                                },
                                Some(ContentBlockDelta::ReasoningContent(
                                    ReasoningContentBlockDelta::Signature(signature),
                                )) => {
                                    // info!("⚠️ SIGNATURE DELTA RECEIVED from Bedrock: '{}'", signature);
                                    // Emit signature_delta immediately - no need to store
                                    Some(Delta::SignatureDelta {
                                        signature: signature.clone(),
                                    })
                                },
                                _ => None,
                            };

                            if let Some(delta) = delta {
                                let event_data = StreamEvent::ContentBlockDelta {
                                    index: event.content_block_index,
                                    delta,
                                };

                                info!("⚠️ Yielding content_block_delta for index {}", event.content_block_index);
                                match create_anthropic_sse_event("content_block_delta", &event_data) {
                                    Ok(event) => yield Ok(event),
                                    Err(e) => yield Err(e),
                                }
                            } else {
                                info!("⚠️ Delta is None for index {}", event.content_block_index);
                            }
                        }

                        ConverseStreamOutput::ContentBlockStop(event) => {
                            // info!("ContentBlockStop event: {:?}", event);

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
                                _ => "unknown",
                            };

                            info!("⚠️ MessageStop received with stop_reason: {}", stop_reason);

                            // Send message_delta immediately without usage (will be 0)
                            let message_delta = StreamEvent::MessageDelta {
                                delta: MessageDeltaData {
                                    stop_reason: Some(stop_reason.to_string()),
                                    stop_sequence: None,
                                },
                                usage: AnthropicUsage::default(),
                            };

                            match create_anthropic_sse_event("message_delta", &message_delta) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }

                            let message_stop = StreamEvent::MessageStop;
                            match create_anthropic_sse_event("message_stop", &message_stop) {
                                Ok(event) => {
                                    info!("⚠️ Yielding message_stop event");
                                    yield Ok(event)
                                },
                                Err(e) => yield Err(e),
                            }
                        }

                        ConverseStreamOutput::Metadata(event) => {
                            info!("⚠️ Metadata event received");
                            if let Some(usage) = &event.usage {
                                info!("⚠️ Usage: input={}, output={}", usage.input_tokens, usage.output_tokens);
                                // Call usage callback
                                usage_callback(usage);
                            }
                            // Metadata comes after MessageStop, just log it
                            // We already sent message_delta and message_stop
                        }

                        _ => {
                            // info!("Unhandled event type: {:?}", std::mem::discriminant(&output));
                        }
                    }
                }
                Ok(None) => {
                    info!("⚠️ Stream finished - received None from Bedrock");
                    break;
                }
                Err(e) => {
                    tracing::error!("Stream receive error: {:?}", e);
                    yield Err(anyhow::anyhow!("Stream receive error: {}", e));
                    break;
                }
            }
        }
    };

    stream.boxed()
}

fn create_anthropic_sse_event(event_name: &str, data: &impl Serialize) -> anyhow::Result<Event> {
    let json = serde_json::to_string(data)?;
    // info!("Creating SSE event '{}' with data: {}", event_name, json);
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
        
        // For streaming, disable operation timeout (keep connection alive as long as data flows)
        // Only timeout if no data is received (read timeout handles this)
        let timeout_config = TimeoutConfig::builder()
            .operation_timeout(std::time::Duration::from_secs(3600)) // 1 hour max
            .build();
        
        let bedrock_config = aws_sdk_bedrockruntime::config::Builder::from(&config)
            .timeout_config(timeout_config)
            .build();
        
        let client = Client::from_conf(bedrock_config);

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
