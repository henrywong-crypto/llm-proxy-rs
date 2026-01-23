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
        let mut thinking_blocks_with_signatures = std::collections::HashSet::new();
        let mut pending_event: Option<ConverseStreamOutput> = None;

        loop {
            // Get the next event (either from pending or from stream)
            let output = if let Some(event) = pending_event.take() {
                event
            } else {
                match stream.recv().await {
                    Ok(Some(output)) => output,
                    Ok(None) => {
                        info!("Anthropic stream finished naturally");
                        break;
                    }
                    Err(e) => {
                        info!("Stream receive error: {}", e);
                        let error_event = StreamEvent::Error {
                            error: anthropic_response::ErrorData {
                                error_type: "stream_error".to_string(),
                                message: format!("Stream receive error: {}", e),
                            },
                        };
                        match create_anthropic_sse_event("error", &error_event) {
                            Ok(event) => yield Ok(event),
                            Err(err) => yield Err(err),
                        }
                        break;
                    }
                }
            };

            info!("Received Bedrock event for Anthropic: {:?}", std::mem::discriminant(&output));
            let event_type = match &output {
                ConverseStreamOutput::MessageStart(_) => "MessageStart",
                ConverseStreamOutput::ContentBlockStart(_) => "ContentBlockStart",
                ConverseStreamOutput::ContentBlockDelta(_) => "ContentBlockDelta",
                ConverseStreamOutput::ContentBlockStop(_) => "ContentBlockStop",
                ConverseStreamOutput::MessageStop(_) => "MessageStop",
                ConverseStreamOutput::Metadata(_) => "Metadata",
                _ => "Unknown",
            };
            info!("Event type: {}", event_type);

            match &output {
                        ConverseStreamOutput::MessageStart(_event) => {
                            info!("Processing MessageStart event");
                            // Usage information comes from Metadata events, not MessageStart
                            let message_start = StreamEvent::MessageStart {
                                message: MessageStartData {
                                    id: id.clone(),
                                    message_type: "message".to_string(),
                                    role: "assistant".to_string(),
                                    content: vec![],
                                    model: model.clone(),
                                    stop_reason: None,
                                    stop_sequence: None,
                                    usage: AnthropicUsage::default(), // Initial usage is 0/0
                                    container: None, // Bedrock doesn't provide container info
                                },
                            };

                            match create_anthropic_sse_event("message_start", &message_start) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }

                            // Peek at next event to determine if we need to send content_block_start
                            match stream.recv().await {
                                Ok(Some(next)) => {
                                    match &next {
                                        ConverseStreamOutput::ContentBlockDelta(delta_event) => {
                                            info!("Peeking next event after MessageStart: ContentBlockDelta at index {}", delta_event.content_block_index);

                                            // Synthesize ContentBlockStart based on delta type
                                            let content_block = match &delta_event.delta {
                                                Some(ContentBlockDelta::ReasoningContent(_)) => {
                                                    info!("⚠️ THINKING BLOCK DETECTED - Synthesizing thinking content block start");
                                                    ContentBlockStartData::Thinking {
                                                        thinking: String::new(),
                                                        signature: None,
                                                    }
                                                }
                                                _ => {
                                                    ContentBlockStartData::Text {
                                                        text: String::new(),
                                                    }
                                                }
                                            };

                                            let event_data = StreamEvent::ContentBlockStart {
                                                index: delta_event.content_block_index,
                                                content_block,
                                            };

                                            info!("⚠️ About to send ContentBlockStart SSE event");
                                            match create_anthropic_sse_event("content_block_start", &event_data) {
                                                Ok(event) => {
                                                    info!("⚠️ Successfully created and yielding ContentBlockStart SSE event");
                                                    yield Ok(event)
                                                },
                                                Err(e) => {
                                                    info!("⚠️ ERROR creating ContentBlockStart SSE event: {}", e);
                                                    yield Err(e)
                                                },
                                            }
                                        }
                                        _ => {
                                            info!("Next event after MessageStart is not ContentBlockDelta, it's: {:?}", std::mem::discriminant(&next));
                                        }
                                    }
                                    // Store the next event for processing in the next iteration
                                    pending_event = Some(next);
                                }
                                Ok(None) => break,
                                Err(e) => {
                                    info!("Stream receive error while peeking: {}", e);
                                    break;
                                }
                            }
                        }

                        ConverseStreamOutput::ContentBlockStart(event) => {
                            info!("Processing ContentBlockStart event at index {}", event.content_block_index);
                            let content_block = match &event.start {
                                Some(ContentBlockStart::ToolUse(tool_use)) => {
                                    let tool_id = tool_use.tool_use_id().to_string();
                                    let tool_name = tool_use.name().to_string();
                                    info!("Tool use block: id='{}', name='{}'", tool_id, tool_name);
                                    ContentBlockStartData::ToolUse {
                                        id: tool_id,
                                        name: tool_name,
                                        input: serde_json::json!({}),
                                        caller: None, // Bedrock doesn't provide caller info in start event
                                    }
                                }
                                _ => unreachable!(),
                            };

                            let event_data = StreamEvent::ContentBlockStart {
                                index: event.content_block_index,
                                content_block: content_block.clone(),
                            };

                            info!("Serialized content_block_start: {:?}", serde_json::to_string(&event_data));

                            match create_anthropic_sse_event("content_block_start", &event_data) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }
                        }

                        ConverseStreamOutput::ContentBlockDelta(event) => {
                            info!("Processing ContentBlockDelta event at index {}", event.content_block_index);
                            info!("Delta content: {:?}", event.delta);

                            let delta = match &event.delta {
                                Some(ContentBlockDelta::Text(text)) => {
                                    info!("Text delta content: '{}'", text);
                                    Some(Delta::TextDelta {
                                        text: text.clone(),
                                    })
                                },
                                Some(ContentBlockDelta::ToolUse(tool_use)) => {
                                    Some(Delta::InputJsonDelta {
                                        partial_json: tool_use.input.clone(),
                                    })
                                }
                                Some(ContentBlockDelta::ReasoningContent(reasoning_delta)) => {
                                    match reasoning_delta {
                                        ReasoningContentBlockDelta::Text(text) => {
                                            info!("⚠️ THINKING DELTA RECEIVED - content: '{}'", text);
                                            Some(Delta::ThinkingDelta {
                                                thinking: text.clone(),
                                            })
                                        },
                                        ReasoningContentBlockDelta::Signature(sig) => {
                                            info!("⚠️ SIGNATURE DELTA RECEIVED from Bedrock for block {}: '{}'", event.content_block_index, sig);
                                            thinking_blocks_with_signatures.insert(event.content_block_index);
                                            Some(Delta::SignatureDelta {
                                                signature: sig.clone(),
                                            })
                                        },
                                        ReasoningContentBlockDelta::RedactedContent(blob) => {
                                            info!("⚠️ REDACTED THINKING CONTENT RECEIVED for block {}: {} bytes",
                                                  event.content_block_index, blob.as_ref().len());
                                            // RedactedContent is the encrypted thinking data blob
                                            // It should also be followed by a Signature delta
                                            // For now, we don't emit a delta for the redacted content itself
                                            // The signature will come in a separate Signature delta event
                                            thinking_blocks_with_signatures.insert(event.content_block_index);
                                            None
                                        },
                                        _ => {
                                            info!("⚠️ Unknown ReasoningContentBlockDelta variant for block {}", event.content_block_index);
                                            None
                                        }
                                    }
                                },
                                _ => None,
                            };

                            if let Some(delta) = delta {
                                let is_thinking = matches!(delta, Delta::ThinkingDelta { .. });
                                if is_thinking {
                                    info!("⚠️ Creating ContentBlockDelta SSE event for THINKING");
                                }

                                let event_data = StreamEvent::ContentBlockDelta {
                                    index: event.content_block_index,
                                    delta,
                                };

                                match create_anthropic_sse_event("content_block_delta", &event_data) {
                                    Ok(event) => {
                                        if is_thinking {
                                            info!("⚠️ Successfully yielding THINKING content_block_delta SSE event");
                                        } else {
                                            info!("Yielding content_block_delta SSE event");
                                        }
                                        yield Ok(event)
                                    },
                                    Err(e) => {
                                        if is_thinking {
                                            info!("⚠️ ERROR yielding THINKING content_block_delta: {}", e);
                                        }
                                        yield Err(e)
                                    },
                                }
                            }
                        }

                        ConverseStreamOutput::ContentBlockStop(event) => {
                            // Check if this is a thinking block that needs signature
                            if !thinking_blocks_with_signatures.contains(&event.content_block_index) {
                                // Check if this was a thinking block by peeking at what came before
                                // Actually, we can't know for sure without tracking, but we can check
                                // if we ever received a signature for this block
                                // For now, we'll just emit the stop event
                            }

                            // Clean up signature tracking for this block
                            thinking_blocks_with_signatures.remove(&event.content_block_index);

                            let event_data = StreamEvent::ContentBlockStop {
                                index: event.content_block_index,
                            };

                            match create_anthropic_sse_event("content_block_stop", &event_data) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(e),
                            }

                            // Peek at next event to determine if we need to send another content_block_start
                            match stream.recv().await {
                                Ok(Some(next)) => {
                                    match &next {
                                        ConverseStreamOutput::ContentBlockDelta(delta_event) => {
                                            info!("Peeking next event after ContentBlockStop: ContentBlockDelta at index {}", delta_event.content_block_index);

                                            // Synthesize ContentBlockStart based on delta type
                                            let content_block = match &delta_event.delta {
                                                Some(ContentBlockDelta::ReasoningContent(_)) => {
                                                    info!("⚠️ THINKING BLOCK DETECTED - Synthesizing thinking content block start");
                                                    ContentBlockStartData::Thinking {
                                                        thinking: String::new(),
                                                        signature: None,
                                                    }
                                                }
                                                _ => {
                                                    ContentBlockStartData::Text {
                                                        text: String::new(),
                                                    }
                                                }
                                            };

                                            let event_data = StreamEvent::ContentBlockStart {
                                                index: delta_event.content_block_index,
                                                content_block,
                                            };

                                            info!("⚠️ About to send ContentBlockStart SSE event after ContentBlockStop");
                                            match create_anthropic_sse_event("content_block_start", &event_data) {
                                                Ok(event) => {
                                                    info!("⚠️ Successfully created and yielding ContentBlockStart SSE event");
                                                    yield Ok(event)
                                                },
                                                Err(e) => {
                                                    info!("⚠️ ERROR creating ContentBlockStart SSE event: {}", e);
                                                    yield Err(e)
                                                },
                                            }
                                        }
                                        _ => {
                                            info!("Next event after ContentBlockStop is not ContentBlockDelta");
                                        }
                                    }
                                    // Store the next event for processing in the next iteration
                                    pending_event = Some(next);
                                }
                                Ok(None) => break,
                                Err(e) => {
                                    info!("Stream receive error while peeking: {}", e);
                                    break;
                                }
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

                            info!("MessageStop with stop_reason: {} (raw: {:?})", stop_reason, event.stop_reason);

                            // Peek at next event to get Metadata with usage info
                            match stream.recv().await {
                                Ok(Some(next)) => {
                                    match &next {
                                        ConverseStreamOutput::Metadata(metadata_event) => {
                                            info!("Peeking next event after MessageStop: Metadata");

                                            // Convert Bedrock usage to Anthropic format directly from metadata
                                            let anthropic_usage = metadata_event.usage.as_ref().map(|u| {
                                                info!("Updated usage: input_tokens={}, output_tokens={}",
                                                    u.input_tokens, u.output_tokens);
                                                AnthropicUsage {
                                                    input_tokens: u.input_tokens,
                                                    output_tokens: u.output_tokens,
                                                    cache_creation_input_tokens: None,
                                                    cache_read_input_tokens: None,
                                                }
                                            }).unwrap_or_default();

                                            // Now emit message_delta with usage
                                            let message_delta = StreamEvent::MessageDelta {
                                                delta: MessageDeltaData {
                                                    stop_reason: Some(stop_reason.to_string()),
                                                    stop_sequence: None,
                                                    container: None,
                                                    context_management: None,
                                                },
                                                usage: anthropic_usage,
                                            };

                                            info!("Serialized message_delta: {:?}", serde_json::to_string(&message_delta));

                                            match create_anthropic_sse_event("message_delta", &message_delta) {
                                                Ok(event) => yield Ok(event),
                                                Err(e) => yield Err(e),
                                            }

                                            // Call usage callback with Bedrock format
                                            if let Some(ref usage) = metadata_event.usage {
                                                usage_callback(usage);
                                            }

                                            // Emit message_stop
                                            let message_stop = StreamEvent::MessageStop;
                                            match create_anthropic_sse_event("message_stop", &message_stop) {
                                                Ok(event) => yield Ok(event),
                                                Err(e) => yield Err(e),
                                            }

                                            break;
                                        }
                                        _ => {
                                            info!("Next event after MessageStop is not Metadata, it's: {:?}", std::mem::discriminant(&next));
                                            // This shouldn't happen in normal flow, but handle gracefully
                                            // Store the event for next iteration
                                            pending_event = Some(next);
                                        }
                                    }
                                }
                                Ok(None) => {
                                    info!("Stream ended after MessageStop without Metadata");
                                    break;
                                }
                                Err(e) => {
                                    info!("Stream receive error while peeking after MessageStop: {}", e);
                                    break;
                                }
                            }
                        }

                        ConverseStreamOutput::Metadata(_event) => {
                            // Metadata should be handled via peek after MessageStop
                            // If we get here, it means we received Metadata without MessageStop
                            info!("Received Metadata event outside of MessageStop flow - ignoring");
                        }

                        _ => {}
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
