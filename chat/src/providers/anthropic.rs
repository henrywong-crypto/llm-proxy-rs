use anthropic_request::V1MessagesRequest;
use anthropic_response::{
    ContentBlockStartData, Delta, MessageDeltaData, MessageStartData, StreamEvent,
    Usage as AnthropicUsage, ResponseContentBlock,
};
use async_stream::try_stream;
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::config::timeout::TimeoutConfig;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockStart, ConverseStreamOutput, ReasoningContentBlockDelta,
    StopReason, TokenUsage,
};
use axum::response::sse::Event;
use futures::stream::BoxStream;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tracing::info;
use uuid::Uuid;
use std::collections::{HashMap, HashSet};

use crate::bedrock::BedrockChatCompletion;

// Helper functions for converting between formats

/// Extract usage information from Anthropic's API response or usage data
pub fn get_usage_from_anthropic(data: &Value) -> anyhow::Result<AnthropicUsage> {
    // Try to get usage from the "usage" field first
    let usage_obj = if let Some(usage) = data.get("usage") {
        usage
    } else {
        // Otherwise treat the data itself as the usage object
        data
    };

    let input_tokens = usage_obj
        .get("input_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let output_tokens = usage_obj
        .get("output_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let cache_creation_tokens = usage_obj
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);

    let cache_read_tokens = usage_obj
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);

    Ok(AnthropicUsage {
        input_tokens,
        output_tokens,
        cache_creation_input_tokens: cache_creation_tokens,
        cache_read_input_tokens: cache_read_tokens,
    })
}

/// Convert Anthropic response to message content blocks
pub fn response_to_content_blocks(response: &Value) -> anyhow::Result<Vec<ResponseContentBlock>> {
    let content_blocks = response
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| anyhow::anyhow!("Invalid response format: missing content array"))?;

    let mut blocks = Vec::new();

    for block in content_blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    blocks.push(ResponseContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(|i| i.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing tool_use id"))?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing tool_use name"))?
                    .to_string();
                let input = block
                    .get("input")
                    .ok_or_else(|| anyhow::anyhow!("Missing tool_use input"))?
                    .clone();

                blocks.push(ResponseContentBlock::ToolUse { id, name, input });
            }
            Some("thinking") => {
                if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                    blocks.push(ResponseContentBlock::Thinking {
                        thinking: thinking.to_string(),
                    });
                }
            }
            _ => continue,
        }
    }

    Ok(blocks)
}

/// Process Bedrock stream and convert to Anthropic SSE format
async fn process_anthropic_stream(
    mut bedrock_stream: aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver<
        ConverseStreamOutput,
        aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError,
    >,
    message_id: String,
    model: String,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
) -> BoxStream<'static, anyhow::Result<Event>> {
    Box::pin(try_stream! {
        // State tracking
        let mut seen_blocks = std::collections::HashSet::new();
        let mut open_blocks = std::collections::HashSet::new();
        let mut message_started = false;
        let mut message_stopped = false;
        let mut usage_tracker = AnthropicUsage::default();
        
        // Track thinking block signatures to emit before content_block_stop
        let mut thinking_block_signatures: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
        let mut thinking_blocks: std::collections::HashSet<i32> = std::collections::HashSet::new();

        loop {
            // Receive next event from Bedrock
            let recv_result = bedrock_stream.recv().await;

            match recv_result {
                Ok(Some(event)) => {
                    info!("Received Bedrock event: {:?}", event);

                    match event {
                        ConverseStreamOutput::MessageStart(_) => {
                            info!("⚠️ Processing MessageStart");
                            message_started = true;

                            let sse_event = create_sse_event("message_start", &StreamEvent::MessageStart {
                                message: MessageStartData {
                                    id: message_id.clone(),
                                    message_type: "message".to_string(),
                                    role: "assistant".to_string(),
                                    content: vec![],
                                    model: model.clone(),
                                    stop_reason: None,
                                    stop_sequence: None,
                                    usage: AnthropicUsage::default(),
                                },
                            })?;

                            yield sse_event;
                        }

                        ConverseStreamOutput::ContentBlockStart(event) => {
                            seen_blocks.insert(event.content_block_index);
                            open_blocks.insert(event.content_block_index);

                            let content_block = match &event.start {
                                Some(ContentBlockStart::ToolUse(tool_use)) => {
                                    info!("⚠️ ContentBlockStart: index={}, type=tool_use, id={}, name={}",
                                        event.content_block_index, tool_use.tool_use_id(), tool_use.name());
                                    ContentBlockStartData::ToolUse {
                                        id: tool_use.tool_use_id().to_string(),
                                        name: tool_use.name().to_string(),
                                        input: serde_json::json!({}),
                                    }
                                }
                                _ => {
                                    info!("⚠️ ContentBlockStart: index={}, type=text", event.content_block_index);
                                    ContentBlockStartData::Text {
                                        text: String::new(),
                                    }
                                }
                            };

                            let sse_event = create_sse_event("content_block_start", &StreamEvent::ContentBlockStart {
                                index: event.content_block_index,
                                content_block,
                            })?;

                            yield sse_event;
                        }

                        ConverseStreamOutput::ContentBlockDelta(event) => {
                            // Log delta content
                            let delta_desc = match &event.delta {
                                Some(ContentBlockDelta::Text(text)) => format!("Text({})", text),
                                Some(ContentBlockDelta::ToolUse(tool_use)) => format!("ToolUse({})", tool_use.input),
                                Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(text))) => {
                                    format!("Thinking({})", text)
                                }
                                Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(sig))) => {
                                    format!("Signature({})", sig)
                                }
                                _ => "Unknown".to_string(),
                            };
                            info!("⚠️ ContentBlockDelta[{}]: {}", event.content_block_index, delta_desc);

                            // Synthesize ContentBlockStart if not seen
                            if !seen_blocks.contains(&event.content_block_index) {
                                info!("⚠️ Synthesizing ContentBlockStart for index {}", event.content_block_index);
                                seen_blocks.insert(event.content_block_index);
                                open_blocks.insert(event.content_block_index);

                                let content_block = match &event.delta {
                                    Some(ContentBlockDelta::ReasoningContent(_)) => {
                                        thinking_blocks.insert(event.content_block_index);
                                        ContentBlockStartData::Thinking { thinking: String::new() }
                                    }
                                    _ => ContentBlockStartData::Text { text: String::new() },
                                };

                                let sse_event = create_sse_event("content_block_start", &StreamEvent::ContentBlockStart {
                                    index: event.content_block_index,
                                    content_block,
                                })?;

                                yield sse_event;
                            }

                            // Convert delta
                            let delta = match &event.delta {
                                Some(ContentBlockDelta::Text(text)) => {
                                    Some(Delta::TextDelta { text: text.clone() })
                                }
                                Some(ContentBlockDelta::ToolUse(tool_use)) => {
                                    Some(Delta::InputJsonDelta { partial_json: tool_use.input.clone() })
                                }
                                Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(text))) => {
                                    Some(Delta::ThinkingDelta { thinking: text.clone() })
                                }
                                Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(sig))) => {
                                    info!("⚠️ Captured signature for thinking block {}: {}", event.content_block_index, sig);
                                    // Store the signature to emit before content_block_stop
                                    thinking_block_signatures.insert(event.content_block_index, sig.clone());
                                    Some(Delta::SignatureDelta { signature: sig.clone() })
                                }
                                _ => None,
                            };

                            if let Some(delta) = delta {
                                let sse_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                    index: event.content_block_index,
                                    delta,
                                })?;

                                yield sse_event;
                            }
                        }

                        ConverseStreamOutput::ContentBlockStop(event) => {
                            info!("⚠️ ContentBlockStop for index {}", event.content_block_index);
                            
                            // CRITICAL: For thinking blocks, ensure signature is sent before closing
                            // If this is a thinking block and we haven't received a signature yet,
                            // we need to wait or synthesize one to avoid client getting stuck
                            if thinking_blocks.contains(&event.content_block_index) {
                                if !thinking_block_signatures.contains_key(&event.content_block_index) {
                                    info!("⚠️ WARNING: Thinking block {} closing without signature - synthesizing placeholder", event.content_block_index);
                                    // Synthesize a placeholder signature to prevent client from getting stuck
                                    let placeholder_sig = format!("bedrock_proxy_sig_{}", uuid::Uuid::new_v4());
                                    let sig_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                        index: event.content_block_index,
                                        delta: Delta::SignatureDelta { signature: placeholder_sig },
                                    })?;
                                    yield sig_event;
                                }
                                thinking_blocks.remove(&event.content_block_index);
                                thinking_block_signatures.remove(&event.content_block_index);
                            }
                            
                            open_blocks.remove(&event.content_block_index);

                            let sse_event = create_sse_event("content_block_stop", &StreamEvent::ContentBlockStop {
                                index: event.content_block_index,
                            })?;

                            yield sse_event;
                        }

                        ConverseStreamOutput::MessageStop(event) => {
                            info!("⚠️ MessageStop received");
                            message_stopped = true;

                            let stop_reason = match event.stop_reason {
                                StopReason::EndTurn => "end_turn",
                                StopReason::ToolUse => "tool_use",
                                StopReason::MaxTokens => "max_tokens",
                                StopReason::StopSequence => "stop_sequence",
                                _ => "unknown",
                            };

                            // Send message_delta with usage
                            let message_delta_event = create_sse_event("message_delta", &StreamEvent::MessageDelta {
                                delta: MessageDeltaData {
                                    stop_reason: Some(stop_reason.to_string()),
                                    stop_sequence: None,
                                },
                                usage: usage_tracker.clone(),
                            })?;

                            yield message_delta_event;

                            // Send message_stop
                            let message_stop_event = create_sse_event("message_stop", &StreamEvent::MessageStop)?;
                            yield message_stop_event;
                        }

                        ConverseStreamOutput::Metadata(event) => {
                            info!("⚠️ Metadata event received");
                            if let Some(usage) = &event.usage {
                                info!("⚠️ Usage: input={}, output={}", usage.input_tokens, usage.output_tokens);
                                usage_tracker.input_tokens = usage.input_tokens;
                                usage_tracker.output_tokens = usage.output_tokens;
                                usage_callback(usage);
                            }
                        }

                        _ => {
                            tracing::warn!("⚠️ Unhandled Bedrock event");
                        }
                    }
                }

                Ok(None) => {
                    info!("⚠️ Bedrock stream finished");

                    // Synthesize missing close events
                    for block_index in open_blocks.iter().copied().collect::<Vec<_>>() {
                        info!("⚠️ Synthesizing ContentBlockStop for index {}", block_index);
                        
                        // For thinking blocks, ensure signature is sent before closing
                        if thinking_blocks.contains(&block_index) {
                            if !thinking_block_signatures.contains_key(&block_index) {
                                info!("⚠️ Synthesizing signature for unclosed thinking block {}", block_index);
                                let placeholder_sig = format!("bedrock_proxy_sig_{}", uuid::Uuid::new_v4());
                                let sig_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                    index: block_index,
                                    delta: Delta::SignatureDelta { signature: placeholder_sig },
                                })?;
                                yield sig_event;
                            }
                        }
                        
                        let sse_event = create_sse_event("content_block_stop", &StreamEvent::ContentBlockStop {
                            index: block_index,
                        })?;
                        yield sse_event;
                    }

                    if message_started && !message_stopped {
                        info!("⚠️ Synthesizing MessageStop");

                        let message_delta_event = create_sse_event("message_delta", &StreamEvent::MessageDelta {
                            delta: MessageDeltaData {
                                stop_reason: Some("end_turn".to_string()),
                                stop_sequence: None,
                            },
                            usage: usage_tracker.clone(),
                        })?;
                        yield message_delta_event;

                        let message_stop_event = create_sse_event("message_stop", &StreamEvent::MessageStop)?;
                        yield message_stop_event;
                    }

                    break;
                }

                Err(e) => {
                    tracing::error!("Bedrock stream error: {:?}", e);
                    Err(anyhow::anyhow!("Bedrock stream error: {}", e))?;
                }
            }
        }
    })
}

/// Create an SSE event from Anthropic StreamEvent
fn create_sse_event(event_name: &str, data: &impl Serialize) -> anyhow::Result<Event> {
    let json = serde_json::to_string(data)?;
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
        info!("Creating Bedrock client for Anthropic v1 messages stream");

        // Create Bedrock client with timeout
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .timeout_config(
                TimeoutConfig::builder()
                    .operation_timeout(Duration::from_secs(3600))
                    .operation_attempt_timeout(Duration::from_secs(3600))
                    .build(),
            )
            .load()
            .await;

        let client = Client::new(&sdk_config);

        // Convert request to Bedrock format
        let bedrock_request = BedrockChatCompletion::try_from(&request)?;

        info!("Sending request to Bedrock");
        info!("Model: {}", bedrock_request.model_id);
        info!("Messages count: {}", bedrock_request.messages.len());
        info!("Inference config max_tokens: {:?}", bedrock_request.inference_config.max_tokens());

        // Clone model_id before moving bedrock_request
        let model = bedrock_request.model_id.clone();

        // Start streaming
        let mut request_builder = client
            .converse_stream()
            .model_id(bedrock_request.model_id)
            .set_messages(Some(bedrock_request.messages))
            .inference_config(bedrock_request.inference_config)
            .set_tool_config(bedrock_request.tool_config);

        // Add system content blocks if present
        if !bedrock_request.system_content_blocks.is_empty() {
            request_builder = request_builder.set_system(Some(bedrock_request.system_content_blocks));
        }

        // Add additional model request fields if present
        if let Some(additional_fields) = bedrock_request.additional_model_request_fields {
            request_builder = request_builder.additional_model_request_fields(additional_fields);
        }

        let response = request_builder
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start Bedrock stream: {}", e))?;

        info!("Successfully connected to Bedrock stream");

        let message_id = format!("msg_{}", Uuid::new_v4());
        let usage_callback = Arc::new(usage_callback);

        let stream = response.stream;
        Ok(process_anthropic_stream(stream, message_id, model, usage_callback).await)
    }
}
