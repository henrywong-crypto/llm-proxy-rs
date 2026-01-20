use anthropic_request::V1MessagesRequest;
use anthropic_response::{
    ContentBlockStartData, Delta, MessageDeltaData, MessageStartData, StreamEvent,
    Usage as AnthropicUsage, ResponseContentBlock,
};
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
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;
use uuid::Uuid;

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
    // Create a channel to decouple Bedrock consumption from client consumption
    // Buffer size of 1000 events should handle most cases
    let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<Event>>(1000);
    
    // Spawn a task that will consume from Bedrock independently
    // This ensures we keep reading from Bedrock even if the client disconnects
    tokio::spawn(async move {
        info!("🚀 Starting independent Bedrock consumer task");
        let result = consume_bedrock_to_channel(
            bedrock_stream,
            message_id,
            model,
            usage_callback,
            tx.clone(),
        ).await;
        
        if let Err(e) = result {
            tracing::error!("❌ Bedrock consumer task failed: {}", e);
            let _ = tx.send(Err(e)).await;
        }
        info!("🏁 Bedrock consumer task completed");
    });
    
    // Return a stream that reads from the channel
    Box::pin(async_stream::stream! {
        let mut rx = rx;
        while let Some(event) = rx.recv().await {
            yield event;
        }
    })
}

/// Consume Bedrock stream and send events to channel
/// This runs in a separate task to decouple from client consumption
async fn consume_bedrock_to_channel(
    mut bedrock_stream: aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver<
        ConverseStreamOutput,
        aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError,
    >,
    message_id: String,
    model: String,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
    tx: tokio::sync::mpsc::Sender<anyhow::Result<Event>>,
) -> anyhow::Result<()> {
    // Helper macro to send events to channel
    macro_rules! send_event {
        ($event:expr) => {
            if tx.send(Ok($event)).await.is_err() {
                // Client disconnected, but we continue consuming from Bedrock
                tracing::warn!("Client disconnected, continuing to consume Bedrock stream");
            }
        };
    }
    
    // State tracking
    let mut seen_blocks = std::collections::HashSet::new();
    let mut open_blocks = std::collections::HashSet::new();
    let mut message_started = false;
    let mut message_stopped = false;
    let mut usage_tracker = AnthropicUsage::default();
    
    // Track thinking block signatures to emit before content_block_stop
    let mut thinking_block_signatures: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
    let mut thinking_blocks: std::collections::HashSet<i32> = std::collections::HashSet::new();
    
    // Tool input buffering to avoid overwhelming client with tiny chunks
    let mut tool_input_buffers: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
    let mut last_tool_send_time: std::collections::HashMap<i32, tokio::time::Instant> = std::collections::HashMap::new();
    const TOOL_INPUT_BUFFER_SIZE: usize = 1024; // Send when buffer reaches 1KB (increased from 512B)
    const TOOL_INPUT_BUFFER_TIMEOUT: Duration = Duration::from_millis(100); // Or after 100ms (increased from 50ms)
    
    // Thinking text buffering to avoid overwhelming client with tiny chunks
    let mut thinking_buffers: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
    let mut last_thinking_send_time: std::collections::HashMap<i32, tokio::time::Instant> = std::collections::HashMap::new();
    const THINKING_BUFFER_SIZE: usize = 512; // Send when buffer reaches 512 bytes (increased from 256B)
    const THINKING_BUFFER_TIMEOUT: Duration = Duration::from_millis(200); // Or after 200ms (increased from 100ms)

    loop {
        // Receive next event from Bedrock
        let recv_result = bedrock_stream.recv().await;

            match recv_result {
                Ok(Some(event)) => {
                    // info!("Received Bedrock event: {:?}", event);

                    match event {
                        ConverseStreamOutput::MessageStart(_) => {
                            // info!("⚠️ Processing MessageStart");
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

                            send_event!(sse_event);
                        }

                        ConverseStreamOutput::ContentBlockStart(event) => {
                            // info!("⚠️ ContentBlockStart received: index={}, seen_blocks={:?}, last_was_thinking={}", 
                            //       event.content_block_index, seen_blocks, thinking_blocks.len() > 0 && thinking_blocks.iter().any(|&i| i == event.content_block_index - 1));
                            
                            seen_blocks.insert(event.content_block_index);
                            open_blocks.insert(event.content_block_index);

                            let content_block = match &event.start {
                                Some(ContentBlockStart::ToolUse(tool_use)) => {
                                    info!("🔧 Tool block {} starting: {} (id: {})", event.content_block_index, tool_use.name(), tool_use.tool_use_id());
                                    ContentBlockStartData::ToolUse {
                                        id: tool_use.tool_use_id().to_string(),
                                        name: tool_use.name().to_string(),
                                        input: serde_json::json!({}),
                                    }
                                }
                                _ => {
                                    // Note: Thinking blocks don't have a ContentBlockStart event
                                    // They are detected in ContentBlockDelta with ReasoningContent
                                    info!("📝 Text block {} starting", event.content_block_index);
                                    ContentBlockStartData::Text {
                                        text: String::new(),
                                    }
                                }
                            };

                            let sse_event = create_sse_event("content_block_start", &StreamEvent::ContentBlockStart {
                                index: event.content_block_index,
                                content_block,
                            })?;

                            send_event!(sse_event);
                        }

                        ConverseStreamOutput::ContentBlockDelta(event) => {
                            // // Log delta content
                            // let delta_desc = match &event.delta {
                            //     Some(ContentBlockDelta::Text(text)) => format!("Text({})", text),
                            //     Some(ContentBlockDelta::ToolUse(tool_use)) => format!("ToolUse({})", tool_use.input),
                            //     Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(text))) => {
                            //         format!("Thinking({})", text)
                            //     }
                            //     Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(sig))) => {
                            //         format!("Signature({})", sig)
                            //     }
                            //     _ => "Unknown".to_string(),
                            // };
                            // info!("⚠️ ContentBlockDelta[{}]: {}", event.content_block_index, delta_desc);

                            // Skip thinking blocks entirely (experimental)
                            if let Some(ContentBlockDelta::ReasoningContent(_)) = &event.delta {
                                info!("💭 Skipping thinking block {} (not sending to client)", event.content_block_index);
                                thinking_blocks.insert(event.content_block_index);
                                continue; // Skip this event entirely
                            }
                            
                            // Synthesize ContentBlockStart if not seen
                            if !seen_blocks.contains(&event.content_block_index) {
                                // info!("⚠️ Synthesizing ContentBlockStart for index {}", event.content_block_index);
                                seen_blocks.insert(event.content_block_index);
                                open_blocks.insert(event.content_block_index);

                                let content_block = match &event.delta {
                                    Some(ContentBlockDelta::ReasoningContent(_)) => {
                                        info!("💭 Thinking block starting");
                                        thinking_blocks.insert(event.content_block_index);
                                        ContentBlockStartData::Thinking { thinking: String::new() }
                                    }
                                    _ => {
                                        info!("📝 Text block starting (synthesized)");
                                        ContentBlockStartData::Text { text: String::new() }
                                    },
                                };

                                let sse_event = create_sse_event("content_block_start", &StreamEvent::ContentBlockStart {
                                    index: event.content_block_index,
                                    content_block,
                                })?;

                                send_event!(sse_event);
                            }

                            // Convert delta
                            let delta = match &event.delta {
                                Some(ContentBlockDelta::Text(text)) => {
                                    // Log text output so user can see responses
                                    if !text.is_empty() {
                                        info!("📝 Text: {}", text);
                                    }
                                    Some(Delta::TextDelta { text: text.clone() })
                                }
                                Some(ContentBlockDelta::ToolUse(tool_use)) => {
                                    // Buffer tool input to avoid overwhelming client with tiny chunks
                                    let block_idx = event.content_block_index;
                                    let buffer = tool_input_buffers.entry(block_idx).or_insert_with(String::new);
                                    buffer.push_str(&tool_use.input);
                                    
                                    // Check if we should flush the buffer
                                    let should_flush = buffer.len() >= TOOL_INPUT_BUFFER_SIZE ||
                                        (last_tool_send_time.get(&block_idx)
                                            .map(|t| t.elapsed() >= TOOL_INPUT_BUFFER_TIMEOUT)
                                            .unwrap_or(false) && !buffer.is_empty());
                                    
                                    if should_flush && !buffer.is_empty() {
                                        let buffered_input = buffer.clone();
                                        buffer.clear();
                                        last_tool_send_time.insert(block_idx, tokio::time::Instant::now());
                                        
                                        info!("🔧 Sending buffered tool input for block {} ({}B)", block_idx, buffered_input.len());
                                        Some(Delta::InputJsonDelta { partial_json: buffered_input })
                                    } else {
                                        // Still buffering
                                        None
                                    }
                                }
                                Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(text))) => {
                                    // Buffer thinking text to avoid overwhelming client with tiny chunks
                                    let block_idx = event.content_block_index;
                                    let buffer = thinking_buffers.entry(block_idx).or_insert_with(String::new);
                                    buffer.push_str(text);
                                    
                                    // Check if we should flush the buffer
                                    let should_flush = buffer.len() >= THINKING_BUFFER_SIZE ||
                                        (last_thinking_send_time.get(&block_idx)
                                            .map(|t| t.elapsed() >= THINKING_BUFFER_TIMEOUT)
                                            .unwrap_or(false) && !buffer.is_empty());
                                    
                                    if should_flush && !buffer.is_empty() {
                                        let buffered_thinking = buffer.clone();
                                        buffer.clear();
                                        last_thinking_send_time.insert(block_idx, tokio::time::Instant::now());
                                        
                                        // Log thinking (but don't spam with every token)
                                        if buffered_thinking.len() > 10 {
                                            info!("💭 Thinking: {}...", &buffered_thinking[..buffered_thinking.len().min(50)]);
                                        }
                                        Some(Delta::ThinkingDelta { thinking: buffered_thinking })
                                    } else {
                                        // Still buffering
                                        None
                                    }
                                }
                                Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(sig))) => {
                                    // info!("⚠️ Captured signature for thinking block {}: {}", event.content_block_index, sig);
                                    // Store the signature to emit before content_block_stop
                                    thinking_block_signatures.insert(event.content_block_index, sig.clone());
                                    Some(Delta::SignatureDelta { signature: sig.clone() })
                                }
                                _ => None,
                            };

                            if let Some(delta) = delta {
                                let event_type = match &delta {
                                    Delta::TextDelta { .. } => "text_delta",
                                    Delta::InputJsonDelta { .. } => "input_json_delta",
                                    Delta::ThinkingDelta { .. } => "thinking_delta",
                                    Delta::SignatureDelta { .. } => "signature_delta",
                                };
                                
                                let sse_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                    index: event.content_block_index,
                                    delta,
                                })?;

                                info!("📤 Sending SSE: {} for block {}", event_type, event.content_block_index);
                                send_event!(sse_event);
                            }
                        }

                        ConverseStreamOutput::ContentBlockStop(event) => {
                            // Skip thinking blocks entirely (experimental)
                            if thinking_blocks.contains(&event.content_block_index) {
                                info!("💭 Skipping ContentBlockStop for thinking block {} (not sending to client)", event.content_block_index);
                                thinking_blocks.remove(&event.content_block_index);
                                thinking_block_signatures.remove(&event.content_block_index);
                                thinking_buffers.remove(&event.content_block_index);
                                last_thinking_send_time.remove(&event.content_block_index);
                                continue; // Skip this event entirely
                            }
                            
                            let block_type = if thinking_blocks.contains(&event.content_block_index) {
                                "thinking"
                            } else {
                                "text/tool"
                            };
                            info!("📍 Bedrock sent ContentBlockStop for {} block {}", block_type, event.content_block_index);
                            
                            // Flush any remaining buffered thinking text for this block
                            if let Some(remaining_thinking) = thinking_buffers.remove(&event.content_block_index) {
                                if !remaining_thinking.is_empty() {
                                    info!("💭 Flushing remaining buffered thinking for block {} ({}B)", event.content_block_index, remaining_thinking.len());
                                    let flush_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                        index: event.content_block_index,
                                        delta: Delta::ThinkingDelta { thinking: remaining_thinking },
                                    })?;
                                    send_event!(flush_event);
                                }
                            }
                            last_thinking_send_time.remove(&event.content_block_index);
                            
                            // Flush any remaining buffered tool input for this block
                            if let Some(remaining_input) = tool_input_buffers.remove(&event.content_block_index) {
                                if !remaining_input.is_empty() {
                                    info!("🔧 Flushing remaining buffered tool input for block {} ({}B)", event.content_block_index, remaining_input.len());
                                    let flush_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                        index: event.content_block_index,
                                        delta: Delta::InputJsonDelta { partial_json: remaining_input },
                                    })?;
                                    send_event!(flush_event);
                                }
                            }
                            last_tool_send_time.remove(&event.content_block_index);
                            
                            // CRITICAL: For thinking blocks, ensure signature is sent before closing
                            // If this is a thinking block and we haven't received a signature yet,
                            // we need to wait or synthesize one to avoid client getting stuck
                            if thinking_blocks.contains(&event.content_block_index) {
                                if !thinking_block_signatures.contains_key(&event.content_block_index) {
                                    tracing::warn!("⚠️ WARNING: Thinking block {} closing without signature - synthesizing placeholder", event.content_block_index);
                                    // Synthesize a placeholder signature to prevent client from getting stuck
                                    let placeholder_sig = format!("bedrock_proxy_sig_{}", uuid::Uuid::new_v4());
                                    let sig_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                        index: event.content_block_index,
                                        delta: Delta::SignatureDelta { signature: placeholder_sig },
                                    })?;
                                    info!("📤 Sending SSE: signature_delta (synthesized) for block {}", event.content_block_index);
                                    send_event!(sig_event);
                                } else {
                                    info!("✓ Thinking block {} has signature (already sent)", event.content_block_index);
                                }
                                thinking_blocks.remove(&event.content_block_index);
                                thinking_block_signatures.remove(&event.content_block_index);
                            }
                            
                            open_blocks.remove(&event.content_block_index);

                            let stop_event_data = StreamEvent::ContentBlockStop {
                                index: event.content_block_index,
                            };
                            let sse_event = create_sse_event("content_block_stop", &stop_event_data)?;

                            // Debug: Log the actual JSON being sent
                            let json_debug = serde_json::to_string(&stop_event_data)?;
                            info!("📤 Sending SSE: content_block_stop for block {} - JSON: {}", event.content_block_index, json_debug);
                            send_event!(sse_event);
                            info!("✅ Block {} closed completely", event.content_block_index);
                        }

                        ConverseStreamOutput::MessageStop(event) => {
                            info!("✓ Message complete (reason: {:?})", event.stop_reason);
                            message_stopped = true;

                            let stop_reason = match event.stop_reason {
                                StopReason::EndTurn => {
                                    // info!("⚠️ Stop reason: END_TURN (normal completion)");
                                    "end_turn"
                                },
                                StopReason::ToolUse => {
                                    // info!("⚠️ Stop reason: TOOL_USE (model wants to call a tool)");
                                    "tool_use"
                                },
                                StopReason::MaxTokens => {
                                    tracing::warn!("⚠️ RESPONSE TRUNCATED: MAX_TOKENS limit reached (input={}, output={})", 
                                                  usage_tracker.input_tokens, usage_tracker.output_tokens);
                                    "max_tokens"
                                },
                                StopReason::StopSequence => {
                                    // info!("⚠️ Stop reason: STOP_SEQUENCE (stop sequence encountered)");
                                    "stop_sequence"
                                },
                                StopReason::ContentFiltered => {
                                    tracing::warn!("⚠️ CONTENT_FILTERED: Content policy violation");
                                    "content_filtered"
                                },
                                _ => {
                                    tracing::warn!("⚠️ Unknown stop reason: {:?}", event.stop_reason);
                                    "unknown"
                                },
                            };

                            // Send message_delta with usage
                            let message_delta_event = create_sse_event("message_delta", &StreamEvent::MessageDelta {
                                delta: MessageDeltaData {
                                    stop_reason: Some(stop_reason.to_string()),
                                    stop_sequence: None,
                                },
                                usage: usage_tracker.clone(),
                            })?;

                            // info!("⚠️ Yielding message_delta event with stop_reason={}", stop_reason);
                            send_event!(message_delta_event);
                            // info!("⚠️ Sent message_delta event");

                            // Send message_stop
                            let message_stop_event = create_sse_event("message_stop", &StreamEvent::MessageStop)?;
                            // info!("⚠️ Yielding message_stop event");
                            send_event!(message_stop_event);
                            // info!("⚠️ Sent message_stop event");
                            // info!("⚠️ ========================================");
                        }

                        ConverseStreamOutput::Metadata(event) => {
                            if let Some(usage) = &event.usage {
                                info!("📊 Usage: input={}, output={}", usage.input_tokens, usage.output_tokens);
                                usage_tracker.input_tokens = usage.input_tokens;
                                usage_tracker.output_tokens = usage.output_tokens;
                                usage_callback(usage);
                            }
                            // Check for trace information that might indicate context issues
                            if let Some(trace) = &event.trace {
                                tracing::warn!("⚠️ Trace information present: {:?}", trace);
                            }
                        }

                        other => {
                            tracing::warn!("⚠️ Unhandled Bedrock event: {:?}", other);
                        }
                    }
                }

                Ok(None) => {
                    info!("🏁 Bedrock stream ended (Ok(None))");
                    
                    // Flush any remaining buffered thinking text
                    for (block_index, remaining_thinking) in thinking_buffers.drain() {
                        if !remaining_thinking.is_empty() {
                            info!("💭 Flushing remaining buffered thinking for block {} ({}B) at stream end", block_index, remaining_thinking.len());
                            let flush_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: Delta::ThinkingDelta { thinking: remaining_thinking },
                            })?;
                            send_event!(flush_event);
                        }
                    }
                    
                    // Flush any remaining buffered tool inputs
                    for (block_index, remaining_input) in tool_input_buffers.drain() {
                        if !remaining_input.is_empty() {
                            info!("🔧 Flushing remaining buffered tool input for block {} ({}B) at stream end", block_index, remaining_input.len());
                            let flush_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: Delta::InputJsonDelta { partial_json: remaining_input },
                            })?;
                            send_event!(flush_event);
                        }
                    }
                    
                    if !open_blocks.is_empty() || !message_stopped {
                        tracing::warn!("⚠️ Stream ended with open blocks or no MessageStop: message_started={}, message_stopped={}, open_blocks={:?}", 
                              message_started, message_stopped, open_blocks);
                    } else {
                        info!("✅ Stream ended cleanly (all blocks closed, message stopped)");
                    }

                    // Synthesize missing close events
                    for block_index in open_blocks.iter().copied().collect::<Vec<_>>() {
                        // info!("⚠️ Synthesizing ContentBlockStop for index {}", block_index);
                        
                        // For thinking blocks, ensure signature is sent before closing
                        if thinking_blocks.contains(&block_index) {
                            if !thinking_block_signatures.contains_key(&block_index) {
                                tracing::warn!("⚠️ Synthesizing signature for unclosed thinking block {}", block_index);
                                let placeholder_sig = format!("bedrock_proxy_sig_{}", uuid::Uuid::new_v4());
                                let sig_event = create_sse_event("content_block_delta", &StreamEvent::ContentBlockDelta {
                                    index: block_index,
                                    delta: Delta::SignatureDelta { signature: placeholder_sig },
                                })?;
                                send_event!(sig_event);
                            }
                        }
                        
                        let sse_event = create_sse_event("content_block_stop", &StreamEvent::ContentBlockStop {
                            index: block_index,
                        })?;
                        send_event!(sse_event);
                    }

                    if message_started && !message_stopped {
                        tracing::warn!("⚠️ Synthesizing MessageStop (stream ended without one)");

                        let message_delta_event = create_sse_event("message_delta", &StreamEvent::MessageDelta {
                            delta: MessageDeltaData {
                                stop_reason: Some("end_turn".to_string()),
                                stop_sequence: None,
                            },
                            usage: usage_tracker.clone(),
                        })?;
                        send_event!(message_delta_event);

                        let message_stop_event = create_sse_event("message_stop", &StreamEvent::MessageStop)?;
                        send_event!(message_stop_event);
                    }

                    break;
                }

                Err(e) => {
                    tracing::error!("❌ Bedrock stream error: {}", e);
                    tracing::error!("   State at error: open_blocks={:?}, message_stopped={}", open_blocks, message_stopped);
                    return Err(anyhow::anyhow!("Bedrock stream error: {}", e));
                }
            }
        }
    
    info!("✅ Bedrock consumer completed successfully");
    info!("   Final state: message_started={}, message_stopped={}, open_blocks={:?}", 
          message_started, message_stopped, open_blocks);
    Ok(())
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
        // info!("Creating Bedrock client for Anthropic v1 messages stream");

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

        // info!("⚠️ ======== Sending request to Bedrock ========");
        // info!("⚠️ Model: {}", bedrock_request.model_id);
        // info!("⚠️ Messages count: {}", bedrock_request.messages.len());
        // info!("⚠️ Inference config max_tokens: {:?}", bedrock_request.inference_config.max_tokens());
        // info!("⚠️ Inference config temperature: {:?}", bedrock_request.inference_config.temperature());
        // info!("⚠️ Inference config top_p: {:?}", bedrock_request.inference_config.top_p());
        // info!("⚠️ System content blocks: {}", bedrock_request.system_content_blocks.len());
        // info!("⚠️ Tool config present: {}", bedrock_request.tool_config.is_some());
        // info!("⚠️ Additional model fields present: {}", bedrock_request.additional_model_request_fields.is_some());
        
        // // Log if thinking is enabled
        // if let Some(ref additional_fields) = bedrock_request.additional_model_request_fields {
        //     info!("⚠️ Additional model request fields: {:?}", additional_fields);
        // }
        // info!("⚠️ ================================================");

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

        // info!("Successfully connected to Bedrock stream");

        let message_id = format!("msg_{}", Uuid::new_v4());
        let usage_callback = Arc::new(usage_callback);

        let stream = response.stream;
        Ok(process_anthropic_stream(stream, message_id, model, usage_callback).await)
    }
}
