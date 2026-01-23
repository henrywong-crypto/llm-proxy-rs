use aws_sdk_bedrockruntime::types::{
    ContentBlockStart as BedrockContentBlockStart, ConverseStreamOutput, StopReason, TokenUsage,
};
use std::sync::Arc;

use crate::{
    bedrock_content_block_delta_to_content_block_delta,
    content_block_delta::ContentBlockDelta,
    event::{ContentBlock, Event, MessageDeltaContent, UsageDelta},
    message::Message,
};

pub struct EventConverter {
    message_id: String,
    model: String,
    stop_reason: Option<String>,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
}

impl EventConverter {
    pub fn new(
        message_id: String,
        model: String,
        usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
    ) -> Self {
        Self {
            message_id,
            model,
            stop_reason: None,
            usage_callback,
        }
    }

    pub fn convert(
        &mut self,
        converse_stream_output: &ConverseStreamOutput,
        previous_event_name: Option<&str>,
    ) -> Option<(&'static str, Vec<(&'static str, Event)>)> {
        match converse_stream_output {
            ConverseStreamOutput::MessageStart(_) => {
                let event = Event::message_start_builder()
                    .message(
                        Message::builder()
                            .id(self.message_id.clone())
                            .model(self.model.clone())
                            .role("assistant".to_string())
                            .message_type("message".to_string())
                            .build(),
                    )
                    .build();
                Some(("message_start", vec![("message_start", event)]))
            }
            ConverseStreamOutput::ContentBlockStart(event) => event
                .start
                .as_ref()
                .and_then(|start| match start {
                    BedrockContentBlockStart::ToolUse(tool_use) => Some(
                        ContentBlock::tool_use_builder()
                            .id(tool_use.tool_use_id().to_string())
                            .name(tool_use.name().to_string())
                            .build(),
                    ),
                    _ => None,
                })
                .map(|content_block| {
                    let event = Event::content_block_start_builder()
                        .content_block(content_block)
                        .index(event.content_block_index)
                        .build();
                    ("content_block_start", vec![("content_block_start", event)])
                }),
            ConverseStreamOutput::ContentBlockDelta(event) => {
                let delta = event
                    .delta
                    .as_ref()
                    .and_then(bedrock_content_block_delta_to_content_block_delta)?;

                let mut events = vec![];

                if matches!(
                    previous_event_name,
                    Some("message_start") | Some("content_block_stop")
                ) && let Some(content_block) = match &delta {
                    ContentBlockDelta::TextDelta { .. } => {
                        Some(ContentBlock::text_builder().text(String::new()).build())
                    }
                    ContentBlockDelta::ThinkingDelta { .. }
                    | ContentBlockDelta::SignatureDelta { .. } => Some(
                        ContentBlock::thinking_builder()
                            .thinking(String::new())
                            .signature(String::new())
                            .build(),
                    ),
                    _ => None,
                } {
                    let event = Event::content_block_start_builder()
                        .content_block(content_block)
                        .index(event.content_block_index)
                        .build();
                    events.push(("content_block_start", event));
                }

                let event = Event::content_block_delta_builder()
                    .delta(delta)
                    .index(event.content_block_index)
                    .build();
                events.push(("content_block_delta", event));

                Some(("content_block_delta", events))
            }
            ConverseStreamOutput::ContentBlockStop(event) => {
                let event = Event::content_block_stop_builder()
                    .index(event.content_block_index)
                    .build();
                Some(("content_block_stop", vec![("content_block_stop", event)]))
            }
            ConverseStreamOutput::MessageStop(event) => {
                self.stop_reason = match event.stop_reason {
                    StopReason::EndTurn => Some("end_turn".to_string()),
                    StopReason::MaxTokens => Some("max_tokens".to_string()),
                    StopReason::StopSequence => Some("stop_sequence".to_string()),
                    StopReason::ToolUse => Some("tool_use".to_string()),
                    _ => None,
                };
                None
            }
            ConverseStreamOutput::Metadata(event) => {
                if let Some(ref usage) = event.usage {
                    (self.usage_callback)(usage);
                }

                let message_delta = Event::message_delta_builder()
                    .delta(MessageDeltaContent {
                        stop_reason: self.stop_reason.clone(),
                        stop_sequence: None,
                    })
                    .usage(UsageDelta {
                        output_tokens: event.usage.as_ref().map_or(0, |u| u.output_tokens),
                    })
                    .build();
                let message_stop = Event::message_stop();

                Some((
                    "metadata",
                    vec![
                        ("message_delta", message_delta),
                        ("message_stop", message_stop),
                    ],
                ))
            }
            _ => None,
        }
    }
}
