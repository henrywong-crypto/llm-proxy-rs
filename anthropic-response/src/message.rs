use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ReasoningContentBlock, StopReason, TokenUsage,
};
use common::document_to_value;
use serde::{Deserialize, Serialize};

use crate::event::ContentBlock;

/// Builds a non-streaming `/v1/messages` response from a Bedrock `Converse`
/// result. Mirrors the streaming `EventConverter`: the matched stop sequence is
/// recovered from the request when exactly one was configured (Bedrock strips
/// it and does not report which matched).
pub fn converse_output_to_message(
    id: String,
    model: String,
    content_blocks: &[BedrockContentBlock],
    stop_reason: &StopReason,
    usage: Option<&TokenUsage>,
    request_stop_sequences: Option<&[String]>,
) -> Message {
    let mut content: Vec<serde_json::Value> = content_blocks
        .iter()
        .filter_map(|block| {
            let content_block = match block {
                BedrockContentBlock::Text(text) => {
                    ContentBlock::text_builder().text(text.clone()).build()
                }
                BedrockContentBlock::ToolUse(tool_use) => ContentBlock::tool_use_builder()
                    .id(tool_use.tool_use_id().to_string())
                    .name(tool_use.name().to_string())
                    .input(document_to_value(tool_use.input()))
                    .build(),
                BedrockContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(
                    reasoning,
                )) => ContentBlock::thinking_builder()
                    .thinking(reasoning.text().to_string())
                    .signature(reasoning.signature().unwrap_or_default().to_string())
                    .build(),
                _ => return None,
            };
            serde_json::to_value(content_block).ok()
        })
        .collect();

    let (stop_reason, stop_sequence) = map_stop_reason(stop_reason, request_stop_sequences);

    // Bedrock strips the matched stop sequence from the output text. Re-inject
    // it into the content so non-streaming consumers that reconstruct output
    // from the body (e.g. Claude Code's auto-mode safety classifier rebuilding
    // `<block>…</block>`) see the closing sequence — matching the streaming
    // `EventConverter`, which injects it as a trailing text delta. The sequence
    // is also surfaced in `stop_sequence` below for clients that read it.
    if let Some(seq) = &stop_sequence {
        inject_stop_sequence_text(&mut content, seq);
    }

    Message::builder()
        .id(id)
        .model(model)
        .role("assistant".to_string())
        .message_type("message".to_string())
        .content(content)
        .stop_reason(stop_reason)
        .stop_sequence(stop_sequence)
        .usage(
            Usage::builder()
                .input_tokens(usage.map_or(0, |u| u.input_tokens))
                .output_tokens(usage.map_or(0, |u| u.output_tokens))
                .cache_creation_input_tokens(usage.and_then(|u| u.cache_write_input_tokens))
                .cache_read_input_tokens(usage.and_then(|u| u.cache_read_input_tokens))
                .build(),
        )
        .build()
}

/// Maps a Bedrock `StopReason` to the Anthropic `stop_reason`, recovering the
/// matched `stop_sequence` only when the request configured exactly one.
fn map_stop_reason(
    stop_reason: &StopReason,
    request_stop_sequences: Option<&[String]>,
) -> (Option<String>, Option<String>) {
    match stop_reason {
        StopReason::EndTurn => (Some("end_turn".to_string()), None),
        StopReason::MaxTokens => (Some("max_tokens".to_string()), None),
        StopReason::ToolUse => (Some("tool_use".to_string()), None),
        StopReason::StopSequence => {
            let matched = match request_stop_sequences {
                Some([only]) => Some(only.clone()),
                _ => None,
            };
            (Some("stop_sequence".to_string()), matched)
        }
        _ => (None, None),
    }
}

/// Appends the matched stop sequence to the message content. Mirrors the
/// streaming `EventConverter`, which injects the stripped sequence as a
/// trailing text delta on the open block. Appends to the last text block when
/// one exists (the stop-sequence case always ends on text), otherwise pushes a
/// new text block so the sequence is never silently dropped.
fn inject_stop_sequence_text(content: &mut Vec<serde_json::Value>, stop_sequence: &str) {
    for block in content.iter_mut().rev() {
        if block.get("type").and_then(|t| t.as_str()) != Some("text") {
            continue;
        }
        if let Some(serde_json::Value::String(text)) = block.get_mut("text") {
            text.push_str(stop_sequence);
            return;
        }
    }
    if let Ok(block) = serde_json::to_value(
        ContentBlock::text_builder()
            .text(stop_sequence.to_string())
            .build(),
    ) {
        content.push(block);
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Message {
    pub content: Vec<serde_json::Value>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(rename = "type")]
    pub message_type: String,
    pub usage: Usage,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<i32>,
}

impl Message {
    pub fn builder() -> MessageBuilder {
        MessageBuilder::default()
    }
}

#[derive(Default)]
pub struct MessageBuilder {
    content: Vec<serde_json::Value>,
    id: String,
    model: String,
    role: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    message_type: String,
    usage: Usage,
}

impl MessageBuilder {
    pub fn content(mut self, content: Vec<serde_json::Value>) -> Self {
        self.content = content;
        self
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = id;
        self
    }

    pub fn model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    pub fn role(mut self, role: String) -> Self {
        self.role = role;
        self
    }

    pub fn stop_reason(mut self, stop_reason: Option<String>) -> Self {
        self.stop_reason = stop_reason;
        self
    }

    pub fn stop_sequence(mut self, stop_sequence: Option<String>) -> Self {
        self.stop_sequence = stop_sequence;
        self
    }

    pub fn message_type(mut self, message_type: String) -> Self {
        self.message_type = message_type;
        self
    }

    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    pub fn build(self) -> Message {
        Message {
            content: self.content,
            id: self.id,
            model: self.model,
            role: self.role,
            stop_reason: self.stop_reason,
            stop_sequence: self.stop_sequence,
            message_type: self.message_type,
            usage: self.usage,
        }
    }
}

impl Usage {
    pub fn builder() -> UsageBuilder {
        UsageBuilder::default()
    }
}

#[derive(Default)]
pub struct UsageBuilder {
    input_tokens: i32,
    output_tokens: i32,
    cache_creation_input_tokens: Option<i32>,
    cache_read_input_tokens: Option<i32>,
}

impl UsageBuilder {
    pub fn input_tokens(mut self, tokens: i32) -> Self {
        self.input_tokens = tokens;
        self
    }

    pub fn output_tokens(mut self, tokens: i32) -> Self {
        self.output_tokens = tokens;
        self
    }

    pub fn cache_creation_input_tokens(mut self, tokens: Option<i32>) -> Self {
        self.cache_creation_input_tokens = tokens;
        self
    }

    pub fn cache_read_input_tokens(mut self, tokens: Option<i32>) -> Self {
        self.cache_read_input_tokens = tokens;
        self
    }

    pub fn build(self) -> Usage {
        Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::types::{ReasoningTextBlock, ToolUseBlock};
    use common::value_to_document;

    #[test]
    fn message_builder_defaults_usage_to_zero() {
        let message = Message::builder()
            .id("msg_1".to_string())
            .model("claude".to_string())
            .role("assistant".to_string())
            .message_type("message".to_string())
            .build();

        assert_eq!(message.usage.input_tokens, 0);
        assert_eq!(message.usage.output_tokens, 0);
    }

    #[test]
    fn converse_output_maps_thinking_text_and_tool_use_in_order() {
        let blocks = vec![
            BedrockContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(
                ReasoningTextBlock::builder()
                    .text("pondering")
                    .signature("sig")
                    .build()
                    .unwrap(),
            )),
            BedrockContentBlock::Text("hello".to_string()),
            BedrockContentBlock::ToolUse(
                ToolUseBlock::builder()
                    .tool_use_id("tu_1")
                    .name("get_weather")
                    .input(value_to_document(&serde_json::json!({"location": "Paris"})))
                    .build()
                    .unwrap(),
            ),
        ];
        let usage = TokenUsage::builder()
            .input_tokens(10)
            .output_tokens(20)
            .total_tokens(30)
            .build()
            .unwrap();

        let message = converse_output_to_message(
            "msg_1".to_string(),
            "claude".to_string(),
            &blocks,
            &StopReason::ToolUse,
            Some(&usage),
            None,
        );

        assert_eq!(message.role, "assistant");
        assert_eq!(message.message_type, "message");
        assert_eq!(message.stop_reason.as_deref(), Some("tool_use"));
        assert!(message.stop_sequence.is_none());
        assert_eq!(message.usage.input_tokens, 10);
        assert_eq!(message.usage.output_tokens, 20);

        assert_eq!(message.content.len(), 3);
        assert_eq!(message.content[0]["type"], "thinking");
        assert_eq!(message.content[0]["thinking"], "pondering");
        assert_eq!(message.content[0]["signature"], "sig");
        assert_eq!(message.content[1]["type"], "text");
        assert_eq!(message.content[1]["text"], "hello");
        assert_eq!(message.content[2]["type"], "tool_use");
        assert_eq!(message.content[2]["id"], "tu_1");
        assert_eq!(message.content[2]["name"], "get_weather");
        assert_eq!(message.content[2]["input"]["location"], "Paris");
    }

    #[test]
    fn converse_output_recovers_single_stop_sequence() {
        let message = converse_output_to_message(
            "m".to_string(),
            "c".to_string(),
            &[],
            &StopReason::StopSequence,
            None,
            Some(&["</block>".to_string()]),
        );
        assert_eq!(message.stop_reason.as_deref(), Some("stop_sequence"));
        assert_eq!(message.stop_sequence.as_deref(), Some("</block>"));
        // No content from Bedrock, so the stripped sequence is surfaced as a
        // standalone text block rather than being dropped.
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.content[0]["type"], "text");
        assert_eq!(message.content[0]["text"], "</block>");
    }

    #[test]
    fn converse_output_appends_matched_stop_sequence_to_trailing_text() {
        let blocks = vec![BedrockContentBlock::Text("<block>no".to_string())];
        let message = converse_output_to_message(
            "m".to_string(),
            "c".to_string(),
            &blocks,
            &StopReason::StopSequence,
            None,
            Some(&["</block>".to_string()]),
        );
        assert_eq!(message.stop_sequence.as_deref(), Some("</block>"));
        // The body reconstructs to valid XML, matching the streaming converter.
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.content[0]["type"], "text");
        assert_eq!(message.content[0]["text"], "<block>no</block>");
    }

    #[test]
    fn converse_output_omits_ambiguous_stop_sequence() {
        let message = converse_output_to_message(
            "m".to_string(),
            "c".to_string(),
            &[],
            &StopReason::StopSequence,
            None,
            Some(&["</block>".to_string(), "STOP".to_string()]),
        );
        assert_eq!(message.stop_reason.as_deref(), Some("stop_sequence"));
        assert!(message.stop_sequence.is_none());
    }
}
