use aws_sdk_bedrockruntime::types::{ContentBlock, ConversationRole, Message as BedrockMessage};
use serde::{Deserialize, Serialize};

use crate::content::{AssistantContent, AssistantContents, UserContents};

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Messages {
    Array(Vec<Message>),
    String(String),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(rename = "assistant")]
    Assistant { content: AssistantContents },
    #[serde(rename = "user")]
    User { content: UserContents },
}

impl TryFrom<&Message> for BedrockMessage {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        match message {
            Message::User { content } => {
                let mut content_blocks = Vec::try_from(content)?;
                apply_document_validation(&mut content_blocks);

                Ok(BedrockMessage::builder()
                    .role(ConversationRole::User)
                    .set_content(Some(content_blocks))
                    .build()?)
            }
            Message::Assistant { content } => {
                let content = Vec::try_from(content)?;

                Ok(BedrockMessage::builder()
                    .role(ConversationRole::Assistant)
                    .set_content(Some(content))
                    .build()?)
            }
        }
    }
}

/// Inserts a minimal text block when documents are present but no text block
/// exists (AWS Bedrock requirement).
fn apply_document_validation(content_blocks: &mut Vec<ContentBlock>) {
    let has_document = content_blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Document(_)));
    let has_text = content_blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Text(_)));

    if has_document && !has_text {
        content_blocks.insert(0, ContentBlock::Text(String::from(" ")));
    }
}

/// Returns true if the assistant content contains any tool_use blocks.
fn assistant_has_tool_use(content: &AssistantContents) -> bool {
    match content {
        AssistantContents::String(_) => false,
        AssistantContents::Array(arr) => arr
            .iter()
            .any(|c| matches!(c, AssistantContent::ToolUse { .. })),
    }
}

fn build_user_message(content: Vec<ContentBlock>) -> anyhow::Result<BedrockMessage> {
    Ok(BedrockMessage::builder()
        .role(ConversationRole::User)
        .set_content(Some(content))
        .build()?)
}

impl TryFrom<&Messages> for Option<Vec<BedrockMessage>> {
    type Error = anyhow::Error;

    fn try_from(messages: &Messages) -> Result<Self, Self::Error> {
        let bedrock_messages: Vec<BedrockMessage> = match messages {
            Messages::String(s) => {
                let content = vec![ContentBlock::Text(s.clone())];
                vec![
                    BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .set_content(Some(content))
                        .build()?,
                ]
            }
            Messages::Array(a) => {
                let mut result = Vec::new();
                let mut deferred_blocks: Vec<ContentBlock> = Vec::new();
                let mut prev_assistant_had_tool_use = false;

                for message in a {
                    match message {
                        Message::User { content } => {
                            let mut content_blocks = Vec::try_from(content)?;

                            // Prepend any deferred blocks from a previous split
                            if !deferred_blocks.is_empty() {
                                let mut merged = std::mem::take(&mut deferred_blocks);
                                merged.append(&mut content_blocks);
                                content_blocks = merged;
                            }

                            if prev_assistant_had_tool_use {
                                // After a tool_use, extract tool_result blocks for the
                                // immediate response and defer everything else to the
                                // next user message to maintain alternation.
                                let (tool_result_blocks, other_blocks): (Vec<_>, Vec<_>) =
                                    content_blocks
                                        .into_iter()
                                        .partition(|b| matches!(b, ContentBlock::ToolResult(_)));

                                if !tool_result_blocks.is_empty() {
                                    result.push(build_user_message(tool_result_blocks)?);
                                }

                                deferred_blocks = other_blocks;
                            } else {
                                apply_document_validation(&mut content_blocks);
                                if !content_blocks.is_empty() {
                                    result.push(build_user_message(content_blocks)?);
                                }
                            }

                            prev_assistant_had_tool_use = false;
                        }
                        Message::Assistant { content } => {
                            prev_assistant_had_tool_use = assistant_has_tool_use(content);

                            let content_blocks = Vec::try_from(content)?;
                            result.push(
                                BedrockMessage::builder()
                                    .role(ConversationRole::Assistant)
                                    .set_content(Some(content_blocks))
                                    .build()?,
                            );
                        }
                    }
                }

                // Flush any remaining deferred blocks as a final user message
                if !deferred_blocks.is_empty() {
                    apply_document_validation(&mut deferred_blocks);
                    result.push(build_user_message(deferred_blocks)?);
                }

                result
            }
        };

        Ok(if bedrock_messages.is_empty() {
            None
        } else {
            Some(bedrock_messages)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::UserContent;

    #[test]
    fn test_user_message_string_content() {
        let json = r#"{
            "role": "user",
            "content": "Hello"
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        match message {
            Message::User { content } => match content {
                UserContents::String(s) => assert_eq!(s, "Hello"),
                _ => panic!("Expected String variant"),
            },
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_user_message_array_content() {
        let json = r#"{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}]
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        match message {
            Message::User { content } => match content {
                UserContents::Array(arr) => {
                    assert_eq!(arr.len(), 1);
                    match &arr[0] {
                        UserContent::Text {
                            cache_control,
                            text,
                        } => {
                            assert_eq!(text, "Hello");
                            assert!(cache_control.is_none());
                        }
                        _ => panic!("Expected Text variant"),
                    }
                }
                _ => panic!("Expected Array variant"),
            },
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_assistant_message_string_content() {
        let json = r#"{
            "role": "assistant",
            "content": "Hi there!"
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        match message {
            Message::Assistant { content } => match content {
                AssistantContents::String(s) => assert_eq!(s, "Hi there!"),
                _ => panic!("Expected String variant"),
            },
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_assistant_message_array_content() {
        let json = r#"{
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there!"}]
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        match message {
            Message::Assistant { content } => match content {
                AssistantContents::Array(arr) => {
                    assert_eq!(arr.len(), 1);
                    match &arr[0] {
                        AssistantContent::Text {
                            cache_control,
                            text,
                        } => {
                            assert_eq!(text, "Hi there!");
                            assert!(cache_control.is_none());
                        }
                        _ => panic!("Expected Text variant"),
                    }
                }
                _ => panic!("Expected Array variant"),
            },
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_messages_array() {
        let json = r#"[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}]
            }
        ]"#;

        let messages: Vec<Message> = serde_json::from_str(json).unwrap();
        assert_eq!(messages.len(), 2);

        match &messages[0] {
            Message::User { content } => match content {
                UserContents::Array(arr) => {
                    assert_eq!(arr.len(), 1);
                }
                _ => panic!("Expected Array variant"),
            },
            _ => panic!("Expected User message"),
        }

        match &messages[1] {
            Message::Assistant { content } => match content {
                AssistantContents::Array(arr) => {
                    assert_eq!(arr.len(), 1);
                }
                _ => panic!("Expected Array variant"),
            },
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_messages_string() {
        let json = r#""Hello world""#;

        let messages: Messages = serde_json::from_str(json).unwrap();
        match messages {
            Messages::String(s) => assert_eq!(s, "Hello world"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_messages_array_wrapper() {
        let json = r#"[
            {
                "role": "user",
                "content": "Hello"
            }
        ]"#;

        let messages: Messages = serde_json::from_str(json).unwrap();
        match messages {
            Messages::Array(arr) => {
                assert_eq!(arr.len(), 1);
            }
            _ => panic!("Expected Array variant"),
        }
    }

    #[test]
    fn test_user_message_with_document_without_text_adds_text_block() {
        use base64::{Engine as _, engine::general_purpose};

        let pdf_bytes = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids []\n/Count 0\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n110\n%%EOF\n";
        let base64_data = general_purpose::STANDARD.encode(pdf_bytes);

        let json = serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_data
                    }
                }
            ]
        });

        let message: Message = serde_json::from_value(json).unwrap();
        let bedrock_message = BedrockMessage::try_from(&message).unwrap();

        let content = bedrock_message.content();
        assert_eq!(content.len(), 2);
        assert!(matches!(content[0], ContentBlock::Text(_)));
        assert!(matches!(content[1], ContentBlock::Document(_)));
    }

    #[test]
    fn test_user_message_with_document_and_text_no_extra_text_added() {
        use base64::{Engine as _, engine::general_purpose};

        let pdf_bytes = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids []\n/Count 0\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n110\n%%EOF\n";
        let base64_data = general_purpose::STANDARD.encode(pdf_bytes);

        let json = serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this document"
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_data
                    }
                }
            ]
        });

        let message: Message = serde_json::from_value(json).unwrap();
        let bedrock_message = BedrockMessage::try_from(&message).unwrap();

        let content = bedrock_message.content();
        assert_eq!(content.len(), 2);
        assert!(matches!(content[0], ContentBlock::Text(_)));
        assert!(matches!(content[1], ContentBlock::Document(_)));

        if let ContentBlock::Text(text) = &content[0] {
            assert_eq!(text, "Analyze this document");
        }
    }

    #[test]
    fn test_tool_result_after_tool_use_splits_and_defers() {
        // assistant has tool_use, user has tool_result + text
        // tool_result should stay, text deferred to next user message
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Initial question"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_123", "name": "search", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "sunny"},
                    {"type": "text", "text": "What should I wear?"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Wear light clothes."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Thanks!"}]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        // Expected:
        // 0: user(Initial question)
        // 1: assistant(text + tool_use)
        // 2: user(tool_result)           -- extracted from msg 2
        // 3: assistant(Wear light...)
        // 4: user(What should I wear? + Thanks!)  -- deferred text merged with msg 4
        assert_eq!(bedrock_messages.len(), 5);

        assert!(matches!(
            bedrock_messages[0].role(),
            ConversationRole::User
        ));
        assert!(matches!(
            bedrock_messages[1].role(),
            ConversationRole::Assistant
        ));
        assert!(matches!(
            bedrock_messages[2].role(),
            ConversationRole::User
        ));
        assert!(matches!(
            bedrock_messages[3].role(),
            ConversationRole::Assistant
        ));
        assert!(matches!(
            bedrock_messages[4].role(),
            ConversationRole::User
        ));

        // msg 2 should only have tool_result
        let msg2_content = bedrock_messages[2].content();
        assert_eq!(msg2_content.len(), 1);
        assert!(matches!(msg2_content[0], ContentBlock::ToolResult(_)));

        // msg 4 should have deferred text + original text
        let msg4_content = bedrock_messages[4].content();
        assert_eq!(msg4_content.len(), 2);
        assert!(matches!(msg4_content[0], ContentBlock::Text(_)));
        assert!(matches!(msg4_content[1], ContentBlock::Text(_)));
        if let ContentBlock::Text(t) = &msg4_content[0] {
            assert_eq!(t, "What should I wear?");
        }
        if let ContentBlock::Text(t) = &msg4_content[1] {
            assert_eq!(t, "Thanks!");
        }
    }

    #[test]
    fn test_tool_result_only_no_split_needed() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_789", "name": "calc", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_789", "content": "42"}
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        assert_eq!(bedrock_messages.len(), 3);
        let content = bedrock_messages[2].content();
        assert_eq!(content.len(), 1);
        assert!(matches!(content[0], ContentBlock::ToolResult(_)));
    }

    #[test]
    fn test_no_tool_use_keeps_mixed_content_together() {
        // When assistant does NOT have tool_use, user content stays together
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_old", "content": "result"},
                    {"type": "text", "text": "Follow-up"}
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        // No preceding assistant with tool_use, so everything stays in one message
        assert_eq!(bedrock_messages.len(), 1);
        let content = bedrock_messages[0].content();
        assert_eq!(content.len(), 2);
        assert!(matches!(content[0], ContentBlock::ToolResult(_)));
        assert!(matches!(content[1], ContentBlock::Text(_)));
    }

    #[test]
    fn test_deferred_blocks_flushed_at_end() {
        // If the conversation ends with deferred blocks, they become a final user message
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Question"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_end", "name": "lookup", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_end", "content": "data"},
                    {"type": "text", "text": "Now explain this"}
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        // Expected:
        // 0: user(Question)
        // 1: assistant(tool_use)
        // 2: user(tool_result)
        // 3: user(Now explain this)  -- deferred, flushed at end
        assert_eq!(bedrock_messages.len(), 4);

        let msg2_content = bedrock_messages[2].content();
        assert_eq!(msg2_content.len(), 1);
        assert!(matches!(msg2_content[0], ContentBlock::ToolResult(_)));

        let msg3_content = bedrock_messages[3].content();
        assert_eq!(msg3_content.len(), 1);
        assert!(matches!(msg3_content[0], ContentBlock::Text(_)));
        if let ContentBlock::Text(t) = &msg3_content[0] {
            assert_eq!(t, "Now explain this");
        }
    }

    #[test]
    fn test_deferred_document_gets_text_validation() {
        use base64::{Engine as _, engine::general_purpose};

        let pdf_bytes = b"%PDF-1.4\nminimal";
        let base64_data = general_purpose::STANDARD.encode(pdf_bytes);

        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Start"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_doc", "name": "fetch", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_doc", "content": "ok"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64_data
                        }
                    }
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        // Deferred doc flushed at end with document validation (adds text block)
        let last = bedrock_messages.last().unwrap();
        let content = last.content();
        assert!(content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text(_))));
        assert!(content
            .iter()
            .any(|b| matches!(b, ContentBlock::Document(_))));
    }

    #[test]
    fn test_multiple_tool_use_cycles_defer_accumulates() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Start"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "a", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                    {"type": "text", "text": "extra1"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "b", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
                    {"type": "text", "text": "extra2"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Final answer"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "thanks"}]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        // Expected:
        // 0: user(Start)
        // 1: assistant(tool_use t1)
        // 2: user(tool_result t1)
        // 3: assistant(tool_use t2)
        // 4: user(tool_result t2)         -- extra1 deferred, prepended here then re-split
        // 5: assistant(Final answer)
        // 6: user(extra1 + extra2 + thanks) -- all deferred merged with final user msg
        assert_eq!(bedrock_messages.len(), 7);

        // Check alternation
        assert!(matches!(
            bedrock_messages[0].role(),
            ConversationRole::User
        ));
        assert!(matches!(
            bedrock_messages[1].role(),
            ConversationRole::Assistant
        ));
        assert!(matches!(
            bedrock_messages[2].role(),
            ConversationRole::User
        ));
        assert!(matches!(
            bedrock_messages[3].role(),
            ConversationRole::Assistant
        ));
        assert!(matches!(
            bedrock_messages[4].role(),
            ConversationRole::User
        ));
        assert!(matches!(
            bedrock_messages[5].role(),
            ConversationRole::Assistant
        ));
        assert!(matches!(
            bedrock_messages[6].role(),
            ConversationRole::User
        ));

        // msg 6 has accumulated deferred text + original
        let msg6_content = bedrock_messages[6].content();
        assert_eq!(msg6_content.len(), 3);
    }

    #[test]
    fn test_messages_user_with_text_only_single_message() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        assert_eq!(bedrock_messages.len(), 1);
        let content = bedrock_messages[0].content();
        assert_eq!(content.len(), 1);
        assert!(matches!(content[0], ContentBlock::Text(_)));
    }
}
