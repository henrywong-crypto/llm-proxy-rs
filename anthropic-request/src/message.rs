use aws_sdk_bedrockruntime::types::{ContentBlock, ConversationRole, Message as BedrockMessage};
use serde::{Deserialize, Serialize};

use crate::content::{AssistantContents, UserContents};

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

                // AWS Bedrock requires at least one text block when documents are present
                let has_document = content_blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Document(_)));
                let has_text = content_blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Text(_)));

                if has_document && !has_text {
                    // Insert a minimal text block at the beginning
                    content_blocks.insert(0, ContentBlock::Text(String::from(" ")));
                }

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

/// Applies document+text validation: inserts minimal text block when documents
/// are present but no text block exists (AWS Bedrock requirement).
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
                for message in a {
                    match message {
                        Message::User { content } => {
                            let content_blocks = Vec::try_from(content)?;

                            let (tool_result_blocks, other_blocks): (Vec<_>, Vec<_>) =
                                content_blocks
                                    .into_iter()
                                    .partition(|b| matches!(b, ContentBlock::ToolResult(_)));

                            if !tool_result_blocks.is_empty() && !other_blocks.is_empty() {
                                // Both groups: create two Bedrock user messages
                                result.push(
                                    BedrockMessage::builder()
                                        .role(ConversationRole::User)
                                        .set_content(Some(tool_result_blocks))
                                        .build()?,
                                );

                                let mut other = other_blocks;
                                apply_document_validation(&mut other);
                                result.push(
                                    BedrockMessage::builder()
                                        .role(ConversationRole::User)
                                        .set_content(Some(other))
                                        .build()?,
                                );
                            } else if !tool_result_blocks.is_empty() {
                                result.push(
                                    BedrockMessage::builder()
                                        .role(ConversationRole::User)
                                        .set_content(Some(tool_result_blocks))
                                        .build()?,
                                );
                            } else if !other_blocks.is_empty() {
                                let mut other = other_blocks;
                                apply_document_validation(&mut other);
                                result.push(
                                    BedrockMessage::builder()
                                        .role(ConversationRole::User)
                                        .set_content(Some(other))
                                        .build()?,
                                );
                            }
                        }
                        Message::Assistant { content } => {
                            let content = Vec::try_from(content)?;
                            result.push(
                                BedrockMessage::builder()
                                    .role(ConversationRole::Assistant)
                                    .set_content(Some(content))
                                    .build()?,
                            );
                        }
                    }
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
    use crate::content::{AssistantContent, UserContent};

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

        // Should have 2 content blocks: auto-inserted text + document
        let content = bedrock_message.content().unwrap();
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

        // Should have 2 content blocks: text + document (no auto-insertion)
        let content = bedrock_message.content().unwrap();
        assert_eq!(content.len(), 2);
        assert!(matches!(content[0], ContentBlock::Text(_)));
        assert!(matches!(content[1], ContentBlock::Document(_)));

        // Verify the text is the original, not auto-inserted
        if let ContentBlock::Text(text) = &content[0] {
            assert_eq!(text, "Analyze this document");
        }
    }

    #[test]
    fn test_messages_user_with_tool_result_and_text_splits_into_two() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "The weather is sunny"
                    },
                    {
                        "type": "text",
                        "text": "What should I wear today?"
                    }
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages).unwrap().unwrap();

        // Should be split into two user messages
        assert_eq!(bedrock_messages.len(), 2);

        // First: tool_result only
        let first = &bedrock_messages[0];
        assert!(matches!(first.role(), Some(ConversationRole::User)));
        let first_content = first.content().unwrap();
        assert_eq!(first_content.len(), 1);
        assert!(matches!(first_content[0], ContentBlock::ToolResult(_)));

        // Second: text only
        let second = &bedrock_messages[1];
        assert!(matches!(second.role(), Some(ConversationRole::User)));
        let second_content = second.content().unwrap();
        assert_eq!(second_content.len(), 1);
        assert!(matches!(second_content[0], ContentBlock::Text(_)));
        if let ContentBlock::Text(t) = &second_content[0] {
            assert_eq!(t, "What should I wear today?");
        }
    }

    #[test]
    fn test_messages_user_with_tool_result_and_document_splits_into_two() {
        use base64::{Engine as _, engine::general_purpose};

        let pdf_bytes = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids []\n/Count 0\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n110\n%%EOF\n";
        let base64_data = general_purpose::STANDARD.encode(pdf_bytes);

        let json = serde_json::json!([
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_456",
                        "content": "Search completed"
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
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages).unwrap().unwrap();

        assert_eq!(bedrock_messages.len(), 2);

        // First: tool_result only
        let first_content = bedrock_messages[0].content().unwrap();
        assert_eq!(first_content.len(), 1);
        assert!(matches!(first_content[0], ContentBlock::ToolResult(_)));

        // Second: document + auto-inserted text (document validation)
        let second_content = bedrock_messages[1].content().unwrap();
        assert_eq!(second_content.len(), 2);
        assert!(matches!(second_content[0], ContentBlock::Text(_)));
        assert!(matches!(second_content[1], ContentBlock::Document(_)));
    }

    #[test]
    fn test_messages_user_with_tool_result_only_single_message() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_789",
                        "content": "Result only"
                    }
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages).unwrap().unwrap();

        assert_eq!(bedrock_messages.len(), 1);
        let content = bedrock_messages[0].content().unwrap();
        assert_eq!(content.len(), 1);
        assert!(matches!(content[0], ContentBlock::ToolResult(_)));
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
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages).unwrap().unwrap();

        assert_eq!(bedrock_messages.len(), 1);
        let content = bedrock_messages[0].content().unwrap();
        assert_eq!(content.len(), 1);
        assert!(matches!(content[0], ContentBlock::Text(_)));
    }

    #[test]
    fn test_messages_mixed_conversation_preserves_order() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "id1", "content": "Tool result"},
                    {"type": "text", "text": "Follow-up question"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Assistant reply"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Another user message"}]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages).unwrap().unwrap();

        // user(tool+text) -> 2 user, assistant -> 1, user -> 1 = 4 total
        assert_eq!(bedrock_messages.len(), 4);

        assert!(matches!(bedrock_messages[0].role(), Some(ConversationRole::User)));
        assert!(matches!(bedrock_messages[1].role(), Some(ConversationRole::User)));
        assert!(matches!(bedrock_messages[2].role(), Some(ConversationRole::Assistant)));
        assert!(matches!(bedrock_messages[3].role(), Some(ConversationRole::User)));
    }
}
