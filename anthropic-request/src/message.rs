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

/// Reorders content blocks so that tool_result blocks come first.
/// Bedrock requires tool_result to appear at the start of the user message
/// that immediately follows an assistant message with tool_use.
fn move_tool_results_to_top(content_blocks: Vec<ContentBlock>) -> Vec<ContentBlock> {
    let (tool_results, others): (Vec<_>, Vec<_>) = content_blocks
        .into_iter()
        .partition(|b| matches!(b, ContentBlock::ToolResult(_)));
    tool_results.into_iter().chain(others).collect()
}

/// Inserts a minimal text block when documents are present but no text block
/// exists (AWS Bedrock requirement). Inserts after any tool_result blocks to
/// preserve Bedrock's requirement that tool_results come first.
fn apply_document_validation(content_blocks: &mut Vec<ContentBlock>) {
    let has_document = content_blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Document(_)));
    let has_text = content_blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Text(_)));

    if has_document && !has_text {
        let insert_pos = content_blocks
            .iter()
            .position(|b| !matches!(b, ContentBlock::ToolResult(_)))
            .unwrap_or(content_blocks.len());
        content_blocks.insert(insert_pos, ContentBlock::Text(String::from(" ")));
    }
}

impl TryFrom<&Message> for BedrockMessage {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        match message {
            Message::User { content } => {
                let content_blocks = Vec::try_from(content)?;
                let mut content_blocks = move_tool_results_to_top(content_blocks);
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
            Messages::Array(a) => a
                .iter()
                .map(BedrockMessage::try_from)
                .collect::<Result<_, _>>()?,
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
    fn test_tool_result_moved_to_top_with_text() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Question"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_123", "name": "search", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What should I wear?"},
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "sunny"}
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        assert_eq!(bedrock_messages.len(), 3);

        // tool_result should be first, text second
        let content = bedrock_messages[2].content();
        assert_eq!(content.len(), 2);
        assert!(matches!(content[0], ContentBlock::ToolResult(_)));
        assert!(matches!(content[1], ContentBlock::Text(_)));
    }

    #[test]
    fn test_tool_result_moved_to_top_with_image() {
        use base64::{Engine as _, engine::general_purpose};

        let png_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let base64_data = general_purpose::STANDARD.encode(&png_bytes);

        let json = serde_json::json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Start"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_456", "name": "screenshot", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    },
                    {"type": "tool_result", "tool_use_id": "toolu_456", "content": "done"},
                    {"type": "text", "text": "What do you see?"}
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        assert_eq!(bedrock_messages.len(), 3);

        // tool_result first, then image, then text
        let content = bedrock_messages[2].content();
        assert_eq!(content.len(), 3);
        assert!(matches!(content[0], ContentBlock::ToolResult(_)));
        assert!(matches!(content[1], ContentBlock::Image(_)));
        assert!(matches!(content[2], ContentBlock::Text(_)));
    }

    #[test]
    fn test_tool_result_only_stays_as_is() {
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
    fn test_no_tool_result_preserves_order() {
        let json = serde_json::json!([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "text", "text": "Second"}
                ]
            }
        ]);

        let messages: Messages = serde_json::from_value(json).unwrap();
        let bedrock_messages = Option::<Vec<BedrockMessage>>::try_from(&messages)
            .unwrap()
            .unwrap();

        assert_eq!(bedrock_messages.len(), 1);
        let content = bedrock_messages[0].content();
        assert_eq!(content.len(), 2);
        if let ContentBlock::Text(t) = &content[0] {
            assert_eq!(t, "First");
        }
        if let ContentBlock::Text(t) = &content[1] {
            assert_eq!(t, "Second");
        }
    }

    #[test]
    fn test_document_with_tool_result_gets_validation() {
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

        // tool_result first, then validation text, then document
        let content = bedrock_messages[2].content();
        assert_eq!(content.len(), 3);
        assert!(matches!(content[0], ContentBlock::ToolResult(_)));
        assert!(matches!(content[1], ContentBlock::Text(_)));
        assert!(matches!(content[2], ContentBlock::Document(_)));
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
