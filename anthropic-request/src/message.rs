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
                // Bedrock requires tool_result blocks to come first.
                let blocks = Vec::try_from(content)?;
                let (tool_results, others): (Vec<_>, Vec<_>) = blocks
                    .into_iter()
                    .partition(|b| matches!(b, ContentBlock::ToolResult(_)));
                let content_blocks: Vec<_> = tool_results.into_iter().chain(others).collect();

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
