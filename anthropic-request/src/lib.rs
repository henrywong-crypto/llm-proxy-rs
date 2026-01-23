use serde::{Deserialize, Serialize};

pub mod cache_control;
pub mod content;
pub mod message;
pub mod system;
pub mod thinking;
pub mod tool;
pub mod tool_result_content;

pub use cache_control::*;
pub use content::*;
pub use message::*;
pub use system::*;
pub use thinking::*;
pub use tool::*;
pub use tool_result_content::*;

#[derive(Debug, Deserialize, Serialize)]
pub struct V1MessagesRequest {
    pub max_tokens: i32,
    pub messages: Messages,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Systems>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_as_string() {
        let json = r#"{
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": "Hello, how are you?"
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        match &request.messages {
            Messages::String(s) => {
                assert_eq!(s, "Hello, how are you?");
            }
            _ => panic!("Expected String messages"),
        }
    }

    #[test]
    fn test_messages_as_array() {
        let json = r#"{
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there!"}]
                }
            ]
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        match &request.messages {
            Messages::Array(arr) => {
                assert_eq!(arr.len(), 2);
                match &arr[0] {
                    Message::User { content } => {
                        assert_eq!(content.len(), 1);
                    }
                    _ => panic!("Expected User message"),
                }
                match &arr[1] {
                    Message::Assistant { content } => {
                        assert_eq!(content.len(), 1);
                    }
                    _ => panic!("Expected Assistant message"),
                }
            }
            _ => panic!("Expected Array messages"),
        }
    }

    #[test]
    fn test_system_as_string() {
        let json = r#"{
            "model": "claude-3",
            "max_tokens": 1024,
            "system": "You are a helpful assistant",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        assert!(request.system.is_some());
        match request.system.unwrap() {
            Systems::String(s) => {
                assert_eq!(s, "You are a helpful assistant");
            }
            _ => panic!("Expected String system"),
        }
    }

    #[test]
    fn test_system_as_array() {
        let json = r#"{
            "model": "claude-3",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful"},
                {"type": "text", "text": "Be concise"}
            ],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        assert!(request.system.is_some());
        match request.system.unwrap() {
            Systems::Array(arr) => {
                assert_eq!(arr.len(), 2);
            }
            _ => panic!("Expected Array system"),
        }
    }

    #[test]
    fn test_combined_string_messages_and_string_system() {
        let json = r#"{
            "model": "claude-3",
            "max_tokens": 1024,
            "system": "You are helpful",
            "messages": "What is 2+2?"
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        // Check system
        assert!(request.system.is_some());
        match request.system.unwrap() {
            Systems::String(s) => {
                assert_eq!(s, "You are helpful");
            }
            _ => panic!("Expected String system"),
        }

        // Check messages
        match &request.messages {
            Messages::String(s) => {
                assert_eq!(s, "What is 2+2?");
            }
            _ => panic!("Expected String messages"),
        }
    }

    #[test]
    fn test_assistant_message_with_thinking() {
        let json = r#"{
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "signature": "test_signature",
                            "thinking": "This is my thinking process"
                        },
                        {
                            "type": "text",
                            "text": "Hello!"
                        }
                    ]
                }
            ]
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        match &request.messages {
            Messages::Array(arr) => {
                assert_eq!(arr.len(), 1);
                match &arr[0] {
                    Message::Assistant { content } => {
                        assert_eq!(content.len(), 2);
                        match &content[0] {
                            AssistantContent::Thinking { thinking, signature } => {
                                assert_eq!(thinking, "This is my thinking process");
                                assert_eq!(signature.as_ref().unwrap(), "test_signature");
                            }
                            _ => panic!("Expected Thinking content"),
                        }
                        match &content[1] {
                            AssistantContent::Text { text, .. } => {
                                assert_eq!(text, "Hello!");
                            }
                            _ => panic!("Expected Text content"),
                        }
                    }
                    _ => panic!("Expected Assistant message"),
                }
            }
            _ => panic!("Expected Array messages"),
        }
    }
}
