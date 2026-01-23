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
                        match content {
                            message::UserContents::Array(arr) => assert_eq!(arr.len(), 1),
                            _ => panic!("Expected Array content"),
                        }
                    }
                    _ => panic!("Expected User message"),
                }
                match &arr[1] {
                    Message::Assistant { content } => {
                        match content {
                            message::AssistantContents::Array(arr) => assert_eq!(arr.len(), 1),
                            _ => panic!("Expected Array content"),
                        }
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
                if let Message::Assistant { content } = &arr[0] {
                    if let message::AssistantContents::Array(content_arr) = content {
                        assert_eq!(content_arr.len(), 2);
                        
                        // Check first item is Thinking
                        match &content_arr[0] {
                            AssistantContent::Thinking { thinking, signature } => {
                                assert_eq!(thinking, "This is my thinking process");
                                assert_eq!(signature, "test_signature");
                            }
                            _ => panic!("Expected Thinking content"),
                        }
                        
                        // Check second item is Text
                        match &content_arr[1] {
                            AssistantContent::Text { text, .. } => {
                                assert_eq!(text, "Hello!");
                            }
                            _ => panic!("Expected Text content"),
                        }
                    } else {
                        panic!("Expected Array content");
                    }
                } else {
                    panic!("Expected Assistant message");
                }
            }
            _ => panic!("Expected Array messages"),
        }
    }

    #[test]
    fn test_message_content_as_string() {
        let json = r#"{
            "model": "claude",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a string content"
                }
            ]
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        match &request.messages {
            Messages::Array(arr) => {
                assert_eq!(arr.len(), 1);
                match &arr[0] {
                    Message::User { content } => match content {
                        message::UserContents::String(s) => {
                            assert_eq!(s, "Hello, this is a string content");
                        }
                        _ => panic!("Expected String content"),
                    },
                    _ => panic!("Expected User message"),
                }
            }
            _ => panic!("Expected Array messages"),
        }
    }

    #[test]
    fn test_assistant_thinking_with_long_signature() {
        let json = r#"{
            "model": "claude",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "signature": "EtMHCkgICxABGAIqQIOvw+YtuOoqaExKEd5ooeSJ8UmcyMIXjMTR9RpzqLFl92UoiONkOqdR5rkGkRn0j7I3e3fLjlySBwq49qm75eoSDMk306krPfNaUuY7lRoMF6qdHd34/6tYsmqhIjAIgvp/NBNX0T3JgY7jnY3ouu0VWw5mOxhaneME6WRdd5CEyFx5Ox6rNb/d196Tj1UquAa146hOOvz3clbnCJQtVCY0HqoIsGPbwBZWCCqgZ9LBg7hJRD1zD2THeTmmL5i4PokBCxGAbiIz40KckFTpO6fvwEkfjE+ITKKugEae1MxlGGQposctlmdS0b6fRxV/GgRh3RjP9165l1hhPBMhH94JY/wUtoAy43KQzU0Vb1jy0q08yPjHBvLeoCDD20X97c/0UD1GUQtNqn6ovmXqWlz53Ndftk9pf1gejI1Mgt1SdRkNFz+wAFT0eBRuAziwBwY6+SejkAdDPaIeCW/fHRQ3JA2JimVanYBZr97SWuGTbTUXb9MhHTX14uIZJ8hP5ziMRlqIgXQZJwprX9UTuq66BtwDj9KHE+JlieXzbUus7uz4m6W27jae0a6X8wO8RW+KKZ+224z2cMbKHZysZux0FKoB3da+/YfjG8Ol5yMOBfunlzl+h4DbjDjstE4MnxRHLTaa+opWvcfDuk2v/Ir5xSSI1LXLy3VX7qn4ki0d6JmYcl7O6q0d9O+JZ2ilbJGl511vdUB2j9PZuUyTPKh9gwchCNLqf0aR5+mPEl1tRv+bZRZiaAAyw07PAxkeQ1Z1+MzKIi7UTbTtq6rWEcAeT2iPthbIoIV9HhQXys4dSwxrAXq0X5kjERpTw7gNxSLDZ9BdKo+SXzj8BGYmUB0b6GVq9zHm5orpKUmk9fts35/gqOAPxwCLSxU/JA0L8sRYL3p1vpfuFK1Gue+8W8QODJxd+aZKlpwVwOxZe5z2suVfbKjyr0nPCxCkTNFaPmjvEHYgNDO7BsogeTj3SBuP866Ee751XpMi5qPE2DQ4rV6zmzsewYRhsrQ1MWrB3sXgUXQzTkmxa2uOf24xSvp4Foz1UE+N4GLpZLfwuKznfmHj0uva1gebkBU/jHK6GsmJ5Ti10VQm+1/6PSRb4kUq6HQj8G4vmUcGKTaANUJFmaJjT2sTlrt48wuLJNjywrBwhDQ0PYZVixuxK6TfE+IbPNK7pzYfpTAv9nkgt4JCmH++cJk6PPZxdBCApYJXOhrjodfDXWyo7X7DHZlgACzkmWqByaBtqI0N2mo+S1B6E6O4MYUOtr1C6+//pwDcNaxvlEHtKS/GIRgB",
                            "thinking": "The user is asking me to implement a detailed migration plan from Flux to ArgoCD. This is a comprehensive plan with multiple phases. Let me break down what needs to be done:\n\n**Phase 1**: Create ArgoCD-specific directories and initial structure\n**Phase 2**: Install ArgoCD in Staging using Flux\n**Phase 3-7**: Create ApplicationSet/App-of-Apps structure and convert HelmReleases to Applications\n**Phase 8**: Validation and testing\n**Phase 9-10**: Production migration and infrastructure migration\n\nThis is a significant implementation task with many files to create and modify. I should:\n1. Use TodoWrite to plan and track all the phases\n2. Start implementing phase by phase\n3. Focus on staging environment first as specified in the plan\n\nLet me start by creating a comprehensive todo list for this implementation."
                        }
                    ]
                }
            ]
        }"#;

        let request: V1MessagesRequest = serde_json::from_str(json).unwrap();

        match &request.messages {
            Messages::Array(arr) => {
                assert_eq!(arr.len(), 1);
                if let Message::Assistant { content } = &arr[0] {
                    if let message::AssistantContents::Array(content_arr) = content {
                        assert_eq!(content_arr.len(), 1);
                        
                        match &content_arr[0] {
                            AssistantContent::Thinking { thinking, signature } => {
                                assert!(thinking.contains("Phase 1"));
                                assert!(signature.to_string().starts_with("EtMHCkgICxABGAIqQIOvw"));
                            }
                            _ => panic!("Expected Thinking content"),
                        }
                    } else {
                        panic!("Expected Array content");
                    }
                } else {
                    panic!("Expected Assistant message");
                }
            }
            _ => panic!("Expected Array messages"),
        }
    }
}
