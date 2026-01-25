use anthropic_request::*;
use serde_json::json;

const TEST_IMAGE: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

#[test]
fn test_basic_request_deserialization() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.model, "claude-3-5-sonnet-20241022");
    assert_eq!(req.max_tokens, 1024);
}

#[test]
fn test_request_with_system_prompt() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(req.system.is_some());
}

#[test]
fn test_request_with_options() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["\n", "END"]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.top_p, Some(0.9));
    assert_eq!(req.top_k, Some(40));
    assert_eq!(
        req.stop_sequences,
        Some(vec!["\n".to_string(), "END".to_string()])
    );
}

#[test]
fn test_request_with_image() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": TEST_IMAGE
                        }
                    }
                ]
            }
        ]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    // Should deserialize successfully
    assert!(matches!(req.messages, Messages::Array(_)));
}

#[test]
fn test_request_with_tools() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(req.tools.is_some());
    assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    assert_eq!(req.tools.as_ref().unwrap()[0].name, "get_weather");
}

#[test]
fn test_request_with_tool_choice() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {}}
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": "get_weather"
        }
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(req.tool_choice.is_some());
    let tool_choice = req.tool_choice.as_ref().unwrap();
    assert_eq!(tool_choice.choice_type, "tool");
    assert_eq!(tool_choice.name, Some("get_weather".to_string()));
}

#[test]
fn test_request_with_thinking() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1000
        }
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(req.thinking.is_some());
}

#[test]
fn test_request_with_metadata() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {
            "user_id": "user123"
        }
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(req.metadata.is_some());
    assert_eq!(
        req.metadata.as_ref().unwrap().user_id,
        Some("user123".to_string())
    );
}

#[test]
fn test_tool_use_in_assistant_message() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }
                ]
            }
        ]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    // Should deserialize successfully
    assert!(matches!(req.messages, Messages::Array(_)));
}

#[test]
fn test_tool_result_in_user_message() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "The weather is sunny, 22°C"
                    }
                ]
            }
        ]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    // Should deserialize successfully
    assert!(matches!(req.messages, Messages::Array(_)));
}

#[test]
fn test_thinking_block_in_assistant_message() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me think...",
                        "signature": "sig123"
                    }
                ]
            }
        ]
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();
    // Should deserialize successfully
    assert!(matches!(req.messages, Messages::Array(_)));
}

#[test]
fn test_full_request_with_all_options() {
    let json = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["STOP"],
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {"type": "object", "properties": {}}
            }
        ],
        "tool_choice": {"type": "auto"}
    });

    let req: V1MessagesRequest = serde_json::from_value(json).unwrap();

    // Verify all fields are present
    assert_eq!(req.model, "claude-3-5-sonnet-20241022");
    assert_eq!(req.max_tokens, 2048);
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.top_p, Some(0.9));
    assert!(req.stop_sequences.is_some());
    assert!(req.tools.is_some());
    assert!(req.tool_choice.is_some());
}
