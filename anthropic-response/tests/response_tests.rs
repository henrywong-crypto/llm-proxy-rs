use anthropic_response::*;
use serde_json::json;

#[test]
fn test_error_response_creation() {
    let err = ErrorResponse::new(400, "Invalid request");
    assert_eq!(err.response_type, "error");
    assert_eq!(err.error.error_type, "invalid_request_error");
    assert_eq!(err.error.message, "Invalid request");
    assert!(err.request_id.is_some());
}

#[test]
fn test_error_response_status_codes() {
    let tests = vec![
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (503, "overloaded_error"),
        (529, "overloaded_error"),
        (500, "api_error"),
    ];

    for (code, expected_type) in tests {
        let err = ErrorResponse::new(code, "test");
        assert_eq!(
            err.error.error_type, expected_type,
            "Failed for code {}",
            code
        );
    }
}

#[test]
fn test_delta_serialization() {
    let delta = Delta::TextDelta {
        text: "Hello".to_string(),
    };
    let json = serde_json::to_value(&delta).unwrap();
    assert_eq!(json["type"], "text_delta");
    assert_eq!(json["text"], "Hello");
}

#[test]
fn test_message_delta_serialization() {
    let delta = MessageDelta {
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
    };
    let json = serde_json::to_value(&delta).unwrap();
    assert_eq!(json["stop_reason"], "end_turn");
}

#[test]
fn test_event_message_start() {
    let message = Message::builder()
        .id("msg_123".to_string())
        .model("claude-3-5-sonnet-20241022".to_string())
        .role("assistant".to_string())
        .message_type("message".to_string())
        .content(vec![])
        .build();

    let event = Event::MessageStart { message };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "message_start");
    assert_eq!(json["message"]["id"], "msg_123");
}

#[test]
fn test_event_content_block_start_text() {
    let content_block = ContentBlock::text_builder().text("".to_string()).build();

    let event = Event::ContentBlockStart {
        content_block,
        index: 0,
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "content_block_start");
    assert_eq!(json["index"], 0);
    assert_eq!(json["content_block"]["type"], "text");
}

#[test]
fn test_event_content_block_delta() {
    let delta = Delta::TextDelta {
        text: "Hello".to_string(),
    };

    let event = Event::ContentBlockDelta { delta, index: 0 };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "content_block_delta");
    assert_eq!(json["delta"]["type"], "text_delta");
    assert_eq!(json["delta"]["text"], "Hello");
}

#[test]
fn test_event_content_block_stop() {
    let event = Event::ContentBlockStop { index: 0 };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "content_block_stop");
    assert_eq!(json["index"], 0);
}

#[test]
fn test_event_message_delta() {
    let delta = MessageDelta {
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
    };
    let usage = DeltaUsage { output_tokens: 10 };

    let event = Event::MessageDelta { delta, usage };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "message_delta");
    assert_eq!(json["delta"]["stop_reason"], "end_turn");
    assert_eq!(json["usage"]["output_tokens"], 10);
}

#[test]
fn test_event_message_stop() {
    let event = Event::MessageStop;
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "message_stop");
}

#[test]
fn test_event_ping() {
    let event = Event::Ping;
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "ping");
}

#[test]
fn test_event_error() {
    let error = Error {
        error_type: "api_error".to_string(),
        message: "Something went wrong".to_string(),
    };

    let event = Event::Error { error };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["type"], "api_error");
}

#[test]
fn test_content_block_text_empty_field_present() {
    let block = ContentBlock::Text {
        text: "".to_string(),
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "text");
    assert!(json.get("text").is_some(), "text field should be present");
}

#[test]
fn test_content_block_thinking_empty_fields_present() {
    let block = ContentBlock::Thinking {
        thinking: "".to_string(),
        signature: "".to_string(),
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "thinking");
    assert!(
        json.get("thinking").is_some(),
        "thinking field should be present"
    );
    assert!(
        json.get("signature").is_some(),
        "signature field should be present"
    );
}

#[test]
fn test_content_block_tool_use() {
    let block = ContentBlock::tool_use_builder()
        .id("call_123".to_string())
        .name("get_weather".to_string())
        .input(json!({"location": "Paris"}))
        .build();

    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "tool_use");
    assert_eq!(json["id"], "call_123");
    assert_eq!(json["name"], "get_weather");
}

#[test]
fn test_all_delta_types() {
    let deltas = vec![
        Delta::TextDelta {
            text: "hello".to_string(),
        },
        Delta::InputJsonDelta {
            partial_json: r#"{"key":"value"}"#.to_string(),
        },
        Delta::ThinkingDelta {
            thinking: "thinking...".to_string(),
        },
        Delta::SignatureDelta {
            signature: "sig123".to_string(),
        },
    ];

    for delta in deltas {
        let json = serde_json::to_value(&delta).unwrap();
        assert!(json.get("type").is_some());
    }
}

#[test]
fn test_message_builder() {
    let message = Message::builder()
        .id("msg_123".to_string())
        .model("claude-3-5-sonnet-20241022".to_string())
        .role("assistant".to_string())
        .message_type("message".to_string())
        .content(vec![])
        .stop_reason(Some("end_turn".to_string()))
        .usage(Usage {
            input_tokens: 10,
            output_tokens: 20,
        })
        .build();

    assert_eq!(message.id, "msg_123");
    assert_eq!(message.stop_reason, Some("end_turn".to_string()));
    assert_eq!(message.usage.input_tokens, 10);
    assert_eq!(message.usage.output_tokens, 20);
}

#[test]
fn test_event_deserialization() {
    let json = json!({
        "type": "message_stop"
    });

    let event: Event = serde_json::from_value(json).unwrap();
    assert!(matches!(event, Event::MessageStop));
}

#[test]
fn test_error_response_serialization_roundtrip() {
    let err = ErrorResponse::new(400, "Bad request");
    let json = serde_json::to_value(&err).unwrap();
    let deserialized: ErrorResponse = serde_json::from_value(json).unwrap();

    assert_eq!(deserialized.response_type, "error");
    assert_eq!(deserialized.error.error_type, "invalid_request_error");
    assert_eq!(deserialized.error.message, "Bad request");
}
