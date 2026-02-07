use serde_json::Value;

pub struct CapturedExchange {
    pub timestamp: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub request_headers: Vec<(String, String)>,
    pub response_headers: Vec<(String, String)>,
    pub request_body: Option<Value>,
    pub response_body: Option<Value>,
    /// Raw SSE events as (event_type, data) pairs. Empty for non-SSE responses.
    pub sse_events: Vec<(String, String)>,
    pub request_summary: String,
    pub response_summary: String,
    pub error: Option<String>,
}

pub fn extract_request_summary(body: &Option<Value>) -> String {
    let Some(body) = body else {
        return String::new();
    };

    let model = body["model"]
        .as_str()
        .map(|m| m.strip_prefix("us.").unwrap_or(m))
        .unwrap_or("unknown");

    let Some(messages) = body["messages"].as_array() else {
        return model.to_string();
    };

    let last_user = messages.iter().rev().find(|m| m["role"] == "user");
    let Some(last_user) = last_user else {
        return model.to_string();
    };

    let text = extract_text_from_content(&last_user["content"]);
    if text.is_empty() {
        return model.to_string();
    }

    let preview = truncate_str(&text, 80);
    format!("{model} | \"{preview}\"")
}

pub fn extract_response_summary(body: &Option<Value>) -> String {
    let Some(body) = body else {
        return String::new();
    };

    // Check for error response
    if let Some(err_type) = body["error"]["type"].as_str() {
        return format!("error: {err_type}");
    }

    let Some(content) = body["content"].as_array() else {
        return "no content".to_string();
    };

    let block_types: Vec<&str> = content
        .iter()
        .filter_map(|block| block["type"].as_str())
        .collect();

    if block_types.is_empty() {
        return "empty".to_string();
    }

    let mut summary = block_types.join(" → ");

    if let Some(usage) = body["usage"].as_object() {
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        summary = format!("{summary} ({input}+{output} tokens)");
    }

    summary
}

/// Accumulate SSE events into a synthetic response JSON body.
pub fn accumulate_sse_events(events: &[(String, String)]) -> Value {
    let mut content_blocks: Vec<Value> = Vec::new();
    let mut current_text = String::new();
    let mut current_thinking = String::new();
    let mut current_input_json = String::new();
    let mut current_type: Option<String> = None;
    let mut model = Value::Null;
    let mut stop_reason = Value::Null;
    let mut usage = serde_json::Map::new();

    for (_event_type, data) in events {
        if data == "[DONE]" {
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<Value>(data) else {
            continue;
        };

        match parsed["type"].as_str() {
            Some("message_start") => {
                if let Some(msg) = parsed.get("message") {
                    if let Some(m) = msg.get("model") {
                        model = m.clone();
                    }
                    if let Some(u) = msg["usage"].as_object() {
                        for (k, v) in u {
                            usage.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
            Some("content_block_start") => {
                if let Some(block) = parsed.get("content_block") {
                    let block_type = block["type"].as_str().unwrap_or("unknown").to_string();
                    current_type = Some(block_type.clone());
                    current_text.clear();
                    current_thinking.clear();
                    current_input_json.clear();

                    match block_type.as_str() {
                        "tool_use" => {
                            // Start with the block info, input will be accumulated
                            let mut tool_block = block.clone();
                            // Remove empty input, will be set at stop
                            tool_block.as_object_mut().map(|m| {
                                m.insert("input".to_string(), Value::Object(Default::default()))
                            });
                            content_blocks.push(tool_block);
                        }
                        _ => {}
                    }
                }
            }
            Some("content_block_delta") => {
                if let Some(delta) = parsed.get("delta") {
                    match delta["type"].as_str() {
                        Some("text_delta") => {
                            if let Some(t) = delta["text"].as_str() {
                                current_text.push_str(t);
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(t) = delta["thinking"].as_str() {
                                current_thinking.push_str(t);
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(j) = delta["partial_json"].as_str() {
                                current_input_json.push_str(j);
                            }
                        }
                        Some("signature_delta") => {
                            // signature deltas are ignored for summary purposes
                        }
                        _ => {}
                    }
                }
            }
            Some("content_block_stop") => {
                match current_type.as_deref() {
                    Some("text") => {
                        content_blocks.push(serde_json::json!({
                            "type": "text",
                            "text": current_text,
                        }));
                    }
                    Some("thinking") => {
                        content_blocks.push(serde_json::json!({
                            "type": "thinking",
                            "thinking": current_thinking,
                        }));
                    }
                    Some("tool_use") => {
                        // Update the last tool_use block with accumulated input
                        if let Some(last) = content_blocks.last_mut() {
                            if let Ok(input_val) =
                                serde_json::from_str::<Value>(&current_input_json)
                            {
                                last.as_object_mut()
                                    .map(|m| m.insert("input".to_string(), input_val));
                            }
                        }
                    }
                    _ => {}
                }
                current_type = None;
            }
            Some("message_delta") => {
                if let Some(delta) = parsed.get("delta") {
                    if let Some(sr) = delta.get("stop_reason") {
                        stop_reason = sr.clone();
                    }
                }
                if let Some(u) = parsed["usage"].as_object() {
                    for (k, v) in u {
                        usage.insert(k.clone(), v.clone());
                    }
                }
            }
            _ => {}
        }
    }

    serde_json::json!({
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "usage": usage,
    })
}

fn extract_text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    // Truncate at char boundary
    let trimmed = s.replace('\n', " ");
    if trimmed.len() <= max_len {
        trimmed
    } else {
        let mut end = max_len;
        while !trimmed.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &trimmed[..end])
    }
}
