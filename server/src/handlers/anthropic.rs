use anthropic_request::V1MessagesRequest;
use axum::{
    Json,
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, sse::Sse},
    extract::FromRequest,
};
use chat::provider::{BedrockV1MessagesProvider, V1MessagesProvider};
use tracing::{debug, error, info};

use crate::{error::AppError, utils::usage_callback};

pub async fn v1_messages(
    body: Bytes,
) -> Result<impl IntoResponse, AppError> {
    // Log the raw JSON for debugging
    let json_str = String::from_utf8_lossy(&body);
    debug!("📥 Raw request body length: {} bytes", body.len());
    
    // Try to parse as JSON Value first to inspect structure
    let json_value: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| {
            error!("❌ Failed to parse JSON: {}", e);
            AppError::from(anyhow::anyhow!("Invalid JSON: {}", e))
        })?;
    
    // Log messages field structure
    if let Some(messages) = json_value.get("messages") {
        debug!("🔍 messages field type: {}", match messages {
            serde_json::Value::String(_) => "String",
            serde_json::Value::Array(arr) => {
                let msg = format!("Array with {} elements", arr.len());
                // Log first few elements
                for (i, item) in arr.iter().take(3).enumerate() {
                    if let Some(role) = item.get("role") {
                        debug!("  messages[{}].role = {:?}", i, role);
                    }
                    if let Some(content) = item.get("content") {
                        match content {
                            serde_json::Value::Array(content_arr) => {
                                debug!("  messages[{}].content: Array with {} items", i, content_arr.len());
                                for (j, c) in content_arr.iter().take(3).enumerate() {
                                    if let Some(content_type) = c.get("type") {
                                        debug!("    content[{}].type = {:?}", j, content_type);
                                    }
                                }
                            }
                            serde_json::Value::String(s) => {
                                debug!("  messages[{}].content: String (length: {})", i, s.len());
                            }
                            _ => {
                                debug!("  messages[{}].content: Other type", i);
                            }
                        }
                    }
                }
                msg
            }
            _ => "Other",
        });
    }
    
    // Now try to deserialize into V1MessagesRequest
    let payload: V1MessagesRequest = serde_json::from_value(json_value)
        .map_err(|e| {
            error!("❌ Failed to deserialize V1MessagesRequest: {}", e);
            AppError::from(anyhow::anyhow!("Failed to deserialize request: {}", e))
        })?;
    info!(
        "Received Anthropic v1/messages request for model: {}",
        payload.model
    );
    
    // Debug log the messages structure
    match &payload.messages {
        anthropic_request::Messages::String(s) => {
            info!("📝 Messages is a String (length: {})", s.len());
        }
        anthropic_request::Messages::Array(arr) => {
            info!("📝 Messages is an Array with {} messages", arr.len());
            for (i, msg) in arr.iter().enumerate() {
                match msg {
                    anthropic_request::Message::User { content } => {
                        info!("  Message[{}]: User with {} content blocks", i, content.len());
                    }
                    anthropic_request::Message::Assistant { content } => {
                        info!("  Message[{}]: Assistant with {} content blocks", i, content.len());
                        for (j, c) in content.iter().enumerate() {
                            match c {
                                anthropic_request::AssistantContent::Text { .. } => {
                                    info!("    Content[{}]: Text", j);
                                }
                                anthropic_request::AssistantContent::ToolUse { .. } => {
                                    info!("    Content[{}]: ToolUse", j);
                                }
                                anthropic_request::AssistantContent::Thinking { .. } => {
                                    info!("    Content[{}]: Thinking", j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if payload.stream == Some(false) {
        error!("Stream is set to false");
        return Err(anyhow::anyhow!("Stream is set to false").into());
    }

    let stream = BedrockV1MessagesProvider::new()
        .await
        .v1_messages_stream(payload, usage_callback)
        .await?;

    Ok((StatusCode::OK, Sse::new(stream)))
}
