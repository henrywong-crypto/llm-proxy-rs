use anthropic_request::V1MessagesRequest;
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, sse::Sse},
};
use chat::provider::{BedrockV1MessagesProvider, V1MessagesProvider};
use tracing::{error, info};

use crate::{error::AppError, utils::usage_callback};

pub async fn v1_messages(
    Json(payload): Json<V1MessagesRequest>,
) -> Result<impl IntoResponse, AppError> {
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
