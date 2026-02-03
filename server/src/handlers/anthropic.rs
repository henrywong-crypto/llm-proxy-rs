use anthropic_request::{
    AssistantContent, AssistantContents, Message, Messages, System, Systems, ToolResultContent,
    ToolResultContents, UserContent, UserContents, V1MessagesCountTokensRequest,
    V1MessagesCountTokensResponse, V1MessagesRequest,
};
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

pub async fn v1_messages_count_tokens(
    Json(payload): Json<V1MessagesCountTokensRequest>,
) -> Result<Json<V1MessagesCountTokensResponse>, AppError> {
    info!(
        "Received Anthropic v1/messages/count_tokens request for model: {}",
        payload.model
    );

    let input_tokens = estimate_tokens(&payload);

    Ok(Json(V1MessagesCountTokensResponse { input_tokens }))
}

fn estimate_tokens(request: &V1MessagesCountTokensRequest) -> i32 {
    let mut total_chars = 0;

    // Count characters in messages
    total_chars += count_messages_chars(&request.messages);

    // Count characters in system prompt
    if let Some(ref system) = request.system {
        total_chars += count_system_chars(system);
    }

    // Count characters in tools (tool definitions add tokens)
    if let Some(ref tools) = request.tools {
        for tool in tools {
            total_chars += tool.name.len();
            total_chars += tool.description.len();
            total_chars += tool.input_schema.to_string().len();
        }
    }

    // Rough estimate: ~4 characters per token for English text
    // Add 10% buffer for tokenization overhead
    let estimated_tokens = (total_chars as f64 / 4.0 * 1.1) as i32;

    estimated_tokens.max(1)
}

fn count_messages_chars(messages: &Messages) -> usize {
    match messages {
        Messages::String(s) => s.len(),
        Messages::Array(arr) => arr.iter().map(count_message_chars).sum(),
    }
}

fn count_message_chars(message: &Message) -> usize {
    match message {
        Message::User { content } => count_user_contents_chars(content),
        Message::Assistant { content } => count_assistant_contents_chars(content),
    }
}

fn count_user_contents_chars(contents: &UserContents) -> usize {
    match contents {
        UserContents::String(s) => s.len(),
        UserContents::Array(arr) => arr.iter().map(count_user_content_chars).sum(),
    }
}

fn count_user_content_chars(content: &UserContent) -> usize {
    match content {
        UserContent::Text { text, .. } => text.len(),
        UserContent::ToolResult { content, .. } => count_tool_result_contents_chars(content),
    }
}

fn count_tool_result_contents_chars(contents: &ToolResultContents) -> usize {
    match contents {
        ToolResultContents::String(s) => s.len(),
        ToolResultContents::Array(arr) => arr.iter().map(count_tool_result_content_chars).sum(),
    }
}

fn count_tool_result_content_chars(content: &ToolResultContent) -> usize {
    match content {
        ToolResultContent::Text { text } => text.len(),
    }
}

fn count_assistant_contents_chars(contents: &AssistantContents) -> usize {
    match contents {
        AssistantContents::String(s) => s.len(),
        AssistantContents::Array(arr) => arr.iter().map(count_assistant_content_chars).sum(),
    }
}

fn count_assistant_content_chars(content: &AssistantContent) -> usize {
    match content {
        AssistantContent::Text { text, .. } => text.len(),
        AssistantContent::ToolUse { name, input, .. } => name.len() + input.to_string().len(),
        AssistantContent::Thinking {
            thinking,
            signature,
        } => thinking.len() + signature.len(),
    }
}

fn count_system_chars(system: &Systems) -> usize {
    match system {
        Systems::String(s) => s.len(),
        Systems::Array(arr) => arr.iter().map(count_system_content_chars).sum(),
    }
}

fn count_system_content_chars(content: &System) -> usize {
    match content {
        System::Text { text, .. } => text.len(),
    }
}
