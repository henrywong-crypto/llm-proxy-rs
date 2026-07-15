use anyhow::anyhow;
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::Sse},
};
use chat::provider::MantleChatCompletionsProvider;
use request::ChatCompletionsRequest;
use std::sync::Arc;
use tracing::{error, info};

use crate::{AppState, error::AppError, utils::log_token_usage};

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<impl IntoResponse, AppError> {
    info!(
        "Received OpenAI chat completions request for model: {}",
        payload.model
    );

    if payload.stream == Some(false) {
        error!("Stream is set to false");
        return Err(anyhow!("Stream is set to false").into());
    }

    let stream = MantleChatCompletionsProvider::new(
        state.http_client.clone(),
        state.aws_region.clone(),
        state.credentials_provider.clone(),
    )
    .chat_completions_stream(payload, log_token_usage)
    .await?;

    Ok((StatusCode::OK, Sse::new(stream)))
}
