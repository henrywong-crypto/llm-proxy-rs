use anyhow::anyhow;
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, sse::Sse},
};
use chat::provider::{BedrockChatCompletionsProvider, ChatCompletionsProvider};
use request::ChatCompletionsRequest;
use std::sync::Arc;
use tracing::{error, info};

use crate::{AppState, error::AppError, utils::usage_callback};
use super::anthropic::filter_anthropic_beta;

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
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

    let anthropic_beta = filter_anthropic_beta(&headers, &state.anthropic_beta);
    info!("anthropic_beta: {:?}", anthropic_beta);

    let stream = BedrockChatCompletionsProvider::new(state.bedrockruntime_client.clone())
        .chat_completions_stream(payload, anthropic_beta, usage_callback)
        .await?;

    Ok((StatusCode::OK, Sse::new(stream)))
}
