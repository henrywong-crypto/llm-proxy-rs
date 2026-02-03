use std::sync::Arc;

use anthropic_request::{
    V1MessagesCountTokensRequest, V1MessagesCountTokensResponse, V1MessagesRequest,
};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::Sse},
};
use chat::provider::{BedrockV1MessagesProvider, V1MessagesProvider};
use tracing::{error, info};

use crate::{AppState, error::AppError, utils::usage_callback};

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
    State(state): State<Arc<AppState>>,
    Json(payload): Json<V1MessagesCountTokensRequest>,
) -> Result<Json<V1MessagesCountTokensResponse>, AppError> {
    info!(
        "Received Anthropic v1/messages/count_tokens request for model: {}",
        payload.model
    );

    let provider = BedrockV1MessagesProvider::new().await;
    let input_tokens = provider
        .count_tokens(
            &payload,
            &state.count_tokens_strip_inference_profile_prefixes,
        )
        .await?;

    Ok(Json(V1MessagesCountTokensResponse { input_tokens }))
}
