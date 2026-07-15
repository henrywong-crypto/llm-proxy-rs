use anyhow::anyhow;
use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response, sse::Sse},
};
use chat::provider::{MantleChatCompletionsProvider, bedrock_mantle_url, forward_mantle_post};
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

    let stream = MantleChatCompletionsProvider::new(
        state.http_client.clone(),
        state.aws_region.clone(),
        state.credentials_provider.clone(),
    )
    .chat_completions_stream(payload, log_token_usage)
    .await?;

    Ok((StatusCode::OK, Sse::new(stream)))
}

/// Transparent passthrough for the OpenAI Responses API. The request body is
/// forwarded verbatim to Bedrock Mantle's `/openai/v1/responses`, and the
/// upstream status, content-type, and (streaming or not) body are relayed back
/// unchanged — the Responses SSE format uses named events, so it must not be
/// reframed the way `/chat/completions` is.
pub async fn handle_responses(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, AppError> {
    let model = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| value["model"].as_str().map(str::to_owned));
    info!(
        "Received OpenAI Responses API request for model: {}",
        model.as_deref().unwrap_or("unknown")
    );

    // `openai.gpt-5.6-sol` (and peers) are served only on the bedrock-mantle host
    // at the `/openai/v1/responses` path — not on bedrock-runtime. The client's
    // request body (including its model) is forwarded unchanged.
    let url = bedrock_mantle_url(&state.aws_region, "/openai/v1/responses");
    let upstream = forward_mantle_post(
        &state.http_client,
        &state.credentials_provider,
        &state.aws_region,
        &url,
        body.to_vec(),
    )
    .await?;

    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    if !status.is_success() {
        error!("Bedrock Mantle Responses request returned {}", status);
    }

    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .body(Body::from_stream(upstream.bytes_stream()))
        .map_err(|e| anyhow!("Failed to build Responses proxy response: {}", e).into())
}
