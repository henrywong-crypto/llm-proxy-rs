use anyhow::anyhow;
use aws_sdk_bedrockruntime::types::TokenUsage;
use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response, sse::Sse},
};
use chat::provider::{MantleChatCompletionsProvider, bedrock_mantle_url, forward_mantle_post};
use futures::{Stream, StreamExt};
use request::ChatCompletionsRequest;
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
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

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    if !status.is_success() {
        error!("Bedrock Mantle Responses request returned {}", status);
    }

    let body_stream = ResponsesUsageStream::new(upstream.bytes_stream());

    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .body(Body::from_stream(body_stream))
        .map_err(|e| anyhow!("Failed to build Responses proxy response: {}", e).into())
}

struct ResponsesUsageStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> ResponsesUsageStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S, E> Stream for ResponsesUsageStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                self.buffer.extend_from_slice(&bytes);
                inspect_complete_sse_frames(&mut self.buffer);
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(None) => {
                inspect_response_json(&self.buffer);
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

fn inspect_complete_sse_frames(buffer: &mut Vec<u8>) {
    while let Some((frame_end, delimiter_len)) = find_sse_frame_end(buffer) {
        let frame = buffer
            .drain(..frame_end + delimiter_len)
            .collect::<Vec<_>>();
        let frame = String::from_utf8_lossy(&frame[..frame_end]);
        for line in frame.lines() {
            if let Some(payload) = line.strip_prefix("data:") {
                inspect_response_json(payload.trim_start().as_bytes());
            }
        }
    }
}

fn find_sse_frame_end(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(lf), Some(crlf)) if lf <= crlf => Some((lf, 2)),
        (Some(_), Some(crlf)) => Some((crlf, 4)),
        (Some(lf), None) => Some((lf, 2)),
        (None, Some(crlf)) => Some((crlf, 4)),
        (None, None) => None,
    }
}

fn inspect_response_json(bytes: &[u8]) {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return;
    };
    let response = value.get("response").unwrap_or(&value);
    if let Some(usage) = response.get("usage").and_then(parse_responses_usage) {
        log_token_usage(&usage);
    }
}

fn parse_responses_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    let input = usage.get("input_tokens")?.as_i64()?;
    let output = usage.get("output_tokens")?.as_i64()?;
    let total = usage.get("total_tokens")?.as_i64()?;

    TokenUsage::builder()
        .input_tokens(i32::try_from(input).ok()?)
        .output_tokens(i32::try_from(output).ok()?)
        .total_tokens(i32::try_from(total).ok()?)
        .build()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_responses_usage() {
        let usage = serde_json::json!({
            "input_tokens": 12,
            "output_tokens": 7,
            "total_tokens": 19
        });
        let usage = parse_responses_usage(&usage).unwrap();
        assert_eq!(usage.input_tokens(), 12);
        assert_eq!(usage.output_tokens(), 7);
        assert_eq!(usage.total_tokens(), 19);
    }

    #[test]
    fn rejects_out_of_range_responses_usage() {
        let usage = serde_json::json!({
            "input_tokens": i64::MAX,
            "output_tokens": 7,
            "total_tokens": i64::MAX
        });
        assert!(parse_responses_usage(&usage).is_none());
    }

    #[test]
    fn finds_crlf_sse_frame() {
        assert_eq!(
            find_sse_frame_end(b"event: done\r\n\r\nnext"),
            Some((11, 4))
        );
    }
}
