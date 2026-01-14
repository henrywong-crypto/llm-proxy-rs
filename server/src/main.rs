use axum::{
    Json, Router,
    http::StatusCode,
    response::{IntoResponse, sse::Sse},
    routing::post,
};
use chat::providers::{BedrockChatCompletionsProvider, ChatCompletionsProvider};
use config::{Config, File};
use request::{AnthropicRequest, ChatCompletionsRequest, StreamOptions};
use response::Usage;
use tracing::{debug, error, info};

mod error;

use crate::error::AppError;

/// Common validation and setup for streaming requests
fn validate_streaming_request(stream: Option<bool>) -> Result<(), AppError> {
    if stream == Some(false) {
        error!("Streaming is required but was disabled");
        return Err(AppError::from(anyhow::anyhow!(
            "Streaming is required but was disabled"
        )));
    }
    Ok(())
}

/// Create a usage callback for logging token usage
fn create_usage_callback() -> impl Fn(&Usage) {
    |usage: &Usage| {
        info!(
            "Usage: prompt_tokens: {}, completion_tokens: {}, total_tokens: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }
}

async fn anthropic_messages(
    Json(payload): Json<AnthropicRequest>,
) -> Result<impl IntoResponse, AppError> {
    debug!(
        "Received Anthropic messages request for model: {}",
        payload.model
    );
    validate_streaming_request(payload.stream)?;

    let usage_callback = create_usage_callback();

    // Use Bedrock provider
    info!("Using Bedrock provider for model: {}", payload.model);
    let anthropic_stream = BedrockChatCompletionsProvider::new()
        .await
        .anthropic_to_bedrock_stream(payload, usage_callback)
        .await
        .map_err(|e| {
            error!("Bedrock provider error: {}", e);
            AppError::from(e)
        })?;

    Ok((StatusCode::OK, Sse::new(anthropic_stream)))
}

async fn chat_completions(
    Json(mut payload): Json<ChatCompletionsRequest>,
) -> Result<impl IntoResponse, AppError> {
    debug!(
        "Received chat completions request for model: {}",
        payload.model
    );
    validate_streaming_request(payload.stream)?;

    payload.stream_options = Some(StreamOptions {
        include_usage: true,
    });
    let usage_callback = create_usage_callback();

    // Use Bedrock provider
    info!("Using Bedrock provider for model: {}", payload.model);
    let stream = BedrockChatCompletionsProvider::new()
        .await
        .chat_completions_stream(payload, usage_callback)
        .await?;

    Ok((StatusCode::OK, Sse::new(stream)))
}

async fn load_config() -> anyhow::Result<(String, u16)> {
    let settings = Config::builder()
        .add_source(File::with_name("config"))
        .build()?;

    let host: String = settings
        .get("host")
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = settings.get("port").unwrap_or(3000);

    Ok((host, port))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("Initializing LLM proxy server");

    let (host, port) = load_config().await?;
    info!("Starting server on {}:{}", host, port);

    let app = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages));

    info!("Routes configured, binding to {}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    info!("Server started successfully, listening for requests");

    axum::serve(listener, app).await?;

    Ok(())
}
