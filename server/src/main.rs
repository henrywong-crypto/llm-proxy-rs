use axum::{
    Json, Router, http::StatusCode, response::IntoResponse, response::sse::Sse, routing::post,
};
use chat::providers::{BedrockChatCompletionsProvider, ChatCompletionsProvider};
use chat::{EventStream, OpenAIChatCompletionsProvider, StreamError};
use config::{Config, File};
use request::ChatCompletionsRequest;
use response::Usage;
use std::env;
use tracing::{debug, error, info};

mod error;

use crate::error::AppError;

#[derive(Debug)]
enum Provider {
    Bedrock,
    OpenAI,
}

async fn chat_completions(
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<impl IntoResponse, AppError> {
    debug!(
        "Received chat completions request for model: {}",
        payload.model
    );

    if payload.stream == Some(false) {
        error!("Streaming is required but was disabled");
        return Err(AppError::from(anyhow::anyhow!(
            "Streaming is required but was disabled"
        )));
    }

    // Load settings to determine which provider to use
    let settings = load_settings().await.map_err(AppError::from)?;
    let provider_type = determine_provider(&payload, &settings)?;

    info!(
        "Processing chat completions request with {} messages using provider {:?}",
        payload.messages.len(),
        provider_type
    );

    // Create a usage callback that can be reused
    let usage_callback = |usage: &Usage| {
        info!(
            "Usage: prompt_tokens: {}, completion_tokens: {}, total_tokens: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    };

    // Process based on provider type
    match provider_type {
        Provider::Bedrock => {
            let provider = BedrockChatCompletionsProvider::new().await;
            let stream = provider
                .chat_completions_stream(payload, usage_callback)
                .await?;
            // We already have a boxed stream from the provider
            Ok((StatusCode::OK, Sse::new(stream)))
        }
        Provider::OpenAI => {
            // Check if the API key is in config
            if let Ok(api_key) = settings.get_string("openai_api_key") {
                // Set environment variable if it's specified in config
                // This is unsafe because it modifies process-wide state that other
                // threads might read concurrently
                unsafe {
                    env::set_var("OPENAI_API_KEY", api_key);
                }
                debug!("Using OpenAI API key from config");
            }
            let provider = OpenAIChatCompletionsProvider::new().await.map_err(|e| {
                AppError::from(anyhow::anyhow!(
                    "Failed to initialize OpenAI provider: {}",
                    e
                ))
            })?;

            let stream = provider
                .chat_completions_stream(payload, usage_callback)
                .await?;
            // We already have a boxed stream from the provider
            Ok((StatusCode::OK, Sse::new(stream)))
        }
    }
}

fn determine_provider(
    request: &ChatCompletionsRequest,
    settings: &Config,
) -> Result<Provider, AppError> {
    // First check if there's an explicit provider setting
    if let Ok(provider) = settings.get_string("provider") {
        match provider.to_lowercase().as_str() {
            "openai" => return Ok(Provider::OpenAI),
            "bedrock" => return Ok(Provider::Bedrock),
            _ => {}
        }
    }

    // Otherwise, determine by model name
    let model = &request.model;
    if model.starts_with("gpt") || model.contains("openai") {
        Ok(Provider::OpenAI)
    } else if model.contains("anthropic") || model.contains("amazon") || model.contains("claude") {
        Ok(Provider::Bedrock)
    } else {
        // Default to Bedrock if we can't determine
        debug!(
            "Could not determine provider from model '{}', defaulting to Bedrock",
            model
        );
        Ok(Provider::Bedrock)
    }
}

async fn load_settings() -> anyhow::Result<Config> {
    Config::builder()
        .add_source(File::with_name("config"))
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))
}

async fn load_config() -> anyhow::Result<(String, u16)> {
    let settings = load_settings().await?;

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

    let app = Router::new().route("/chat/completions", post(chat_completions));

    info!("Routes configured, binding to {}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;
    info!("Server started successfully, listening for requests");

    axum::serve(listener, app).await?;

    Ok(())
}
