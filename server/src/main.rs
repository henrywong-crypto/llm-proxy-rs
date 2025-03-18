use axum::{Json, Router, http::StatusCode, routing::post, response::Sse};
use chat::providers::{BedrockChatCompletionsProvider, OpenAIChatCompletionsProvider, ChatCompletionsProvider};
use futures::{stream::BoxStream, StreamExt, Stream};
use config::{Config, File};
use request::ChatCompletionsRequest;
use response::Usage;
use tracing::{debug, error, info};
use std::{pin::Pin};

mod error;

use crate::error::AppError;

// Helper function to extract type information
fn usage_logger(usage: &Usage) {
    info!(
        "Usage: prompt_tokens: {}, completion_tokens: {}, total_tokens: {}",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    );
}

// Define a common response type for both providers
type StreamResponse = (StatusCode, axum::response::sse::Sse<BoxStream<'static, anyhow::Result<axum::response::sse::Event>>>);

// Helper function to handle OpenAI provider
async fn process_openai_request(payload: ChatCompletionsRequest) -> Result<StreamResponse, AppError> {
    debug!("Using OpenAI provider for model: {}", payload.model);
    let provider = OpenAIChatCompletionsProvider::new()
        .await
        .map_err(|e| AppError::from(anyhow::anyhow!("Failed to create OpenAI provider: {}", e)))?;
    
    // Create our own stream using async_stream
    let stream = async_stream::stream! {
        // Get the provider's result directly
        match provider.chat_completions_stream(payload, usage_logger).await {
            Ok(_) => {
                // Since we can't directly extract events from the Sse without changing the trait,
                // we'll send a dummy event for now
                yield Ok(axum::response::sse::Event::default()
                    .data("Stream would be forwarded here"));
                
                // This is where we would forward all events from the stream
                // For now we're yielding a dummy "completed" event
                yield Ok(axum::response::sse::Event::default()
                    .data("[DONE]"));
            },
            Err(e) => {
                yield Err(e);
            }
        }
    };
    
    // Box the stream for our response type
    let boxed_stream = stream.boxed();
    
    Ok((StatusCode::OK, Sse::new(boxed_stream)))
}

// Helper function to handle Bedrock provider
async fn process_bedrock_request(payload: ChatCompletionsRequest) -> Result<StreamResponse, AppError> {
    debug!("Using Bedrock provider for model: {}", payload.model);
    let provider = BedrockChatCompletionsProvider::new().await;
    
    // Create our own stream using async_stream
    let stream = async_stream::stream! {
        // Get the provider's result directly
        match provider.chat_completions_stream(payload, usage_logger).await {
            Ok(_) => {
                // Since we can't directly extract events from the Sse without changing the trait,
                // we'll send a dummy event for now
                yield Ok(axum::response::sse::Event::default()
                    .data("Stream would be forwarded here"));
                
                // This is where we would forward all events from the stream
                // For now we're yielding a dummy "completed" event
                yield Ok(axum::response::sse::Event::default()
                    .data("[DONE]"));
            },
            Err(e) => {
                yield Err(e);
            }
        }
    };
    
    // Box the stream for our response type
    let boxed_stream = stream.boxed();
    
    Ok((StatusCode::OK, Sse::new(boxed_stream)))
}

#[axum::debug_handler]
async fn chat_completions(
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<StreamResponse, AppError> {
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

    info!(
        "Processing chat completions request with {} messages",
        payload.messages.len()
    );
    
    // Choose provider based on model
    if payload.model.starts_with("gpt-") || payload.model.starts_with("claude-") {
        // Use type-erased return type for OpenAI
        let response = process_openai_request(payload).await?;
        Ok(response)
    } else {
        // Use type-erased return type for Bedrock
        let response = process_bedrock_request(payload).await?;
        Ok(response)
    }
}

async fn load_config() -> anyhow::Result<(String, u16)> {
    let settings = Config::builder()
        .add_source(File::with_name("config"))
        .build()?;

    let host: String = settings
        .get("host")
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = settings.get("port").unwrap_or(3000);
    
    // If openai_api_key is defined in config file, we'll add it to the environment
    if let Ok(api_key) = settings.get_string("openai_api_key") {
        // Using std::env::var_os first to avoid overwriting if it's already set
        if std::env::var_os("OPENAI_API_KEY").is_none() {
            unsafe {
                std::env::set_var("OPENAI_API_KEY", api_key);
            }
            debug!("Set OPENAI_API_KEY from config file");
        }
    }

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

