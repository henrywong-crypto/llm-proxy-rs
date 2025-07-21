use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::Sse},
    routing::post,
};
use chat::{
    openai::OpenAIChatCompletionsProvider,
    providers::{BedrockChatCompletionsProvider, ChatCompletionsProvider},
};
use config::{Config, File};
use request::{ChatCompletionsRequest, StreamOptions};
use response::Usage;
use tracing::{debug, error, info};

mod error;

use crate::error::AppError;

#[derive(Clone)]
struct AppState {
    openai_api_key: Option<String>,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(mut payload): Json<ChatCompletionsRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Only log tool results
    for msg in payload.messages.iter() {
        if msg.role == request::Role::Tool {
            if let Some(tool_call_id) = &msg.tool_call_id {
                if tool_call_id.contains("get_current_time") || tool_call_id.starts_with("tooluse_") {
                    info!("🕐 get_current_time result: {:?}", msg.contents);
                } else {
                    info!("🔧 Tool result: {:?}", msg.contents);
                }
            } else {
                info!("🔧 Tool result: {:?}", msg.contents);
            }
        }
    }

    if payload.stream == Some(false) {
        error!("Streaming is required but was disabled");
        return Err(AppError::from(anyhow::anyhow!(
            "Streaming is required but was disabled"
        )));
    }

    let model_name = payload.model.to_lowercase();

    payload.stream_options = Some(StreamOptions {
        include_usage: true,
    });

    let usage_callback = |_usage: &Usage| {};

    let stream = if model_name.starts_with("gpt-") {
        if let Some(openai_api_key) = state.openai_api_key {
            if openai_api_key.is_empty() {
                error!("OpenAI API key is empty but OpenAI model was requested");
                return Err(AppError::from(anyhow::anyhow!(
                    "OpenAI API key is empty but OpenAI model was requested"
                )));
            }
            OpenAIChatCompletionsProvider::new(&openai_api_key)
                .chat_completions_stream(payload, usage_callback)
                .await?
        } else {
            error!("OpenAI API key is not configured but OpenAI model was requested");
            return Err(AppError::from(anyhow::anyhow!(
                "OpenAI API key is not configured but OpenAI model was requested"
            )));
        }
    } else {
        match BedrockChatCompletionsProvider::new()
            .await
            .chat_completions_stream(payload, usage_callback)
            .await
        {
            Ok(stream) => {
                debug!("Successfully created Bedrock stream");
                stream
            }
            Err(e) => {
                error!("Failed to create Bedrock stream: {}", e);
                return Err(AppError::from(e));
            }
        }
    };

    Ok((StatusCode::OK, Sse::new(stream)))
}

async fn load_config() -> anyhow::Result<(String, u16, Option<String>)> {
    let settings = Config::builder()
        .add_source(File::with_name("config"))
        .build()?;

    let host: String = settings
        .get("host")
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = settings.get("port").unwrap_or(3000);

    let openai_api_key = settings.get::<String>("openai_api_key").ok();

    if openai_api_key.is_some() {
        info!("OpenAI API key found in configuration");
    } else {
        info!("No OpenAI API key found in configuration, OpenAI models will not be available");
    }

    Ok((host, port, openai_api_key))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("Initializing LLM proxy server");

    let (host, port, openai_api_key) = load_config().await?;
    info!("Starting server on {}:{}", host, port);

    let app_state = AppState { openai_api_key };

    let app = Router::new()
        .route("/chat/completions", post(chat_completions))
        .with_state(app_state);

    info!("Routes configured, binding to {}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    info!("Server started successfully, listening for requests");

    axum::serve(listener, app).await?;

    Ok(())
}
