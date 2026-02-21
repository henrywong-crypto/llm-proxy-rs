use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use config::{Config, File};
use server::{AppState, app};
use std::sync::Arc;
use tracing::info;

async fn load_config() -> anyhow::Result<(String, u16, Vec<String>, Vec<String>)> {
    let settings = Config::builder()
        .add_source(File::with_name("config"))
        .build()?;

    let host: String = settings
        .get("host")
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = settings.get("port").unwrap_or(3000);

    let inference_profile_prefixes: Vec<String> = settings
        .get("inference_profile_prefixes")
        .unwrap_or_else(|_| vec!["us.".to_string()]);

    info!(
        "inference_profile_prefixes: {:?}",
        inference_profile_prefixes
    );

    let anthropic_beta_whitelist: Vec<String> = settings
        .get("anthropic_beta_whitelist")
        .unwrap_or_else(|_| vec![]);

    info!("anthropic_beta_whitelist: {:?}", anthropic_beta_whitelist);

    Ok((host, port, inference_profile_prefixes, anthropic_beta_whitelist))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("Initializing LLM proxy server");

    let (host, port, inference_profile_prefixes, anthropic_beta_whitelist) = load_config().await?;
    info!("Starting server on {}:{}", host, port);

    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let bedrockruntime_client = Client::new(&config);
    info!("AWS Bedrock client initialized");

    let state = Arc::new(AppState {
        bedrockruntime_client,
        inference_profile_prefixes,
        anthropic_beta_whitelist,
    });

    let app = app(state);

    info!("Routes configured, binding to {}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    info!("Server started successfully, listening for requests");

    axum::serve(listener, app).await?;

    Ok(())
}
