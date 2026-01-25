use axum::{Router, routing::post};
use chat::bedrock::ReasoningEffortToThinkingBudgetTokens;
use std::sync::Arc;

pub mod error;
pub mod handlers;
pub mod utils;

pub struct AppState {
    pub reasoning_effort_to_thinking_budget_tokens: ReasoningEffortToThinkingBudgetTokens,
}

pub async fn create_router() -> anyhow::Result<Router> {
    let state = Arc::new(AppState {
        reasoning_effort_to_thinking_budget_tokens: ReasoningEffortToThinkingBudgetTokens::default(
        ),
    });

    let app = Router::new()
        .route(
            "/chat/completions",
            post(handlers::openai::chat_completions),
        )
        .route("/v1/messages", post(handlers::anthropic::v1_messages))
        .with_state(state);

    Ok(app)
}
