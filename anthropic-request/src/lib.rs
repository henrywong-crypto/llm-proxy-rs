use serde::{Deserialize, Serialize};

pub mod cache_control;
pub mod content;
pub mod message;
pub mod system;
pub mod thinking;
pub mod tool;
pub mod tool_result_content;

pub use cache_control::*;
pub use content::*;
pub use message::*;
pub use system::*;
pub use thinking::*;
pub use tool::*;
pub use tool_result_content::*;

#[derive(Debug, Deserialize, Serialize)]
pub struct V1MessagesRequest {
    pub max_tokens: i32,
    pub messages: Messages,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Systems>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct V1MessagesCountTokensRequest {
    pub model: String,
    pub messages: Messages,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Systems>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct V1MessagesCountTokensResponse {
    pub input_tokens: i32,
}
