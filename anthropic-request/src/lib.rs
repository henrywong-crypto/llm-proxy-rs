use serde::{Deserialize, Serialize};

pub mod cache_control;
pub mod content;
pub mod content_block;
pub mod image_source;
pub mod message;
pub mod metadata;
pub mod system;
pub mod thinking;
pub mod tool;
pub mod tool_choice;
pub mod tool_result_content;

pub use cache_control::*;
pub use content::*;
pub use content_block::*;
pub use image_source::*;
pub use message::*;
pub use metadata::*;
pub use system::*;
pub use thinking::*;
pub use tool::*;
pub use tool_choice::*;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}
