pub mod content_block_delta;
pub mod event;
pub mod message;
mod stream;

pub use content_block_delta::*;
pub use event::*;
pub use message::*;
pub use stream::*;

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: AnthropicError,
}
