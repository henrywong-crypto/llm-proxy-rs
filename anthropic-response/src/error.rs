use serde::{Deserialize, Serialize};

/// Error types matching Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl Error {
    pub fn new(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error_type: error_type.into(),
            message: message.into(),
        }
    }

    /// Creates an API error
    pub fn api_error(message: impl Into<String>) -> Self {
        Self::new("api_error", message)
    }

    /// Creates an invalid request error
    pub fn invalid_request_error(message: impl Into<String>) -> Self {
        Self::new("invalid_request_error", message)
    }

    /// Creates an overloaded error
    pub fn overloaded_error(message: impl Into<String>) -> Self {
        Self::new("overloaded_error", message)
    }
}
