use serde::Serialize;

/// Anthropic SSE `error` event payload.
/// `{"type": "error", "error": {"type": "<error_type>", "message": "..."}}`
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ErrorEvent {
    #[serde(rename = "error")]
    Error { error: ApiError },
}

impl ErrorEvent {
    pub fn new(error_type: ErrorType, message: impl Into<String>) -> Self {
        Self::Error {
            error: ApiError {
                error_type,
                message: message.into(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    error_type: ErrorType,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    InvalidRequestError,
    PermissionError,
    NotFoundError,
    RateLimitError,
    ApiError,
    OverloadedError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_anthropic_wire_shape() {
        let event = ErrorEvent::new(ErrorType::OverloadedError, "Too busy");
        assert_eq!(
            serde_json::to_value(&event).expect("serialize"),
            serde_json::json!({
                "type": "error",
                "error": { "type": "overloaded_error", "message": "Too busy" }
            })
        );
    }
}
