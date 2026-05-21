use serde::Serialize;

/// Anthropic SSE `error` event payload.
/// `{"type": "error", "error": {"type": "<code>", "message": "..."}}`
///
/// `code` is the upstream HTTP status (when known) or the Bedrock exception name.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ErrorEvent {
    #[serde(rename = "error")]
    Error { error: ApiError },
}

impl ErrorEvent {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error {
            error: ApiError {
                code: code.into(),
                message: message.into(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    code: String,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_bedrock_code_and_message() {
        let event = ErrorEvent::new("400", "input is too long for requested model");
        assert_eq!(
            serde_json::to_value(&event).expect("serialize"),
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "400",
                    "message": "input is too long for requested model"
                }
            })
        );
    }
}
