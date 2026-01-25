use serde::{Deserialize, Serialize};

/// Error types matching Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String, // always "error"
    pub error: Error,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl ErrorResponse {
    /// Creates a new ErrorResponse with the appropriate error type based on HTTP status code
    pub fn new(code: u16, message: impl Into<String>) -> Self {
        let error_type = match code {
            400 => "invalid_request_error",
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            429 => "rate_limit_error",
            503 | 529 => "overloaded_error",
            _ => "api_error",
        };

        ErrorResponse {
            response_type: "error".to_string(),
            error: Error {
                error_type: error_type.to_string(),
                message: message.into(),
            },
            request_id: Some(generate_request_id()),
        }
    }
}

/// Generates a unique request ID
fn generate_request_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let random_bytes: [u8; 12] = rng.random();
    format!("req_{}", hex_encode(random_bytes))
}

fn hex_encode(bytes: [u8; 12]) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}
