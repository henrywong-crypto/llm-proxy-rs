use anthropic_response::{AnthropicError, AnthropicErrorResponse};
use axum::{Json, http::StatusCode, response::IntoResponse};
use tracing::error;

pub struct AppError {
    pub status_code: StatusCode,
    pub error_type: String,
    pub message: String,
}

impl AppError {
    pub fn new(status_code: StatusCode, error_type: String, message: String) -> Self {
        Self {
            status_code,
            error_type,
            message,
        }
    }

    pub fn internal_server_error(message: String) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error".to_string(),
            message,
        )
    }

    pub fn bad_request(message: String) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error".to_string(),
            message,
        )
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!("Request error [{}]: {}", self.error_type, self.message);

        let error_response = AnthropicErrorResponse {
            response_type: "error".to_string(),
            error: AnthropicError {
                error_type: self.error_type,
                message: self.message,
            },
        };

        (self.status_code, Json(error_response)).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        let err_string = err.to_string();

        // Check for AWS Bedrock validation errors
        if err_string.contains("ValidationException") || err_string.contains("Input is too long") {
            return Self::bad_request(err_string);
        }

        // Check for other AWS Bedrock errors that might indicate client errors
        if err_string.contains("ResourceNotFoundException") {
            return Self::new(
                StatusCode::NOT_FOUND,
                "not_found_error".to_string(),
                err_string,
            );
        }

        if err_string.contains("ThrottlingException") {
            return Self::new(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error".to_string(),
                err_string,
            );
        }

        // Default to internal server error
        Self::internal_server_error(err_string)
    }
}
