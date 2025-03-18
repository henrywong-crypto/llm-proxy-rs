use axum::{http::StatusCode, response::IntoResponse};
use chat::StreamError;
use std::fmt;

pub struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        // Determine the appropriate status code based on the error
        let status_code = if self.0.to_string().contains("Streaming is required") {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };

        (status_code, format!("Error: {}", self.0)).into_response()
    }
}

// Implement Display for AppError
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Implement Debug for AppError
impl fmt::Debug for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AppError({:?})", self.0)
    }
}

// Implement std::error::Error for AppError
impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

// Implement From trait for specific error types, not generic over all types.
// This avoids the conflict with the blanket impl<T> From<T> for T
impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

impl From<StreamError> for AppError {
    fn from(err: StreamError) -> Self {
        Self(anyhow::anyhow!("{}", err))
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        Self(err.into())
    }
}

// Add more specific implementations as needed
