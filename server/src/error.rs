use anthropic_response::ErrorResponse;
use axum::{Json, http::StatusCode, response::IntoResponse};
use tracing::error;

pub struct AppError {
    status: StatusCode,
    error: anyhow::Error,
}

impl AppError {
    pub fn new(status: StatusCode, error: anyhow::Error) -> Self {
        Self { status, error }
    }

    pub fn bad_request(error: anyhow::Error) -> Self {
        Self::new(StatusCode::BAD_REQUEST, error)
    }

    pub fn internal_error(error: anyhow::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, error)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!("Request error: {:?}", self.error);

        let error_response = ErrorResponse::new(self.status.as_u16(), self.error.to_string());

        (self.status, Json(error_response)).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self::internal_error(err.into())
    }
}
