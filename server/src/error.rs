use anyhow::Error as AnyhowError;
use axum::{http::StatusCode, response::IntoResponse};
use tracing::error;

pub struct AppError(AnyhowError);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!("Request error: {}", self.0.root_cause());

        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error: {}", self.0),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<AnyhowError>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
