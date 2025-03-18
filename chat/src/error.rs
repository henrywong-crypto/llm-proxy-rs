use axum::response::sse::Event;
use std::fmt;

/// Error type for streaming operations
#[derive(Debug)]
pub enum StreamError {
    /// Error when serializing/deserializing data
    Serialization(serde_json::Error),
    /// Error with the underlying stream
    StreamReceive(String),
    /// Error from AWS SDK
    AwsSdk(aws_sdk_bedrockruntime::Error),
    /// Error from HTTP client
    HttpClient(reqwest::Error),
    /// Error with API response
    ApiResponse { status: u16, message: String },
    /// Any other error
    Other(String),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::Serialization(err) => write!(f, "Serialization error: {}", err),
            StreamError::StreamReceive(err) => write!(f, "Stream receive error: {}", err),
            StreamError::AwsSdk(err) => write!(f, "AWS SDK error: {}", err),
            StreamError::HttpClient(err) => write!(f, "HTTP client error: {}", err),
            StreamError::ApiResponse { status, message } => {
                write!(f, "API error: HTTP {}: {}", status, message)
            }
            StreamError::Other(err) => write!(f, "Error: {}", err),
        }
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamError::Serialization(err) => Some(err),
            StreamError::AwsSdk(err) => Some(err),
            StreamError::HttpClient(err) => Some(err),
            _ => None,
        }
    }
}

// Implement conversions from specific error types
impl From<serde_json::Error> for StreamError {
    fn from(err: serde_json::Error) -> Self {
        StreamError::Serialization(err)
    }
}

impl From<aws_sdk_bedrockruntime::Error> for StreamError {
    fn from(err: aws_sdk_bedrockruntime::Error) -> Self {
        StreamError::AwsSdk(err)
    }
}

impl From<reqwest::Error> for StreamError {
    fn from(err: reqwest::Error) -> Self {
        StreamError::HttpClient(err)
    }
}

/// Create an SSE event from a response
pub fn create_sse_event(
    response: &response::ChatCompletionsResponse,
) -> Result<Event, StreamError> {
    match serde_json::to_string(response) {
        Ok(data) => Ok(Event::default().data(data)),
        Err(e) => Err(StreamError::Serialization(e)),
    }
}

/// Type alias for the stream type to make it easier to work with
pub type EventStream =
    std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<Event, StreamError>> + Send>>;
