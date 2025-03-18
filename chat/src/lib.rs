mod bedrock;
pub mod error;
pub mod openai; // Make openai module public
pub mod providers;

pub trait ProcessChatCompletionsRequest<T> {
    fn process_chat_completions_request(&self, request: &request::ChatCompletionsRequest) -> T;
}

// Re-export OpenAIChatCompletionsProvider for convenient access
pub use openai::OpenAIChatCompletionsProvider;
// Re-export error types
pub use error::{EventStream, StreamError, create_sse_event};
