pub mod bedrock;
pub mod providers;

use axum::response::sse::Event;
use response::ChatCompletionsResponse;

pub const DONE_MESSAGE: &str = "[DONE]";

pub trait ProcessChatCompletionsRequest<T> {
    fn process_chat_completions_request(
        &self,
        request: &request::ChatCompletionsRequest,
    ) -> anyhow::Result<T>;
}

pub fn create_sse_event(response: &ChatCompletionsResponse) -> anyhow::Result<Event> {
    match serde_json::to_string(response) {
        Ok(data) => Ok(Event::default().data(data)),
        Err(e) => anyhow::bail!("Failed to serialize response: {}", e),
    }
}

pub struct AnthropicEvent {
    pub event: Event,
    pub event_type: String,
}

pub fn create_anthropic_sse_events(
    response: &ChatCompletionsResponse,
) -> impl Iterator<Item = anyhow::Result<(AnthropicEvent, bool)>> + '_ {
    response.to_anthropic_events().map(|anthropic_response| {
        let event_type = anthropic_response.event_type.clone();
        let is_text_delta = matches!(
            anthropic_response.delta.as_ref(),
            Some(response::AnthropicDelta::TextDelta { .. })
        );
        
        eprintln!("DEBUG: Creating SSE event: type={}, index={:?}, is_text_delta={}", event_type, anthropic_response.index, is_text_delta);
        match serde_json::to_string(&anthropic_response) {
            Ok(data) => {
                eprintln!("DEBUG: Event data: {}", data);
                let event = Event::default().event(&event_type).data(data);
                Ok((AnthropicEvent { event, event_type }, is_text_delta))
            }
            Err(e) => Err(anyhow::anyhow!(
                "Failed to serialize Anthropic response: {}",
                e
            )),
        }
    })
}
