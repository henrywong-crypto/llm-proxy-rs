pub mod bedrock;
pub mod openai;
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
) -> anyhow::Result<Vec<AnthropicEvent>> {
    let anthropic_events = response.to_anthropic_events();
    let mut events = Vec::new();

    for anthropic_response in anthropic_events {
        let event_type = anthropic_response.event_type.clone();
        match serde_json::to_string(&anthropic_response) {
            Ok(data) => {
                let event = Event::default()
                    .event(&event_type)
                    .data(data);
                events.push(AnthropicEvent { event, event_type });
            }
            Err(e) => anyhow::bail!("Failed to serialize Anthropic response: {}", e),
        }
    }

    Ok(events)
}
