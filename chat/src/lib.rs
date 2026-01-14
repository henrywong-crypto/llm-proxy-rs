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
) -> impl Iterator<Item = anyhow::Result<AnthropicEvent>> + '_ {
    let mut events: Vec<_> = response.to_anthropic_events().collect();
    
    // Bedrock doesn't send ContentBlockStart for text, only for tools
    // If the first content event is a delta (not a start), inject a start event
    let needs_text_start = events.iter().enumerate().find_map(|(i, event)| {
        if event.event_type == "content_block_delta" && event.index == Some(0) {
            // Check if there's a content_block_start before this
            let has_start = events[..i].iter().any(|e| {
                e.event_type == "content_block_start" && e.index == Some(0)
            });
            Some(!has_start)
        } else {
            None
        }
    }).unwrap_or(false);
    
    if needs_text_start {
        // Find the position to insert (right before first content_block_delta at index 0)
        if let Some(pos) = events.iter().position(|e| {
            e.event_type == "content_block_delta" && e.index == Some(0)
        }) {
            events.insert(pos, response::AnthropicStreamResponse {
                event_type: "content_block_start".to_string(),
                message: None,
                index: Some(0),
                content_block: Some(response::AnthropicContentBlock::Text {
                    text: String::new(),
                }),
                delta: None,
                usage: None,
            });
        }
    }
    
    events.into_iter().map(|anthropic_response| {
        let event_type = anthropic_response.event_type.clone();
        eprintln!("DEBUG: Creating SSE event: type={}, index={:?}", event_type, anthropic_response.index);
        match serde_json::to_string(&anthropic_response) {
            Ok(data) => {
                eprintln!("DEBUG: Event data: {}", data);
                let event = Event::default().event(&event_type).data(data);
                Ok(AnthropicEvent { event, event_type })
            }
            Err(e) => Err(anyhow::anyhow!(
                "Failed to serialize Anthropic response: {}",
                e
            )),
        }
    })
}
