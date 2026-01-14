use crate::{create_anthropic_sse_events, create_sse_event, DONE_MESSAGE};
use async_stream::stream;
use axum::response::sse::Event;
use futures::stream::BoxStream;
use futures::StreamExt;
use response::ChatCompletionsResponse;
use tracing::error;

/// Wraps an OpenAI-format stream and converts it to Anthropic format
pub fn wrap_with_anthropic_format(
    openai_stream: BoxStream<'static, anyhow::Result<Event>>,
) -> BoxStream<'static, anyhow::Result<Event>> {
    let stream = stream! {
        let mut openai_stream = openai_stream;

        while let Some(event_result) = openai_stream.next().await {
            match event_result {
                Ok(event) => {
                    // We need to serialize the event to get its data
                    // The Event struct doesn't expose its data, so we work with the response objects
                    // This is a workaround - ideally we'd modify the providers directly
                    
                    // For now, we'll just pass through the event as-is
                    // A proper implementation would require modifying the provider layer
                    yield Ok(event);
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    yield Err(e);
                }
            }
        }
    };

    stream.boxed()
}

/// Creates a stream that converts ChatCompletionsResponse to Anthropic SSE events
pub fn create_anthropic_stream(
    response_stream: BoxStream<'static, anyhow::Result<ChatCompletionsResponse>>,
) -> BoxStream<'static, anyhow::Result<Event>> {
    let stream = stream! {
        let mut response_stream = response_stream;

        while let Some(response_result) = response_stream.next().await {
            match response_result {
                Ok(response) => {
                    // Convert to Anthropic events
                    match create_anthropic_sse_events(&response) {
                        Ok(anthropic_events) => {
                            for anthropic_event in anthropic_events {
                                yield Ok(anthropic_event);
                            }
                        }
                        Err(e) => {
                            error!("Failed to convert to Anthropic format: {}", e);
                            yield Err(anyhow::anyhow!("Conversion error: {}", e));
                        }
                    }
                }
                Err(e) => {
                    error!("Response stream error: {}", e);
                    yield Err(e);
                }
            }
        }
    };

    stream.boxed()
}

