use crate::{
    DONE_MESSAGE, ProcessChatCompletionsRequest,
    bedrock::{BedrockChatCompletion, process_chat_completions_request_to_bedrock_chat_completion},
    create_sse_event,
};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use axum::response::sse::Event;
use chrono::offset::Utc;
use futures::stream::{BoxStream, StreamExt};
use request::ChatCompletionsRequest;
use response::{Usage, converse_stream_output_to_chat_completions_response_builder};
use std::sync::Arc;
use tracing::debug;
use uuid::Uuid;

#[async_trait]
pub trait ChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static;
}

pub struct BedrockChatCompletionsProvider {}

impl BedrockChatCompletionsProvider {
    pub async fn new() -> Self {
        Self {}
    }
}

impl ProcessChatCompletionsRequest<BedrockChatCompletion> for BedrockChatCompletionsProvider {
    fn process_chat_completions_request(
        &self,
        request: &ChatCompletionsRequest,
    ) -> anyhow::Result<BedrockChatCompletion> {
        process_chat_completions_request_to_bedrock_chat_completion(request)
    }
}

#[async_trait]
impl ChatCompletionsProvider for BedrockChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        debug!(
            "Processing chat completions request for model: {}",
            request.model
        );
        let bedrock_chat_completion = self.process_chat_completions_request(&request)?;

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        let converse_builder = client
            .converse_stream()
            .model_id(&bedrock_chat_completion.model_id)
            .set_system(Some(bedrock_chat_completion.system_content_blocks))
            .set_messages(Some(bedrock_chat_completion.messages))
            .set_tool_config(bedrock_chat_completion.tool_config);

        let mut stream = converse_builder.send().await?.stream;

        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();

        let usage_callback = Arc::new(usage_callback);

        let stream = async_stream::stream! {
            loop {
                match stream.recv().await {
                    Ok(Some(output)) => {
                        let usage_callback = usage_callback.clone();
                        let builder = converse_stream_output_to_chat_completions_response_builder(&output, usage_callback);
                        let response = builder
                            .id(Some(id.clone()))
                            .created(Some(created))
                            .build();

                        match create_sse_event(&response) {
                            Ok(event) => yield Ok(event),
                            Err(e) => yield Err(e),
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Stream receive error: {}", e));
                    }
                }
            }

            yield Ok(Event::default().data(DONE_MESSAGE));
        };

        Ok(stream.boxed())
    }
}
