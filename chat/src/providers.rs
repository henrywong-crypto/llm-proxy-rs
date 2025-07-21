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
use tracing::{debug, error, info, trace};
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
        debug!("Request has {} messages, {} tools", request.messages.len(), request.tools.as_ref().map(|t| t.len()).unwrap_or(0));
        
        let bedrock_chat_completion = match self.process_chat_completions_request(&request) {
            Ok(completion) => {
                debug!("Successfully processed request to Bedrock format");
                completion
            }
            Err(e) => {
                error!("Failed to process request to Bedrock format: {}", e);
                return Err(e);
            }
        };
        info!(
            "Processed request to Bedrock format with {} messages",
            bedrock_chat_completion.messages.len()
        );

        debug!("Loading AWS config");
        let config = match aws_config::load_defaults(BehaviorVersion::latest()).await {
            config => {
                debug!("Successfully loaded AWS config");
                config
            }
        };
        let client = Client::new(&config);
        debug!("Created Bedrock client");

        info!(
            "Sending request to Bedrock API for model: {}",
            bedrock_chat_completion.model_id
        );

        let mut converse_builder = client
            .converse_stream()
            .model_id(&bedrock_chat_completion.model_id)
            .set_system(Some(bedrock_chat_completion.system_content_blocks))
            .set_messages(Some(bedrock_chat_completion.messages));

        if let Some(tool_config) = bedrock_chat_completion.tool_config {
            debug!("Adding tool configuration to Bedrock request with {} tools", 
                tool_config.tools().len());
            converse_builder = converse_builder.tool_config(tool_config);
        } else {
            debug!("No tool configuration to add");
        }

        debug!("Sending request to Bedrock API...");
        let mut stream = match converse_builder.send().await {
            Ok(response) => {
                debug!("Successfully initiated Bedrock stream");
                response.stream
            }
            Err(e) => {
                error!("Failed to send request to Bedrock API: {}", e);
                return Err(anyhow::anyhow!("Bedrock API error: {}", e));
            }
        };
        info!("Successfully connected to Bedrock stream");

        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();
        debug!("Created response with id: {}", id);

        let usage_callback = Arc::new(usage_callback);

        let stream = async_stream::stream! {
            trace!("Starting to process stream");
            loop {
                match stream.recv().await {
                    Ok(Some(output)) => {
                        trace!("Received output from Bedrock stream");
                        let usage_callback = usage_callback.clone();
                        let builder = converse_stream_output_to_chat_completions_response_builder(&output, usage_callback);
                        let response = builder
                            .id(Some(id.clone()))
                            .created(Some(created))
                            .build();

                        match create_sse_event(&response) {
                            Ok(event) => {
                                trace!("Created SSE event");
                                yield Ok(event);
                            },
                            Err(e) => {
                                error!("Failed to create SSE event: {}", e);
                                yield Err(e);
                            }
                        }
                    }
                    Ok(None) => {
                        debug!("Stream completed");
                        break;
                    }
                    Err(e) => {
                        error!("Error receiving from stream: {}", e);
                        yield Err(anyhow::anyhow!(
                            "Stream receive error: {}",
                            e
                        ));
                    }
                }
            }

            info!("Stream finished, sending DONE message");
            yield Ok(Event::default().data(DONE_MESSAGE));
        };

        Ok(stream.boxed())
    }
}
