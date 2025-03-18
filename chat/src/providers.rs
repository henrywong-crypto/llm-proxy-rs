use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use axum::response::sse::{Event, Sse};
use chrono::offset::Utc;
use futures::stream::Stream;
use request::{ChatCompletionsRequest, Role, Contents};
use response::{
    ChatCompletionsResponse, Delta, Usage, UsageBuilder, ChoiceBuilder,
    converse_stream_output_to_chat_completions_response_builder,
};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::env;
use tracing::{debug, error, info, trace};
use uuid::Uuid;

use crate::ProcessChatCompletionsRequest;
use crate::bedrock::{
    BedrockChatCompletion, process_chat_completions_request_to_bedrock_chat_completion,
};

const DONE_MESSAGE: &str = "[DONE]";
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

#[async_trait]
pub trait ChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<Sse<impl Stream<Item = anyhow::Result<Event>>>>
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
    ) -> BedrockChatCompletion {
        process_chat_completions_request_to_bedrock_chat_completion(request)
    }
}

fn create_sse_event(response: &ChatCompletionsResponse) -> anyhow::Result<Event> {
    match serde_json::to_string(response) {
        Ok(data) => Ok(Event::default().data(data)),
        Err(e) => Err(anyhow::anyhow!("Failed to serialize response: {}", e)),
    }
}

#[async_trait]
impl ChatCompletionsProvider for BedrockChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<Sse<impl Stream<Item = anyhow::Result<Event>>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        debug!(
            "Processing chat completions request for model: {}",
            request.model
        );
        let bedrock_chat_completion = self.process_chat_completions_request(&request);
        info!(
            "Processed request to Bedrock format with {} messages",
            bedrock_chat_completion.messages.len()
        );

        debug!("Loading AWS config");
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        info!(
            "Sending request to Bedrock API for model: {}",
            bedrock_chat_completion.model_id
        );
        let mut stream = client
            .converse_stream()
            .model_id(&bedrock_chat_completion.model_id)
            .set_system(Some(bedrock_chat_completion.system_content_blocks))
            .set_messages(Some(bedrock_chat_completion.messages))
            .send()
            .await?
            .stream;
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

        Ok(Sse::new(stream))
    }
}

// OpenAI Provider Structures

#[derive(Deserialize)]
struct OpenAIStreamingChunk {
    id: Option<String>,
    #[allow(dead_code)]
    object: Option<String>,
    created: Option<i64>,
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAIChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    index: i32,
    delta: Option<OpenAIDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIDelta {
    role: Option<String>,
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

#[derive(Serialize)]
struct OpenAIRequestMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIRequestMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

// OpenAI provider implementation
pub struct OpenAIChatCompletionsProvider {
    client: reqwest::Client,
    api_key: String,
}

impl OpenAIChatCompletionsProvider {
    pub async fn new() -> anyhow::Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;

        let client = reqwest::Client::builder()
            .build()?;

        Ok(Self { client, api_key })
    }

    fn process_request(&self, request: &ChatCompletionsRequest) -> OpenAIRequest {
        let messages = request.messages.iter().map(|message| {
            let role = match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            
            // Extract content from the Contents enum
            let content = match &message.contents {
                Contents::String(text) => text.clone(),
                Contents::Array(content_array) => {
                    // For simplicity, just concat all text in the array
                    content_array.iter().filter_map(|content| {
                        match content {
                            request::Content::Text { text } => Some(text.as_str()),
                        }
                    }).collect::<Vec<_>>().join(" ")
                }
            };
            
            OpenAIRequestMessage {
                role: role.to_string(),
                content,
            }
        }).collect();

        OpenAIRequest {
            model: request.model.clone(),
            messages,
            stream: true, // Always stream with OpenAI
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            n: request.n,
            user: request.user.clone(),
        }
    }
}

#[async_trait]
impl ChatCompletionsProvider for OpenAIChatCompletionsProvider {
    async fn chat_completions_stream<F>(
        self,
        request: ChatCompletionsRequest,
        usage_callback: F,
    ) -> anyhow::Result<Sse<impl Stream<Item = anyhow::Result<Event>>>>
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        debug!("Processing OpenAI chat completions request for model: {}", request.model);
        
        let openai_request = self.process_request(&request);
        debug!("Converted to OpenAI request format");
        
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|e| anyhow::anyhow!("Invalid API key format: {}", e))?,
        );

        info!("Sending request to OpenAI API");
        let response = self.client
            .post(OPENAI_API_URL)
            .headers(headers)
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "OpenAI API error: HTTP {}: {}",
                status,
                error_text
            ));
        }

        let stream = response.bytes_stream();
        let usage_callback = Arc::new(usage_callback);
        let id = Uuid::new_v4().to_string();
        let created = Utc::now().timestamp();
        let model = request.model.clone();

        let stream = async_stream::stream! {
            let mut line_buffer = String::new();
            let mut bytes_stream = stream;
            
            use futures::StreamExt;
            
            while let Some(chunk_result) = bytes_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Convert chunk to string and append to buffer
                        if let Ok(chunk_str) = std::str::from_utf8(&chunk) {
                            line_buffer.push_str(chunk_str);
                            
                            // Process complete lines
                            while let Some(pos) = line_buffer.find('\n') {
                                let line = line_buffer[..pos].trim().to_string();
                                let remainder = line_buffer[pos + 1..].to_string();
                                line_buffer = remainder;
                                
                                if line.is_empty() {
                                    continue;
                                }
                                
                                if line == "data: [DONE]" {
                                    debug!("Received DONE from OpenAI");
                                    yield Ok(Event::default().data(DONE_MESSAGE));
                                    continue;
                                }
                                
                                if let Some(data) = line.strip_prefix("data: ") {
                                    match serde_json::from_str::<OpenAIStreamingChunk>(data) {
                                        Ok(chunk) => {
                                            let mut builder = ChatCompletionsResponse::builder()
                                                .id(chunk.id.or(Some(id.clone())))
                                                .created(chunk.created.or(Some(created)))
                                                .model(chunk.model.or(Some(model.clone())));
                                            
                                            for choice in &chunk.choices {
                                                let delta = match &choice.delta {
                                                    Some(d) => {
                                                        if let Some(role) = &d.role {
                                                            Some(Delta::Role { role: role.clone() })
                                                        } else if let Some(content) = &d.content {
                                                            Some(Delta::Content { content: content.clone() })
                                                        } else {
                                                            None
                                                        }
                                                    },
                                                    None => None,
                                                };
                                                
                                                let choice = ChoiceBuilder::default()
                                                    .delta(delta)
                                                    .index(choice.index)
                                                    .finish_reason(choice.finish_reason.clone())
                                                    .build();
                                                
                                                builder = builder.choice(choice);
                                            }
                                            
                                            if let Some(usage) = chunk.usage {
                                                let usage_obj = UsageBuilder::default()
                                                    .prompt_tokens(usage.prompt_tokens)
                                                    .completion_tokens(usage.completion_tokens)
                                                    .total_tokens(usage.total_tokens)
                                                    .build();
                                                
                                                usage_callback(&usage_obj);
                                                builder = builder.usage(Some(usage_obj));
                                            }
                                            
                                            let response = builder.build();
                                            match create_sse_event(&response) {
                                                Ok(event) => {
                                                    trace!("Created SSE event from OpenAI chunk");
                                                    yield Ok(event);
                                                },
                                                Err(e) => {
                                                    error!("Failed to create SSE event: {}", e);
                                                    yield Err(e);
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            error!("Failed to parse OpenAI chunk: {}", e);
                                            error!("Raw chunk: {}", data);
                                            yield Err(anyhow::anyhow!("Failed to parse OpenAI chunk: {}", e));
                                        }
                                    }
                                }
                            }
                        }
                    },
                    Err(e) => {
                        error!("Error receiving from OpenAI stream: {}", e);
                        yield Err(anyhow::anyhow!("Stream receive error: {}", e));
                    }
                }
            }
            
            info!("OpenAI stream completed");
        };

        Ok(Sse::new(stream))
    }
}
