use crate::{ChatCompletionsRequest, Content, Contents, Message, SystemContents};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

impl From<AnthropicRequest> for ChatCompletionsRequest {
    fn from(req: AnthropicRequest) -> Self {
        // Convert Anthropic messages to OpenAI messages
        let mut messages: Vec<Message> = req
            .messages
            .into_iter()
            .map(|msg| {
                let content = Contents::Array(
                    msg.content
                        .into_iter()
                        .map(|block| {
                            match block {
                                ContentBlock::Text { text } => Content::Text { text },
                                ContentBlock::Image { source } => {
                                    // Convert Anthropic image format to OpenAI format
                                    Content::ImageUrl {
                                        image_url: crate::ImageUrl {
                                            url: format!(
                                                "data:{};base64,{}",
                                                source.media_type, source.data
                                            ),
                                        },
                                    }
                                }
                            }
                        })
                        .collect(),
                );

                match msg.role.as_str() {
                    "user" => Message::User {
                        contents: Some(content),
                    },
                    "assistant" => Message::Assistant {
                        contents: Some(content),
                        tool_calls: None,
                    },
                    _ => Message::User {
                        contents: Some(content),
                    },
                }
            })
            .collect();

        // Add system message at the beginning if present
        if let Some(system_messages) = req.system
            && !system_messages.is_empty()
        {
            let system_text = system_messages
                .into_iter()
                .map(|s| s.text)
                .collect::<Vec<_>>()
                .join("\n\n");

            messages.insert(
                0,
                Message::System {
                    contents: Some(SystemContents::String(system_text)),
                },
            );
        }

        ChatCompletionsRequest {
            model: req.model,
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop_sequences,
            stream: req.stream,
            frequency_penalty: None,
            logit_bias: None,
            n: None,
            presence_penalty: None,
            stream_options: None,
            user: None,
            tools: None, // Anthropic tools format is different, skip for now
            tool_choice: None,
            reasoning_effort: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SystemMessage {
    #[serde(rename = "type")]
    pub message_type: String,
    pub text: String,
}
