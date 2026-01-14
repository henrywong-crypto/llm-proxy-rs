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
                let content_blocks: Vec<Content> = msg
                    .content
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
                    .collect();

                // If there's only one text block, use a simple string; otherwise use array
                let content = if content_blocks.len() == 1 {
                    if let Some(Content::Text { text }) = content_blocks.first() {
                        Contents::String(text.clone())
                    } else {
                        Contents::Array(content_blocks)
                    }
                } else {
                    Contents::Array(content_blocks)
                };

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

        // Convert tools if present
        let tools = req.tools.and_then(|anthropic_tools| {
            if anthropic_tools.is_empty() {
                None
            } else {
                // Convert Anthropic tools to OpenAI format
                let converted_tools: Vec<crate::Tool> = anthropic_tools
                    .into_iter()
                    .filter_map(|tool| {
                        // Anthropic format: {"name": "...", "description": "...", "input_schema": {...}}
                        // OpenAI format: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

                        let obj = tool.as_object()?;
                        let name = obj.get("name")?.as_str()?.to_string();
                        let description = obj
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|s| s.to_string());
                        let parameters = obj.get("input_schema")?.clone();

                        Some(crate::Tool {
                            tool_type: "function".to_string(),
                            function: crate::ToolFunction {
                                name,
                                description,
                                parameters,
                            },
                        })
                    })
                    .collect();

                if converted_tools.is_empty() {
                    None
                } else {
                    Some(converted_tools)
                }
            }
        });

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
            tools,
            tool_choice: None,
            reasoning_effort: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    #[serde(deserialize_with = "deserialize_content")]
    pub content: Vec<ContentBlock>,
}

fn deserialize_content<'de, D>(deserializer: D) -> Result<Vec<ContentBlock>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let value = Value::deserialize(deserializer)?;

    match value {
        // If it's a string, convert to a single text block
        Value::String(s) => Ok(vec![ContentBlock::Text { text: s }]),
        // If it's an array, deserialize normally
        Value::Array(_) => serde_json::from_value(value).map_err(D::Error::custom),
        _ => Err(D::Error::custom("content must be a string or array")),
    }
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
