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
        let mut messages: Vec<Message> = Vec::new();

        for msg in req.messages {
            match msg.role.as_str() {
                "assistant" => {
                    // Extract text/image content and tool_use blocks separately
                    let mut content_blocks: Vec<Content> = Vec::new();
                    let mut tool_calls: Vec<crate::ToolCall> = Vec::new();

                    for (_index, block) in msg.content.into_iter().enumerate() {
                        match block {
                            ContentBlock::Text { text } => {
                                content_blocks.push(Content::Text { text });
                            }
                            ContentBlock::Image { source } => {
                                content_blocks.push(Content::ImageUrl {
                                    image_url: crate::ImageUrl {
                                        url: format!(
                                            "data:{};base64,{}",
                                            source.media_type, source.data
                                        ),
                                    },
                                });
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                tool_calls.push(crate::ToolCall {
                                    id,
                                    tool_type: "function".to_string(),
                                    function: crate::FunctionCall {
                                        name,
                                        arguments: input.to_string(),
                                    },
                                });
                            }
                            ContentBlock::ToolResult { .. } => {
                                // Tool results should not appear in assistant messages
                                eprintln!("WARNING: tool_result in assistant message, skipping");
                            }
                        }
                    }

                    let content = if !content_blocks.is_empty() {
                        if content_blocks.len() == 1 {
                            if let Some(Content::Text { text }) = content_blocks.first() {
                                Some(Contents::String(text.clone()))
                            } else {
                                Some(Contents::Array(content_blocks))
                            }
                        } else {
                            Some(Contents::Array(content_blocks))
                        }
                    } else {
                        None
                    };

                    messages.push(Message::Assistant {
                        contents: content,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                    });
                }
                "user" => {
                    // Check if this message contains tool_result blocks
                    let has_tool_results = msg
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult { .. }));

                    if has_tool_results {
                        // Convert each tool_result to a separate Tool message
                        for block in msg.content {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } = block
                            {
                                messages.push(Message::Tool {
                                    contents: Some(Contents::String(content.to_string())),
                                    tool_call_id: Some(tool_use_id),
                                });
                            }
                        }
                    } else {
                        // Regular user message
                        let content_blocks: Vec<Content> = msg
                            .content
                            .into_iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text { text } => Some(Content::Text { text }),
                                ContentBlock::Image { source } => Some(Content::ImageUrl {
                                    image_url: crate::ImageUrl {
                                        url: format!(
                                            "data:{};base64,{}",
                                            source.media_type, source.data
                                        ),
                                    },
                                }),
                                _ => None,
                            })
                            .collect();

                        if !content_blocks.is_empty() {
                            let content = if content_blocks.len() == 1 {
                                if let Some(Content::Text { text }) = content_blocks.first() {
                                    Contents::String(text.clone())
                                } else {
                                    Contents::Array(content_blocks)
                                }
                            } else {
                                Contents::Array(content_blocks)
                            };

                            messages.push(Message::User {
                                contents: Some(content),
                            });
                        }
                    }
                }
                _ => {
                    // Unknown role, treat as user
                    let content_blocks: Vec<Content> = msg
                        .content
                        .into_iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => Some(Content::Text { text }),
                            _ => None,
                        })
                        .collect();

                    if !content_blocks.is_empty() {
                        messages.push(Message::User {
                            contents: Some(if content_blocks.len() == 1 {
                                if let Some(Content::Text { text }) = content_blocks.first() {
                                    Contents::String(text.clone())
                                } else {
                                    Contents::Array(content_blocks)
                                }
                            } else {
                                Contents::Array(content_blocks)
                            }),
                        });
                    }
                }
            }
        }

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
            eprintln!("DEBUG: Received {} Anthropic tools", anthropic_tools.len());
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

                        eprintln!("DEBUG: Converted tool: {}", name);

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

                eprintln!("DEBUG: Converted {} tools to OpenAI format", converted_tools.len());
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
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
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
