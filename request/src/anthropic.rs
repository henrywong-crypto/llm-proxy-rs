use crate::{ChatCompletionsRequest, Content, Contents, Message, SystemContents, value_to_document};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, InferenceConfiguration, Message as BedrockMessage,
    SystemContentBlock, Tool as BedrockTool, ToolConfiguration, ToolInputSchema, ToolSpecification,
    ToolResultContentBlock,
};
use base64::Engine;

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

/// Direct conversion from Anthropic format to Bedrock Converse API
pub struct BedrockConverseRequest {
    pub model_id: String,
    pub messages: Vec<BedrockMessage>,
    pub system: Vec<SystemContentBlock>,
    pub inference_config: InferenceConfiguration,
    pub tool_config: Option<ToolConfiguration>,
}

impl AnthropicRequest {
    pub fn to_bedrock_converse(&self) -> Result<BedrockConverseRequest> {
        eprintln!("DEBUG: Starting to_bedrock_converse conversion");
        eprintln!("DEBUG: Model: {}", self.model);
        eprintln!("DEBUG: Messages count: {}", self.messages.len());
        
        // Convert system messages
        let system: Vec<SystemContentBlock> = self
            .system
            .as_ref()
            .map(|sys_msgs| {
                eprintln!("DEBUG: Converting {} system messages", sys_msgs.len());
                sys_msgs
                    .iter()
                    .map(|msg| SystemContentBlock::Text(msg.text.clone()))
                    .collect()
            })
            .unwrap_or_default();
        
        eprintln!("DEBUG: System messages converted: {} blocks", system.len());

        // Convert messages
        eprintln!("DEBUG: Starting message conversion");
        eprintln!("DEBUG: Total messages to convert: {}", self.messages.len());
        
        // Log all messages first for debugging
        for (idx, msg) in self.messages.iter().enumerate() {
            eprintln!("DEBUG: Message {}: role={}, content_blocks={}", idx, msg.role, msg.content.len());
            for (block_idx, block) in msg.content.iter().enumerate() {
                match block {
                    ContentBlock::Text { text } => eprintln!("  Block {}: Text ({} chars)", block_idx, text.len()),
                    ContentBlock::Image { .. } => eprintln!("  Block {}: Image", block_idx),
                    ContentBlock::ToolUse { id, name, .. } => eprintln!("  Block {}: ToolUse (id={}, name={})", block_idx, id, name),
                    ContentBlock::ToolResult { tool_use_id, .. } => eprintln!("  Block {}: ToolResult (tool_use_id={})", block_idx, tool_use_id),
                }
            }
        }
        
        let mut messages = Vec::new();
        let mut i = 0;
        while i < self.messages.len() {
            let msg = &self.messages[i];
            eprintln!("DEBUG: Processing message {}: role={}", i, msg.role);
            
            match msg.role.as_str() {
                "user" => {
                    eprintln!("DEBUG: Converting user message with {} content blocks", msg.content.len());
                    let mut content_blocks = Vec::new();
                    let mut seen_tool_result_ids = std::collections::HashSet::new();
                    
                    for block in &msg.content {
                        match block {
                            ContentBlock::Text { text } => {
                                eprintln!("DEBUG: User text block: {} chars", text.len());
                                content_blocks.push(BedrockContentBlock::Text(text.clone()));
                            }
                            ContentBlock::Image { source } => {
                                eprintln!("DEBUG: User image block");
                                let image_block = BedrockContentBlock::Image(
                                    aws_sdk_bedrockruntime::types::ImageBlock::builder()
                                        .source(
                                            aws_sdk_bedrockruntime::types::ImageSource::Bytes(
                                                aws_smithy_types::Blob::new(
                                                    base64::engine::general_purpose::STANDARD
                                                        .decode(&source.data)
                                                        .map_err(|e| anyhow::anyhow!("Failed to decode base64 image: {}", e))?,
                                                ),
                                            ),
                                        )
                                        .build()
                                        .map_err(|e| anyhow::anyhow!("Failed to build ImageBlock: {}", e))?,
                                );
                                content_blocks.push(image_block);
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => {
                                // Deduplicate tool_result blocks by tool_use_id (client bug workaround)
                                if seen_tool_result_ids.contains(tool_use_id) {
                                    eprintln!("DEBUG: WARNING - Skipping duplicate tool_result block: tool_use_id={}", tool_use_id);
                                    continue;
                                }
                                seen_tool_result_ids.insert(tool_use_id.clone());
                                
                                eprintln!("DEBUG: User tool_result block: tool_use_id={}", tool_use_id);
                                let tool_result_content = if let Some(text) = content.as_str() {
                                    vec![ToolResultContentBlock::Text(text.to_string())]
                                } else if let Some(arr) = content.as_array() {
                                    arr.iter()
                                        .filter_map(|v| {
                                            v.get("text")
                                                .and_then(|t| t.as_str())
                                                .map(|s| ToolResultContentBlock::Text(s.to_string()))
                                        })
                                        .collect()
                                } else {
                                    vec![]
                                };

                                let tool_result = BedrockContentBlock::ToolResult(
                                    aws_sdk_bedrockruntime::types::ToolResultBlock::builder()
                                        .tool_use_id(tool_use_id)
                                        .set_content(Some(tool_result_content))
                                        .set_status(is_error.and_then(|err| {
                                            if err {
                                                Some(
                                                    aws_sdk_bedrockruntime::types::ToolResultStatus::Error,
                                                )
                                            } else {
                                                None
                                            }
                                        }))
                                        .build()
                                        .map_err(|e| anyhow::anyhow!("Failed to build ToolResultBlock: {}", e))?,
                                );
                                content_blocks.push(tool_result);
                            }
                            ContentBlock::ToolUse { .. } => {
                                eprintln!("DEBUG: WARNING - ToolUse in user message (skipping)");
                            }
                        }
                    }
                    
                    if !content_blocks.is_empty() {
                        eprintln!("DEBUG: Adding user message with {} content blocks", content_blocks.len());
                        messages.push(
                            BedrockMessage::builder()
                                .role(ConversationRole::User)
                                .set_content(Some(content_blocks))
                                .build()
                                .map_err(|e| anyhow::anyhow!("Failed to build user Message: {}", e))?,
                        );
                    } else {
                        eprintln!("DEBUG: WARNING - User message has no content blocks");
                    }
                }
                "assistant" => {
                    eprintln!("DEBUG: Converting assistant message with {} content blocks", msg.content.len());
                    let mut content_blocks = Vec::new();
                    let mut seen_tool_use = false;
                    let mut seen_tool_ids = std::collections::HashSet::new();
                    
                    for block in &msg.content {
                        match block {
                            ContentBlock::Text { text } => {
                                eprintln!("DEBUG: Assistant text block: {} chars", text.len());
                                // Bedrock requires that tool_use blocks must come after all text
                                // Once we've seen a tool_use, we cannot add more text
                                if seen_tool_use {
                                    eprintln!("DEBUG: WARNING - Skipping text after tool_use blocks (Bedrock requirement)");
                                } else {
                                    content_blocks.push(BedrockContentBlock::Text(text.clone()));
                                }
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                // Deduplicate tool_use blocks by ID (client bug workaround)
                                if seen_tool_ids.contains(id) {
                                    eprintln!("DEBUG: WARNING - Skipping duplicate tool_use block: id={}", id);
                                    continue;
                                }
                                seen_tool_ids.insert(id.clone());
                                
                                eprintln!("DEBUG: Assistant tool_use block: name={}, id={}", name, id);
                                seen_tool_use = true;
                                let tool_use = BedrockContentBlock::ToolUse(
                                    aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
                                        .tool_use_id(id)
                                        .name(name)
                                        .input(value_to_document(input))
                                        .build()
                                        .map_err(|e| anyhow::anyhow!("Failed to build ToolUseBlock: {}", e))?,
                                );
                                content_blocks.push(tool_use);
                                // Continue to process more tool_use blocks if present (parallel tool calls)
                            }
                            ContentBlock::Image { .. } => {
                                eprintln!("DEBUG: WARNING - Image in assistant message (skipping)");
                            }
                            ContentBlock::ToolResult { .. } => {
                                eprintln!("DEBUG: WARNING - ToolResult in assistant message (skipping)");
                            }
                        }
                    }
                    
                    if !content_blocks.is_empty() {
                        eprintln!("DEBUG: Adding assistant message with {} content blocks", content_blocks.len());
                        messages.push(
                            BedrockMessage::builder()
                                .role(ConversationRole::Assistant)
                                .set_content(Some(content_blocks))
                                .build()
                                .map_err(|e| anyhow::anyhow!("Failed to build assistant Message: {}", e))?,
                        );
                    } else {
                        eprintln!("DEBUG: WARNING - Assistant message has no content blocks");
                    }
                }
                _ => {}
            }
            
            i += 1;
        }

        // Convert tools
        eprintln!("DEBUG: Starting tool conversion");
        let tool_config = if let Some(tools) = &self.tools {
            eprintln!("DEBUG: Found {} tools to convert", tools.len());
            if !tools.is_empty() {
                let tool_specs: Vec<ToolSpecification> = tools
                    .iter()
                    .filter_map(|tool| {
                        let obj = tool.as_object()?;
                        let name = obj.get("name")?.as_str()?.to_string();
                        eprintln!("DEBUG: Converting tool: {}", name);
                        let description = obj
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|s| s.to_string());
                        let input_schema = obj.get("input_schema")?;

                        let result = ToolSpecification::builder()
                            .name(&name)
                            .set_description(description)
                            .input_schema(
                                ToolInputSchema::Json(value_to_document(input_schema)),
                            )
                            .build();
                        
                        match result {
                            Ok(spec) => {
                                eprintln!("DEBUG: Successfully converted tool: {}", name);
                                Some(spec)
                            }
                            Err(e) => {
                                eprintln!("DEBUG: ERROR - Failed to build ToolSpecification for {}: {}", name, e);
                                None
                            }
                        }
                    })
                    .collect();

                eprintln!("DEBUG: Converted {} tool specifications", tool_specs.len());

                let tool_config_result = ToolConfiguration::builder()
                    .set_tools(Some(
                        tool_specs
                            .into_iter()
                            .map(BedrockTool::ToolSpec)
                            .collect(),
                    ))
                    .build();
                
                match tool_config_result {
                    Ok(config) => {
                        eprintln!("DEBUG: Successfully built ToolConfiguration");
                        Some(config)
                    }
                    Err(e) => {
                        eprintln!("DEBUG: ERROR - Failed to build ToolConfiguration: {}", e);
                        return Err(anyhow::anyhow!("Failed to build ToolConfiguration: {}", e));
                    }
                }
            } else {
                eprintln!("DEBUG: No tools to convert (empty list)");
                None
            }
        } else {
            eprintln!("DEBUG: No tools field in request");
            None
        };

        // Build inference configuration
        eprintln!("DEBUG: Building inference configuration");
        let inference_config = InferenceConfiguration::builder()
            .set_max_tokens(self.max_tokens)
            .set_temperature(self.temperature)
            .set_top_p(self.top_p)
            .build();
        
        eprintln!("DEBUG: Successfully built inference configuration");

        Ok(BedrockConverseRequest {
            model_id: self.model.clone(),
            messages,
            system,
            inference_config,
            tool_config,
        })
    }
}
