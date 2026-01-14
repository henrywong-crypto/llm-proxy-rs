use crate::value_to_document;
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, InferenceConfiguration,
    Message as BedrockMessage, SystemContentBlock, Tool as BedrockTool, ToolConfiguration,
    ToolInputSchema, ToolResultContentBlock, ToolSpecification,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

impl From<&SystemMessage> for SystemContentBlock {
    fn from(msg: &SystemMessage) -> Self {
        SystemContentBlock::Text(msg.text.clone())
    }
}

impl TryFrom<&ContentBlock> for Option<BedrockContentBlock> {
    type Error = anyhow::Error;

    fn try_from(block: &ContentBlock) -> Result<Self, Self::Error> {
        match block {
            ContentBlock::Text { text } => Ok(Some(BedrockContentBlock::Text(text.clone()))),
            ContentBlock::Image { source } => {
                let image_block = BedrockContentBlock::Image(
                    aws_sdk_bedrockruntime::types::ImageBlock::builder()
                        .source(aws_sdk_bedrockruntime::types::ImageSource::Bytes(
                            aws_smithy_types::Blob::new(
                                base64::engine::general_purpose::STANDARD.decode(&source.data)?,
                            ),
                        ))
                        .build()?,
                );
                Ok(Some(image_block))
            }
            ContentBlock::ToolUse { id, name, input } => {
                let tool_use = BedrockContentBlock::ToolUse(
                    aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
                        .tool_use_id(id)
                        .name(name)
                        .input(value_to_document(input))
                        .build()?,
                );
                Ok(Some(tool_use))
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
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
                                Some(aws_sdk_bedrockruntime::types::ToolResultStatus::Error)
                            } else {
                                None
                            }
                        }))
                        .build()?,
                );
                Ok(Some(tool_result))
            }
        }
    }
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
        // Convert system messages
        let system: Vec<SystemContentBlock> = self
            .system
            .as_ref()
            .map(|sys_msgs| sys_msgs.iter().map(SystemContentBlock::from).collect())
            .unwrap_or_default();

        // Convert messages
        let mut messages = Vec::new();
        for msg in &self.messages {
            match msg.role.as_str() {
                "user" => {
                    let mut content_blocks = Vec::new();
                    let mut seen_tool_result_ids = HashSet::new();

                    for block in &msg.content {
                        // Deduplicate tool_result blocks (client bug workaround)
                        if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                            if seen_tool_result_ids.contains(tool_use_id) {
                                eprintln!(
                                    "WARNING: Skipping duplicate tool_result block: tool_use_id={}",
                                    tool_use_id
                                );
                                continue;
                            }
                            seen_tool_result_ids.insert(tool_use_id.clone());
                        }

                        // Skip ToolUse in user messages
                        if matches!(block, ContentBlock::ToolUse { .. }) {
                            eprintln!("WARNING: ToolUse in user message (skipping)");
                            continue;
                        }

                        if let Some(bedrock_block) = Option::<BedrockContentBlock>::try_from(block)?
                        {
                            content_blocks.push(bedrock_block);
                        }
                    }

                    if !content_blocks.is_empty() {
                        messages.push(
                            BedrockMessage::builder()
                                .role(ConversationRole::User)
                                .set_content(Some(content_blocks))
                                .build()?,
                        );
                    }
                }
                "assistant" => {
                    let mut content_blocks = Vec::new();
                    let mut seen_tool_use = false;
                    let mut seen_tool_ids = HashSet::new();

                    for block in &msg.content {
                        // Deduplicate tool_use blocks (client bug workaround)
                        if let ContentBlock::ToolUse { id, .. } = block {
                            if seen_tool_ids.contains(id) {
                                eprintln!("WARNING: Skipping duplicate tool_use block: id={}", id);
                                continue;
                            }
                            seen_tool_ids.insert(id.clone());
                            seen_tool_use = true;
                        }

                        // Bedrock requires tool_use blocks come after all text
                        if matches!(block, ContentBlock::Text { .. }) && seen_tool_use {
                            eprintln!(
                                "WARNING: Skipping text after tool_use blocks (Bedrock requirement)"
                            );
                            continue;
                        }

                        // Skip invalid blocks in assistant messages
                        if matches!(block, ContentBlock::Image { .. }) {
                            eprintln!("WARNING: Image in assistant message (skipping)");
                            continue;
                        }
                        if matches!(block, ContentBlock::ToolResult { .. }) {
                            eprintln!("WARNING: ToolResult in assistant message (skipping)");
                            continue;
                        }

                        if let Some(bedrock_block) = Option::<BedrockContentBlock>::try_from(block)?
                        {
                            content_blocks.push(bedrock_block);
                        }
                    }

                    if !content_blocks.is_empty() {
                        messages.push(
                            BedrockMessage::builder()
                                .role(ConversationRole::Assistant)
                                .set_content(Some(content_blocks))
                                .build()?,
                        );
                    }
                }
                _ => {}
            }
        }

        // Convert tools
        let tool_config = if let Some(tools) = &self.tools {
            if !tools.is_empty() {
                let tool_specs: Vec<ToolSpecification> = tools
                    .iter()
                    .filter_map(|tool| {
                        let obj = tool.as_object()?;
                        let name = obj.get("name")?.as_str()?.to_string();
                        let description = obj
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|s| s.to_string());
                        let input_schema = obj.get("input_schema")?;

                        ToolSpecification::builder()
                            .name(&name)
                            .set_description(description)
                            .input_schema(ToolInputSchema::Json(value_to_document(input_schema)))
                            .build()
                            .ok()
                    })
                    .collect();

                Some(
                    ToolConfiguration::builder()
                        .set_tools(Some(
                            tool_specs.into_iter().map(BedrockTool::ToolSpec).collect(),
                        ))
                        .build()
                        .map_err(|e| anyhow::anyhow!("Failed to build ToolConfiguration: {}", e))?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // Build inference configuration
        let inference_config = InferenceConfiguration::builder()
            .set_max_tokens(self.max_tokens)
            .set_temperature(self.temperature)
            .set_top_p(self.top_p)
            .build();

        Ok(BedrockConverseRequest {
            model_id: self.model.clone(),
            messages,
            system,
            inference_config,
            tool_config,
        })
    }
}
