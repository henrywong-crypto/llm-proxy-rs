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

impl From<&ImageSource> for Option<aws_sdk_bedrockruntime::types::ImageBlock> {
    fn from(source: &ImageSource) -> Self {
        let image_bytes = base64::engine::general_purpose::STANDARD
            .decode(&source.data)
            .ok()?;

        aws_sdk_bedrockruntime::types::ImageBlock::builder()
            .source(aws_sdk_bedrockruntime::types::ImageSource::Bytes(
                aws_smithy_types::Blob::new(image_bytes),
            ))
            .build()
            .ok()
    }
}

impl TryFrom<&ContentBlock> for Option<BedrockContentBlock> {
    type Error = anyhow::Error;

    fn try_from(block: &ContentBlock) -> Result<Self, Self::Error> {
        match block {
            ContentBlock::Text { text } => Ok(Some(BedrockContentBlock::Text(text.clone()))),
            ContentBlock::Image { source } => {
                Ok(Option::<aws_sdk_bedrockruntime::types::ImageBlock>::from(source)
                    .map(BedrockContentBlock::Image))
            }
            ContentBlock::ToolUse { id, name, input } => {
                let tool_use = aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
                    .tool_use_id(id)
                    .name(name)
                    .input(value_to_document(input))
                    .build()?;
                Ok(Some(BedrockContentBlock::ToolUse(tool_use)))
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

                let tool_result = aws_sdk_bedrockruntime::types::ToolResultBlock::builder()
                    .tool_use_id(tool_use_id)
                    .set_content(Some(tool_result_content))
                    .set_status(is_error.and_then(|err| {
                        if err {
                            Some(aws_sdk_bedrockruntime::types::ToolResultStatus::Error)
                        } else {
                            None
                        }
                    }))
                    .build()?;
                Ok(Some(BedrockContentBlock::ToolResult(tool_result)))
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

impl From<&[SystemMessage]> for Vec<SystemContentBlock> {
    fn from(messages: &[SystemMessage]) -> Self {
        messages.iter().map(SystemContentBlock::from).collect()
    }
}

impl From<&InferenceParams> for InferenceConfiguration {
    fn from(params: &InferenceParams) -> Self {
        InferenceConfiguration::builder()
            .set_max_tokens(params.max_tokens)
            .set_temperature(params.temperature)
            .set_top_p(params.top_p)
            .build()
    }
}

/// Helper struct to group inference parameters
struct InferenceParams {
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

impl TryFrom<&serde_json::Value> for ToolSpecification {
    type Error = anyhow::Error;

    fn try_from(tool: &serde_json::Value) -> Result<Self, Self::Error> {
        let obj = tool
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Tool must be an object"))?;
        let name = obj
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow::anyhow!("Tool missing 'name' field"))?;
        let description = obj
            .get("description")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());
        let input_schema = obj
            .get("input_schema")
            .ok_or_else(|| anyhow::anyhow!("Tool missing 'input_schema' field"))?;

        ToolSpecification::builder()
            .name(name)
            .set_description(description)
            .input_schema(ToolInputSchema::Json(value_to_document(input_schema)))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build ToolSpecification: {}", e))
    }
}

impl TryFrom<&[serde_json::Value]> for ToolConfiguration {
    type Error = anyhow::Error;

    fn try_from(tools: &[serde_json::Value]) -> Result<Self, Self::Error> {
        let tool_specs: Vec<ToolSpecification> = tools
            .iter()
            .map(ToolSpecification::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        ToolConfiguration::builder()
            .set_tools(Some(
                tool_specs.into_iter().map(BedrockTool::ToolSpec).collect(),
            ))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build ToolConfiguration: {}", e))
    }
}

impl TryFrom<&AnthropicRequest> for BedrockConverseRequest {
    type Error = anyhow::Error;

    fn try_from(request: &AnthropicRequest) -> Result<Self, Self::Error> {
        // Convert system messages
        let system = request
            .system
            .as_ref()
            .map(|msgs| Vec::<SystemContentBlock>::from(msgs.as_slice()))
            .unwrap_or_default();

        // Convert messages
        let mut messages = Vec::new();
        for msg in &request.messages {
            if let Some(bedrock_msg) = convert_anthropic_message(msg)? {
                messages.push(bedrock_msg);
            }
        }

        // Convert tools
        let tool_config = request
            .tools
            .as_ref()
            .filter(|tools| !tools.is_empty())
            .map(|tools| ToolConfiguration::try_from(tools.as_slice()))
            .transpose()?;

        // Build inference configuration
        let inference_params = InferenceParams {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
        };
        let inference_config = InferenceConfiguration::from(&inference_params);

        Ok(BedrockConverseRequest {
            model_id: request.model.clone(),
            messages,
            system,
            inference_config,
            tool_config,
        })
    }
}

/// Convert an Anthropic message to a Bedrock message with role-specific logic
fn convert_anthropic_message(msg: &AnthropicMessage) -> Result<Option<BedrockMessage>> {
    match msg.role.as_str() {
        "user" => convert_user_message(&msg.content),
        "assistant" => convert_assistant_message(&msg.content),
        _ => Ok(None),
    }
}

/// Convert user message content blocks
fn convert_user_message(content: &[ContentBlock]) -> Result<Option<BedrockMessage>> {
    let mut content_blocks = Vec::new();
    let mut seen_tool_result_ids = HashSet::new();

    for block in content {
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

        if let Some(bedrock_block) = Option::<BedrockContentBlock>::try_from(block)? {
            content_blocks.push(bedrock_block);
        }
    }

    if content_blocks.is_empty() {
        return Ok(None);
    }

    Ok(Some(
        BedrockMessage::builder()
            .role(ConversationRole::User)
            .set_content(Some(content_blocks))
            .build()?,
    ))
}

/// Convert assistant message content blocks
fn convert_assistant_message(content: &[ContentBlock]) -> Result<Option<BedrockMessage>> {
    let mut content_blocks = Vec::new();
    let mut seen_tool_use = false;
    let mut seen_tool_ids = HashSet::new();

    for block in content {
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
            eprintln!("WARNING: Skipping text after tool_use blocks (Bedrock requirement)");
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

        if let Some(bedrock_block) = Option::<BedrockContentBlock>::try_from(block)? {
            content_blocks.push(bedrock_block);
        }
    }

    if content_blocks.is_empty() {
        return Ok(None);
    }

    Ok(Some(
        BedrockMessage::builder()
            .role(ConversationRole::Assistant)
            .set_content(Some(content_blocks))
            .build()?,
    ))
}

impl AnthropicRequest {
    pub fn to_bedrock_converse(&self) -> Result<BedrockConverseRequest> {
        BedrockConverseRequest::try_from(self)
    }
}

