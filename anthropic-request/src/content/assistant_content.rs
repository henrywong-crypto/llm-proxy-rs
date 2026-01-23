use aws_sdk_bedrockruntime::types::{
    ContentBlock, ReasoningContentBlock, ReasoningTextBlock, ToolUseBlock,
};
use common::value_to_document;
use serde::{Deserialize, Deserializer, Serialize};
use tracing::debug;

use crate::cache_control::CacheControl;

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum AssistantContent {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

impl<'de> Deserialize<'de> for AssistantContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        use serde_json::Value;

        let value = Value::deserialize(deserializer)?;
        
        let content_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
        debug!("🔍 Deserializing AssistantContent with type: {}", content_type);

        match content_type {
            "text" => {
                debug!("  ✅ Deserializing as Text");
                serde_json::from_value(value).map_err(|e| {
                    debug!("  ❌ Failed to deserialize Text: {}", e);
                    Error::custom(format!("Failed to deserialize Text content: {}", e))
                })
            }
            "tool_use" => {
                debug!("  ✅ Deserializing as ToolUse");
                serde_json::from_value(value).map_err(|e| {
                    debug!("  ❌ Failed to deserialize ToolUse: {}", e);
                    Error::custom(format!("Failed to deserialize ToolUse content: {}", e))
                })
            }
            "thinking" => {
                debug!("  ✅ Deserializing as Thinking");
                let has_signature = value.get("signature").is_some();
                let has_thinking = value.get("thinking").is_some();
                debug!("    has_signature: {}, has_thinking: {}", has_signature, has_thinking);
                
                serde_json::from_value(value).map_err(|e| {
                    debug!("  ❌ Failed to deserialize Thinking: {}", e);
                    Error::custom(format!("Failed to deserialize Thinking content: {}", e))
                })
            }
            _ => {
                debug!("  ❌ Unknown content type: {}", content_type);
                Err(Error::custom(format!("Unknown assistant content type: {}", content_type)))
            }
        }
    }
}

impl TryFrom<&AssistantContent> for Vec<ContentBlock> {
    type Error = anyhow::Error;

    fn try_from(content: &AssistantContent) -> Result<Self, Self::Error> {
        match content {
            AssistantContent::Text {
                text,
                cache_control,
            } => {
                let mut blocks = vec![ContentBlock::Text(text.clone())];

                if let Some(cache_control) = cache_control {
                    let cache_point = cache_control.try_into()?;
                    blocks.push(ContentBlock::CachePoint(cache_point));
                }

                Ok(blocks)
            }
            AssistantContent::ToolUse { id, name, input } => {
                let tool_use_block = ToolUseBlock::builder()
                    .tool_use_id(id)
                    .name(name)
                    .input(value_to_document(input))
                    .build()?;

                Ok(vec![ContentBlock::ToolUse(tool_use_block)])
            }
            AssistantContent::Thinking {
                thinking,
                signature,
            } => {
                let mut reasoning_text_builder = ReasoningTextBlock::builder().text(thinking);

                if let Some(sig) = signature {
                    reasoning_text_builder = reasoning_text_builder.signature(sig);
                }

                let reasoning_text_block = reasoning_text_builder.build()?;
                let reasoning_content_block =
                    ReasoningContentBlock::ReasoningText(reasoning_text_block);

                Ok(vec![ContentBlock::ReasoningContent(
                    reasoning_content_block,
                )])
            }
        }
    }
}
