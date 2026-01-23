use aws_sdk_bedrockruntime::types::{
    ContentBlock, ReasoningContentBlock, ReasoningTextBlock, ToolUseBlock,
};
use common::value_to_document;
use serde::{Deserialize, Serialize};

use crate::cache_control::CacheControl;

#[derive(Debug, Deserialize, Serialize)]
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
