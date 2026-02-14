use aws_sdk_bedrockruntime::types::{ContentBlock, ToolResultBlock};
use serde::{Deserialize, Serialize};

use crate::cache_control::CacheControl;
use crate::document_source::DocumentSource;
use crate::image_source::ImageSource;
use crate::tool_result_content::ToolResultContents;

/// Bedrock requires at least one text block when documents are present.
/// Empty strings are rejected, so we use a single space as a minimal placeholder.
/// See: https://github.com/BerriAI/litellm/issues/7169
const BEDROCK_DOCUMENT_PLACEHOLDER_TEXT: &str = " ";

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum UserContents {
    Array(Vec<UserContent>),
    String(String),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum UserContent {
    #[serde(rename = "text")]
    Text {
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        text: String,
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
    },
    #[serde(rename = "document")]
    Document {
        source: DocumentSource,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        content: ToolResultContents,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        tool_use_id: String,
    },
}

impl TryFrom<&UserContents> for Vec<ContentBlock> {
    type Error = anyhow::Error;

    fn try_from(contents: &UserContents) -> Result<Self, Self::Error> {
        match contents {
            UserContents::String(s) => Ok(vec![ContentBlock::Text(s.clone())]),
            UserContents::Array(arr) => Ok(arr
                .iter()
                .map(Option::<Vec<ContentBlock>>::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .flatten()
                .collect()),
        }
    }
}

impl TryFrom<&UserContent> for Option<Vec<ContentBlock>> {
    type Error = anyhow::Error;

    fn try_from(content: &UserContent) -> Result<Self, Self::Error> {
        match content {
            UserContent::Text {
                text,
                cache_control,
            } => {
                let mut blocks = vec![ContentBlock::Text(text.clone())];

                if let Some(cache_control) = cache_control {
                    let cache_point = cache_control.try_into()?;
                    blocks.push(ContentBlock::CachePoint(cache_point));
                }

                Ok(Some(blocks))
            }
            UserContent::Image { source } => {
                if let Some(image_block) =
                    Option::<aws_sdk_bedrockruntime::types::ImageBlock>::from(source)
                {
                    Ok(Some(vec![ContentBlock::Image(image_block)]))
                } else {
                    Ok(None)
                }
            }
            UserContent::Document { source } => {
                if let Some(document_block) =
                    Option::<aws_sdk_bedrockruntime::types::DocumentBlock>::from(source)
                {
                    Ok(Some(vec![
                        ContentBlock::Document(document_block),
                        ContentBlock::Text(BEDROCK_DOCUMENT_PLACEHOLDER_TEXT.into()),
                    ]))
                } else {
                    Ok(None)
                }
            }
            UserContent::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let tool_result_block = ToolResultBlock::builder()
                    .tool_use_id(tool_use_id)
                    .set_content(Some(content.into()))
                    .set_status(is_error.map(|is_error| {
                        if is_error {
                            aws_sdk_bedrockruntime::types::ToolResultStatus::Error
                        } else {
                            aws_sdk_bedrockruntime::types::ToolResultStatus::Success
                        }
                    }))
                    .build()?;

                Ok(Some(vec![ContentBlock::ToolResult(tool_result_block)]))
            }
        }
    }
}