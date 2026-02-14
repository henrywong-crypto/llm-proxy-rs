use aws_sdk_bedrockruntime::types::{
    ContentBlock, DocumentBlock, ImageBlock, ToolResultBlock, ToolResultStatus,
};
use serde::{Deserialize, Serialize};

use crate::cache_control::CacheControl;
use crate::document_source::DocumentSource;
use crate::image_source::ImageSource;
use crate::tool_result_content::ToolResultContents;

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
    Image { source: ImageSource },
    #[serde(rename = "document")]
    Document { source: DocumentSource },
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
            UserContents::Array(arr) => {
                let mut blocks = Vec::new();
                for content in arr {
                    if let Some(content_blocks) =
                        Option::<Vec<ContentBlock>>::try_from(content)?
                    {
                        blocks.extend(content_blocks);
                    }
                }
                Ok(blocks)
            }
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
                Ok(Some(vec![ContentBlock::Image(ImageBlock::from(source))]))
            }
            UserContent::Document { source } => {
                if let Some(document_block) = Option::<DocumentBlock>::from(source) {
                    Ok(Some(vec![
                        ContentBlock::Document(document_block),
                        ContentBlock::Text(" ".into()),
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
                            ToolResultStatus::Error
                        } else {
                            ToolResultStatus::Success
                        }
                    }))
                    .build()?;

                Ok(Some(vec![ContentBlock::ToolResult(tool_result_block)]))
            }
        }
    }
}
