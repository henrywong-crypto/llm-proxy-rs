use aws_sdk_bedrockruntime::types::{ContentBlock, ToolResultBlock};
use serde::{Deserialize, Serialize};

use crate::cache_control::CacheControl;
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
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
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
                .map(Vec::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect()),
        }
    }
}

impl TryFrom<&UserContent> for Vec<ContentBlock> {
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

                Ok(blocks)
            }
            UserContent::Image {
                source,
                cache_control,
            } => {
                // Convert image source to Bedrock format
                let image_block = match source.source_type.as_str() {
                    "base64" => {
                        if let Some(data) = &source.data {
                            let format = source
                                .media_type
                                .as_ref()
                                .and_then(|mt| match mt.as_str() {
                                    "image/jpeg" | "image/jpg" => {
                                        Some(aws_sdk_bedrockruntime::types::ImageFormat::Jpeg)
                                    }
                                    "image/png" => {
                                        Some(aws_sdk_bedrockruntime::types::ImageFormat::Png)
                                    }
                                    "image/gif" => {
                                        Some(aws_sdk_bedrockruntime::types::ImageFormat::Gif)
                                    }
                                    "image/webp" => {
                                        Some(aws_sdk_bedrockruntime::types::ImageFormat::Webp)
                                    }
                                    _ => None,
                                })
                                .ok_or_else(|| anyhow::anyhow!("Unsupported image format"))?;

                            let decoded = base64::Engine::decode(
                                &base64::engine::general_purpose::STANDARD,
                                data,
                            )?;

                            let image_source = aws_sdk_bedrockruntime::types::ImageSource::Bytes(
                                aws_smithy_types::Blob::new(decoded),
                            );

                            ContentBlock::Image(
                                aws_sdk_bedrockruntime::types::ImageBlock::builder()
                                    .format(format)
                                    .source(image_source)
                                    .build()?,
                            )
                        } else {
                            return Err(anyhow::anyhow!("Base64 image missing data field"));
                        }
                    }
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Unsupported image source type: {}",
                            source.source_type
                        ));
                    }
                };

                let mut blocks = vec![image_block];

                if let Some(cache_control) = cache_control {
                    let cache_point = cache_control.try_into()?;
                    blocks.push(ContentBlock::CachePoint(cache_point));
                }

                Ok(blocks)
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

                Ok(vec![ContentBlock::ToolResult(tool_result_block)])
            }
        }
    }
}
