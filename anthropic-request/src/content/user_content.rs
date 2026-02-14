use aws_sdk_bedrockruntime::types::{ContentBlock, ToolResultBlock};
use serde::{Deserialize, Serialize};

use crate::cache_control::CacheControl;
use crate::image_source::AnthropicImageSource;
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
        source: AnthropicImageSource,
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
            UserContent::Image { source } => {
                if let Some(image_block) = Option::<aws_sdk_bedrockruntime::types::ImageBlock>::from(source) {
                    Ok(vec![ContentBlock::Image(image_block)])
                } else {
                    // If image conversion fails (e.g., unsupported format or URL/file sources),
                    // return an empty vec to filter it out
                    Ok(vec![])
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

                Ok(vec![ContentBlock::ToolResult(tool_result_block)])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine as _, engine::general_purpose};

    #[test]
    fn test_user_content_image_base64_png() {
        // Create a simple 1x1 red PNG image
        let png_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let base64_data = general_purpose::STANDARD.encode(&png_bytes);

        let json = serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_data
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ContentBlock>::try_from(&content).unwrap();

        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ContentBlock::Image(_)));
    }

    #[test]
    fn test_user_content_image_base64_jpeg() {
        // Create a minimal JPEG header
        let jpeg_bytes = vec![
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9,
        ];
        let base64_data = general_purpose::STANDARD.encode(&jpeg_bytes);

        let json = serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_data
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ContentBlock>::try_from(&content).unwrap();

        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ContentBlock::Image(_)));
    }

    #[test]
    fn test_user_content_image_url_not_supported() {
        let json = serde_json::json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.png"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ContentBlock>::try_from(&content).unwrap();

        // URL images are not supported by Bedrock, should return empty vec
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_user_content_image_file_not_supported() {
        let json = serde_json::json!({
            "type": "image",
            "source": {
                "type": "file",
                "file_id": "file_abc123"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ContentBlock>::try_from(&content).unwrap();

        // File images are not supported by Bedrock, should return empty vec
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_user_contents_with_mixed_content() {
        let png_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let base64_data = general_purpose::STANDARD.encode(&png_bytes);

        let json = serde_json::json!([
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_data
                }
            }
        ]);

        let contents: UserContents = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ContentBlock>::try_from(&contents).unwrap();

        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], ContentBlock::Text(_)));
        assert!(matches!(blocks[1], ContentBlock::Image(_)));
    }
}
