use aws_sdk_bedrockruntime::types::{ContentBlock, ToolResultBlock};
use serde::{Deserialize, Serialize};

use crate::cache_control::CacheControl;
use crate::document_source::AnthropicDocumentSource;
use crate::image_source::AnthropicImageSource;
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
        source: AnthropicImageSource,
    },
    #[serde(rename = "document")]
    Document {
        source: AnthropicDocumentSource,
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

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine as _, engine::general_purpose};

    #[test]
    fn test_user_content_image_base64_png() {
        let png_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
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
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content)
            .unwrap()
            .unwrap();

        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ContentBlock::Image(_)));
    }

    #[test]
    fn test_user_content_image_base64_jpeg() {
        let jpeg_bytes = vec![
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9,
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
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content)
            .unwrap()
            .unwrap();

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
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content).unwrap();
        assert!(blocks.is_none());
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
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content).unwrap();
        assert!(blocks.is_none());
    }

    #[test]
    fn test_user_contents_with_mixed_content() {
        let png_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
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

    #[test]
    fn test_user_content_document_base64_pdf() {
        let pdf_bytes = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids []\n/Count 0\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n110\n%%EOF\n";
        let base64_data = general_purpose::STANDARD.encode(pdf_bytes);

        let json = serde_json::json!({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64_data
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content)
            .unwrap()
            .unwrap();

        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], ContentBlock::Document(_)));
        assert!(matches!(blocks[1], ContentBlock::Text(_)));
    }

    #[test]
    fn test_user_content_document_base64_txt() {
        let txt_bytes = b"Hello, this is a text document.";
        let base64_data = general_purpose::STANDARD.encode(txt_bytes);

        let json = serde_json::json!({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "text/plain",
                "data": base64_data
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content)
            .unwrap()
            .unwrap();

        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], ContentBlock::Document(_)));
        assert!(matches!(blocks[1], ContentBlock::Text(_)));
    }

    #[test]
    fn test_user_content_document_url_not_supported() {
        let json = serde_json::json!({
            "type": "document",
            "source": {
                "type": "url",
                "url": "https://example.com/document.pdf"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content).unwrap();
        assert!(blocks.is_none());
    }

    #[test]
    fn test_user_content_document_file_not_supported() {
        let json = serde_json::json!({
            "type": "document",
            "source": {
                "type": "file",
                "file_id": "file_xyz789"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        let blocks = Option::<Vec<ContentBlock>>::try_from(&content).unwrap();
        assert!(blocks.is_none());
    }

    #[test]
    fn test_user_contents_with_image_and_document() {
        let png_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let image_base64 = general_purpose::STANDARD.encode(&png_bytes);

        let pdf_bytes = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids []\n/Count 0\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n110\n%%EOF\n";
        let pdf_base64 = general_purpose::STANDARD.encode(pdf_bytes);

        let json = serde_json::json!([
            {
                "type": "text",
                "text": "Analyze this image and document:"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64
                }
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_base64
                }
            }
        ]);

        let contents: UserContents = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ContentBlock>::try_from(&contents).unwrap();

        assert_eq!(blocks.len(), 4);
        assert!(matches!(blocks[0], ContentBlock::Text(_)));
        assert!(matches!(blocks[1], ContentBlock::Image(_)));
        assert!(matches!(blocks[2], ContentBlock::Document(_)));
        assert!(matches!(blocks[3], ContentBlock::Text(_)));
    }
}
