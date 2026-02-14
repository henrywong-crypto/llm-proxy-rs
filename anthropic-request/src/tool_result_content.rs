use aws_sdk_bedrockruntime::types::ToolResultContentBlock;
use serde::{Deserialize, Serialize};

use crate::document_source::AnthropicDocumentSource;
use crate::image_source::AnthropicImageSource;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolResultContents {
    String(String),
    Array(Vec<ToolResultContent>),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ToolResultContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "document")]
    Document { source: AnthropicDocumentSource },
}

impl From<&ToolResultContent> for Option<ToolResultContentBlock> {
    fn from(content: &ToolResultContent) -> Self {
        match content {
            ToolResultContent::Text { text } => Some(ToolResultContentBlock::Text(text.clone())),
            ToolResultContent::Image { source } => {
                Option::<aws_sdk_bedrockruntime::types::ImageBlock>::from(source)
                    .map(ToolResultContentBlock::Image)
            }
            ToolResultContent::Document { source } => {
                Option::<aws_sdk_bedrockruntime::types::DocumentBlock>::from(source)
                    .map(ToolResultContentBlock::Document)
            }
        }
    }
}

impl From<&ToolResultContents> for Vec<ToolResultContentBlock> {
    fn from(contents: &ToolResultContents) -> Self {
        match contents {
            ToolResultContents::String(s) => vec![ToolResultContentBlock::Text(s.clone())],
            ToolResultContents::Array(a) => a
                .iter()
                .filter_map(|content| Option::<ToolResultContentBlock>::from(content))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine as _, engine::general_purpose};

    #[test]
    fn test_tool_result_content_text() {
        let json = serde_json::json!({
            "type": "text",
            "text": "Tool execution result"
        });

        let content: ToolResultContent = serde_json::from_value(json).unwrap();
        let block = Option::<ToolResultContentBlock>::from(&content).unwrap();
        assert!(matches!(block, ToolResultContentBlock::Text(_)));
    }

    #[test]
    fn test_tool_result_content_image() {
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

        let content: ToolResultContent = serde_json::from_value(json).unwrap();
        let block = Option::<ToolResultContentBlock>::from(&content).unwrap();
        assert!(matches!(block, ToolResultContentBlock::Image(_)));
    }

    #[test]
    fn test_tool_result_content_document() {
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

        let content: ToolResultContent = serde_json::from_value(json).unwrap();
        let block = Option::<ToolResultContentBlock>::from(&content).unwrap();
        assert!(matches!(block, ToolResultContentBlock::Document(_)));
    }

    #[test]
    fn test_tool_result_contents_string() {
        let json = serde_json::json!("Simple text result");

        let contents: ToolResultContents = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ToolResultContentBlock>::from(&contents);

        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolResultContentBlock::Text(_)));
    }

    #[test]
    fn test_tool_result_contents_array_mixed() {
        let png_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let image_base64 = general_purpose::STANDARD.encode(&png_bytes);

        let json = serde_json::json!([
            {
                "type": "text",
                "text": "Here's the result:"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64
                }
            }
        ]);

        let contents: ToolResultContents = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ToolResultContentBlock>::from(&contents);

        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], ToolResultContentBlock::Text(_)));
        assert!(matches!(blocks[1], ToolResultContentBlock::Image(_)));
    }

    #[test]
    fn test_tool_result_content_url_filtered_out() {
        let json = serde_json::json!([
            {
                "type": "text",
                "text": "Result"
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png"
                }
            }
        ]);

        let contents: ToolResultContents = serde_json::from_value(json).unwrap();
        let blocks = Vec::<ToolResultContentBlock>::from(&contents);

        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolResultContentBlock::Text(_)));
    }
}
