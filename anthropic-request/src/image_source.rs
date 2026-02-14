use aws_sdk_bedrockruntime::types::{
    DocumentBlock, DocumentFormat, DocumentSource, ImageBlock, ImageFormat, ImageSource,
};
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicImageSource {
    #[serde(rename = "base64")]
    Base64 {
        media_type: String,
        data: String,
    },
    #[serde(rename = "url")]
    Url {
        url: String,
    },
    #[serde(rename = "file")]
    File {
        file_id: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicDocumentSource {
    #[serde(rename = "base64")]
    Base64 {
        media_type: String,
        data: String,
    },
    #[serde(rename = "url")]
    Url {
        url: String,
    },
    #[serde(rename = "file")]
    File {
        file_id: String,
    },
}

impl From<&AnthropicImageSource> for Option<ImageBlock> {
    fn from(source: &AnthropicImageSource) -> Self {
        match source {
            AnthropicImageSource::Base64 { media_type, data } => {
                let format = match media_type.as_str() {
                    "image/jpeg" => ImageFormat::Jpeg,
                    "image/png" => ImageFormat::Png,
                    "image/gif" => ImageFormat::Gif,
                    "image/webp" => ImageFormat::Webp,
                    _ => return None,
                };

                let image_bytes = general_purpose::STANDARD.decode(data).ok()?;

                ImageBlock::builder()
                    .format(format)
                    .source(ImageSource::Bytes(image_bytes.into()))
                    .build()
                    .ok()
            }
            AnthropicImageSource::Url { .. } => {
                // URL-based images are not supported by AWS Bedrock's Converse API
                // The client should convert URLs to base64 before sending to this proxy
                None
            }
            AnthropicImageSource::File { .. } => {
                // File-based images from Anthropic's Files API are not supported
                // The client should convert file references to base64 before sending to this proxy
                None
            }
        }
    }
}

impl From<&AnthropicDocumentSource> for Option<DocumentBlock> {
    fn from(source: &AnthropicDocumentSource) -> Self {
        match source {
            AnthropicDocumentSource::Base64 { media_type, data } => {
                let format = match media_type.as_str() {
                    "application/pdf" => DocumentFormat::Pdf,
                    "text/csv" => DocumentFormat::Csv,
                    "application/msword" => DocumentFormat::Doc,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
                        DocumentFormat::Docx
                    }
                    "application/vnd.ms-excel" => DocumentFormat::Xls,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
                        DocumentFormat::Xlsx
                    }
                    "text/html" => DocumentFormat::Html,
                    "text/plain" => DocumentFormat::Txt,
                    "text/markdown" => DocumentFormat::Md,
                    _ => return None,
                };

                let document_bytes = general_purpose::STANDARD.decode(data).ok()?;

                DocumentBlock::builder()
                    .format(format)
                    .name("document") // Default name, could be made configurable
                    .source(DocumentSource::Bytes(document_bytes.into()))
                    .build()
                    .ok()
            }
            AnthropicDocumentSource::Url { .. } => {
                // URL-based documents are not supported by AWS Bedrock's Converse API
                // The client should convert URLs to base64 before sending to this proxy
                None
            }
            AnthropicDocumentSource::File { .. } => {
                // File-based documents from Anthropic's Files API are not supported
                // The client should convert file references to base64 before sending to this proxy
                None
            }
        }
    }
}
