use aws_sdk_bedrockruntime::types::{DocumentBlock, DocumentFormat, DocumentSource};
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};

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
                    .name("document")
                    .source(DocumentSource::Bytes(document_bytes.into()))
                    .build()
                    .ok()
            }
            AnthropicDocumentSource::Url { .. } => None,
            AnthropicDocumentSource::File { .. } => None,
        }
    }
}
