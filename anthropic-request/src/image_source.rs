use aws_sdk_bedrockruntime::types::{ImageBlock, ImageFormat, ImageSource as BedrockImageSource};
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ImageSource {
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

impl From<&ImageSource> for Option<ImageBlock> {
    fn from(source: &ImageSource) -> Self {
        match source {
            ImageSource::Base64 { media_type, data } => {
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
                    .source(BedrockImageSource::Bytes(image_bytes.into()))
                    .build()
                    .ok()
            }
            ImageSource::Url { .. } => None,
            ImageSource::File { .. } => None,
        }
    }
}
