use aws_sdk_bedrockruntime::types::{
    ErrorBlock, ImageBlock, ImageFormat, ImageSource as BedrockImageSource,
};
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

fn error_image_block(format: ImageFormat, message: impl Into<String>) -> ImageBlock {
    ImageBlock::builder()
        .format(format)
        .error(ErrorBlock::builder().message(message).build())
        .build()
        .expect("failed to build error ImageBlock")
}

impl From<&ImageSource> for ImageBlock {
    fn from(source: &ImageSource) -> Self {
        match source {
            ImageSource::Base64 { media_type, data } => {
                let format = match media_type.as_str() {
                    "image/jpeg" => ImageFormat::Jpeg,
                    "image/png" => ImageFormat::Png,
                    "image/gif" => ImageFormat::Gif,
                    "image/webp" => ImageFormat::Webp,
                    _ => {
                        return error_image_block(
                            ImageFormat::Png,
                            format!("unsupported media type: {media_type}"),
                        );
                    }
                };

                match general_purpose::STANDARD.decode(data) {
                    Ok(image_bytes) => ImageBlock::builder()
                        .format(format)
                        .source(BedrockImageSource::Bytes(image_bytes.into()))
                        .build()
                        .expect("failed to build ImageBlock"),
                    Err(e) => error_image_block(format, format!("failed to decode base64: {e}")),
                }
            }
        }
    }
}
