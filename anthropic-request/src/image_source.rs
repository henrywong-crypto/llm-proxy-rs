use serde::{Deserialize, Serialize};

/// ImageSource represents the source of an image
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64" or "url"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}
