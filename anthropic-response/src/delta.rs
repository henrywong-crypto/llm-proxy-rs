use serde::{Deserialize, Serialize};

/// Delta represents an incremental update in streaming
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum Delta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

/// MessageDelta contains stop information
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// DeltaUsage contains cumulative token usage
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DeltaUsage {
    pub output_tokens: i32,
}
