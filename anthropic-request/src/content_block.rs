use crate::image_source::ImageSource;
use serde::{Deserialize, Serialize};

/// ContentBlock represents a content block in a message.
/// Text and Thinking use Option so they serialize as the field being present (even if empty)
/// only when set, which is required for SDK streaming accumulation.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<serde_json::Value>, // string or []ContentBlock
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}
