use serde::{Deserialize, Serialize};

/// ToolChoice controls how the model uses tools
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String, // "auto", "any", "tool", "none"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}
