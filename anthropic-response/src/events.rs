use crate::delta::Delta;
use crate::{ResponseContentBlock, Usage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: i32,
        content_block: ContentBlockStartData,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: i32, delta: Delta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: i32 },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: Usage,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "error")]
    Error { error: ErrorData },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ErrorData {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Container {
    pub id: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<Skill>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Skill {
    #[serde(rename = "type")]
    pub skill_type: String,
    pub skill_id: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContextManagement {
    pub applied_edits: Vec<ContextEdit>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContextEdit {
    #[serde(rename = "clear_tool_uses_20250919")]
    ClearToolUses {
        cleared_tool_uses: i32,
        cleared_input_tokens: i32,
    },
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking {
        cleared_thinking_turns: i32,
        cleared_input_tokens: i32,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageStartData {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String, // "message"
    pub role: String, // "assistant"
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Container>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Caller {
    #[serde(rename = "type")]
    pub caller_type: String, // "code_execution_20250825" or "direct"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlockStartData {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        caller: Option<Caller>,
    },

    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },

    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "mcp_tool_use")]
    McpToolUse {
        id: String,
        name: String,
        server_name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "mcp_tool_result")]
    McpToolResult {
        tool_use_id: String,
        content: serde_json::Value,
        is_error: bool,
    },

    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        content: WebSearchContent,
    },

    #[serde(rename = "web_fetch_tool_result")]
    WebFetchToolResult {
        tool_use_id: String,
        content: WebFetchContent,
    },

    #[serde(rename = "code_execution_tool_result")]
    CodeExecutionToolResult {
        tool_use_id: String,
        content: CodeExecutionContent,
    },

    #[serde(rename = "bash_code_execution_tool_result")]
    BashCodeExecutionToolResult {
        tool_use_id: String,
        content: BashCodeExecutionContent,
    },

    #[serde(rename = "text_editor_code_execution_tool_result")]
    TextEditorCodeExecutionToolResult {
        tool_use_id: String,
        content: TextEditorCodeExecutionContent,
    },

    #[serde(rename = "tool_search_tool_result")]
    ToolSearchToolResult {
        tool_use_id: String,
        content: ToolSearchContent,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum WebSearchContent {
    Results(Vec<WebSearchResult>),
    Error(WebSearchError),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebSearchResult {
    #[serde(rename = "type")]
    pub result_type: String, // "web_search_result"
    pub url: String,
    pub title: String,
    pub encrypted_content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebSearchError {
    #[serde(rename = "type")]
    pub error_type: String, // "web_search_tool_result_error"
    pub error_code: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebFetchContent {
    #[serde(rename = "type")]
    pub content_type: String, // "web_fetch_result"
    pub url: String,
    pub retrieved_at: String,
    pub content: WebFetchDocument,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebFetchDocument {
    #[serde(rename = "type")]
    pub doc_type: String, // "document"
    pub title: String,
    pub citations: Vec<serde_json::Value>,
    pub source: WebFetchSource,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebFetchSource {
    #[serde(rename = "type")]
    pub source_type: String, // "text"
    pub media_type: String,
    pub data: String, // base64 encoded
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CodeExecutionContent {
    #[serde(rename = "type")]
    pub content_type: String, // "code_execution_result"
    pub stdout: String,
    pub stderr: String,
    pub return_code: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BashCodeExecutionContent {
    #[serde(rename = "type")]
    pub content_type: String, // "bash_code_execution_result"
    pub stdout: String,
    pub stderr: String,
    pub return_code: i32,
    pub content: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextEditorCodeExecutionContent {
    #[serde(rename = "type")]
    pub content_type: String, // "text_editor_code_execution_create_result" or similar
    pub is_file_update: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolSearchContent {
    #[serde(rename = "type")]
    pub content_type: String, // "tool_search_tool_search_result"
    pub tool_references: Vec<ToolReference>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolReference {
    #[serde(rename = "type")]
    pub ref_type: String, // "tool_reference"
    pub tool_name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageDeltaData {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Container>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<ContextManagement>,
}
