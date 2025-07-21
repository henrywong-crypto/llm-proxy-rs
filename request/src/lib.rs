use aws_sdk_bedrockruntime::types::{ContentBlock, ConversationRole, SystemContentBlock};
use serde::{
    Deserialize, Serialize,
    de::{self, SeqAccess, Visitor},
};
use std::{collections::HashMap, fmt};

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionsRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIToolFunction,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String),
    Object {
        #[serde(rename = "type")]
        tool_type: String,
        function: OpenAIToolChoiceFunction,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    #[serde(rename = "content")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<Contents>,
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunctionCall,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Assistant,
    System,
    User,
    Tool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Contents {
    Array(Vec<Content>),
    String(String),
}

impl Contents {
    fn is_empty(&self) -> bool {
        match self {
            Contents::String(s) => s.is_empty(),
            Contents::Array(arr) => arr.is_empty(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
}

impl<'de> Visitor<'de> for Contents {
    type Value = Contents;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("string or array")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Contents::String(value.to_string()))
    }

    fn visit_seq<S>(self, seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let content_vec: Vec<Content> =
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))?;
        Ok(Contents::Array(content_vec))
    }
}

impl From<&Contents> for Vec<ContentBlock> {
    fn from(contents: &Contents) -> Self {
        match contents {
            Contents::Array(arr) => arr
                .iter()
                .map(|c| match c {
                    Content::Text { text } => ContentBlock::Text(text.clone()),
                })
                .collect(),
            Contents::String(s) => vec![ContentBlock::Text(s.clone())],
        }
    }
}

impl From<&Contents> for Vec<SystemContentBlock> {
    fn from(contents: &Contents) -> Self {
        match contents {
            Contents::Array(arr) => arr
                .iter()
                .map(|c| match c {
                    Content::Text { text } => SystemContentBlock::Text(text.clone()),
                })
                .collect(),
            Contents::String(s) => vec![SystemContentBlock::Text(s.clone())],
        }
    }
}

impl TryFrom<&Contents> for Vec<aws_sdk_bedrockruntime::types::ToolResultContentBlock> {
    type Error = anyhow::Error;

    fn try_from(contents: &Contents) -> Result<Self, Self::Error> {
        if contents.is_empty() {
            return Err(anyhow::anyhow!("Tool message contents cannot be empty"));
        }

        match contents {
            Contents::String(result) => Ok(vec![
                aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(result.clone()),
            ]),
            Contents::Array(content_blocks) => Ok(content_blocks
                .iter()
                .map(|block| match block {
                    Content::Text { text } => {
                        aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(text.clone())
                    }
                })
                .collect()),
        }
    }
}

// Convert OpenAI tool call to Bedrock ToolUseBlock
impl TryFrom<&OpenAIToolCall> for aws_sdk_bedrockruntime::types::ToolUseBlock {
    type Error = anyhow::Error;

    fn try_from(tool_call: &OpenAIToolCall) -> Result<Self, Self::Error> {
        let input = serde_json::from_str(&tool_call.function.arguments)
            .map(|value| value_to_document(&value))
            .unwrap_or(aws_smithy_types::Document::Object(
                std::collections::HashMap::new(),
            ));

        Ok(aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
            .tool_use_id(&tool_call.id)
            .name(&tool_call.function.name)
            .input(input)
            .build()?)
    }
}

pub fn value_to_document(value: &serde_json::Value) -> aws_smithy_types::Document {
    match value {
        serde_json::Value::Null => aws_smithy_types::Document::Null,
        serde_json::Value::Bool(b) => aws_smithy_types::Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                aws_smithy_types::Document::Number(if i >= 0 {
                    aws_smithy_types::Number::PosInt(i as u64)
                } else {
                    aws_smithy_types::Number::NegInt(i)
                })
            } else {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::Float(
                    n.as_f64().unwrap_or(0.0),
                ))
            }
        }
        serde_json::Value::String(s) => aws_smithy_types::Document::String(s.clone()),
        serde_json::Value::Array(a) => {
            aws_smithy_types::Document::Array(a.iter().map(value_to_document).collect())
        }
        serde_json::Value::Object(o) => aws_smithy_types::Document::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), value_to_document(v)))
                .collect(),
        ),
    }
}

// Convert tool result message to Bedrock ToolResultBlock
impl TryFrom<&Message> for aws_sdk_bedrockruntime::types::ToolResultBlock {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        let contents = message
            .contents
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tool message must have contents"))?;
        let tool_call_id = message
            .tool_call_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tool message must have tool_call_id"))?;

        let result_content_blocks =
            Vec::<aws_sdk_bedrockruntime::types::ToolResultContentBlock>::try_from(contents)?;

        aws_sdk_bedrockruntime::types::ToolResultBlock::builder()
            .tool_use_id(tool_call_id.clone())
            .set_content(Some(result_content_blocks))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build ToolResultBlock: {e}"))
    }
}

impl TryFrom<&Message> for aws_sdk_bedrockruntime::types::Message {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        match message.role {
            Role::Assistant => {
                let mut content_blocks: Vec<ContentBlock> = message
                    .contents
                    .as_ref()
                    .map(|contents| contents.into())
                    .unwrap_or_default();

                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        let tool_use =
                            aws_sdk_bedrockruntime::types::ToolUseBlock::try_from(tool_call)?;
                        content_blocks.push(ContentBlock::ToolUse(tool_use));
                    }
                }

                Ok(aws_sdk_bedrockruntime::types::Message::builder()
                    .role(ConversationRole::Assistant)
                    .set_content(Some(content_blocks))
                    .build()?)
            }
            Role::Tool => {
                let tool_result =
                    aws_sdk_bedrockruntime::types::ToolResultBlock::try_from(message)?;

                Ok(aws_sdk_bedrockruntime::types::Message::builder()
                    .role(ConversationRole::User)
                    .content(ContentBlock::ToolResult(tool_result))
                    .build()?)
            }
            Role::User | Role::System => {
                let content_blocks: Option<Vec<ContentBlock>> =
                    message.contents.as_ref().map(|contents| contents.into());

                Ok(aws_sdk_bedrockruntime::types::Message::builder()
                    .role(ConversationRole::User)
                    .set_content(content_blocks)
                    .build()?)
            }
        }
    }
}
