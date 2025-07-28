use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    Tool as BedrockTool, ToolChoice as BedrockToolChoice, ToolConfiguration,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod tool;
pub use tool::*;

pub mod content;
pub use content::*;

pub mod message;
pub use message::*;

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
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

impl TryFrom<&ChatCompletionsRequest> for ToolConfiguration {
    type Error = anyhow::Error;

    fn try_from(request: &ChatCompletionsRequest) -> Result<Self, Self::Error> {
        let mut builder = ToolConfiguration::builder();

        if let Some(tools) = &request.tools {
            for tool in tools {
                let bedrock_tool = BedrockTool::try_from(tool)?;
                builder = builder.tools(bedrock_tool);
            }
        }

        if let Some(tool_choice) = &request.tool_choice {
            let bedrock_tool_choice = Option::<BedrockToolChoice>::try_from(tool_choice)?;
            builder = builder.set_tool_choice(bedrock_tool_choice);
        }

        Ok(builder.build()?)
    }
}
