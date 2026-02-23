use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    AutoToolChoice, ImageBlock, Tool as BedrockTool, ToolChoice as BedrockToolChoice,
    ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolSpecification,
    ToolUseBlock,
};
use common::value_to_document;
use serde::{Deserialize, Serialize};

use crate::{ChatCompletionsRequest, Content, Contents, Message};

#[derive(Debug, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_call_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl From<&Contents> for Vec<ToolResultContentBlock> {
    fn from(contents: &Contents) -> Self {
        match contents {
            Contents::String(s) => {
                vec![ToolResultContentBlock::Text(s.clone())]
            }
            Contents::Array(a) => a
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } => Some(ToolResultContentBlock::Text(text.clone())),
                    Content::ImageUrl { image_url } => ImageBlock::try_from(image_url)
                        .ok()
                        .map(ToolResultContentBlock::Image),
                })
                .collect(),
        }
    }
}

impl TryFrom<&Message> for ToolResultBlock {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        let Message::Tool {
            contents,
            tool_call_id,
        } = message
        else {
            unreachable!()
        };

        Ok(ToolResultBlock::builder()
            .set_tool_use_id(tool_call_id.clone())
            .set_content(contents.as_ref().map(|contents| contents.into()))
            .build()?)
    }
}

impl TryFrom<&ToolCall> for ToolUseBlock {
    type Error = anyhow::Error;

    fn try_from(tool_call: &ToolCall) -> Result<Self, Self::Error> {
        let input = serde_json::from_str(&tool_call.function.arguments)
            .map(|value| value_to_document(&value))?;

        Ok(ToolUseBlock::builder()
            .tool_use_id(&tool_call.id)
            .name(&tool_call.function.name)
            .input(input)
            .build()?)
    }
}

impl TryFrom<&Tool> for BedrockTool {
    type Error = anyhow::Error;

    fn try_from(tool: &Tool) -> Result<Self, Self::Error> {
        let description = tool
            .function
            .description
            .as_ref()
            .filter(|d| !d.is_empty())
            .cloned();

        let tool_spec = ToolSpecification::builder()
            .name(&tool.function.name)
            .set_description(description)
            .input_schema(ToolInputSchema::Json(value_to_document(
                &tool.function.parameters,
            )))
            .build()?;

        Ok(BedrockTool::ToolSpec(tool_spec))
    }
}

impl TryFrom<&ChatCompletionsRequest> for Option<ToolConfiguration> {
    type Error = anyhow::Error;

    fn try_from(request: &ChatCompletionsRequest) -> Result<Self, Self::Error> {
        if request.tools.is_none() && request.tool_choice.is_none() {
            return Ok(None);
        }

        let mut builder = ToolConfiguration::builder();

        if let Some(tools) = &request.tools {
            for tool in tools {
                let bedrock_tool = BedrockTool::try_from(tool)?;
                builder = builder.tools(bedrock_tool);
            }
        }

        if request.tool_choice.is_some() {
            builder =
                builder.tool_choice(BedrockToolChoice::Auto(AutoToolChoice::builder().build()));
        }

        Ok(Some(builder.build()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_block_from_tool_message() {
        let json = serde_json::json!({
            "role": "tool",
            "content": "result text",
            "tool_call_id": "call_123"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let block = ToolResultBlock::try_from(&message).unwrap();
        assert_eq!(block.tool_use_id(), "call_123");
        assert_eq!(block.content().len(), 1);
        match &block.content()[0] {
            ToolResultContentBlock::Text(text) => assert_eq!(text, "result text"),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn tool_result_block_from_tool_message_with_array_content() {
        let json = serde_json::json!({
            "role": "tool",
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"}
            ],
            "tool_call_id": "call_123"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let block = ToolResultBlock::try_from(&message).unwrap();
        assert_eq!(block.tool_use_id(), "call_123");
        assert_eq!(block.content().len(), 2);
        match &block.content()[0] {
            ToolResultContentBlock::Text(text) => assert_eq!(text, "line one"),
            other => panic!("expected Text, got {:?}", other),
        }
        match &block.content()[1] {
            ToolResultContentBlock::Text(text) => assert_eq!(text, "line two"),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn tool_use_block_from_tool_call() {
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            tool_call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"city":"NYC"}"#.to_string(),
            },
        };
        let block = ToolUseBlock::try_from(&tool_call).unwrap();
        assert_eq!(block.tool_use_id(), "call_1");
        assert_eq!(block.name(), "get_weather");
    }

    #[test]
    fn tool_use_block_from_invalid_json_arguments_errors() {
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            tool_call_type: "function".to_string(),
            function: FunctionCall {
                name: "test".to_string(),
                arguments: "not json".to_string(),
            },
        };
        assert!(ToolUseBlock::try_from(&tool_call).is_err());
    }

    #[test]
    fn bedrock_tool_from_tool() {
        let tool = Tool {
            tool_call_type: "function".to_string(),
            function: ToolFunction {
                name: "calculator".to_string(),
                description: Some("Does math".to_string()),
                parameters: serde_json::json!({"type": "object"}),
            },
        };
        let bedrock_tool = BedrockTool::try_from(&tool).unwrap();
        match bedrock_tool {
            BedrockTool::ToolSpec(spec) => {
                assert_eq!(spec.name(), "calculator");
                assert_eq!(spec.description(), Some("Does math"));
            }
            other => panic!("expected ToolSpec, got {:?}", other),
        }
    }

    #[test]
    fn bedrock_tool_empty_description_is_none() {
        let tool = Tool {
            tool_call_type: "function".to_string(),
            function: ToolFunction {
                name: "calculator".to_string(),
                description: Some("".to_string()),
                parameters: serde_json::json!({"type": "object"}),
            },
        };
        let bedrock_tool = BedrockTool::try_from(&tool).unwrap();
        match bedrock_tool {
            BedrockTool::ToolSpec(spec) => {
                assert_eq!(spec.name(), "calculator");
                assert!(spec.description().is_none());
            }
            other => panic!("expected ToolSpec, got {:?}", other),
        }
    }

    #[test]
    fn tool_configuration_from_request_with_no_tools_returns_none() {
        let request = ChatCompletionsRequest {
            frequency_penalty: None,
            logit_bias: None,
            messages: vec![],
            max_tokens: None,
            model: "test".to_string(),
            n: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            temperature: None,
            top_p: None,
            user: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
        };
        let config = Option::<ToolConfiguration>::try_from(&request).unwrap();
        assert!(config.is_none());
    }

    #[test]
    fn tool_configuration_from_request_with_tools() {
        let request = ChatCompletionsRequest {
            frequency_penalty: None,
            logit_bias: None,
            messages: vec![],
            max_tokens: None,
            model: "test".to_string(),
            n: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            temperature: None,
            top_p: None,
            user: None,
            tools: Some(vec![Tool {
                tool_call_type: "function".to_string(),
                function: ToolFunction {
                    name: "test_fn".to_string(),
                    description: Some("A test function".to_string()),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
            tool_choice: Some("auto".to_string()),
            reasoning_effort: None,
        };
        let config = Option::<ToolConfiguration>::try_from(&request).unwrap();
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.tools().len(), 1);
        match &config.tools()[0] {
            BedrockTool::ToolSpec(spec) => {
                assert_eq!(spec.name(), "test_fn");
                assert_eq!(spec.description(), Some("A test function"));
            }
            other => panic!("expected ToolSpec, got {:?}", other),
        }
    }

    #[test]
    fn tool_result_block_with_image_content() {
        let json = serde_json::json!({
            "role": "tool",
            "content": [
                {"type": "text", "text": "image result"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
            ],
            "tool_call_id": "call_123"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let block = ToolResultBlock::try_from(&message).unwrap();
        assert_eq!(block.tool_use_id(), "call_123");
        assert_eq!(block.content().len(), 2);
        match &block.content()[0] {
            ToolResultContentBlock::Text(text) => assert_eq!(text, "image result"),
            other => panic!("expected Text, got {:?}", other),
        }
        assert!(matches!(
            block.content()[1],
            ToolResultContentBlock::Image(_)
        ));
    }

    #[test]
    fn tool_result_block_with_unsupported_image_skips_it() {
        let json = serde_json::json!({
            "role": "tool",
            "content": [
                {"type": "text", "text": "result"},
                {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,AAAA"}}
            ],
            "tool_call_id": "call_123"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let block = ToolResultBlock::try_from(&message).unwrap();
        assert_eq!(block.content().len(), 1);
        match &block.content()[0] {
            ToolResultContentBlock::Text(text) => assert_eq!(text, "result"),
            other => panic!("expected Text, got {:?}", other),
        }
    }
}
