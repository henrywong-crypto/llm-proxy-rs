use anthropic_request::{V1MessagesRequest, build_bedrock_tools};
use anyhow::{Result, bail};
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, InferenceConfiguration, OutputConfig as BedrockOutputConfig,
    SpecificToolChoice, SystemContentBlock, ToolChoice as BedrockToolChoice, ToolConfiguration,
};

use crate::bedrock::BedrockChatCompletion;

fn tool_choice_from_value(value: &serde_json::Value) -> Result<BedrockToolChoice> {
    match value.get("type").and_then(|t| t.as_str()) {
        Some("auto") | Some("none") | None => {
            Ok(BedrockToolChoice::Auto(AutoToolChoice::builder().build()))
        }
        Some("any") => Ok(BedrockToolChoice::Any(AnyToolChoice::builder().build())),
        Some("tool") => {
            let name = value
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_default();
            Ok(BedrockToolChoice::Tool(
                SpecificToolChoice::builder().name(name).build()?,
            ))
        }
        Some(other) => bail!("Unsupported tool_choice type: {other}"),
    }
}

impl TryFrom<&V1MessagesRequest> for BedrockChatCompletion {
    type Error = anyhow::Error;

    fn try_from(request: &V1MessagesRequest) -> Result<Self, Self::Error> {
        let messages = Option::<Vec<_>>::try_from(&request.messages)?;

        let system_content_blocks = request
            .system
            .as_ref()
            .map(Vec::<SystemContentBlock>::try_from)
            .transpose()?;

        let tool_config = request
            .tools
            .as_deref()
            .map(build_bedrock_tools)
            .transpose()?
            .flatten()
            .map(|tools| {
                let choice = request
                    .tool_choice
                    .as_ref()
                    .map(tool_choice_from_value)
                    .transpose()?
                    .unwrap_or_else(|| BedrockToolChoice::Auto(AutoToolChoice::builder().build()));
                ToolConfiguration::builder()
                    .set_tools(Some(tools))
                    .tool_choice(choice)
                    .build()
                    .map_err(anyhow::Error::from)
            })
            .transpose()?;

        let inference_config = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens)
            .set_temperature(request.temperature)
            .set_top_p(request.top_p)
            .set_stop_sequences(request.stop_sequences.clone())
            .build();

        let output_config = request
            .output_config
            .as_ref()
            .map(Option::<BedrockOutputConfig>::try_from)
            .transpose()?
            .flatten();

        Ok(BedrockChatCompletion {
            model_id: request.model.clone(),
            messages,
            system_content_blocks,
            tool_config,
            inference_config,
            output_config,
        })
    }
}
