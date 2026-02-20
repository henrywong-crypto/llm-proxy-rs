use anthropic_request::{Thinking, V1MessagesRequest, additional_model_request_fields};
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, OutputConfig as BedrockOutputConfig, SystemContentBlock,
};

use super::BedrockChatCompletion;

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
            .map(anthropic_request::tools_to_tool_configuration)
            .transpose()?
            .flatten();

        let inference_config = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens)
            .set_temperature(request.temperature)
            .build();

        let output_config = request
            .output_config
            .as_ref()
            .map(Option::<BedrockOutputConfig>::try_from)
            .transpose()?
            .flatten();

        let additional_model_request_fields = {
            let fields = additional_model_request_fields(
                request.thinking.as_ref(),
                request.output_config.as_ref(),
            );
            let fields = super::add_anthropic_beta(fields, "context-1m-2025-08-07");
            // Testing: claude-code-20250219
            let fields = super::add_anthropic_beta(fields, "claude-code-20250219");
            let fields = if request.thinking.is_some() {
                super::add_anthropic_beta(fields, "interleaved-thinking-2025-05-14")
            } else {
                fields
            };
            let fields = super::add_anthropic_beta(fields, "adaptive-thinking-2026-01-28");
            fields
        };

        Ok(BedrockChatCompletion {
            model_id: request.model.clone(),
            messages,
            system_content_blocks,
            tool_config,
            inference_config,
            additional_model_request_fields,
            output_config,
        })
    }
}
