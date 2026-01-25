use anthropic_request::V1MessagesRequest;
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{InferenceConfiguration, SystemContentBlock};
use aws_smithy_types::Document;

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
            .map(|tools| {
                anthropic_request::tools_to_tool_configuration_with_choice(
                    tools,
                    request.tool_choice.as_ref(),
                )
            })
            .transpose()?
            .flatten();

        let inference_config = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens)
            .set_temperature(request.temperature)
            .set_top_p(request.top_p)
            .set_stop_sequences(request.stop_sequences.clone())
            .build();

        let additional_model_request_fields = request.thinking.as_ref().map(Document::from);

        Ok(BedrockChatCompletion {
            model_id: request.model.clone(),
            messages,
            system_content_blocks,
            tool_config,
            inference_config,
            additional_model_request_fields,
        })
    }
}
