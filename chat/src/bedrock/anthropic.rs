use anthropic_request::V1MessagesRequest;
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{InferenceConfiguration, SystemContentBlock};
use aws_smithy_types::Document;

use super::BedrockChatCompletion;

impl TryFrom<&V1MessagesRequest> for BedrockChatCompletion {
    type Error = anyhow::Error;

    fn try_from(request: &V1MessagesRequest) -> Result<Self, Self::Error> {
        let messages =
            Option::<Vec<aws_sdk_bedrockruntime::types::Message>>::try_from(&request.messages)?;

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

        // Only include thinking if explicitly requested and max_tokens is sufficient
        let additional_model_request_fields = request.thinking.as_ref().and_then(|thinking| {
            // Bedrock requires max_tokens > budget_tokens
            if request.max_tokens > thinking.budget_tokens {
                Some(Document::from(thinking))
            } else {
                // Skip thinking if max_tokens is not sufficient
                None
            }
        });

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
