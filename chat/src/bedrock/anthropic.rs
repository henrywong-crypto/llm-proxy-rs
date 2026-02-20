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
            let mut fields = additional_model_request_fields(
                request.thinking.as_ref(),
                request.output_config.as_ref(),
            );
            
            // Add beta flags from the request header, filtering out known invalid ones
            if let Some(betas) = &request.betas {
                // Known invalid beta flags for Bedrock (confirmed through testing)
                const INVALID_BETAS: &[&str] = &[
                    "prompt-caching-scope-2026-01-05",  // Tested: returns "invalid beta flag" error
                ];
                
                for beta in betas {
                    if !INVALID_BETAS.contains(&beta.as_str()) {
                        fields = super::add_anthropic_beta(fields, beta);
                    } else {
                        // Log that we're filtering out an invalid beta
                        tracing::warn!("Filtering out invalid beta flag for Bedrock: {}", beta);
                    }
                }
            }
            
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
