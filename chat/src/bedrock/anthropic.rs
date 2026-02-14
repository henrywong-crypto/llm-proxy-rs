use anthropic_request::V1MessagesRequest;
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{InferenceConfiguration, OutputConfig, SystemContentBlock};
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
            .map(anthropic_request::tools_to_tool_configuration)
            .transpose()?
            .flatten();

        let inference_config = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens)
            .set_temperature(request.temperature)
            .build();

        // Handle additional_model_request_fields
        // Priority: thinking > output_config.effort
        let mut additional_model_request_fields = request.thinking.as_ref().map(Document::from);
        
        // If output_config has effort, merge it into additional_model_request_fields
        if let Some(output_cfg) = &request.output_config {
            if let Some(effort_doc) = output_cfg.to_additional_model_request_fields() {
                additional_model_request_fields = match additional_model_request_fields.take() {
                    Some(existing) => {
                        // Merge the two Documents
                        match (existing, effort_doc) {
                            (Document::Object(mut existing_map), Document::Object(effort_map)) => {
                                existing_map.extend(effort_map);
                                Some(Document::Object(existing_map))
                            }
                            (_, effort_doc) => Some(effort_doc),
                        }
                    }
                    None => Some(effort_doc),
                };
            }
        }

        // Convert output_config to Bedrock OutputConfig only if it has format (JSON schema)
        let output_config = request
            .output_config
            .as_ref()
            .filter(|cfg| cfg.has_format())
            .map(OutputConfig::try_from)
            .transpose()?;

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
