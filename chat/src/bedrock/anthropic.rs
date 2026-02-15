use anthropic_request::{OutputConfig, V1MessagesRequest};
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, OutputConfig as BedrockOutputConfig, SystemContentBlock,
};
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

        let mut additional_model_request_fields = request.thinking.as_ref().map(Document::from);
        let mut output_config = None;

        if let Some(output_cfg) = &request.output_config {
            match output_cfg {
                OutputConfig::Format { format } => {
                    output_config = Some(BedrockOutputConfig::try_from(format)?);
                }
                OutputConfig::Effort { effort } => {
                    let effort_doc = Document::Object(
                        [
                            (
                                "output_config".to_string(),
                                Document::Object(
                                    [("effort".to_string(), Document::String(effort.clone()))]
                                        .into_iter()
                                        .collect(),
                                ),
                            ),
                            (
                                "anthropic_beta".to_string(),
                                Document::Array(vec![Document::String(
                                    "effort-2025-11-24".to_string(),
                                )]),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    );

                    additional_model_request_fields =
                        match additional_model_request_fields.take() {
                            Some(Document::Object(mut existing_map)) => {
                                if let Document::Object(effort_map) = effort_doc {
                                    existing_map.extend(effort_map);
                                }
                                Some(Document::Object(existing_map))
                            }
                            _ => Some(effort_doc),
                        };
                }
                OutputConfig::Other(_) => {}
            }
        }

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
