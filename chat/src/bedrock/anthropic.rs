use anthropic_request::V1MessagesRequest;
use anyhow::Result;
use aws_sdk_bedrockruntime::types::{InferenceConfiguration, SystemContentBlock};
use aws_smithy_types::Document;
use tracing::{info, warn};

use super::BedrockChatCompletion;

impl TryFrom<&V1MessagesRequest> for BedrockChatCompletion {
    type Error = anyhow::Error;

    fn try_from(request: &V1MessagesRequest) -> Result<Self, Self::Error> {
        let mut messages = Option::<Vec<_>>::try_from(&request.messages)?;
        
        // Filter out messages with empty content
        if let Some(ref mut msgs) = messages {
            let original_count = msgs.len();
            let mut removed_indices = Vec::new();
            
            msgs.retain_mut(|msg| {
                let content = msg.content();
                let is_empty = content.is_empty();
                
                if is_empty {
                    // Find the index by counting how many we've kept
                    let current_index = original_count - msgs.len() - removed_indices.len();
                    removed_indices.push(current_index);
                    warn!(
                        "Filtering out message at index {} with empty content. Role: {:?}",
                        current_index,
                        msg.role()
                    );
                }
                
                !is_empty
            });
            
            let filtered_count = msgs.len();
            if filtered_count < original_count {
                info!(
                    "Filtered {} messages with empty content. Original: {}, After filtering: {}. Removed indices: {:?}",
                    original_count - filtered_count,
                    original_count,
                    filtered_count,
                    removed_indices
                );
            }
        }

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
