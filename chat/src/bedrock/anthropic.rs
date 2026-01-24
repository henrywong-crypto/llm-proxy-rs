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
            
            // First pass: identify empty messages and log them
            let mut indices_to_remove = Vec::new();
            for (index, msg) in msgs.iter().enumerate() {
                let content = msg.content();
                if content.is_empty() {
                    warn!(
                        "Found message at index {} with empty content. Role: {:?}",
                        index,
                        msg.role()
                    );
                    indices_to_remove.push(index);
                }
            }
            
            // Second pass: remove empty messages
            if !indices_to_remove.is_empty() {
                let mut removed_count = 0;
                msgs.retain(|msg| {
                    let should_keep = !msg.content().is_empty();
                    if !should_keep {
                        removed_count += 1;
                    }
                    should_keep
                });
                
                info!(
                    "Filtered {} messages with empty content. Original: {}, After filtering: {}. Removed indices: {:?}",
                    removed_count,
                    original_count,
                    msgs.len(),
                    indices_to_remove
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
