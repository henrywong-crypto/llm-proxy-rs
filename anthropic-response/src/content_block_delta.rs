use crate::delta::Delta;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta as BedrockContentBlockDelta, ReasoningContentBlockDelta,
};

/// Converts Bedrock ContentBlockDelta to Anthropic Delta
pub fn bedrock_content_block_delta_to_delta(delta: &BedrockContentBlockDelta) -> Option<Delta> {
    match delta {
        BedrockContentBlockDelta::ReasoningContent(reasoning_content) => match reasoning_content {
            ReasoningContentBlockDelta::Signature(signature) => Some(Delta::SignatureDelta {
                signature: signature.clone(),
            }),
            ReasoningContentBlockDelta::Text(text) => Some(Delta::ThinkingDelta {
                thinking: text.clone(),
            }),
            _ => None,
        },
        BedrockContentBlockDelta::Text(text) => Some(Delta::TextDelta { text: text.clone() }),
        BedrockContentBlockDelta::ToolUse(tool_use) => Some(Delta::InputJsonDelta {
            partial_json: tool_use.input.clone(),
        }),
        _ => None,
    }
}

// Keep the old name for backward compatibility
#[deprecated(note = "Use bedrock_content_block_delta_to_delta instead")]
pub fn bedrock_content_block_delta_to_content_block_delta(
    delta: &BedrockContentBlockDelta,
) -> Option<Delta> {
    bedrock_content_block_delta_to_delta(delta)
}
