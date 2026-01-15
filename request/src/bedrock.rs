use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, Message as BedrockMessage, SystemContentBlock, ToolConfiguration,
};

/// Unified Bedrock request format
///
/// This struct represents a request to AWS Bedrock's Converse API,
/// and can be constructed from either Anthropic or OpenAI-compatible formats.
pub struct BedrockRequest {
    pub model_id: String,
    pub messages: Vec<BedrockMessage>,
    pub system: Vec<SystemContentBlock>,
    pub inference_config: InferenceConfiguration,
    pub tool_config: Option<ToolConfiguration>,
    pub additional_fields: Option<aws_smithy_types::Document>,
}

