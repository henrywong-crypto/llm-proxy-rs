use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, Message, OutputConfig, SystemContentBlock, ToolConfiguration,
};
use aws_smithy_types::Document;

pub mod anthropic;
pub mod anthropic_beta;
pub mod openai;

pub struct BedrockChatCompletion {
    pub model_id: String,
    pub messages: Option<Vec<Message>>,
    pub system_content_blocks: Option<Vec<SystemContentBlock>>,
    pub tool_config: Option<ToolConfiguration>,
    pub inference_config: InferenceConfiguration,
    pub additional_model_request_fields: Option<Document>,
    pub output_config: Option<OutputConfig>,
}

