use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, Message, OutputConfig, SystemContentBlock, ToolConfiguration,
};
use aws_smithy_types::Document;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod anthropic;
pub mod openai;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ReasoningEffortToThinkingBudgetTokens {
    pub low: i32,
    pub medium: i32,
    pub high: i32,
}

impl Default for ReasoningEffortToThinkingBudgetTokens {
    fn default() -> Self {
        Self {
            low: 1024,
            medium: 2048,
            high: 4096,
        }
    }
}

pub struct BedrockChatCompletion {
    pub model_id: String,
    pub messages: Option<Vec<Message>>,
    pub system_content_blocks: Option<Vec<SystemContentBlock>>,
    pub tool_config: Option<ToolConfiguration>,
    pub inference_config: InferenceConfiguration,
    pub additional_model_request_fields: Option<Document>,
    pub output_config: Option<OutputConfig>,
}

/// Adds an `anthropic_version` entry to `additional_model_request_fields`.
pub fn add_anthropic_version(fields: Option<Document>, version: &str) -> Option<Document> {
    match fields {
        Some(Document::Object(mut map)) => {
            map.insert(
                "anthropic_version".to_string(),
                Document::String(version.to_string()),
            );
            Some(Document::Object(map))
        }
        Some(other) => Some(other),
        None => {
            let mut map = HashMap::new();
            map.insert(
                "anthropic_version".to_string(),
                Document::String(version.to_string()),
            );
            Some(Document::Object(map))
        }
    }
}

/// Adds an `anthropic_beta` entry to `additional_model_request_fields`.
/// If the document already contains an `anthropic_beta` array, the new beta is appended.
pub fn add_anthropic_beta(fields: Option<Document>, beta: &str) -> Option<Document> {
    match fields {
        Some(Document::Object(mut map)) => {
            if let Some(Document::Array(existing)) = map.get_mut("anthropic_beta") {
                existing.push(Document::String(beta.to_string()));
            } else {
                map.insert(
                    "anthropic_beta".to_string(),
                    Document::Array(vec![Document::String(beta.to_string())]),
                );
            }
            Some(Document::Object(map))
        }
        Some(other) => Some(other),
        None => {
            let mut map = HashMap::new();
            map.insert(
                "anthropic_beta".to_string(),
                Document::Array(vec![Document::String(beta.to_string())]),
            );
            Some(Document::Object(map))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_anthropic_beta_creates_field_when_none() {
        let result = add_anthropic_beta(None, "context-1m-2025-08-07");
        let Document::Object(map) = result.unwrap() else {
            panic!("expected Document::Object");
        };

        let Document::Array(betas) = &map["anthropic_beta"] else {
            panic!("expected anthropic_beta to be an array");
        };
        assert_eq!(betas.len(), 1);
        assert_eq!(
            betas[0],
            Document::String("context-1m-2025-08-07".to_string())
        );
    }

    #[test]
    fn add_anthropic_beta_appends_to_existing_beta_array() {
        let existing = Some(Document::Object(
            [(
                "anthropic_beta".to_string(),
                Document::Array(vec![Document::String(
                    "effort-2025-11-24".to_string(),
                )]),
            )]
            .into_iter()
            .collect(),
        ));

        let result = add_anthropic_beta(existing, "context-1m-2025-08-07");
        let Document::Object(map) = result.unwrap() else {
            panic!("expected Document::Object");
        };

        let Document::Array(betas) = &map["anthropic_beta"] else {
            panic!("expected anthropic_beta to be an array");
        };
        assert_eq!(betas.len(), 2);
        assert_eq!(
            betas[0],
            Document::String("effort-2025-11-24".to_string())
        );
        assert_eq!(
            betas[1],
            Document::String("context-1m-2025-08-07".to_string())
        );
    }
}
