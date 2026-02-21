use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, Message, OutputConfig, SystemContentBlock, ToolConfiguration,
};
use aws_smithy_types::Document;

pub mod anthropic;
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

pub fn with_anthropic_beta(doc: Option<Document>, beta: Vec<String>) -> Option<Document> {
    if beta.is_empty() {
        return doc;
    }
    let beta_doc = Document::Array(beta.into_iter().map(Document::String).collect());
    Some(match doc {
        None => Document::Object(
            [("anthropic_beta".to_string(), beta_doc)]
                .into_iter()
                .collect(),
        ),
        Some(Document::Object(mut map)) => {
            map.insert("anthropic_beta".to_string(), beta_doc);
            Document::Object(map)
        }
        Some(d) => d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_anthropic_beta_empty_beta_none_doc_returns_none() {
        assert_eq!(with_anthropic_beta(None, vec![]), None);
    }

    #[test]
    fn with_anthropic_beta_empty_beta_some_doc_returns_doc_unchanged() {
        let doc = Some(Document::Object(
            [("thinking".to_string(), Document::String("foo".to_string()))]
                .into_iter()
                .collect(),
        ));
        assert_eq!(with_anthropic_beta(doc.clone(), vec![]), doc);
    }

    #[test]
    fn with_anthropic_beta_none_doc_creates_beta_doc() {
        let result = with_anthropic_beta(None, vec!["effort-2025-11-24".to_string()]);
        let Document::Object(map) = result.unwrap() else {
            panic!("expected Document::Object");
        };
        assert_eq!(
            map["anthropic_beta"],
            Document::Array(vec![Document::String("effort-2025-11-24".to_string())])
        );
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn with_anthropic_beta_merges_into_existing_doc() {
        let doc = Some(Document::Object(
            [("thinking".to_string(), Document::String("foo".to_string()))]
                .into_iter()
                .collect(),
        ));
        let result = with_anthropic_beta(doc, vec!["effort-2025-11-24".to_string()]);
        let Document::Object(map) = result.unwrap() else {
            panic!("expected Document::Object");
        };
        assert!(map.contains_key("thinking"));
        assert_eq!(
            map["anthropic_beta"],
            Document::Array(vec![Document::String("effort-2025-11-24".to_string())])
        );
        assert_eq!(map.len(), 2);
    }
}
