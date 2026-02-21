use aws_smithy_types::Document;

pub fn get_anthropic_beta_document(beta: Vec<String>) -> Option<Document> {
    if beta.is_empty() {
        return None;
    }
    Some(Document::Array(
        beta.into_iter().map(Document::String).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_beta_returns_none() {
        assert_eq!(get_anthropic_beta_document(vec![]), None);
    }

    #[test]
    fn single_beta_returns_array() {
        let result = get_anthropic_beta_document(vec!["effort-2025-11-24".to_string()]);
        assert_eq!(
            result.unwrap(),
            Document::Array(vec![Document::String("effort-2025-11-24".to_string())])
        );
    }

    #[test]
    fn multiple_betas_returns_array() {
        let result = get_anthropic_beta_document(vec![
            "effort-2025-11-24".to_string(),
            "interleaved-thinking-2025-05-14".to_string(),
        ]);
        assert_eq!(
            result.unwrap(),
            Document::Array(vec![
                Document::String("effort-2025-11-24".to_string()),
                Document::String("interleaved-thinking-2025-05-14".to_string()),
            ])
        );
    }
}
