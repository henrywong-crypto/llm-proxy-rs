use axum::http::HeaderMap;
use std::borrow::Cow;
use tracing::warn;

/// Bedrock caps tool names at 64 characters. Names within the limit are passed
/// through untouched; over-long names are shortened deterministically to a
/// readable prefix plus a stable hash suffix, so the same source name always
/// maps to the same Bedrock-safe name — keeping a request's tool specs,
/// `tool_use` blocks, and `tool_choice` in agreement. CRC-32 gives a
/// version-stable, reproducible 32-bit digest (8 hex chars).
pub fn bedrock_tool_name(name: &str) -> Cow<'_, str> {
    const MAX: usize = 64;
    const SUFFIX_LEN: usize = 9; // '_' + 8 hex digits (CRC-32, u32)
    if name.chars().count() <= MAX {
        return Cow::Borrowed(name);
    }
    let digest = crc32fast::hash(name.as_bytes());
    let prefix: String = name.chars().take(MAX - SUFFIX_LEN).collect();
    Cow::Owned(format!("{prefix}_{digest:08x}"))
}

pub fn filter_anthropic_beta(headers: &HeaderMap, whitelist: &[String]) -> Option<Vec<String>> {
    let requested: Vec<&str> = headers
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|s| s.trim())
        .collect();

    let filtered_out: Vec<&str> = requested
        .iter()
        .filter(|r| !whitelist.iter().any(|b| b.as_str() == **r))
        .copied()
        .collect();

    if !filtered_out.is_empty() {
        warn!("anthropic_beta filtered out: {:?}", filtered_out);
    }

    let v: Vec<String> = whitelist
        .iter()
        .filter(|b| requested.contains(&b.as_str()))
        .cloned()
        .collect();

    if v.is_empty() { None } else { Some(v) }
}

pub fn value_to_document(value: &serde_json::Value) -> aws_smithy_types::Document {
    match value {
        serde_json::Value::Null => aws_smithy_types::Document::Null,
        serde_json::Value::Bool(b) => aws_smithy_types::Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                aws_smithy_types::Document::Number(if i >= 0 {
                    aws_smithy_types::Number::PosInt(i as u64)
                } else {
                    aws_smithy_types::Number::NegInt(i)
                })
            } else {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::Float(
                    n.as_f64().unwrap_or(0.0),
                ))
            }
        }
        serde_json::Value::String(s) => aws_smithy_types::Document::String(s.clone()),
        serde_json::Value::Array(a) => {
            aws_smithy_types::Document::Array(a.iter().map(value_to_document).collect())
        }
        serde_json::Value::Object(o) => aws_smithy_types::Document::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), value_to_document(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_anthropic_beta_only_whitelisted_pass_through() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            "context-1m-2025-08-07,prompt-caching-scope-2026-01-05,effort-2025-11-24"
                .parse()
                .unwrap(),
        );
        let whitelist = vec![
            "context-1m-2025-08-07".to_string(),
            "effort-2025-11-24".to_string(),
        ];
        let result = filter_anthropic_beta(&headers, &whitelist);
        assert_eq!(
            result.unwrap(),
            vec![
                "context-1m-2025-08-07".to_string(),
                "effort-2025-11-24".to_string(),
            ]
        );
    }

    #[test]
    fn filter_anthropic_beta_non_whitelisted_all_filtered_out() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            "prompt-caching-scope-2026-01-05".parse().unwrap(),
        );
        let whitelist = vec!["effort-2025-11-24".to_string()];
        let result = filter_anthropic_beta(&headers, &whitelist);
        assert!(result.is_none());
    }

    #[test]
    fn filter_anthropic_beta_no_header_returns_none() {
        let headers = HeaderMap::new();
        let whitelist = vec![
            "context-1m-2025-08-07".to_string(),
            "effort-2025-11-24".to_string(),
        ];
        let result = filter_anthropic_beta(&headers, &whitelist);
        assert!(result.is_none());
    }

    #[test]
    fn bedrock_tool_name_within_limit_is_borrowed_unchanged() {
        let name = "get_weather";
        let out = bedrock_tool_name(name);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, "get_weather");

        let exactly_64 = "a".repeat(64);
        assert_eq!(bedrock_tool_name(&exactly_64), exactly_64);
    }

    #[test]
    fn bedrock_tool_name_over_limit_is_shortened_to_64_chars() {
        let name = "mcp__some_server__".to_string() + &"x".repeat(80);
        let out = bedrock_tool_name(&name);
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out.chars().count(), 64);
        assert!(out.starts_with("mcp__some_server__"));
    }

    #[test]
    fn bedrock_tool_name_is_deterministic() {
        let name = "z".repeat(100);
        assert_eq!(bedrock_tool_name(&name), bedrock_tool_name(&name));
    }

    #[test]
    fn bedrock_tool_name_distinguishes_names_sharing_a_prefix() {
        let prefix = "p".repeat(70);
        let a = format!("{prefix}_alpha");
        let b = format!("{prefix}_beta");
        assert_ne!(bedrock_tool_name(&a), bedrock_tool_name(&b));
    }
}
