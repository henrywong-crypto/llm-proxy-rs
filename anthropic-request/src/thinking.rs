use aws_smithy_types::Document;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Thinking {
    Enabled {
        budget_tokens: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<String>,
    },
    Adaptive {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<String>,
    },
    Disabled,
}

impl From<&Thinking> for Document {
    fn from(thinking: &Thinking) -> Self {
        let inner: std::collections::HashMap<String, Document> = match thinking {
            Thinking::Enabled {
                budget_tokens,
                display,
            } => {
                let mut map = std::collections::HashMap::from([
                    ("type".to_string(), Document::String("enabled".to_string())),
                    ("budget_tokens".to_string(), Document::from(*budget_tokens)),
                ]);
                if let Some(display) = display {
                    map.insert("display".to_string(), Document::String(display.clone()));
                }
                map
            }
            Thinking::Adaptive { display } => {
                let mut map = std::collections::HashMap::from([(
                    "type".to_string(),
                    Document::String("adaptive".to_string()),
                )]);
                if let Some(display) = display {
                    map.insert("display".to_string(), Document::String(display.clone()));
                }
                map
            }
            Thinking::Disabled => std::collections::HashMap::from([(
                "type".to_string(),
                Document::String("disabled".to_string()),
            )]),
        };

        Document::Object(
            [("thinking".to_string(), Document::Object(inner))]
                .into_iter()
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inner_thinking(doc: Document) -> std::collections::HashMap<String, Document> {
        let Document::Object(map) = doc else {
            panic!("expected Document::Object");
        };
        let Some(Document::Object(inner)) = map.get("thinking") else {
            panic!("expected thinking object");
        };
        inner.clone()
    }

    #[test]
    fn enabled_with_display_includes_display_in_document() {
        let thinking = Thinking::Enabled {
            budget_tokens: 4000,
            display: Some("summarized".to_string()),
        };
        let inner = inner_thinking(Document::from(&thinking));
        assert_eq!(inner["type"], Document::String("enabled".to_string()));
        assert_eq!(inner["display"], Document::String("summarized".to_string()));
    }

    #[test]
    fn enabled_without_display_omits_display_in_document() {
        let thinking = Thinking::Enabled {
            budget_tokens: 4000,
            display: None,
        };
        let inner = inner_thinking(Document::from(&thinking));
        assert!(!inner.contains_key("display"));
    }

    #[test]
    fn adaptive_with_display_includes_display_in_document() {
        let thinking = Thinking::Adaptive {
            display: Some("raw".to_string()),
        };
        let inner = inner_thinking(Document::from(&thinking));
        assert_eq!(inner["type"], Document::String("adaptive".to_string()));
        assert_eq!(inner["display"], Document::String("raw".to_string()));
    }

    #[test]
    fn enabled_with_display_deserializes() {
        let json =
            serde_json::json!({"type": "enabled", "budget_tokens": 4000, "display": "summarized"});
        let thinking: Thinking = serde_json::from_value(json).unwrap();
        match thinking {
            Thinking::Enabled {
                budget_tokens,
                display,
            } => {
                assert_eq!(budget_tokens, 4000);
                assert_eq!(display.as_deref(), Some("summarized"));
            }
            _ => panic!("expected Enabled"),
        }
    }

    #[test]
    fn enabled_without_display_deserializes_to_none() {
        let json = serde_json::json!({"type": "enabled", "budget_tokens": 4000});
        let thinking: Thinking = serde_json::from_value(json).unwrap();
        match thinking {
            Thinking::Enabled { display, .. } => assert!(display.is_none()),
            _ => panic!("expected Enabled"),
        }
    }

    #[test]
    fn enabled_with_display_serializes_round_trip() {
        let thinking = Thinking::Enabled {
            budget_tokens: 4000,
            display: Some("summarized".to_string()),
        };
        let value = serde_json::to_value(&thinking).unwrap();
        assert_eq!(value["type"], "enabled");
        assert_eq!(value["budget_tokens"], 4000);
        assert_eq!(value["display"], "summarized");
    }

    #[test]
    fn enabled_without_display_omits_display_on_serialize() {
        let thinking = Thinking::Enabled {
            budget_tokens: 4000,
            display: None,
        };
        let value = serde_json::to_value(&thinking).unwrap();
        assert!(value.get("display").is_none());
    }

    #[test]
    fn adaptive_without_display_still_deserializes() {
        let json = serde_json::json!({"type": "adaptive"});
        let thinking: Thinking = serde_json::from_value(json).unwrap();
        match thinking {
            Thinking::Adaptive { display } => assert!(display.is_none()),
            _ => panic!("expected Adaptive"),
        }
    }
}
