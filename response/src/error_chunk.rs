use serde::Serialize;

/// OpenAI-shaped streaming error chunk, sent as a `data:` line ahead of the
/// usual `data: [DONE]` sentinel:
/// `{"error": {"message": "...", "type": "...", "code": null, "param": null}}`
#[derive(Debug, Serialize)]
pub struct ChatCompletionsErrorChunk {
    pub error: ChatCompletionsError,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionsError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
    param: Option<String>,
}

impl ChatCompletionsErrorChunk {
    pub fn new(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ChatCompletionsError {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
                param: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_openai_wire_shape() {
        let chunk = ChatCompletionsErrorChunk::new("server_error", "boom");
        assert_eq!(
            serde_json::to_value(&chunk).expect("serialize"),
            serde_json::json!({
                "error": {
                    "message": "boom",
                    "type": "server_error",
                    "code": null,
                    "param": null,
                }
            })
        );
    }
}
