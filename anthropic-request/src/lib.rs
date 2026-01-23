use serde::{Deserialize, Deserializer, Serialize};

pub mod cache_control;
pub mod content;
pub mod message;
pub mod system;
pub mod thinking;
pub mod tool;
pub mod tool_result_content;

pub use cache_control::*;
pub use content::*;
pub use message::*;
pub use system::*;
pub use thinking::*;
pub use tool::*;
pub use tool_result_content::*;

// Custom deserializer that accepts either a string or an array of messages
fn deserialize_messages<'de, D>(deserializer: D) -> Result<Vec<Message>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let value = Value::deserialize(deserializer)?;
    
    match value {
        // If it's a string, convert it to a single user message
        Value::String(text) => {
            Ok(vec![Message::User {
                content: vec![UserContent::Text { 
                    text,
                    cache_control: None,
                }],
            }])
        }
        // If it's an array, deserialize normally
        Value::Array(_) => {
            serde_json::from_value(value).map_err(D::Error::custom)
        }
        _ => Err(D::Error::custom("messages must be either a string or an array")),
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct V1MessagesRequest {
    pub max_tokens: i32,
    #[serde(deserialize_with = "deserialize_messages")]
    pub messages: Vec<Message>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Systems>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}
