use aws_sdk_bedrockruntime::types::{ConversationRole, Message as BedrockMessage};
use serde::{Deserialize, Deserializer, Serialize};
use tracing::debug;

use crate::content::{AssistantContent, UserContent};

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Messages {
    String(String),
    Array(Vec<Message>),
}

impl<'de> Deserialize<'de> for Messages {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        use serde_json::Value;

        let value = Value::deserialize(deserializer)?;
        
        debug!("🔍 Deserializing Messages, value type: {}", match &value {
            serde_json::Value::String(_) => "String",
            serde_json::Value::Array(_) => "Array",
            serde_json::Value::Object(_) => "Object",
            serde_json::Value::Number(_) => "Number",
            serde_json::Value::Bool(_) => "Bool",
            serde_json::Value::Null => "Null",
        });

        if let serde_json::Value::String(s) = value {
            debug!("✅ Messages is a String");
            return Ok(Messages::String(s));
        }

        if let serde_json::Value::Array(arr) = &value {
            debug!("🔍 Messages is an Array with {} elements", arr.len());
            
            for (i, item) in arr.iter().enumerate() {
                debug!("  Message[{}]: {:?}", i, item.get("role"));
                if let Some(content) = item.get("content") {
                    match content {
                        serde_json::Value::Array(content_arr) => {
                            debug!("    content is Array with {} items", content_arr.len());
                            for (j, content_item) in content_arr.iter().enumerate() {
                                debug!("      content[{}] type: {:?}", j, content_item.get("type"));
                            }
                        }
                        serde_json::Value::String(s) => {
                            debug!("    content is String: {}", s);
                        }
                        _ => {
                            debug!("    content is other type");
                        }
                    }
                }
            }

            match serde_json::from_value::<Vec<Message>>(value) {
                Ok(messages) => {
                    debug!("✅ Successfully deserialized {} messages", messages.len());
                    return Ok(Messages::Array(messages));
                }
                Err(e) => {
                    debug!("❌ Failed to deserialize messages array: {}", e);
                    return Err(Error::custom(format!("Failed to deserialize messages array: {}", e)));
                }
            }
        }

        debug!("❌ Messages value is neither String nor Array");
        Err(Error::custom("messages must be either a string or an array of message objects"))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User { content: Vec<UserContent> },
    Assistant { content: Vec<AssistantContent> },
}

impl TryFrom<&Message> for BedrockMessage {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        match message {
            Message::User { content } => {
                let content = content
                    .iter()
                    .map(Vec::try_from)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                Ok(BedrockMessage::builder()
                    .role(ConversationRole::User)
                    .set_content(Some(content))
                    .build()?)
            }
            Message::Assistant { content } => {
                let content = content
                    .iter()
                    .map(Vec::try_from)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                Ok(BedrockMessage::builder()
                    .role(ConversationRole::Assistant)
                    .set_content(Some(content))
                    .build()?)
            }
        }
    }
}

impl TryFrom<&Messages> for Option<Vec<BedrockMessage>> {
    type Error = anyhow::Error;

    fn try_from(messages: &Messages) -> Result<Self, Self::Error> {
        let bedrock_messages: Vec<BedrockMessage> = match messages {
            Messages::String(s) => {
                // Create a temporary message for string case
                let temp_message = Message::User {
                    content: vec![UserContent::Text {
                        text: s.clone(),
                        cache_control: None,
                    }],
                };
                vec![BedrockMessage::try_from(&temp_message)?]
            }
            Messages::Array(arr) => {
                // Process array directly without cloning
                arr.iter()
                    .map(BedrockMessage::try_from)
                    .collect::<Result<_, _>>()?
            }
        };

        Ok(if bedrock_messages.is_empty() {
            None
        } else {
            Some(bedrock_messages)
        })
    }
}
