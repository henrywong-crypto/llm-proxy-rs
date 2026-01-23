use aws_sdk_bedrockruntime::types::{ConversationRole, Message as BedrockMessage};
use serde::{Deserialize, Serialize};

use crate::content::{AssistantContent, UserContent};

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Messages {
    String(String),
    Array(Vec<Message>),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(rename = "user")]
    User {
        #[serde(deserialize_with = "deserialize_user_content")]
        content: Vec<UserContent>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(deserialize_with = "deserialize_assistant_content")]
        content: Vec<AssistantContent>,
    },
}

fn deserialize_user_content<'de, D>(deserializer: D) -> Result<Vec<UserContent>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let value = Value::deserialize(deserializer)?;

    match value {
        Value::String(s) => {
            // Convert string to a Text content block
            Ok(vec![UserContent::Text {
                text: s,
                cache_control: None,
            }])
        }
        Value::Array(_) => {
            // Deserialize as array of UserContent
            serde_json::from_value(value).map_err(|e| {
                Error::custom(format!("Failed to deserialize user content array: {}", e))
            })
        }
        _ => Err(Error::custom(
            "User content must be either a string or an array",
        )),
    }
}

fn deserialize_assistant_content<'de, D>(deserializer: D) -> Result<Vec<AssistantContent>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let value = Value::deserialize(deserializer)?;

    match value {
        Value::String(s) => {
            // Convert string to a Text content block
            Ok(vec![AssistantContent::Text {
                text: s,
                cache_control: None,
            }])
        }
        Value::Array(_) => {
            // Deserialize as array of AssistantContent
            serde_json::from_value(value).map_err(|e| {
                Error::custom(format!(
                    "Failed to deserialize assistant content array: {}",
                    e
                ))
            })
        }
        _ => Err(Error::custom(
            "Assistant content must be either a string or an array",
        )),
    }
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
