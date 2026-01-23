use aws_sdk_bedrockruntime::types::{ConversationRole, Message as BedrockMessage};
use serde::{Deserialize, Serialize};

use crate::content::{AssistantContent, UserContent};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Messages {
    String(String),
    Array(Vec<Message>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

impl From<Messages> for Vec<Message> {
    fn from(messages: Messages) -> Self {
        match messages {
            Messages::String(s) => vec![Message::User {
                content: vec![UserContent::Text {
                    text: s,
                    cache_control: None,
                }],
            }],
            Messages::Array(arr) => arr,
        }
    }
}

pub fn messages_to_bedrock_messages(
    messages: &Messages,
) -> anyhow::Result<Option<Vec<BedrockMessage>>> {
    let message_vec: Vec<Message> = match messages {
        Messages::String(s) => vec![Message::User {
            content: vec![UserContent::Text {
                text: s.clone(),
                cache_control: None,
            }],
        }],
        Messages::Array(arr) => arr.clone(),
    };

    let bedrock_messages: Vec<BedrockMessage> = message_vec
        .iter()
        .map(BedrockMessage::try_from)
        .collect::<Result<_, _>>()?;

    Ok(if bedrock_messages.is_empty() {
        None
    } else {
        Some(bedrock_messages)
    })
}
