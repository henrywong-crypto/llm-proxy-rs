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
#[serde(untagged)]
pub enum UserContents {
    String(String),
    Array(Vec<UserContent>),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum AssistantContents {
    String(String),
    Array(Vec<AssistantContent>),
}

impl From<UserContents> for Vec<UserContent> {
    fn from(wrapper: UserContents) -> Self {
        match wrapper {
            UserContents::String(s) => vec![UserContent::Text {
                text: s,
                cache_control: None,
            }],
            UserContents::Array(arr) => arr,
        }
    }
}

impl From<AssistantContents> for Vec<AssistantContent> {
    fn from(wrapper: AssistantContents) -> Self {
        match wrapper {
            AssistantContents::String(s) => vec![AssistantContent::Text {
                text: s,
                cache_control: None,
            }],
            AssistantContents::Array(arr) => arr,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(rename = "user")]
    User { content: UserContents },
    #[serde(rename = "assistant")]
    Assistant { content: AssistantContents },
}

impl TryFrom<&Message> for BedrockMessage {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        match message {
            Message::User { content } => {
                let content = match content {
                    UserContents::String(s) => [UserContent::Text {
                        text: s.clone(),
                        cache_control: None,
                    }]
                    .iter()
                    .map(Vec::try_from)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect(),
                    UserContents::Array(arr) => arr
                        .iter()
                        .map(Vec::try_from)
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .flatten()
                        .collect(),
                };

                Ok(BedrockMessage::builder()
                    .role(ConversationRole::User)
                    .set_content(Some(content))
                    .build()?)
            }
            Message::Assistant { content } => {
                let content = match content {
                    AssistantContents::String(s) => [AssistantContent::Text {
                        text: s.clone(),
                        cache_control: None,
                    }]
                    .iter()
                    .map(Vec::try_from)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect(),
                    AssistantContents::Array(arr) => arr
                        .iter()
                        .map(Vec::try_from)
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .flatten()
                        .collect(),
                };

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
                    content: UserContents::String(s.clone()),
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
