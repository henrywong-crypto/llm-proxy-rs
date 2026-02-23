use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, Message as BedrockMessage, ToolResultBlock, ToolUseBlock,
};
use serde::{Deserialize, Serialize};

use crate::content::Contents;
use crate::system_content::SystemContents;

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        #[serde(rename = "content")]
        #[serde(skip_serializing_if = "Option::is_none")]
        contents: Option<SystemContents>,
    },
    User {
        #[serde(rename = "content")]
        #[serde(skip_serializing_if = "Option::is_none")]
        contents: Option<Contents>,
    },
    Assistant {
        #[serde(rename = "content")]
        #[serde(skip_serializing_if = "Option::is_none")]
        contents: Option<Contents>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<crate::ToolCall>>,
    },
    Tool {
        #[serde(rename = "content")]
        #[serde(skip_serializing_if = "Option::is_none")]
        contents: Option<Contents>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    },
}

impl TryFrom<&Message> for Option<Vec<ContentBlock>> {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        match message {
            Message::Tool { .. } => Ok(Some(vec![ContentBlock::ToolResult(
                ToolResultBlock::try_from(message)?,
            )])),
            Message::Assistant {
                contents,
                tool_calls,
            } => Ok(Some(
                contents
                    .iter()
                    .map(Vec::<ContentBlock>::try_from)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .chain(
                        tool_calls
                            .iter()
                            .flatten()
                            .map(ToolUseBlock::try_from)
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .map(ContentBlock::ToolUse),
                    )
                    .collect::<Vec<_>>(),
            )),
            Message::User { contents } => Ok(contents
                .as_ref()
                .map(Vec::<ContentBlock>::try_from)
                .transpose()?),
            Message::System { .. } => unreachable!(),
        }
    }
}

impl TryFrom<&Message> for BedrockMessage {
    type Error = anyhow::Error;

    fn try_from(message: &Message) -> Result<Self, Self::Error> {
        match message {
            Message::Assistant { .. } => Ok(BedrockMessage::builder()
                .role(ConversationRole::Assistant)
                .set_content(Option::<Vec<ContentBlock>>::try_from(message)?)
                .build()?),
            Message::Tool { .. } => unreachable!(),
            Message::User { .. } => Ok(BedrockMessage::builder()
                .role(ConversationRole::User)
                .set_content(Option::<Vec<ContentBlock>>::try_from(message)?)
                .build()?),
            Message::System { .. } => unreachable!(),
        }
    }
}

pub fn tool_messages_to_bedrock_message(messages: &[&Message]) -> anyhow::Result<BedrockMessage> {
    let mut contents = Vec::new();

    for message in messages {
        if let Message::Tool { .. } = message
            && let Some(content_blocks) = Option::<Vec<ContentBlock>>::try_from(*message)?
        {
            contents.extend(content_blocks);
        }
    }

    Ok(BedrockMessage::builder()
        .role(ConversationRole::User)
        .set_content(Some(contents))
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_to_bedrock_message() {
        let json = serde_json::json!({
            "role": "user",
            "content": "hello"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let bedrock = BedrockMessage::try_from(&message).unwrap();
        assert_eq!(bedrock.role(), &ConversationRole::User);
        assert_eq!(bedrock.content().len(), 1);
        match &bedrock.content()[0] {
            ContentBlock::Text(text) => assert_eq!(text, "hello"),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn assistant_message_to_bedrock_message() {
        let json = serde_json::json!({
            "role": "assistant",
            "content": "hi there"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let bedrock = BedrockMessage::try_from(&message).unwrap();
        assert_eq!(bedrock.role(), &ConversationRole::Assistant);
        assert_eq!(bedrock.content().len(), 1);
        match &bedrock.content()[0] {
            ContentBlock::Text(text) => assert_eq!(text, "hi there"),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn assistant_with_tool_calls_to_content_blocks() {
        let json = serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"NYC\"}"
                }
            }]
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let content_blocks = Option::<Vec<ContentBlock>>::try_from(&message)
            .unwrap()
            .unwrap();
        assert_eq!(content_blocks.len(), 1);
        match &content_blocks[0] {
            ContentBlock::ToolUse(tool_use) => {
                assert_eq!(tool_use.tool_use_id(), "call_1");
                assert_eq!(tool_use.name(), "get_weather");
            }
            other => panic!("expected ToolUse, got {:?}", other),
        }
    }

    #[test]
    fn assistant_with_content_and_tool_calls() {
        let json = serde_json::json!({
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"NYC\"}"
                }
            }]
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let content_blocks = Option::<Vec<ContentBlock>>::try_from(&message)
            .unwrap()
            .unwrap();
        assert_eq!(content_blocks.len(), 2);
        match &content_blocks[0] {
            ContentBlock::Text(text) => assert_eq!(text, "Let me check"),
            other => panic!("expected Text, got {:?}", other),
        }
        match &content_blocks[1] {
            ContentBlock::ToolUse(tool_use) => {
                assert_eq!(tool_use.tool_use_id(), "call_1");
                assert_eq!(tool_use.name(), "get_weather");
            }
            other => panic!("expected ToolUse, got {:?}", other),
        }
    }

    #[test]
    fn assistant_with_multiple_tool_calls() {
        let json = serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\": \"NYC\"}"
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": "{\"timezone\": \"EST\"}"
                    }
                }
            ]
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let content_blocks = Option::<Vec<ContentBlock>>::try_from(&message)
            .unwrap()
            .unwrap();
        assert_eq!(content_blocks.len(), 2);
        match &content_blocks[0] {
            ContentBlock::ToolUse(tool_use) => {
                assert_eq!(tool_use.tool_use_id(), "call_1");
                assert_eq!(tool_use.name(), "get_weather");
            }
            other => panic!("expected ToolUse, got {:?}", other),
        }
        match &content_blocks[1] {
            ContentBlock::ToolUse(tool_use) => {
                assert_eq!(tool_use.tool_use_id(), "call_2");
                assert_eq!(tool_use.name(), "get_time");
            }
            other => panic!("expected ToolUse, got {:?}", other),
        }
    }

    #[test]
    fn tool_message_to_content_blocks() {
        let json = serde_json::json!({
            "role": "tool",
            "content": "sunny",
            "tool_call_id": "call_1"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let content_blocks = Option::<Vec<ContentBlock>>::try_from(&message)
            .unwrap()
            .unwrap();
        assert_eq!(content_blocks.len(), 1);
        match &content_blocks[0] {
            ContentBlock::ToolResult(result) => {
                assert_eq!(result.tool_use_id(), "call_1");
            }
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }

    #[test]
    fn user_message_with_none_content_to_content_blocks() {
        let json = serde_json::json!({
            "role": "user"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let content_blocks = Option::<Vec<ContentBlock>>::try_from(&message).unwrap();
        assert!(content_blocks.is_none());
    }

    #[test]
    fn user_message_with_empty_content() {
        let json = serde_json::json!({
            "role": "user",
            "content": ""
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let bedrock = BedrockMessage::try_from(&message).unwrap();
        assert_eq!(bedrock.content().len(), 1);
        match &bedrock.content()[0] {
            ContentBlock::Text(text) => assert_eq!(text, ""),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn tool_messages_to_bedrock_message_merges_multiple() {
        let json1 = serde_json::json!({
            "role": "tool",
            "content": "result1",
            "tool_call_id": "call_1"
        });
        let json2 = serde_json::json!({
            "role": "tool",
            "content": "result2",
            "tool_call_id": "call_2"
        });
        let msg1: Message = serde_json::from_value(json1).unwrap();
        let msg2: Message = serde_json::from_value(json2).unwrap();
        let messages: Vec<&Message> = vec![&msg1, &msg2];
        let bedrock = tool_messages_to_bedrock_message(&messages).unwrap();
        assert_eq!(bedrock.role(), &ConversationRole::User);
        assert_eq!(bedrock.content().len(), 2);
        match &bedrock.content()[0] {
            ContentBlock::ToolResult(result) => assert_eq!(result.tool_use_id(), "call_1"),
            other => panic!("expected ToolResult, got {:?}", other),
        }
        match &bedrock.content()[1] {
            ContentBlock::ToolResult(result) => assert_eq!(result.tool_use_id(), "call_2"),
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }

    #[test]
    fn tool_messages_to_bedrock_message_skips_non_tool() {
        let tool_json = serde_json::json!({
            "role": "tool",
            "content": "result1",
            "tool_call_id": "call_1"
        });
        let user_json = serde_json::json!({
            "role": "user",
            "content": "hello"
        });
        let msg1: Message = serde_json::from_value(tool_json).unwrap();
        let msg2: Message = serde_json::from_value(user_json).unwrap();
        let messages: Vec<&Message> = vec![&msg1, &msg2];
        let bedrock = tool_messages_to_bedrock_message(&messages).unwrap();
        assert_eq!(bedrock.content().len(), 1);
        match &bedrock.content()[0] {
            ContentBlock::ToolResult(result) => assert_eq!(result.tool_use_id(), "call_1"),
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }

    #[test]
    fn tool_message_with_image_e2e() {
        let json = serde_json::json!({
            "role": "tool",
            "content": [
                {"type": "text", "text": "image analysis"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ],
            "tool_call_id": "call_img"
        });
        let message: Message = serde_json::from_value(json).unwrap();
        let content_blocks = Option::<Vec<ContentBlock>>::try_from(&message)
            .unwrap()
            .unwrap();
        assert_eq!(content_blocks.len(), 1);
        match &content_blocks[0] {
            ContentBlock::ToolResult(result) => {
                assert_eq!(result.tool_use_id(), "call_img");
                assert_eq!(result.content().len(), 2);
            }
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }
}
