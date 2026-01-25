use serde::{Deserialize, Serialize};

use crate::delta::{Delta, DeltaUsage, MessageDelta};
use crate::error::Error;
use crate::message::Message;

/// Streaming event types for Anthropic API
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    /// MessageStartEvent is sent at the start of streaming
    #[serde(rename = "message_start")]
    MessageStart { message: Message },

    /// ContentBlockStartEvent signals the start of a content block
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        content_block: ContentBlock,
        index: i32,
    },

    /// ContentBlockDeltaEvent contains incremental content updates
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: Delta, index: i32 },

    /// ContentBlockStopEvent signals the end of a content block
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: i32 },

    /// MessageDeltaEvent contains updates to the message
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        usage: DeltaUsage,
    },

    /// MessageStopEvent signals the end of the message
    #[serde(rename = "message_stop")]
    MessageStop,

    /// PingEvent is a keepalive event
    #[serde(rename = "ping")]
    Ping,

    /// StreamErrorEvent is an error during streaming
    #[serde(rename = "error")]
    Error { error: Error },
}

impl Event {
    pub fn content_block_delta_builder() -> ContentBlockDeltaEventBuilder {
        ContentBlockDeltaEventBuilder::default()
    }

    pub fn content_block_start_builder() -> ContentBlockStartEventBuilder {
        ContentBlockStartEventBuilder::default()
    }

    pub fn content_block_stop_builder() -> ContentBlockStopEventBuilder {
        ContentBlockStopEventBuilder::default()
    }

    pub fn message_delta_builder() -> MessageDeltaEventBuilder {
        MessageDeltaEventBuilder::default()
    }

    pub fn message_start_builder() -> MessageStartEventBuilder {
        MessageStartEventBuilder::default()
    }

    pub fn message_stop() -> Self {
        Event::MessageStop
    }

    pub fn ping() -> Self {
        Event::Ping
    }
}

#[derive(Default)]
pub struct ContentBlockDeltaEventBuilder {
    delta: Option<Delta>,
    index: i32,
}

impl ContentBlockDeltaEventBuilder {
    pub fn delta(mut self, delta: Delta) -> Self {
        self.delta = Some(delta);
        self
    }

    pub fn index(mut self, index: i32) -> Self {
        self.index = index;
        self
    }

    pub fn build(self) -> Event {
        Event::ContentBlockDelta {
            delta: self.delta.expect("delta is required"),
            index: self.index,
        }
    }
}

#[derive(Default)]
pub struct ContentBlockStartEventBuilder {
    content_block: Option<ContentBlock>,
    index: i32,
}

impl ContentBlockStartEventBuilder {
    pub fn content_block(mut self, content_block: ContentBlock) -> Self {
        self.content_block = Some(content_block);
        self
    }

    pub fn index(mut self, index: i32) -> Self {
        self.index = index;
        self
    }

    pub fn build(self) -> Event {
        Event::ContentBlockStart {
            content_block: self.content_block.expect("content_block is required"),
            index: self.index,
        }
    }
}

#[derive(Default)]
pub struct ContentBlockStopEventBuilder {
    index: i32,
}

impl ContentBlockStopEventBuilder {
    pub fn index(mut self, index: i32) -> Self {
        self.index = index;
        self
    }

    pub fn build(self) -> Event {
        Event::ContentBlockStop { index: self.index }
    }
}

#[derive(Default)]
pub struct MessageDeltaEventBuilder {
    delta: Option<MessageDelta>,
    usage: Option<DeltaUsage>,
}

impl MessageDeltaEventBuilder {
    pub fn delta(mut self, delta: MessageDelta) -> Self {
        self.delta = Some(delta);
        self
    }

    pub fn usage(mut self, usage: DeltaUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn build(self) -> Event {
        Event::MessageDelta {
            delta: self.delta.expect("delta is required"),
            usage: self.usage.expect("usage is required"),
        }
    }
}

#[derive(Default)]
pub struct MessageStartEventBuilder {
    message: Option<Message>,
}

impl MessageStartEventBuilder {
    pub fn message(mut self, message: Message) -> Self {
        self.message = Some(message);
        self
    }

    pub fn build(self) -> Event {
        Event::MessageStart {
            message: self.message.expect("message is required"),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { signature: String, thinking: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        input: serde_json::Value,
        name: String,
    },
}

impl ContentBlock {
    pub fn text_builder() -> TextBlockBuilder {
        TextBlockBuilder::default()
    }

    pub fn thinking_builder() -> ThinkingBlockBuilder {
        ThinkingBlockBuilder::default()
    }

    pub fn tool_use_builder() -> ToolUseBlockBuilder {
        ToolUseBlockBuilder::default()
    }
}

#[derive(Default)]
pub struct TextBlockBuilder {
    text: String,
}

impl TextBlockBuilder {
    pub fn text(mut self, text: String) -> Self {
        self.text = text;
        self
    }

    pub fn build(self) -> ContentBlock {
        ContentBlock::Text { text: self.text }
    }
}

#[derive(Default)]
pub struct ThinkingBlockBuilder {
    signature: String,
    thinking: String,
}

impl ThinkingBlockBuilder {
    pub fn signature(mut self, signature: String) -> Self {
        self.signature = signature;
        self
    }

    pub fn thinking(mut self, thinking: String) -> Self {
        self.thinking = thinking;
        self
    }

    pub fn build(self) -> ContentBlock {
        ContentBlock::Thinking {
            signature: self.signature,
            thinking: self.thinking,
        }
    }
}

#[derive(Default)]
pub struct ToolUseBlockBuilder {
    id: String,
    input: Option<serde_json::Value>,
    name: String,
}

impl ToolUseBlockBuilder {
    pub fn id(mut self, id: String) -> Self {
        self.id = id;
        self
    }

    pub fn input(mut self, input: serde_json::Value) -> Self {
        self.input = Some(input);
        self
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn build(self) -> ContentBlock {
        ContentBlock::ToolUse {
            id: self.id,
            input: self.input.unwrap_or(serde_json::json!({})),
            name: self.name,
        }
    }
}
