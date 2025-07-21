use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockStart, ConversationRole, ConverseStreamOutput, StopReason,
    ToolUseBlockDelta, ToolUseBlockStart,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing;

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionsResponse {
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<Delta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub index: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<Function>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Function {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    pub completion_tokens: i32,
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

impl ChatCompletionsResponse {
    pub fn builder() -> ChatCompletionsResponseBuilder {
        ChatCompletionsResponseBuilder::default()
    }
}

#[derive(Default)]
pub struct ChatCompletionsResponseBuilder {
    choices: Vec<Choice>,
    created: Option<i64>,
    id: Option<String>,
    model: Option<String>,
    object: Option<String>,
    usage: Option<Usage>,
}

impl ChatCompletionsResponseBuilder {
    pub fn choice(mut self, choice: Choice) -> Self {
        self.choices.push(choice);
        self
    }

    pub fn created(mut self, created: Option<i64>) -> Self {
        self.created = created;
        self
    }

    pub fn id(mut self, id: Option<String>) -> Self {
        self.id = id;
        self
    }

    pub fn model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }

    pub fn object(mut self, object: Option<String>) -> Self {
        self.object = object;
        self
    }

    pub fn usage(mut self, usage: Option<Usage>) -> Self {
        self.usage = usage;
        self
    }

    pub fn build(self) -> ChatCompletionsResponse {
        let mut choices = self.choices;
        
        // Ensure there's always at least one choice for streaming compatibility
        if choices.is_empty() {
            choices.push(Choice {
                delta: Some(Delta {
                    content: None,
                    role: None,
                    tool_calls: None,
                }),
                finish_reason: None,
                index: 0,
                logprobs: None,
            });
        }

        ChatCompletionsResponse {
            choices,
            created: self.created,
            id: self.id,
            model: self.model,
            object: self.object,
            usage: self.usage,
        }
    }
}

#[derive(Default)]
pub struct ChoiceBuilder {
    pub delta: Option<Delta>,
    pub finish_reason: Option<String>,
    pub index: i32,
    pub logprobs: Option<String>,
}

impl ChoiceBuilder {
    pub fn delta(mut self, delta: Option<Delta>) -> Self {
        self.delta = delta;
        self
    }

    pub fn finish_reason(mut self, reason: Option<String>) -> Self {
        self.finish_reason = reason;
        self
    }

    pub fn index(mut self, index: i32) -> Self {
        self.index = index;
        self
    }

    pub fn logprobs(mut self, logprobs: Option<String>) -> Self {
        self.logprobs = logprobs;
        self
    }

    pub fn build(self) -> Choice {
        Choice {
            delta: self.delta,
            finish_reason: self.finish_reason,
            index: self.index,
            logprobs: self.logprobs,
        }
    }
}

#[derive(Default)]
pub struct UsageBuilder {
    pub completion_tokens: i32,
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

impl UsageBuilder {
    pub fn completion_tokens(mut self, tokens: i32) -> Self {
        self.completion_tokens = tokens;
        self
    }

    pub fn prompt_tokens(mut self, tokens: i32) -> Self {
        self.prompt_tokens = tokens;
        self
    }

    pub fn total_tokens(mut self, tokens: i32) -> Self {
        self.total_tokens = tokens;
        self
    }

    pub fn build(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
        }
    }
}

fn tool_use_block_delta_to_tool_call(tool_use_block_delta: &ToolUseBlockDelta, index: i32) -> ToolCall {
    ToolCall {
        id: None,
        tool_type: "function".to_string(),
        function: Some(Function {
            name: None,
            arguments: Some(tool_use_block_delta.input.clone()),
        }),
        index: Some(index),
    }
}

fn tool_use_block_start_to_tool_call(tool_use_block_start: &ToolUseBlockStart, index: i32) -> ToolCall {
    ToolCall {
        id: Some(tool_use_block_start.tool_use_id().to_string()),
        tool_type: "function".to_string(),
        function: Some(Function {
            name: Some(tool_use_block_start.name().to_string()),
            arguments: Some("".to_string()),
        }),
        index: Some(index),
    }
}

pub fn converse_stream_output_to_chat_completions_response_builder(
    output: &ConverseStreamOutput,
    usage_callback: Arc<dyn Fn(&Usage)>,
) -> ChatCompletionsResponseBuilder {
    let mut builder = ChatCompletionsResponse::builder()
        .object(Some("chat.completion.chunk".to_string()));

    tracing::info!("Processing Bedrock stream output: {:?}", 
        match output {
            ConverseStreamOutput::ContentBlockDelta(_) => "ContentBlockDelta",
            ConverseStreamOutput::ContentBlockStart(_) => "ContentBlockStart",
            ConverseStreamOutput::MessageStart(_) => "MessageStart", 
            ConverseStreamOutput::MessageStop(_) => "MessageStop",
            ConverseStreamOutput::Metadata(_) => "Metadata",
            _ => "Other"
        }
    );

    match output {
        ConverseStreamOutput::ContentBlockDelta(event) => {
            let delta = event.delta.as_ref().and_then(|d| match d {
                ContentBlockDelta::Text(text) => {
                    tracing::info!("Bedrock text delta: {}", text);
                    Some(Delta {
                        content: Some(text.clone()),
                        role: None,
                        tool_calls: None,
                    })
                }
                ContentBlockDelta::ToolUse(tool_use) => {
                    tracing::info!("Bedrock tool use delta: input={}", tool_use.input);
                    Some(Delta {
                        content: None,
                        role: None,
                        tool_calls: Some(vec![tool_use_block_delta_to_tool_call(tool_use, 0)]),
                    })
                }
                _ => None,
            });

            let choice = ChoiceBuilder::default()
                .delta(delta)
                .index(0) // Always use index 0 for streaming
                .build();

            builder = builder.choice(choice);
        }
        ConverseStreamOutput::ContentBlockStart(event) => {
            let delta = event.start.as_ref().and_then(|start| match start {
                ContentBlockStart::ToolUse(tool_use) => {
                    tracing::info!("Bedrock tool use start: name={}, id={}", 
                        tool_use.name(), tool_use.tool_use_id());
                    Some(Delta {
                        content: None,
                        role: None,
                        tool_calls: Some(vec![tool_use_block_start_to_tool_call(tool_use, 0)]),
                    })
                }
                _ => {
                    tracing::debug!("Bedrock content block start (non-tool)");
                    None
                }
            });

            let choice = ChoiceBuilder::default()
                .delta(delta)
                .index(0) // Always use index 0 for streaming
                .build();

            builder = builder.choice(choice);
        }
        ConverseStreamOutput::MessageStart(event) => {
            let delta = Some(Delta {
                content: match event.role {
                    ConversationRole::Assistant => Some("".to_string()),
                    _ => None,
                },
                role: match event.role {
                    ConversationRole::Assistant => Some("assistant".to_string()),
                    _ => None,
                },
                tool_calls: None,
            });

            let choice = ChoiceBuilder::default()
                .delta(delta)
                .index(0)
                .build();

            builder = builder.choice(choice);
        }
        ConverseStreamOutput::MessageStop(event) => {
            tracing::info!("Bedrock message stop with reason: {:?}", event.stop_reason);
            let (content, finish_reason) = match event.stop_reason {
                StopReason::EndTurn => (None, Some("stop".to_string())),
                StopReason::ToolUse => {
                    tracing::info!("Message stopped due to tool use - should have tool calls");
                    (None, Some("tool_calls".to_string()))
                }
                StopReason::MaxTokens => (None, Some("length".to_string())),
                StopReason::StopSequence => (None, Some("stop".to_string())),
                _ => (None, Some("stop".to_string())),
            };

            let choice = ChoiceBuilder::default()
                .delta(Some(Delta {
                    content,
                    role: None,
                    tool_calls: None,
                }))
                .finish_reason(finish_reason)
                .index(0)
                .build();

            builder = builder.choice(choice);
        }
        ConverseStreamOutput::Metadata(event) => {
            let usage = event.usage.as_ref().map(|u| {
                let usage = UsageBuilder::default()
                    .completion_tokens(u.output_tokens)
                    .prompt_tokens(u.input_tokens)
                    .total_tokens(u.total_tokens)
                    .build();

                usage_callback(&usage);

                usage
            });

            // Still need to provide a choice even for usage events
            let choice = ChoiceBuilder::default()
                .delta(Some(Delta {
                    content: None,
                    role: None,
                    tool_calls: None,
                }))
                .index(0)
                .build();

            builder = builder.choice(choice).usage(usage);
        }
        _ => {
            // For any unhandled events, still provide a minimal choice
            let choice = ChoiceBuilder::default()
                .delta(Some(Delta {
                    content: None,
                    role: None,
                    tool_calls: None,
                }))
                .index(0)
                .build();

            builder = builder.choice(choice);
        }
    }

    builder
}
