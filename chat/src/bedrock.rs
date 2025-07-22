use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, ContentBlock, ConversationRole, Message, SpecificToolChoice,
    SystemContentBlock, Tool, ToolChoice, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolSpecification,
};
use aws_smithy_types::Document;
use request::{ChatCompletionsRequest, Contents, OpenAITool, OpenAIToolChoice, Role};
use serde_json::Value;

pub struct BedrockChatCompletion {
    pub model_id: String,
    pub system_content_blocks: Vec<SystemContentBlock>,
    pub messages: Vec<Message>,
    pub tool_config: Option<ToolConfiguration>,
}

pub fn process_chat_completions_request_to_bedrock_chat_completion(
    request: &ChatCompletionsRequest,
) -> Result<BedrockChatCompletion> {
    let mut system_content_blocks = Vec::new();
    let mut messages = Vec::new();
    let model_id = request.model.clone();

    for request_message in &request.messages {
        match request_message.role {
            Role::Assistant | Role::User => {
                if request_message.role == Role::Assistant && request_message.tool_calls.is_some() {
                    let mut content_blocks = Vec::new();

                    // Add text content if present
                    if let Some(contents) = &request_message.contents {
                        match contents {
                            Contents::String(text) if !text.is_empty() => {
                                content_blocks.push(ContentBlock::Text(text.clone()));
                            }
                            Contents::Array(blocks) => {
                                for block in blocks {
                                    let request::Content::Text { text } = block;
                                    if !text.is_empty() {
                                        content_blocks.push(ContentBlock::Text(text.clone()));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    // Add ToolUse blocks for each tool call
                    if let Some(tool_calls) = &request_message.tool_calls {
                        for tool_call in tool_calls {
                            let input_document = if tool_call.function.arguments.is_empty() {
                                Document::Object(std::collections::HashMap::new())
                            } else {
                                value_to_document(
                                    &serde_json::from_str(&tool_call.function.arguments).unwrap_or(
                                        serde_json::Value::Object(serde_json::Map::new()),
                                    ),
                                )
                            };

                            let tool_use = aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
                                .tool_use_id(&tool_call.id)
                                .name(&tool_call.function.name)
                                .input(input_document)
                                .build()
                                .map_err(|e| {
                                    anyhow::anyhow!("Failed to build ToolUse block: {e}")
                                })?;

                            content_blocks.push(ContentBlock::ToolUse(tool_use));
                        }
                    }

                    let message = aws_sdk_bedrockruntime::types::Message::builder()
                        .role(ConversationRole::Assistant)
                        .set_content(Some(content_blocks))
                        .build()
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "Failed to build assistant message with tool calls: {e}"
                            )
                        })?;

                    messages.push(message);
                } else {
                    messages.push(Message::try_from(request_message)?);
                }
            }
            Role::System => {
                if let Some(contents) = &request_message.contents {
                    let new_system_content_blocks: Vec<SystemContentBlock> = contents.into();
                    system_content_blocks.extend(new_system_content_blocks);
                }
            }
            Role::Tool => {
                if let (Some(contents), Some(tool_call_id)) =
                    (&request_message.contents, &request_message.tool_call_id)
                {
                    let result_text = match contents {
                        Contents::String(result) => result.clone(),
                        Contents::Array(content_blocks) => content_blocks
                            .iter()
                            .map(|block| match block {
                                request::Content::Text { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join(" "),
                    };

                    let tool_result = ToolResultBlock::builder()
                        .tool_use_id(tool_call_id.clone())
                        .content(ToolResultContentBlock::Text(result_text))
                        .build()
                        .map_err(|e| anyhow::anyhow!("Failed to build tool result block: {e}"))?;

                    let tool_result_message = aws_sdk_bedrockruntime::types::Message::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::ToolResult(tool_result))
                        .build()
                        .map_err(|e| anyhow::anyhow!("Failed to build tool result message: {e}"))?;

                    messages.push(tool_result_message);
                }
            }
        }
    }

    let tool_config = request
        .tools
        .as_ref()
        .map(|tools| openai_tools_to_bedrock_tool_config(tools, &request.tool_choice))
        .transpose()?;

    Ok(BedrockChatCompletion {
        model_id,
        system_content_blocks,
        messages,
        tool_config,
    })
}

fn openai_tools_to_bedrock_tool_config(
    openai_tools: &[OpenAITool],
    openai_tool_choice: &Option<OpenAIToolChoice>,
) -> Result<ToolConfiguration> {
    let mut builder = ToolConfiguration::builder();

    for openai_tool in openai_tools {
        let tool_spec = ToolSpecification::builder()
            .name(&openai_tool.function.name)
            .set_description(openai_tool.function.description.clone())
            .input_schema(ToolInputSchema::Json(value_to_document(
                &openai_tool.function.parameters,
            )))
            .build()?;

        builder = builder.tools(Tool::ToolSpec(tool_spec));
    }

    if let Some(openai_tool_choice) = openai_tool_choice {
        let bedrock_tool_choice = match openai_tool_choice {
            OpenAIToolChoice::String(s) => match s.as_str() {
                "none" => None,
                "required" => Some(ToolChoice::Any(AnyToolChoice::builder().build())),
                _ => Some(ToolChoice::Auto(AutoToolChoice::builder().build())),
            },
            OpenAIToolChoice::Object { function, .. } => Some(ToolChoice::Tool(
                SpecificToolChoice::builder().name(&function.name).build()?,
            )),
        };
        builder = builder.set_tool_choice(bedrock_tool_choice);
    } else {
        builder = builder.tool_choice(ToolChoice::Auto(AutoToolChoice::builder().build()));
    }

    Ok(builder.build()?)
}

fn value_to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Document::Number(if i >= 0 {
                    aws_smithy_types::Number::PosInt(i as u64)
                } else {
                    aws_smithy_types::Number::NegInt(i)
                })
            } else {
                Document::Number(aws_smithy_types::Number::Float(n.as_f64().unwrap_or(0.0)))
            }
        }
        Value::String(s) => Document::String(s.clone()),
        Value::Array(a) => Document::Array(a.iter().map(value_to_document).collect()),
        Value::Object(o) => Document::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), value_to_document(v)))
                .collect(),
        ),
    }
}
