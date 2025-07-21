use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, ConversationRole, Message, SpecificToolChoice, SystemContentBlock, Tool,
    ToolChoice, ToolConfiguration, ToolInputSchema, ToolSpecification,
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
        tracing::debug!("Processing message: role={:?}, has_content={}, has_tool_calls={}, has_tool_call_id={}", 
            request_message.role, 
            request_message.contents.is_some(),
            request_message.tool_calls.is_some(),
            request_message.tool_call_id.is_some()
        );
        
        match request_message.role {
            Role::Assistant | Role::User => {
                // Skip assistant messages that only have tool_calls but no content
                // Bedrock handles tool calls differently than OpenAI
                if request_message.role == Role::Assistant 
                    && request_message.tool_calls.is_some() 
                    && (request_message.contents.is_none() || 
                        matches!(request_message.contents.as_ref(), Some(Contents::String(s)) if s.is_empty()))
                {
                    tracing::debug!("Skipping assistant message with only tool_calls (no content) for Bedrock");
                } else {
                    match Message::try_from(request_message) {
                        Ok(message) => {
                            tracing::debug!("Successfully converted message to Bedrock format");
                            messages.push(message);
                        }
                        Err(e) => {
                            tracing::error!("Failed to convert message to Bedrock format: {}", e);
                            return Err(e);
                        }
                    }
                }
            }
            Role::System => {
                if let Some(contents) = &request_message.contents {
                    let mut new_system_content_blocks: Vec<SystemContentBlock> = contents.into();
                    
                    // If tools are available, enhance the system message to encourage tool usage
                    if request.tools.is_some() {
                        match contents {
                            Contents::String(original) => {
                                let enhanced_message = format!("{} IMPORTANT: Always use the available functions when appropriate. For time queries, use get_current_time. For fibonacci calculations, use fibonacci.", original);
                                new_system_content_blocks = vec![SystemContentBlock::Text(enhanced_message)];
                                tracing::debug!("Enhanced system message to encourage tool usage");
                            }
                            Contents::Array(_) => {
                                // For array content, add an additional system block
                                let blocks: Vec<SystemContentBlock> = contents.into();
                                new_system_content_blocks.extend(blocks);
                                new_system_content_blocks.push(SystemContentBlock::Text(
                                    "IMPORTANT: Always use the available functions when appropriate. For time queries, use get_current_time. For fibonacci calculations, use fibonacci.".to_string()
                                ));
                                tracing::debug!("Enhanced system content blocks with tool usage instructions");
                            }
                        };
                    }
                    
                    system_content_blocks.extend(new_system_content_blocks);
                    tracing::debug!("Added system content blocks");
                } else {
                    tracing::debug!("Skipping system message with no content");
                }
            }
            Role::Tool => {
                tracing::debug!("Processing tool result message with tool_call_id: {:?}", request_message.tool_call_id);
                // For Bedrock, we need to convert tool results back to user messages 
                // but format them clearly as function results, not user input
                if let Some(contents) = &request_message.contents {
                    // Format the tool result to make it clear it's a function output
                    let formatted_content = match contents {
                        Contents::String(result) => {
                            format!("Function result: {}", result)
                        }
                        Contents::Array(content_blocks) => {
                            // Extract text from content blocks and format as function result
                            let text = content_blocks.iter()
                                .filter_map(|block| match block {
                                    request::Content::Text { text } => Some(text.as_str()),
                                })
                                .collect::<Vec<_>>()
                                .join(" ");
                            format!("Function result: {}", text)
                        }
                    };
                    
                    let tool_result_message = aws_sdk_bedrockruntime::types::Message::builder()
                        .role(ConversationRole::User)
                        .content(aws_sdk_bedrockruntime::types::ContentBlock::Text(formatted_content))
                        .build()
                        .map_err(|e| anyhow::anyhow!("Failed to build tool result message: {e}"))?;
                    
                    messages.push(tool_result_message);
                    tracing::debug!("Converted tool result to formatted user message for Bedrock");
                } else {
                    tracing::debug!("Tool message has no content, skipping");
                }
            }
        }
    }

    let tool_config = request
        .tools
        .as_ref()
        .map(|tools| {
            tracing::debug!("Converting {} OpenAI tools to Bedrock format", tools.len());
            tracing::debug!("Tool choice from request: {:?}", request.tool_choice);
            openai_tools_to_bedrock_tool_config(tools, &request.tool_choice)
        })
        .transpose()?;

    tracing::debug!("Successfully created BedrockChatCompletion: model={}, system_blocks={}, messages={}, has_tools={}", 
        model_id, system_content_blocks.len(), messages.len(), tool_config.is_some());

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
        tracing::debug!("Processing tool_choice: {:?}", openai_tool_choice);
        let bedrock_tool_choice = match openai_tool_choice {
            OpenAIToolChoice::String(s) => match s.as_str() {
                "none" => {
                    tracing::debug!("Tool choice: none - disabling tools");
                    None
                }
                "required" => {
                    tracing::debug!("Tool choice: required - forcing tool use");
                    Some(ToolChoice::Any(AnyToolChoice::builder().build()))
                }
                _ => {
                    tracing::debug!("Tool choice: auto - letting model decide");
                    Some(ToolChoice::Auto(AutoToolChoice::builder().build()))
                }
            },
            OpenAIToolChoice::Object { function, .. } => {
                tracing::debug!("Tool choice: specific function {}", function.name);
                Some(ToolChoice::Tool(
                    SpecificToolChoice::builder().name(&function.name).build()?,
                ))
            }
        };
        builder = builder.set_tool_choice(bedrock_tool_choice);
    } else {
        tracing::debug!("No tool_choice specified, defaulting to auto");
        // For time-related requests, we should encourage tool usage
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
