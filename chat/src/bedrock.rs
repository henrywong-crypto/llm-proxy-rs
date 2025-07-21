use anyhow::Result;
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, Message, SpecificToolChoice, SystemContentBlock, Tool,
    ToolChoice, ToolConfiguration, ToolInputSchema, ToolSpecification,
};
use request::{ChatCompletionsRequest, OpenAITool, OpenAIToolChoice, Role, value_to_document};

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

    for message in &request.messages {
        match message.role {
            Role::Assistant | Role::User | Role::Tool => {
                messages.push(Message::try_from(message)?);
            }
            Role::System => {
                if let Some(contents) = &message.contents {
                    let new_system_content_blocks: Vec<SystemContentBlock> = contents.into();
                    system_content_blocks.extend(new_system_content_blocks);
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

// Convert OpenAI tool to Bedrock Tool
fn openai_tool_to_bedrock_tool(openai_tool: &OpenAITool) -> Result<Tool> {
    let tool_spec = ToolSpecification::builder()
        .name(&openai_tool.function.name)
        .set_description(openai_tool.function.description.clone())
        .input_schema(ToolInputSchema::Json(value_to_document(
            &openai_tool.function.parameters,
        )))
        .build()?;

    Ok(Tool::ToolSpec(tool_spec))
}

// Convert OpenAI tool choice to Bedrock ToolChoice
fn openai_tool_choice_to_bedrock_tool_choice(
    openai_tool_choice: &OpenAIToolChoice,
) -> Result<Option<ToolChoice>> {
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
    Ok(bedrock_tool_choice)
}

fn openai_tools_to_bedrock_tool_config(
    openai_tools: &[OpenAITool],
    openai_tool_choice: &Option<OpenAIToolChoice>,
) -> Result<ToolConfiguration> {
    let mut builder = ToolConfiguration::builder();

    for openai_tool in openai_tools {
        let tool = openai_tool_to_bedrock_tool(openai_tool)?;
        builder = builder.tools(tool);
    }

    if let Some(openai_tool_choice) = openai_tool_choice {
        let bedrock_tool_choice = openai_tool_choice_to_bedrock_tool_choice(openai_tool_choice)?;
        builder = builder.set_tool_choice(bedrock_tool_choice);
    } else {
        builder = builder.tool_choice(ToolChoice::Auto(AutoToolChoice::builder().build()));
    }

    Ok(builder.build()?)
}
