use anyhow::{Context, Result, bail};
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, CachePointBlock, SpecificToolChoice, Tool as BedrockTool,
    ToolChoice, ToolConfiguration, ToolInputSchema, ToolSpecification,
};
use common::{ToolNameMap, alias_tool_name, value_to_document};
use serde::{Deserialize, Serialize};

use crate::cache_control::CacheControl;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Tool {
    Custom {
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        input_schema: serde_json::Value,
        name: String,
    },
    Server(serde_json::Value),
}

impl TryFrom<&Tool> for Option<Vec<BedrockTool>> {
    type Error = anyhow::Error;

    fn try_from(tool: &Tool) -> Result<Self, Self::Error> {
        match tool {
            Tool::Custom {
                cache_control,
                description,
                input_schema,
                name,
            } => {
                let mut tools = vec![BedrockTool::ToolSpec(
                    ToolSpecification::builder()
                        .name(alias_tool_name(name))
                        .set_description(
                            description
                                .as_deref()
                                .filter(|d| !d.is_empty())
                                .map(str::to_owned),
                        )
                        .input_schema(ToolInputSchema::Json(value_to_document(input_schema)))
                        .build()?,
                )];

                if let Some(cc) = cache_control {
                    tools.push(BedrockTool::CachePoint(CachePointBlock::try_from(cc)?));
                }

                Ok(Some(tools))
            }
            Tool::Server(_) => Ok(None),
        }
    }
}

pub fn build_bedrock_tools(tools: &[Tool]) -> anyhow::Result<Option<Vec<BedrockTool>>> {
    let bedrock_tools: Vec<BedrockTool> = tools
        .iter()
        .map(Option::<Vec<_>>::try_from)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();

    Ok(if bedrock_tools.is_empty() {
        None
    } else {
        Some(bedrock_tools)
    })
}

pub fn tool_choice_from_value(value: &serde_json::Value) -> Result<Option<ToolChoice>> {
    match value.get("type").and_then(|t| t.as_str()) {
        Some("none") | None => Ok(None),
        Some("auto") => Ok(Some(ToolChoice::Auto(AutoToolChoice::builder().build()))),
        Some("any") => Ok(Some(ToolChoice::Any(AnyToolChoice::builder().build()))),
        Some("tool") => {
            let name = value
                .get("name")
                .and_then(|n| n.as_str())
                .context("tool_choice type 'tool' requires a 'name' field")?;
            Ok(Some(ToolChoice::Tool(
                SpecificToolChoice::builder()
                    .name(alias_tool_name(name))
                    .build()?,
            )))
        }
        Some(other) => bail!("Unsupported tool_choice type: {other}"),
    }
}

pub fn build_tool_configuration(
    tools: &[Tool],
    tool_choice: Option<&serde_json::Value>,
) -> Result<Option<ToolConfiguration>> {
    let bedrock_tools = build_bedrock_tools(tools)?;

    bedrock_tools
        .map(|tools| {
            let choice = tool_choice
                .map(tool_choice_from_value)
                .transpose()?
                .flatten();
            ToolConfiguration::builder()
                .set_tools(Some(tools))
                .set_tool_choice(choice)
                .build()
                .map_err(anyhow::Error::from)
        })
        .transpose()
}

/// Builds the reverse map used to restore Bedrock tool-name aliases on the
/// response. Only custom tools carry a name we alias; server tools pass through
/// Bedrock untouched.
pub fn build_tool_name_map(tools: &[Tool]) -> ToolNameMap {
    ToolNameMap::from_originals(tools.iter().filter_map(|tool| match tool {
        Tool::Custom { name, .. } => Some(name.clone()),
        Tool::Server(_) => None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::BEDROCK_TOOL_NAME_MAX_LEN;

    #[test]
    fn tool_to_bedrock_tools_without_cache() {
        let tool = Tool::Custom {
            cache_control: None,
            description: Some("Gets the weather".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
            name: "get_weather".to_string(),
        };
        let bedrock_tools = Option::<Vec<BedrockTool>>::try_from(&tool)
            .unwrap()
            .unwrap();
        assert_eq!(bedrock_tools.len(), 1);
        match &bedrock_tools[0] {
            BedrockTool::ToolSpec(spec) => {
                assert_eq!(spec.name(), "get_weather");
                assert_eq!(spec.description(), Some("Gets the weather"));
            }
            other => panic!("expected ToolSpec, got {:?}", other),
        }
    }

    #[test]
    fn tool_to_bedrock_tools_with_cache() {
        let tool = Tool::Custom {
            cache_control: Some(CacheControl {
                cache_control_type: "ephemeral".to_string(),
                ttl: None,
            }),
            description: Some("Gets the weather".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
            name: "get_weather".to_string(),
        };
        let bedrock_tools = Option::<Vec<BedrockTool>>::try_from(&tool)
            .unwrap()
            .unwrap();
        assert_eq!(bedrock_tools.len(), 2);
        match &bedrock_tools[0] {
            BedrockTool::ToolSpec(spec) => assert_eq!(spec.name(), "get_weather"),
            other => panic!("expected ToolSpec, got {:?}", other),
        }
        assert!(matches!(bedrock_tools[1], BedrockTool::CachePoint(_)));
    }

    #[test]
    fn tool_with_empty_description_is_none() {
        let tool = Tool::Custom {
            cache_control: None,
            description: Some("".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
            name: "get_weather".to_string(),
        };
        let bedrock_tools = Option::<Vec<BedrockTool>>::try_from(&tool)
            .unwrap()
            .unwrap();
        match &bedrock_tools[0] {
            BedrockTool::ToolSpec(spec) => assert!(spec.description().is_none()),
            other => panic!("expected ToolSpec, got {:?}", other),
        }
    }

    #[test]
    fn build_bedrock_tools_empty_returns_none() {
        let tools: Vec<Tool> = vec![];
        let result = build_bedrock_tools(&tools).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn build_tool_configuration_with_cache() {
        let tools = vec![Tool::Custom {
            cache_control: Some(CacheControl {
                cache_control_type: "ephemeral".to_string(),
                ttl: None,
            }),
            description: Some("A tool".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
            name: "my_tool".to_string(),
        }];
        let config = build_tool_configuration(&tools, None).unwrap().unwrap();
        assert_eq!(config.tools().len(), 2);
        match &config.tools()[0] {
            BedrockTool::ToolSpec(spec) => assert_eq!(spec.name(), "my_tool"),
            other => panic!("expected ToolSpec, got {:?}", other),
        }
        assert!(matches!(config.tools()[1], BedrockTool::CachePoint(_)));
    }

    #[test]
    fn tool_choice_tool_with_name() {
        let value = serde_json::json!({"type": "tool", "name": "get_weather"});
        let choice = tool_choice_from_value(&value).unwrap().unwrap();
        match choice {
            ToolChoice::Tool(specific) => assert_eq!(specific.name(), "get_weather"),
            other => panic!("expected ToolChoice::Tool, got {:?}", other),
        }
    }

    #[test]
    fn tool_choice_tool_without_name_errors() {
        let value = serde_json::json!({"type": "tool"});
        let result = tool_choice_from_value(&value);
        assert!(result.is_err());
    }

    /// A name Anthropic accepts (≤128) but Bedrock rejects (>64).
    fn over_limit_name() -> String {
        let name = format!("search_knowledge_base_{}", "x".repeat(60));
        assert!(name.len() > BEDROCK_TOOL_NAME_MAX_LEN);
        name
    }

    #[test]
    fn long_tool_name_is_aliased_within_bedrock_limit() {
        let name = over_limit_name();
        let tool = Tool::Custom {
            cache_control: None,
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
            name: name.clone(),
        };
        let bedrock_tools = Option::<Vec<BedrockTool>>::try_from(&tool)
            .unwrap()
            .unwrap();
        match &bedrock_tools[0] {
            BedrockTool::ToolSpec(spec) => {
                assert_eq!(spec.name(), alias_tool_name(&name));
                assert!(spec.name().len() <= BEDROCK_TOOL_NAME_MAX_LEN);
                assert_ne!(spec.name(), name);
            }
            other => panic!("expected ToolSpec, got {other:?}"),
        }
    }

    #[test]
    fn long_tool_choice_name_is_aliased_to_match_the_spec() {
        let name = over_limit_name();
        let value = serde_json::json!({"type": "tool", "name": name});
        match tool_choice_from_value(&value).unwrap().unwrap() {
            ToolChoice::Tool(specific) => assert_eq!(specific.name(), alias_tool_name(&name)),
            other => panic!("expected ToolChoice::Tool, got {other:?}"),
        }
    }

    #[test]
    fn build_tool_name_map_restores_aliased_custom_tool() {
        let name = over_limit_name();
        let tools = vec![
            Tool::Custom {
                cache_control: None,
                description: None,
                input_schema: serde_json::json!({"type": "object"}),
                name: name.clone(),
            },
            Tool::Server(serde_json::json!({"type": "web_search"})),
        ];
        let map = build_tool_name_map(&tools);
        assert_eq!(map.restore(&alias_tool_name(&name)), name);
    }
}
