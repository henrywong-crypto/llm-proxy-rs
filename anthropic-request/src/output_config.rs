use anyhow::{bail, Result};
use aws_sdk_bedrockruntime::types::{
    JsonSchemaDefinition, OutputConfig as BedrockOutputConfig, OutputFormat as BedrockOutputFormat,
    OutputFormatStructure, OutputFormatType,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OutputConfig {
    Format { format: OutputConfigFormat },
    Effort { effort: String },
    Other(serde_json::Value),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OutputConfigFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub schema: serde_json::Value,
}

impl TryFrom<&OutputConfigFormat> for BedrockOutputConfig {
    type Error = anyhow::Error;

    fn try_from(format: &OutputConfigFormat) -> Result<Self, Self::Error> {
        if format.format_type != "json_schema" {
            bail!("Unsupported output format type: {}", format.format_type);
        }

        let schema_str = serde_json::to_string(&format.schema)?;

        let json_schema = JsonSchemaDefinition::builder()
            .schema(schema_str)
            .build()?;

        let bedrock_format = BedrockOutputFormat::builder()
            .r#type(OutputFormatType::JsonSchema)
            .structure(OutputFormatStructure::JsonSchema(json_schema))
            .build()?;

        Ok(BedrockOutputConfig::builder()
            .text_format(bedrock_format)
            .build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_output_config_deserializes_as_other() {
        let json = serde_json::json!({"foo": "bar"});
        let config: OutputConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(config, OutputConfig::Other(_)));
    }
}
