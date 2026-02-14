use anyhow::{anyhow, Result};
use aws_sdk_bedrockruntime::types::{
    JsonSchemaDefinition, OutputConfig as BedrockOutputConfig, OutputFormat as BedrockOutputFormat,
    OutputFormatStructure, OutputFormatType,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OutputFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub schema: serde_json::Value,
}

impl TryFrom<&OutputConfig> for BedrockOutputConfig {
    type Error = anyhow::Error;

    fn try_from(config: &OutputConfig) -> Result<Self, Self::Error> {
        // Validate format type
        if config.format.format_type != "json_schema" {
            return Err(anyhow!(
                "Unsupported output format type: {}",
                config.format.format_type
            ));
        }

        // Convert the schema to a JSON string
        let schema_str = serde_json::to_string(&config.format.schema)
            .map_err(|e| anyhow!("Failed to serialize schema: {}", e))?;

        // Build the JsonSchemaDefinition
        let json_schema = JsonSchemaDefinition::builder()
            .schema(schema_str)
            .build()
            .map_err(|e| anyhow!("Failed to build JsonSchemaDefinition: {}", e))?;

        // Build the OutputFormatStructure
        let structure = OutputFormatStructure::JsonSchema(json_schema);

        // Build the BedrockOutputFormat
        let format = BedrockOutputFormat::builder()
            .r#type(OutputFormatType::JsonSchema)
            .structure(structure)
            .build()
            .map_err(|e| anyhow!("Failed to build OutputFormat: {}", e))?;

        // Build the BedrockOutputConfig
        let output_config = BedrockOutputConfig::builder()
            .text_format(format)
            .build();

        Ok(output_config)
    }
}
