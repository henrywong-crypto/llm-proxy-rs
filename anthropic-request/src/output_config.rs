use anyhow::{anyhow, Result};
use aws_sdk_bedrockruntime::types::{
    JsonSchemaDefinition, OutputConfig as BedrockOutputConfig, OutputFormat as BedrockOutputFormat,
    OutputFormatStructure, OutputFormatType,
};
use aws_smithy_types::Document;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OutputConfig {
    // Format with JSON schema: { "format": { "type": "json_schema", "schema": {...} } }
    WithFormat { format: OutputFormat },
    // Format with effort: { "effort": "medium" }
    WithEffort { effort: String },
    // Catch-all for any other format - silently accept but ignore
    Other(serde_json::Value),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OutputFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub schema: serde_json::Value,
}

impl OutputConfig {
    /// Extract effort value if present
    pub fn effort(&self) -> Option<&str> {
        match self {
            OutputConfig::WithEffort { effort } => Some(effort.as_str()),
            _ => None,
        }
    }

    /// Check if this config contains a format (JSON schema)
    pub fn has_format(&self) -> bool {
        matches!(self, OutputConfig::WithFormat { .. })
    }

    /// Convert output_config to additional model request fields Document
    /// This handles effort and other output_config variants that need to be passed to Bedrock
    pub fn to_additional_model_request_fields(&self) -> Option<Document> {
        match self {
            OutputConfig::WithEffort { effort } => {
                // For effort, we need to pass both output_config and anthropic_beta
                let mut fields = std::collections::HashMap::new();
                
                // TESTING: Send invalid beta header to see if Bedrock validates it
                // TODO: Remove this test and use actual values
                let test_beta = "INVALID_BETA_HEADER_TEST";
                
                // Add output_config with effort (use actual value for now)
                fields.insert(
                    "output_config".to_string(),
                    Document::Object(
                        [("effort".to_string(), Document::String(effort.clone()))]
                            .into_iter()
                            .collect(),
                    ),
                );
                
                // Add anthropic_beta header for effort feature (TESTING with invalid value)
                fields.insert(
                    "anthropic_beta".to_string(),
                    Document::Array(vec![Document::String(test_beta.to_string())]),
                );
                
                Some(Document::Object(fields))
            }
            OutputConfig::Other(_) => {
                // Silently ignore unknown output_config formats
                None
            }
            _ => None,
        }
    }
}

impl TryFrom<&OutputConfig> for BedrockOutputConfig {
    type Error = anyhow::Error;

    fn try_from(config: &OutputConfig) -> Result<Self, Self::Error> {
        match config {
            OutputConfig::WithFormat { format } => {
                // Validate format type
                if format.format_type != "json_schema" {
                    return Err(anyhow!(
                        "Unsupported output format type: {}",
                        format.format_type
                    ));
                }

                // Convert the schema to a JSON string
                let schema_str = serde_json::to_string(&format.schema)
                    .map_err(|e| anyhow!("Failed to serialize schema: {}", e))?;

                // Build the JsonSchemaDefinition
                let json_schema = JsonSchemaDefinition::builder()
                    .schema(schema_str)
                    .build()
                    .map_err(|e| anyhow!("Failed to build JsonSchemaDefinition: {}", e))?;

                // Build the OutputFormatStructure
                let structure = OutputFormatStructure::JsonSchema(json_schema);

                // Build the BedrockOutputFormat
                let bedrock_format = BedrockOutputFormat::builder()
                    .r#type(OutputFormatType::JsonSchema)
                    .structure(structure)
                    .build()
                    .map_err(|e| anyhow!("Failed to build OutputFormat: {}", e))?;

                // Build the BedrockOutputConfig
                let output_config = BedrockOutputConfig::builder()
                    .text_format(bedrock_format)
                    .build();

                Ok(output_config)
            }
            OutputConfig::WithEffort { .. } | OutputConfig::Other(_) => {
                // These fields are not part of Bedrock's OutputConfig
                // They should be passed via additionalModelRequestFields instead
                Err(anyhow!(
                    "output_config variant should not be converted to Bedrock OutputConfig"
                ))
            }
        }
    }
}
