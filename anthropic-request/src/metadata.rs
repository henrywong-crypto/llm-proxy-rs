use serde::{Deserialize, Serialize};

/// Metadata for the request
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}
