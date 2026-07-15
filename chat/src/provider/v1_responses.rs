use aws_credential_types::provider::SharedCredentialsProvider;

use crate::mantle::{bedrock_mantle_url, forward_mantle_post};

pub struct V1ResponsesProvider {
    http_client: reqwest::Client,
    region: String,
    credentials_provider: SharedCredentialsProvider,
}

impl V1ResponsesProvider {
    pub fn new(
        http_client: reqwest::Client,
        region: String,
        credentials_provider: SharedCredentialsProvider,
    ) -> Self {
        Self {
            http_client,
            region,
            credentials_provider,
        }
    }

    pub async fn v1_responses_stream(&self, body: Vec<u8>) -> anyhow::Result<reqwest::Response> {
        let url = bedrock_mantle_url(&self.region, "/openai/v1/responses");
        forward_mantle_post(
            &self.http_client,
            &self.credentials_provider,
            &self.region,
            &url,
            body,
        )
        .await
    }
}
