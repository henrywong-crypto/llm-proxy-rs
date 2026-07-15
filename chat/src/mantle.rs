use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{
    PayloadChecksumKind, SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use std::time::SystemTime;

pub(crate) fn bedrock_runtime_openai_url(region: &str, path: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com{path}")
}

pub(crate) fn bedrock_mantle_url(region: &str, path: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws{path}")
}

pub(crate) async fn sign_bedrock_json_post(
    credentials_provider: &SharedCredentialsProvider,
    region: &str,
    url: &str,
    body: &[u8],
) -> anyhow::Result<Vec<(String, String)>> {
    let credentials = credentials_provider
        .provide_credentials()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to resolve AWS credentials: {}", e))?;
    let identity: Identity = credentials.into();

    let mut signing_settings = SigningSettings::default();
    signing_settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;

    let signing_params: SigningParams = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name("bedrock")
        .time(SystemTime::now())
        .settings(signing_settings)
        .build()?
        .into();

    let signable_request = SignableRequest::new(
        "POST",
        url,
        std::iter::once(("content-type", "application/json")),
        SignableBody::Bytes(body),
    )?;

    let (instructions, _signature) = sign(signable_request, &signing_params)?.into_parts();

    Ok(instructions
        .headers()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect())
}

pub(crate) async fn forward_mantle_post(
    http_client: &reqwest::Client,
    credentials_provider: &SharedCredentialsProvider,
    region: &str,
    url: &str,
    body: Vec<u8>,
) -> anyhow::Result<reqwest::Response> {
    let signed_headers = sign_bedrock_json_post(credentials_provider, region, url, &body).await?;

    let mut request = http_client
        .post(url)
        .header("content-type", "application/json");
    for (name, value) in signed_headers {
        request = request.header(name, value);
    }

    Ok(request.body(body).send().await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bedrock_runtime_openai_url_builds_regional_host() {
        assert_eq!(
            bedrock_runtime_openai_url("us-east-1", "/openai/v1/chat/completions"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn bedrock_mantle_url_builds_regional_host() {
        assert_eq!(
            bedrock_mantle_url("us-east-1", "/openai/v1/responses"),
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
        );
    }
}
