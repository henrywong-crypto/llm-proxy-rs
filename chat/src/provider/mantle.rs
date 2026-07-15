use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{
    PayloadChecksumKind, SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use std::time::SystemTime;

/// Default model used when `config.toml` does not set `mantle_model`.
pub const DEFAULT_MANTLE_MODEL: &str = "openai.gpt-5.6-sol";

/// Overwrites the `model` field of an OpenAI request body with `model`. No-op if
/// the body is not a JSON object.
pub fn force_mantle_model(body: &mut serde_json::Value, model: &str) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
    }
}

/// URL on the `bedrock-runtime` OpenAI-compatible host, e.g.
/// `https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions`.
/// Used for models/APIs served by the `bedrock-runtime` endpoint.
pub fn mantle_url(region: &str, path: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com{path}")
}

/// URL on the dedicated `bedrock-mantle` host, e.g.
/// `https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses`. Some models
/// (e.g. `openai.gpt-5.6-sol`) are served only here, not on `bedrock-runtime`.
pub fn bedrock_mantle_url(region: &str, path: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws{path}")
}

/// SigV4-signs a JSON POST to the `bedrock` service and returns the headers to
/// attach to the outgoing request (authorization, x-amz-date, x-amz-content-sha256,
/// and x-amz-security-token when the credentials carry a session token).
///
/// The request is signed with a `content-type: application/json` header and the
/// exact `body` bytes, so the caller must send those unchanged.
pub async fn sign_bedrock_json_post(
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

/// Signs and forwards a JSON POST to a full Mantle `url`, returning the raw
/// upstream response so the caller can relay it transparently. Unlike the
/// chat-completions provider, this does not inspect the status or reframe the
/// body — it is a pass-through for APIs (such as the Responses API) whose
/// request and streaming formats should reach the client unchanged.
pub async fn forward_mantle_post(
    http_client: &reqwest::Client,
    credentials_provider: &SharedCredentialsProvider,
    region: &str,
    url: &str,
    body: Vec<u8>,
) -> anyhow::Result<reqwest::Response> {
    let signed_headers = sign_bedrock_json_post(credentials_provider, region, url, &body).await?;

    let mut req = http_client
        .post(url)
        .header("content-type", "application/json");
    for (name, value) in signed_headers {
        req = req.header(name, value);
    }

    Ok(req.body(body).send().await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_mantle_model_overrides_client_model() {
        let mut value = serde_json::json!({ "model": "gpt-4o-mini", "input": "hi" });
        force_mantle_model(&mut value, "openai.gpt-oss-120b");
        assert_eq!(value["model"], serde_json::json!("openai.gpt-oss-120b"));
        // Other fields are left untouched.
        assert_eq!(value["input"], serde_json::json!("hi"));
    }

    #[test]
    fn force_mantle_model_inserts_when_absent() {
        let mut value = serde_json::json!({ "input": "hi" });
        force_mantle_model(&mut value, DEFAULT_MANTLE_MODEL);
        assert_eq!(value["model"], serde_json::json!(DEFAULT_MANTLE_MODEL));
    }

    #[test]
    fn force_mantle_model_noop_on_non_object() {
        let mut value = serde_json::json!("not an object");
        force_mantle_model(&mut value, DEFAULT_MANTLE_MODEL);
        assert!(value.is_string());
    }

    #[test]
    fn mantle_url_builds_regional_host() {
        assert_eq!(
            mantle_url("us-east-1", "/openai/v1/responses"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
        );
    }
}

