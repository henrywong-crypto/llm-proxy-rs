use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{
    PayloadChecksumKind, SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use std::time::SystemTime;

/// Base host for Bedrock's OpenAI-compatible ("Mantle") operations. The region
/// is interpolated to form e.g. `bedrock-runtime.us-east-1.amazonaws.com`.
pub fn mantle_url(region: &str, path: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com{path}")
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

/// Signs and forwards a JSON POST to a Mantle path (e.g. `/openai/v1/responses`),
/// returning the raw upstream response so the caller can relay it transparently.
/// Unlike the chat-completions provider, this does not inspect the status or
/// reframe the body — it is a pass-through for APIs (such as the Responses API)
/// whose request and streaming formats should reach the client unchanged.
pub async fn forward_mantle_post(
    http_client: &reqwest::Client,
    credentials_provider: &SharedCredentialsProvider,
    region: &str,
    path: &str,
    body: Vec<u8>,
) -> anyhow::Result<reqwest::Response> {
    let url = mantle_url(region, path);
    let signed_headers = sign_bedrock_json_post(credentials_provider, region, &url, &body).await?;

    let mut req = http_client
        .post(&url)
        .header("content-type", "application/json");
    for (name, value) in signed_headers {
        req = req.header(name, value);
    }

    Ok(req.body(body).send().await?)
}
