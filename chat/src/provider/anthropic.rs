use anthropic_request::{
    AssistantContent, AssistantContents, Message, Messages, UserContent, UserContents,
    V1MessagesCountTokensRequest, V1MessagesRequest, build_tool_configuration,
    get_additional_model_request_fields,
};
use anthropic_response::{ErrorEvent, ErrorType, EventConverter};
use async_trait::async_trait;
use aws_sdk_bedrockruntime::{
    Client,
    error::SdkError,
    operation::converse_stream::{
        ConverseStreamError, ConverseStreamOutput as ConverseStreamResponse,
    },
    primitives::event_stream::EventReceiver,
    types::{
        ContentBlock, ConverseStreamOutput, ConverseTokensRequest, CountTokensInput,
        Message as BedrockMessage, SystemContentBlock, TokenUsage,
        error::ConverseStreamOutputError,
    },
};
use aws_smithy_types::{Document, error::metadata::ProvideErrorMetadata, event_stream::RawMessage};
use axum::response::sse::Event;
use futures::stream::{BoxStream, StreamExt};
use std::{future::Future, pin::Pin, sync::Arc, time::Duration};
use tokio::{
    sync::mpsc,
    time::{Instant, interval_at, timeout},
};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};
use uuid::Uuid;

use crate::bedrock::BedrockChatCompletion;

const PING_INTERVAL: Duration = Duration::from_secs(20);
const EVENT_TX_SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// Synchronous window before falling to SSE + pings. Bedrock returns
/// validation/throttle/access errors in <200ms; this catches them so they
/// flow through `AppError` as proper HTTP 4xx with the upstream status code.
const CONNECT_ERROR_WINDOW: Duration = Duration::from_secs(1);

type SendFut = Pin<
    Box<dyn Future<Output = Result<ConverseStreamResponse, SdkError<ConverseStreamError>>> + Send>,
>;

enum ConnectOutcome {
    Ok(ConverseStreamResponse),
    Err(SdkError<ConverseStreamError>),
    /// Window expired with the connect still pending — caller commits to
    /// a 200 SSE response and pings while finishing the connect.
    Pending(SendFut),
}

/// Race a Bedrock connect against `window`. Resolved within the window
/// becomes `Ok`/`Err`; otherwise the still-pending future is handed back.
async fn race_connect(mut send_fut: SendFut, window: Duration) -> ConnectOutcome {
    match timeout(window, &mut send_fut).await {
        Ok(Ok(r)) => ConnectOutcome::Ok(r),
        Ok(Err(e)) => ConnectOutcome::Err(e),
        Err(_) => ConnectOutcome::Pending(send_fut),
    }
}

/// Sends a ping SSE event. Returns false if the consumer is gone or stuck.
async fn send_ping(event_tx: &mpsc::Sender<anyhow::Result<Event>>) -> bool {
    info!("Sending ping event");
    let ping_event = Ok(Event::default().event("ping").data(r#"{"type": "ping"}"#));
    match timeout(EVENT_TX_SEND_TIMEOUT, event_tx.send(ping_event)).await {
        Ok(Ok(())) => true,
        Ok(Err(_)) => {
            info!("SSE client disconnected, stopping Bedrock stream");
            false
        }
        Err(_) => {
            error!("Channel send timed out, consumer likely stuck");
            false
        }
    }
}

async fn process_bedrock_stream_events(
    mut stream: EventReceiver<ConverseStreamOutput, ConverseStreamOutputError>,
    model: String,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
    event_tx: mpsc::Sender<anyhow::Result<Event>>,
    mut ping_interval: tokio::time::Interval,
) {
    let id = format!("msg_{}", Uuid::new_v4());
    let mut event_converter = EventConverter::new(id, model, usage_callback);
    loop {
        tokio::select! {
            biased;
            result = stream.recv() => {
                match result {
                    Ok(Some(output)) => {
                        if let Some(events) = event_converter.convert(&output) {
                            for (event_name, event) in events {
                                let sse_event = match serde_json::to_string(&event) {
                                    Ok(json) => Event::default().event(event_name).data(json),
                                    Err(e) => {
                                        error!("Failed to serialize event: {}", e);
                                        send_error_event(
                                            &event_tx,
                                            ErrorEvent::new(
                                                ErrorType::ApiError,
                                                format!("Failed to serialize event: {e}"),
                                            ),
                                        )
                                        .await;
                                        return;
                                    }
                                };
                                match timeout(EVENT_TX_SEND_TIMEOUT, event_tx.send(Ok(sse_event)))
                                    .await
                                {
                                    Ok(Ok(())) => {}
                                    Ok(Err(_)) => {
                                        info!("SSE client disconnected, stopping Bedrock stream");
                                        return;
                                    }
                                    Err(_) => {
                                        error!("Channel send timed out, consumer likely stuck");
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        send_error_event(&event_tx, map_stream_output_error(&e)).await;
                        break;
                    }
                }
            }
            _ = ping_interval.tick() => {
                if !send_ping(&event_tx).await {
                    return;
                }
            }
        }
    }
    info!("Bedrock stream finished");
}

#[async_trait]
pub trait V1MessagesProvider {
    async fn v1_messages_stream<F>(
        self,
        request: V1MessagesRequest,
        response_model_id: Option<String>,
        anthropic_beta: Option<Vec<String>>,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&TokenUsage) + Send + Sync + 'static;

    async fn v1_messages_count_tokens(
        &self,
        request: &V1MessagesCountTokensRequest,
        inference_profile_prefixes: &[String],
    ) -> anyhow::Result<i32>;
}

fn log_v1_messages_request(request: &V1MessagesRequest) {
    match &request.messages {
        Messages::String(s) => {
            info!(
                "V1 Messages Request: single string message, len={}",
                s.len()
            );
        }
        Messages::Array(messages) => {
            for (i, message) in messages.iter().enumerate() {
                match message {
                    Message::User { content } => {
                        let user_content_types = match content {
                            UserContents::String(s) => format!("String(len={})", s.len()),
                            UserContents::Array(arr) => arr
                                .iter()
                                .map(|c| match c {
                                    UserContent::Document { .. } => "Document",
                                    UserContent::Image { .. } => "Image",
                                    UserContent::Text { .. } => "Text",
                                    UserContent::ToolResult { .. } => "ToolResult",
                                    UserContent::Thinking { .. } => "Thinking",
                                    UserContent::RedactedThinking { .. } => "RedactedThinking",
                                    UserContent::ServerToolResult { .. } => "ServerToolResult",
                                })
                                .collect::<Vec<_>>()
                                .join(", "),
                        };
                        info!(
                            "V1 Messages Request Message {}: role=user, content=[{}]",
                            i, user_content_types
                        );
                    }
                    Message::Assistant { content } => {
                        let assistant_content_types = match content {
                            AssistantContents::String(s) => format!("String(len={})", s.len()),
                            AssistantContents::Array(arr) => arr
                                .iter()
                                .map(|c| match c {
                                    AssistantContent::Text { .. } => "Text",
                                    AssistantContent::Thinking { .. } => "Thinking",
                                    AssistantContent::ToolUse { .. } => "ToolUse",
                                    AssistantContent::RedactedThinking { .. } => "RedactedThinking",
                                    AssistantContent::ServerToolUse { .. } => "ServerToolUse",
                                })
                                .collect::<Vec<_>>()
                                .join(", "),
                        };
                        info!(
                            "V1 Messages Request Message {}: role=assistant, content=[{}]",
                            i, assistant_content_types
                        );
                    }
                }
            }
        }
    }
}

fn log_bedrock_messages(messages: &[BedrockMessage]) {
    for (i, message) in messages.iter().enumerate() {
        let content_block_types = message
            .content()
            .iter()
            .map(|content_block| match content_block {
                ContentBlock::Document(_) => "Document",
                ContentBlock::GuardContent(_) => "GuardContent",
                ContentBlock::Image(_) => "Image",
                ContentBlock::ReasoningContent(_) => "ReasoningContent",
                ContentBlock::Text(_) => "Text",
                ContentBlock::ToolResult(_) => "ToolResult",
                ContentBlock::ToolUse(_) => "ToolUse",
                _ => "Unknown",
            })
            .collect::<Vec<_>>()
            .join(", ");
        info!(
            "Bedrock Message {}: role={:?}, content=[{}]",
            i,
            message.role(),
            content_block_types
        );
    }
}

pub struct BedrockV1MessagesProvider {
    bedrockruntime_client: Client,
}

/// Returns true if this is the Bedrock "thinking block modified" validation error.
fn is_thinking_block_modified_error(err: &SdkError<ConverseStreamError>) -> bool {
    let Some(message) = err.as_service_error().and_then(|e| e.meta().message()) else {
        return false;
    };

    message.contains(
        "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified",
    )
}

/// Formats a Bedrock error for the client. Prefers the service-error message
/// over the SDK's "service error" Display. When `status` is `Some`, embeds
/// the upstream HTTP code so it round-trips over a 200 SSE response.
pub(super) fn format_bedrock_error<E, R>(err: &SdkError<E, R>, status: Option<u16>) -> String
where
    E: ProvideErrorMetadata + std::fmt::Debug,
    R: std::fmt::Debug,
{
    let prefix = match status {
        Some(code) => format!("Bedrock API error (HTTP {code})"),
        None => "Bedrock API error".to_string(),
    };
    let body = err
        .as_service_error()
        .and_then(|e| e.meta().message())
        .map(String::from)
        .unwrap_or_else(|| format!("{err:?}"));
    format!("{prefix}: {body}")
}

/// Maps a Bedrock connect-time error to an Anthropic SSE error event.
fn map_converse_stream_error(err: &SdkError<ConverseStreamError>) -> ErrorEvent {
    let error_type = match err.as_service_error() {
        Some(ConverseStreamError::ValidationException(_)) => ErrorType::InvalidRequestError,
        Some(ConverseStreamError::AccessDeniedException(_)) => ErrorType::PermissionError,
        Some(ConverseStreamError::ResourceNotFoundException(_)) => ErrorType::NotFoundError,
        Some(ConverseStreamError::ThrottlingException(_)) => ErrorType::RateLimitError,
        Some(ConverseStreamError::ModelNotReadyException(_)) => ErrorType::OverloadedError,
        Some(ConverseStreamError::ServiceUnavailableException(_)) => ErrorType::OverloadedError,
        _ => ErrorType::ApiError,
    };
    let status = err.raw_response().map(|r| r.status().as_u16());
    ErrorEvent::new(error_type, format_bedrock_error(err, status))
}

/// Maps a Bedrock mid-stream error to an Anthropic SSE error event.
/// Mid-stream errors carry `RawMessage`, not an HTTP response — no status.
fn map_stream_output_error(err: &SdkError<ConverseStreamOutputError, RawMessage>) -> ErrorEvent {
    let error_type = match err.as_service_error() {
        Some(ConverseStreamOutputError::ValidationException(_)) => ErrorType::InvalidRequestError,
        Some(ConverseStreamOutputError::ThrottlingException(_)) => ErrorType::RateLimitError,
        Some(ConverseStreamOutputError::ServiceUnavailableException(_)) => {
            ErrorType::OverloadedError
        }
        _ => ErrorType::ApiError,
    };
    ErrorEvent::new(error_type, format_bedrock_error(err, None))
}

/// Sends an Anthropic-shaped `event: error` SSE frame. Always sends `Ok(_)` —
/// pushing `Err(_)` would close the TCP socket, which is the very failure
/// mode this exists to prevent.
async fn send_error_event(
    event_tx: &mpsc::Sender<anyhow::Result<Event>>,
    error_event: ErrorEvent,
) {
    let json = serde_json::to_string(&error_event).unwrap_or_else(|e| {
        error!("Failed to serialize ErrorEvent (using fallback): {}", e);
        r#"{"type":"error","error":{"type":"api_error","message":"internal serialization failure"}}"#
            .to_string()
    });
    let event = Event::default().event("error").data(json);
    let _ = timeout(EVENT_TX_SEND_TIMEOUT, event_tx.send(Ok(event))).await;
}

/// Spawns the streaming processor with a fresh ping interval.
fn spawn_stream_processor(
    stream: EventReceiver<ConverseStreamOutput, ConverseStreamOutputError>,
    model: String,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
    event_tx: mpsc::Sender<anyhow::Result<Event>>,
) {
    let ping_interval = interval_at(Instant::now() + PING_INTERVAL, PING_INTERVAL);
    tokio::spawn(process_bedrock_stream_events(
        stream,
        model,
        usage_callback,
        event_tx,
        ping_interval,
    ));
}

/// Continues an already-pending Bedrock connect after the sync window has
/// expired. Pings the SSE consumer while waiting; on resolve, hands the
/// stream to the processor or emits an in-band `event: error` frame (the
/// HTTP status is already lost — we have committed to 200).
async fn drive_pending_connect(
    mut send_fut: SendFut,
    event_tx: mpsc::Sender<anyhow::Result<Event>>,
    model: String,
    usage_callback: Arc<dyn Fn(&TokenUsage) + Send + Sync>,
) {
    let mut ping_interval = interval_at(Instant::now() + PING_INTERVAL, PING_INTERVAL);
    let result = loop {
        tokio::select! {
            biased;
            r = &mut send_fut => break r,
            _ = ping_interval.tick() => {
                if !send_ping(&event_tx).await { return; }
            }
        }
    };
    match result {
        Ok(response) => {
            process_bedrock_stream_events(
                response.stream,
                model,
                usage_callback,
                event_tx,
                ping_interval,
            )
            .await;
        }
        Err(e) => {
            error!("Bedrock API error after connect window: {:?}", e);
            send_error_event(&event_tx, map_converse_stream_error(&e)).await;
        }
    }
}

impl BedrockV1MessagesProvider {
    pub fn new(bedrockruntime_client: Client) -> Self {
        Self {
            bedrockruntime_client,
        }
    }
}

#[async_trait]
impl V1MessagesProvider for BedrockV1MessagesProvider {
    async fn v1_messages_stream<F>(
        self,
        request: V1MessagesRequest,
        response_model_id: Option<String>,
        anthropic_beta: Option<Vec<String>>,
        usage_callback: F,
    ) -> anyhow::Result<BoxStream<'async_trait, anyhow::Result<Event>>>
    where
        F: Fn(&TokenUsage) + Send + Sync + 'static,
    {
        let model = response_model_id.unwrap_or(request.model.clone());
        log_v1_messages_request(&request);
        let bedrock_chat_completion = BedrockChatCompletion::try_from(&request)?;
        let additional_model_request_fields = get_additional_model_request_fields(
            request.thinking.as_ref(),
            request.output_config.as_ref(),
            anthropic_beta.as_deref(),
            request.context_management.as_ref(),
        );
        if let Some(messages) = &bedrock_chat_completion.messages {
            log_bedrock_messages(messages);
        }
        info!(
            "Processed Anthropic request to Bedrock format with {} messages",
            bedrock_chat_completion
                .messages
                .as_ref()
                .map_or(0, |m| m.len())
        );

        info!(
            "Sending Anthropic request to Bedrock API for model: {}",
            bedrock_chat_completion.model_id
        );

        let (event_tx, event_rx) = mpsc::channel::<anyhow::Result<Event>>(1);
        let usage_callback = Arc::new(usage_callback);
        let client = self.bedrockruntime_client;

        // Race the connect against a short window: errors caught here flow
        // through `AppError` as proper HTTP 4xx with the upstream Bedrock
        // status. Slower connects fall to a 200 SSE response with pings.
        let send_fut: SendFut = Box::pin(
            client
                .converse_stream()
                .model_id(&bedrock_chat_completion.model_id)
                .set_system(bedrock_chat_completion.system_content_blocks.clone())
                .set_messages(bedrock_chat_completion.messages.clone())
                .set_tool_config(bedrock_chat_completion.tool_config.clone())
                .set_inference_config(Some(bedrock_chat_completion.inference_config.clone()))
                .set_additional_model_request_fields(additional_model_request_fields.clone())
                .set_output_config(bedrock_chat_completion.output_config.clone())
                .send(),
        );

        match race_connect(send_fut, CONNECT_ERROR_WINDOW).await {
            ConnectOutcome::Ok(response) => {
                info!("Successfully connected to Bedrock stream for Anthropic format");
                spawn_stream_processor(response.stream, model, usage_callback, event_tx);
            }
            ConnectOutcome::Err(e) if is_thinking_block_modified_error(&e) => {
                info!(
                    "Thinking block was modified; retrying with prior thinking text blanked"
                );
                let mut retry_request = request;
                retry_request.blank_assistant_thinking_text();
                let retry_bcc = BedrockChatCompletion::try_from(&retry_request)?;
                let retry_fut: SendFut = Box::pin(
                    client
                        .converse_stream()
                        .model_id(&retry_bcc.model_id)
                        .set_system(retry_bcc.system_content_blocks)
                        .set_messages(retry_bcc.messages)
                        .set_tool_config(retry_bcc.tool_config)
                        .set_inference_config(Some(retry_bcc.inference_config))
                        .set_additional_model_request_fields(additional_model_request_fields)
                        .set_output_config(retry_bcc.output_config)
                        .send(),
                );
                match race_connect(retry_fut, CONNECT_ERROR_WINDOW).await {
                    ConnectOutcome::Ok(response) => {
                        info!("Retry succeeded with prior thinking text blanked");
                        spawn_stream_processor(response.stream, model, usage_callback, event_tx);
                    }
                    ConnectOutcome::Err(e) => {
                        error!("Bedrock API error on retry: {:?}", e);
                        return Err(e.into());
                    }
                    ConnectOutcome::Pending(fut) => {
                        tokio::spawn(drive_pending_connect(fut, event_tx, model, usage_callback));
                    }
                }
            }
            ConnectOutcome::Err(e) => {
                error!("Bedrock API error: {:?}", e);
                return Err(e.into());
            }
            ConnectOutcome::Pending(fut) => {
                tokio::spawn(drive_pending_connect(fut, event_tx, model, usage_callback));
            }
        }

        Ok(ReceiverStream::new(event_rx).boxed())
    }

    async fn v1_messages_count_tokens(
        &self,
        request: &V1MessagesCountTokensRequest,
        inference_profile_prefixes: &[String],
    ) -> anyhow::Result<i32> {
        let messages: Option<Vec<BedrockMessage>> = Option::try_from(&request.messages)?;

        let system: Option<Vec<SystemContentBlock>> = request
            .system
            .as_ref()
            .map(Vec::<SystemContentBlock>::try_from)
            .transpose()?;

        let tool_config = request
            .tools
            .as_deref()
            .map(|tools| build_tool_configuration(tools, request.tool_choice.as_ref()))
            .transpose()?
            .flatten();

        let additional_model_request_fields = request.thinking.as_ref().map(Document::from);

        let converse_tokens_request = ConverseTokensRequest::builder()
            .set_additional_model_request_fields(additional_model_request_fields)
            .set_messages(messages)
            .set_system(system)
            .set_tool_config(tool_config)
            .build();

        let count_tokens_input = CountTokensInput::Converse(converse_tokens_request);

        let model_id = inference_profile_prefixes
            .iter()
            .find_map(|inference_profile_prefix| {
                request.model.strip_prefix(inference_profile_prefix)
            })
            .unwrap_or(&request.model);

        let result = self
            .bedrockruntime_client
            .count_tokens()
            .model_id(model_id)
            .input(count_tokens_input)
            .send()
            .await;

        match result {
            Ok(response) => Ok(response.input_tokens),
            Err(e) => {
                error!("Bedrock API error: {:?}", e);
                Err(e.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::types::error::ValidationException;
    use aws_smithy_runtime_api::http::{
        Response as SmithyResponse, StatusCode as SmithyStatusCode,
    };
    use aws_smithy_types::body::SdkBody;
    use aws_smithy_types::error::ErrorMetadata;

    fn make_sdk_error(message: &str) -> SdkError<ConverseStreamError> {
        let raw = SmithyResponse::new(
            SmithyStatusCode::try_from(400).unwrap(),
            SdkBody::from("error"),
        );
        let err = ConverseStreamError::ValidationException(
            ValidationException::builder()
                .message(message)
                .meta(ErrorMetadata::builder().message(message).build())
                .build(),
        );
        SdkError::service_error(err, raw)
    }

    #[test]
    fn is_thinking_block_modified_error_returns_true_for_matching_error() {
        let err = make_sdk_error(
            "The model returned the following errors: messages.3.content.1: \
             `thinking` or `redacted_thinking` blocks in the latest assistant \
             message cannot be modified.",
        );
        assert!(is_thinking_block_modified_error(&err));
    }

    #[test]
    fn is_thinking_block_modified_error_returns_false_for_unrelated() {
        let err = make_sdk_error("Some other validation error");
        assert!(!is_thinking_block_modified_error(&err));
    }

    #[test]
    fn is_thinking_block_modified_error_returns_false_for_partial_message() {
        let err = make_sdk_error("thinking blocks cannot be modified but no message index here");
        assert!(!is_thinking_block_modified_error(&err));
    }

    #[test]
    fn is_thinking_block_modified_error_works_without_indices() {
        let err = make_sdk_error(
            "The model returned the following errors: \
             `thinking` or `redacted_thinking` blocks in the latest assistant \
             message cannot be modified.",
        );
        assert!(is_thinking_block_modified_error(&err));
    }

    #[test]
    fn map_converse_stream_error_validation_to_invalid_request() {
        // Context-window-overflow surfaces here as a ValidationException.
        let err = make_sdk_error("input is too long for requested model");
        let event = map_converse_stream_error(&err);
        let json = serde_json::to_value(&event).expect("serialize");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("input is too long"),
            "expected upstream message preserved: {json}"
        );
        // The Bedrock HTTP status code must round-trip to the client. SSE
        // commits a 200 before the error is known, so the upstream code is
        // otherwise lost — embedding it in the message is what surfaces it.
        assert!(
            message.contains("HTTP 400"),
            "expected upstream HTTP code in message: {json}"
        );
    }

    #[test]
    fn map_converse_stream_error_unhandled_falls_back_to_api_error() {
        let raw = SmithyResponse::new(
            SmithyStatusCode::try_from(500).unwrap(),
            SdkBody::from("error"),
        );
        let sdk_err: SdkError<ConverseStreamError> =
            SdkError::service_error(ConverseStreamError::unhandled("boom"), raw);
        let event = map_converse_stream_error(&sdk_err);
        let json = serde_json::to_value(&event).expect("serialize");
        assert_eq!(json["error"]["type"], "api_error");
    }

    #[test]
    fn format_bedrock_error_omits_status_when_none() {
        // Mid-stream errors carry RawMessage, not an HTTP response, so
        // callers pass `status = None`. The prefix must degrade cleanly
        // (no "(HTTP …)" segment) instead of e.g. "(HTTP 0)" or panicking.
        let err = make_sdk_error("transport hiccup");
        let formatted = format_bedrock_error(&err, None);
        assert!(formatted.starts_with("Bedrock API error:"), "{formatted}");
        assert!(!formatted.contains("HTTP"), "{formatted}");
        assert!(formatted.contains("transport hiccup"), "{formatted}");
    }

    /// Regression for the original "socket connection closed unexpectedly" bug:
    /// `send_error_event` must push `Ok(Event)` (a real SSE frame) on the
    /// channel — never `Err(_)` — otherwise Axum's `Sse` aborts the body.
    #[tokio::test]
    async fn send_error_event_pushes_ok_sse_frame_to_channel() {
        let (tx, mut rx) = mpsc::channel::<anyhow::Result<Event>>(1);
        let event = ErrorEvent::new(ErrorType::InvalidRequestError, "input is too long");

        send_error_event(&tx, event).await;

        let received = rx
            .recv()
            .await
            .expect("channel must yield exactly one frame");
        let sse = received.expect("must be Ok(Event), never Err — see send_error_event docs");

        // The Event surface is opaque, so assert via the wire format.
        let rendered = format!("{sse:?}");
        assert!(
            rendered.contains("error"),
            "frame should be `event: error`: {rendered}"
        );
        assert!(
            rendered.contains("invalid_request_error"),
            "payload should carry the mapped error type: {rendered}"
        );
    }
}
