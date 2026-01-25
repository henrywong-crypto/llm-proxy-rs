use anthropic_response::ErrorResponse;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

#[tokio::test]
async fn test_error_response_format() {
    // This test verifies that errors are returned in Anthropic API format

    // Create a test request with stream=false (which should trigger an error)
    let app = server::create_router().await.unwrap();

    let request = Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    // Should return 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Parse response body
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let error_response: ErrorResponse = serde_json::from_slice(&body).unwrap();

    // Verify Anthropic API error format
    assert_eq!(error_response.response_type, "error");
    assert_eq!(error_response.error.error_type, "invalid_request_error");
    assert!(
        error_response
            .error
            .message
            .contains("Streaming is required")
    );
    assert!(error_response.request_id.is_some());
}

#[test]
fn test_error_response_status_code_mapping() {
    // Test that different status codes map to correct error types
    let tests = vec![
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (503, "overloaded_error"),
        (500, "api_error"),
    ];

    for (code, expected_type) in tests {
        let err = ErrorResponse::new(code, "test message");
        assert_eq!(
            err.error.error_type, expected_type,
            "Status code {} should map to {}",
            code, expected_type
        );
        assert_eq!(err.response_type, "error");
        assert!(err.request_id.is_some());
    }
}
