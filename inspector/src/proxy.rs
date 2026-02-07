use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use chrono::Local;
use futures::StreamExt;
use http_body_util::BodyExt;
use tokio::sync::mpsc;

use crate::exchange::{
    CapturedExchange, accumulate_sse_events, extract_request_summary, extract_response_summary,
};

pub struct AppState {
    pub client: reqwest::Client,
    pub target: String,
    pub tx: mpsc::Sender<CapturedExchange>,
}

fn mask_sensitive(name: &str, value: &str) -> String {
    let lower = name.to_lowercase();
    if lower == "authorization" || lower == "x-api-key" || lower == "proxy-authorization" {
        if value.len() > 12 {
            format!("{}...{}", &value[..8], &value[value.len() - 4..])
        } else {
            "****".to_string()
        }
    } else {
        value.to_string()
    }
}

pub async fn proxy_handler(
    State(state): State<Arc<AppState>>,
    request: Request,
) -> impl IntoResponse {
    let timestamp = Local::now().format("%H:%M:%S").to_string();

    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path().to_string();
    let query = uri.query().map(|q| format!("?{}", q)).unwrap_or_default();

    // Collect and capture request headers
    let mut req_headers = reqwest::header::HeaderMap::new();
    let mut captured_req_headers: Vec<(String, String)> = Vec::new();
    for (name, value) in request.headers() {
        if name.as_str().eq_ignore_ascii_case("host") {
            continue;
        }
        let val_str = value.to_str().unwrap_or("");
        captured_req_headers.push((
            name.as_str().to_string(),
            mask_sensitive(name.as_str(), val_str),
        ));
        if let Ok(v) = reqwest::header::HeaderValue::from_str(val_str) {
            req_headers.insert(name.clone(), v);
        }
    }

    // Read request body
    let body_bytes = match request.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => Bytes::new(),
    };

    // Parse request JSON
    let request_json: Option<serde_json::Value> = if !body_bytes.is_empty() {
        serde_json::from_slice(&body_bytes).ok()
    } else {
        None
    };

    let request_summary = extract_request_summary(&request_json);

    // Forward request to target
    let target_url = format!("{}{}{}", state.target.trim_end_matches('/'), path, query);

    let forward = state
        .client
        .request(method.clone(), &target_url)
        .headers(req_headers)
        .body(body_bytes);

    let upstream_resp = match forward.send().await {
        Ok(resp) => resp,
        Err(e) => {
            let exchange = CapturedExchange {
                timestamp,
                method: method.to_string(),
                path: path.clone(),
                status: 502,
                request_headers: captured_req_headers,
                response_headers: Vec::new(),
                request_body: request_json,
                response_body: None,
                sse_events: Vec::new(),
                request_summary,
                response_summary: String::new(),
                error: Some(format!("{}", e)),
            };
            let _ = state.tx.send(exchange).await;

            return Response::builder()
                .status(502)
                .body(Body::from(format!("Proxy error: {}", e)))
                .unwrap();
        }
    };

    let status = upstream_resp.status();
    let status_code = status.as_u16();

    // Build response headers and detect SSE
    let mut resp_builder = Response::builder().status(status);
    let mut is_sse = false;
    let mut captured_res_headers: Vec<(String, String)> = Vec::new();
    for (name, value) in upstream_resp.headers() {
        let val_str = value.to_str().unwrap_or("<binary>");
        captured_res_headers.push((
            name.as_str().to_string(),
            mask_sensitive(name.as_str(), val_str),
        ));
        if name.as_str().eq_ignore_ascii_case("content-type")
            && val_str.contains("text/event-stream")
        {
            is_sse = true;
        }
        resp_builder = resp_builder.header(name.clone(), value.clone());
    }

    if is_sse {
        let byte_stream = upstream_resp.bytes_stream();
        let (tx_body, rx_body) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(32);

        let tx_exchange = state.tx.clone();
        let exchange_meta = (
            timestamp,
            method.to_string(),
            path,
            status_code,
            captured_req_headers,
            captured_res_headers,
            request_json,
            request_summary,
        );

        tokio::spawn(async move {
            let (
                timestamp,
                method,
                path,
                status_code,
                req_headers,
                res_headers,
                request_json,
                request_summary,
            ) = exchange_meta;

            let mut events: Vec<(String, String)> = Vec::new();
            let mut buffer = String::new();

            tokio::pin!(byte_stream);

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let _ = tx_body.send(Ok(chunk.clone())).await;

                        let text = String::from_utf8_lossy(&chunk);
                        buffer.push_str(&text);

                        while let Some(pos) = buffer.find("\n\n") {
                            let event_text = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            if event_text.trim().is_empty() {
                                continue;
                            }

                            let mut event_type = String::new();
                            let mut data_parts = Vec::new();

                            for line in event_text.lines() {
                                if let Some(et) = line.strip_prefix("event:") {
                                    event_type = et.trim().to_string();
                                } else if let Some(d) = line.strip_prefix("data:") {
                                    data_parts.push(d.trim().to_string());
                                }
                            }

                            let data_str = data_parts.join("");
                            events.push((event_type, data_str));
                        }
                    }
                    Err(e) => {
                        let _ = tx_body
                            .send(Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                e.to_string(),
                            )))
                            .await;
                        break;
                    }
                }
            }

            // Process remaining buffer
            if !buffer.trim().is_empty() {
                let mut event_type = String::new();
                let mut data_parts = Vec::new();

                for line in buffer.lines() {
                    if let Some(et) = line.strip_prefix("event:") {
                        event_type = et.trim().to_string();
                    } else if let Some(d) = line.strip_prefix("data:") {
                        data_parts.push(d.trim().to_string());
                    }
                }

                let data_str = data_parts.join("");
                if !data_str.is_empty() {
                    events.push((event_type, data_str));
                }
            }

            let response_json = accumulate_sse_events(&events);
            let response_summary = extract_response_summary(&Some(response_json.clone()));

            let exchange = CapturedExchange {
                timestamp,
                method,
                path,
                status: status_code,
                request_headers: req_headers,
                response_headers: res_headers,
                request_body: request_json,
                response_body: Some(response_json),
                sse_events: events,
                request_summary,
                response_summary,
                error: None,
            };
            let _ = tx_exchange.send(exchange).await;
        });

        let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx_body);
        let body = Body::from_stream(body_stream);

        resp_builder.body(body).unwrap()
    } else {
        let body_bytes = upstream_resp.bytes().await.unwrap_or_default();

        let response_json: Option<serde_json::Value> = if !body_bytes.is_empty() {
            serde_json::from_slice(&body_bytes).ok()
        } else {
            None
        };

        let response_summary = extract_response_summary(&response_json);

        let exchange = CapturedExchange {
            timestamp,
            method: method.to_string(),
            path,
            status: status_code,
            request_headers: captured_req_headers,
            response_headers: captured_res_headers,
            request_body: request_json,
            response_body: response_json,
            sse_events: Vec::new(),
            request_summary,
            response_summary,
            error: None,
        };
        let _ = state.tx.send(exchange).await;

        resp_builder.body(Body::from(body_bytes)).unwrap()
    }
}
