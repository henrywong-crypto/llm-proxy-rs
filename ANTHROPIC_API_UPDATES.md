# Anthropic API Updates

This document summarizes the updates made to the `anthropic-request` and `anthropic-response` packages to fully support the Anthropic Messages API specification, including SSE streaming events.

## anthropic-request Package

### New Files Added

1. **`src/tool_choice.rs`** - ToolChoice configuration
   - Controls how the model uses tools
   - Supports types: "auto", "any", "tool", "none"
   - Optional fields: name, disable_parallel_tool_use

2. **`src/metadata.rs`** - Request metadata
   - Optional user_id field for tracking

3. **`src/image_source.rs`** - Image source representation
   - Supports "base64" and "url" source types
   - Includes media_type, data, and url fields

4. **`src/content_block.rs`** - Unified ContentBlock enum
   - Text blocks with optional text field
   - Image blocks with ImageSource
   - ToolUse blocks with id, name, and input
   - ToolResult blocks with tool_use_id, content, and is_error
   - Thinking blocks with optional thinking and signature fields

### Updated Files

1. **`src/content/user_content.rs`**
   - Added Image variant to UserContent enum
   - Added conversion logic for base64 images to Bedrock format
   - Supports multiple image formats: JPEG, PNG, GIF, WebP

2. **`src/lib.rs`**
   - Added exports for new modules
   - Extended V1MessagesRequest with additional fields:
     - tool_choice
     - top_p
     - top_k
     - stop_sequences
     - metadata

## anthropic-response Package

### New Files Added

1. **`src/error.rs`** - Error response types
   - Error struct with type and message
   - ErrorResponse with appropriate error types based on HTTP status codes
   - Error types: invalid_request_error, authentication_error, permission_error, not_found_error, rate_limit_error, overloaded_error, api_error
   - Automatic request ID generation

2. **`src/delta.rs`** - Streaming delta types
   - Delta enum with variants:
     - TextDelta - incremental text updates
     - InputJsonDelta - incremental JSON for tool inputs
     - ThinkingDelta - incremental thinking content
     - SignatureDelta - signature updates
   - MessageDelta - stop reason and sequence information
   - DeltaUsage - cumulative output token counts

### Updated Files

1. **`src/event.rs`** - Complete streaming event types
   - MessageStart - sent at the start of streaming
   - ContentBlockStart - signals the start of a content block
   - ContentBlockDelta - incremental content updates
   - ContentBlockStop - signals the end of a content block
   - MessageDelta - message-level updates
   - MessageStop - signals the end of the message
   - Ping - keepalive event
   - Error - error during streaming
   - Updated all builders to use new Delta types

2. **`src/content_block_delta.rs`**
   - Added `bedrock_content_block_delta_to_delta()` function
   - Converts AWS Bedrock deltas to Anthropic Delta format
   - Deprecated old function name for backward compatibility

3. **`src/stream.rs`**
   - Updated to use new Delta types instead of ContentBlockDelta
   - Updated imports and type references

4. **`src/lib.rs`**
   - Added exports for delta and error modules

5. **`Cargo.toml`**
   - Added `rand = "0.8"` dependency for error ID generation

## Key Features

### Complete Anthropic API Support

- ✅ Full request parameter support (model, max_tokens, messages, system, stream, temperature, top_p, top_k, stop_sequences, tools, tool_choice, thinking, metadata)
- ✅ All content block types (text, image, tool_use, tool_result, thinking)
- ✅ Image support with base64 encoding
- ✅ Tool configuration and tool choice
- ✅ Extended thinking support

### SSE Streaming Events

All Anthropic streaming event types are now supported:

1. **message_start** - Initial message with metadata
2. **content_block_start** - Start of each content block (text, thinking, tool_use)
3. **content_block_delta** - Incremental updates with different delta types
4. **content_block_stop** - End of content block
5. **message_delta** - Final message metadata (stop_reason, token usage)
6. **message_stop** - End of stream
7. **ping** - Keepalive
8. **error** - Error events

### Error Handling

- Proper error response structure matching Anthropic API
- HTTP status code to error type mapping
- Unique request ID generation for tracking

## Usage Example

```rust
use anthropic_request::{V1MessagesRequest, ToolChoice, Metadata};
use anthropic_response::{Event, Delta, ErrorResponse};

// Create a request with all features
let request = V1MessagesRequest {
    model: "claude-3-5-sonnet-20241022".to_string(),
    max_tokens: 1024,
    messages: /* ... */,
    stream: Some(true),
    temperature: Some(0.7),
    top_p: Some(0.9),
    tool_choice: Some(ToolChoice {
        choice_type: "auto".to_string(),
        name: None,
        disable_parallel_tool_use: None,
    }),
    metadata: Some(Metadata {
        user_id: Some("user123".to_string()),
    }),
    // ... other fields
};

// Handle streaming events
match event {
    Event::MessageStart { message } => { /* ... */ },
    Event::ContentBlockDelta { delta, index } => {
        match delta {
            Delta::TextDelta { text } => { /* ... */ },
            Delta::ThinkingDelta { thinking } => { /* ... */ },
            Delta::InputJsonDelta { partial_json } => { /* ... */ },
            Delta::SignatureDelta { signature } => { /* ... */ },
        }
    },
    Event::Error { error } => { /* ... */ },
    // ... other events
}

// Create error responses
let error = ErrorResponse::new(400, "Invalid request");
```

## Compatibility

- All changes are backward compatible
- Existing code using the old API will continue to work
- New features are opt-in through optional fields
- Deprecated functions are marked but still functional

