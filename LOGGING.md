# Logging Strategy

## Privacy-First Approach

This proxy is designed with privacy in mind. We follow these principles:

### What We LOG:
- ✅ Request/response counts and structure (number of messages, tools, etc.)
- ✅ Model names and routing decisions
- ✅ Error conditions and warnings
- ✅ Performance metrics (token counts, latency)
- ✅ Stream lifecycle events (start, stop)

### What We DON'T LOG:
- ❌ Message content (user prompts, assistant responses)
- ❌ Tool inputs/outputs (only tool names and IDs)
- ❌ System prompts or instructions
- ❌ Any personally identifiable information (PII)

## Log Levels

- **ERROR**: Critical failures that prevent request processing
- **WARN**: Recoverable issues (e.g., skipping invalid content, client bugs)
- **INFO**: High-level request flow (model selection, token usage, stream lifecycle)
- **DEBUG**: Structural information (message counts, tool counts) - NO CONTENT
- **TRACE**: Reserved for future detailed debugging (disabled by default)

## Examples

### Good Logging (Privacy-Preserving):
```rust
info!("Converting Anthropic request to Bedrock format: {} messages, {} tools", 
    messages.len(), tools.len());
warn!("Skipping duplicate tool_use block: id={}", id);
debug!("Message {}: role={}, {} content blocks", idx, role, content.len());
```

### Bad Logging (Privacy-Violating):
```rust
// DON'T DO THIS:
debug!("User message: {}", text);  // ❌ Logs user content
debug!("Tool input: {:?}", input);  // ❌ Logs tool parameters
info!("System prompt: {}", system);  // ❌ Logs instructions
```

## Enabling Detailed Debugging

For troubleshooting, set the `RUST_LOG` environment variable:

```bash
# Production (default)
RUST_LOG=info

# Development
RUST_LOG=debug

# Troubleshooting
RUST_LOG=trace
```

## Future Enhancements

- [ ] Add structured logging with request IDs for tracing
- [ ] Add metrics export (Prometheus/OpenTelemetry)
- [ ] Add optional audit logging (opt-in, with user consent)
- [ ] Add performance profiling hooks

