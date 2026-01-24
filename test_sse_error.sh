#!/bin/bash
# Test script to see the actual SSE output
curl -N -H "Content-Type: application/json" \
  -d '{"model":"us.anthropic.claude-haiku-4-5-20251001-v1:0","max_tokens":100,"messages":[{"role":"user","content":"test"}],"stream":true}' \
  http://localhost:3000/v1/messages 2>&1 | head -20
