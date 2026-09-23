# gut-check-mcp

MCP server exposing gut-check's System-1 verification gate as a single tool,
`verify_claim(claim_text, context) -> {choice, confidence, escalate, reason}`.

Works with any MCP-capable host: Claude Code, Codex, Gemini CLI,
hermes-agent (via `adapters/hermes_plugin`, which calls this server as an
MCP client rather than the raw HTTP sidecar it used previously).

This server never calls a System-2 LLM itself -- it classifies and signals
low confidence via `escalate: true`; the calling host decides what to do
about it.

## Run locally

    cd adapters/mcp_server
    pip install -e .
    python -m gut_check_mcp.server

## Publish to Smithery

See `smithery.yaml` in this directory.
