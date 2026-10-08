# imagine-backend

Ollama-compatible adapter that forwards chat requests to `llama-server`
running a quantized Imagine GGUF. Used by `pin-clientd` to serve Imagine
models on the PIN network.

## What it does

- Speaks the Ollama API (`/api/tags`, `/api/chat`, `/api/generate`) on `:11434`
- Forwards to `llama-server` (default `http://localhost:8081`)
- Translates llama-server's **SSE** event stream into Ollama **NDJSON** —
  without this, `json.loads()` on the SSE body fails and every streaming
  request 500s
- Stdlib only — no dependencies

## Usage

```bash
python3 v10_backend.py [--port 11434] [--llama-url http://localhost:8081]
```

## Chain

```
llama-server (:8081, Imagine GGUF)
      ↓ SSE
v10_backend.py (:11434, Ollama-compatible)
      ↓
pin-clientd → PIN network
```

## SSE fix

llama-server's OpenAI-compatible endpoint speaks Server-Sent Events
(`data: {...}` lines). The adapter reads the event stream line by line,
strips the `data:` prefix, and re-emits each chunk as a newline-delimited
JSON object per the Ollama protocol. Non-streaming requests go through
the regular JSON path unchanged.
