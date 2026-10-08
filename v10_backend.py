#!/usr/bin/env python3
"""Imagine v10 backend for pin-clientd.

Ollama-compatible adapter that forwards chat requests to llama-server
running the quantized Imagine v10 GGUF. Advertises as "imagine-v10".

Usage:
  python3 v9_backend.py [--port 11434] [--llama-url http://localhost:8081]

Stdlib only.
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "imagine-v10"
LLAMA_URL = "http://localhost:8081"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} {fmt % args}", flush=True)

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send(200, {"models": [{"name": MODEL, "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]})
        return self._send(404, {"error": "not found"})

    def _handle_stream(self, llama_req):
        """Proxy a streaming request: llama-server SSE -> Ollama NDJSON."""
        llama_req = dict(llama_req, stream=True)
        req = urllib.request.Request(
            f"{LLAMA_URL}/v1/chat/completions",
            data=json.dumps(llama_req).encode(),
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.URLError as e:
            return self._send(502, {"error": f"llama-server unreachable: {e}"})
        except Exception as e:
            return self._send(500, {"error": f"inference failed: {e}"})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            buf = b""
            while True:
                data = resp.read(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                        delta = ev["choices"][0]["delta"].get("content") or ""
                    except Exception:
                        continue
                    if delta:
                        chunk = {
                            "model": MODEL,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime()),
                            "message": {"role": "assistant", "content": delta},
                            "done": False,
                        }
                        self.wfile.write((json.dumps(chunk) + "\n").encode())
                        self.wfile.flush()
            final = {
                "model": MODEL,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }
            self.wfile.write((json.dumps(final) + "\n").encode())
            self.wfile.flush()
        finally:
            resp.close()
        return None

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length))
            except Exception:
                return self._send(400, {"error": "bad json"})

            messages = req.get("messages", [])
            stream = req.get("stream", False)

            # Forward to llama-server OpenAI-compatible endpoint
            llama_req = {
                "messages": messages,
                "max_tokens": req.get("options", {}).get("num_predict", 512),
                "temperature": req.get("options", {}).get("temperature", 0.7),
                "stream": stream,
            }

            # pin-clientd ALWAYS requests stream:true and expects Ollama-style
            # newline-delimited JSON. llama-server's OpenAI endpoint speaks SSE
            # instead, so translate its event stream into Ollama NDJSON here.
            # Without this, json.loads() on the SSE body fails and every real
            # PIN request 500s ("Expecting value: line 1 column 1 (char 0)").
            if stream:
                return self._handle_stream(llama_req)

            try:
                llama_body = json.dumps(llama_req).encode()
                llama_http = urllib.request.Request(
                    f"{LLAMA_URL}/v1/chat/completions",
                    data=llama_body,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(llama_http, timeout=120) as resp:
                    llama_resp = json.loads(resp.read())
            except urllib.error.URLError as e:
                return self._send(502, {"error": f"llama-server unreachable: {e}"})
            except Exception as e:
                return self._send(500, {"error": f"inference failed: {e}"})

            # Extract response
            try:
                content = llama_resp["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                return self._send(500, {"error": "bad llama-server response"})

            # Ollama chat response format
            ollama_resp = {
                "model": MODEL,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": content},
                "done": True,
            }
            return self._send(200, ollama_resp)

        return self._send(404, {"error": "not found"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--llama-url", default="http://localhost:8081")
    args = p.parse_args()

    global LLAMA_URL
    LLAMA_URL = args.llama_url.rstrip("/")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"v9 backend on :{args.port} -> {LLAMA_URL} (model: {MODEL})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
