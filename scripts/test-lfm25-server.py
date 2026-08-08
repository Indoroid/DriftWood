#!/usr/bin/env python3
"""Live bmoe-server contract test against the official LFM2.5-8B-A1B GGUF."""

import argparse
import http.client
import json
import os
from pathlib import Path
import subprocess
import time


def request(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, str]:
    payload = None if body is None else json.dumps(body)
    headers = {} if payload is None else {"Content-Type": "application/json"}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    text = response.read().decode("utf-8", errors="replace")
    connection.close()
    return response.status, text


def require_json(port: int, path: str, body: dict) -> dict:
    status, text = request(port, "POST", path, body)
    if status != 200:
        raise RuntimeError(f"{path} returned HTTP {status}: {text}")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--out-dir", type=Path, default=Path("lfm25-server-test"))
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "server.log"
    server_args = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    command = [
        str(args.server.resolve()),
        "-m",
        str(args.model.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-c",
        "4096",
        "-t",
        "2",
        "-n",
        "256",
        "--temp",
        "0.2",
        "--top-k",
        "80",
        "--top-p",
        "0.95",
        "--dense-weights",
        "mmap",
        "--csv",
        str((args.out_dir / "metrics.csv").resolve()),
        *server_args,
    ]

    with log_path.open("w", encoding="utf-8") as log:
        env = os.environ.copy()
        # Normal hosts resolve the executable's $ORIGIN RUNPATH. Minimal containers may have no
        # /proc/self/exe, so make the same sibling library directory explicit for the test runner.
        lib_dir = args.server.resolve().parent.parent / "bin"
        env["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup; see {log_path}")
                try:
                    status, text = request(args.port, "GET", "/v1/models")
                    if status == 200:
                        models = json.loads(text)
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("server did not become ready within 600 seconds")

            model = models["data"][0]
            assert model["meta"]["arch"] == "lfm2moe", models
            assert model["meta"]["n_ctx"] == 4096, models

            chat = require_json(
                args.port,
                "/v1/chat/completions",
                {
                    "model": model["id"],
                    "messages": [
                        {"role": "system", "content": "Answer briefly."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                    "max_tokens": 160,
                    "temperature": 0.2,
                    "top_p": 0.95,
                },
            )
            chat_message = chat["choices"][0]["message"]
            assert chat_message.get("content") or chat_message.get("reasoning_content"), chat

            completion = require_json(
                args.port,
                "/v1/completions",
                {"model": model["id"], "prompt": "The capital of Japan is", "max_tokens": 8, "temperature": 0},
            )
            assert completion["choices"][0]["text"], completion

            status, sse = request(
                args.port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": model["id"],
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "max_tokens": 64,
                    "stream": True,
                    "temperature": 0,
                },
            )
            assert status == 200 and "data: [DONE]" in sse, sse[-1000:]

            tool = {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
            tool_user = {
                "role": "user",
                "content": "Use get_weather to check Paris. Call the tool; do not answer from memory.",
            }
            tool_request = {
                "model": model["id"],
                "messages": [tool_user],
                "tools": [tool],
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "max_tokens": 256,
                "temperature": 0.2,
            }
            tool_response = require_json(args.port, "/v1/chat/completions", tool_request)
            assistant = tool_response["choices"][0]["message"]
            calls = assistant.get("tool_calls", [])
            assert calls and calls[0]["id"] and calls[0]["function"]["name"] == "get_weather", tool_response

            followup = require_json(
                args.port,
                "/v1/chat/completions",
                {
                    "model": model["id"],
                    "messages": [
                        tool_user,
                        assistant,
                        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "21 C and sunny"},
                        {
                            "role": "user",
                            "content": "Do not call another tool. Summarize the supplied tool result in one sentence.",
                        },
                    ],
                    "tools": [tool],
                    "tool_choice": "none",
                    "max_tokens": 384,
                    "temperature": 0.2,
                },
            )
            followup_message = followup["choices"][0]["message"]
            assert followup_message.get("content"), followup

            tool_request["stream"] = True
            status, tool_sse = request(args.port, "POST", "/v1/chat/completions", tool_request)
            assert status == 200 and '"tool_calls"' in tool_sse and "data: [DONE]" in tool_sse, tool_sse[-2000:]

            outputs = {
                "models": models,
                "chat": chat,
                "completion": completion,
                "tool_call": tool_response,
                "tool_followup": followup,
            }
            (args.out_dir / "responses.json").write_text(
                json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (args.out_dir / "chat.sse").write_text(sse, encoding="utf-8")
            (args.out_dir / "tool.sse").write_text(tool_sse, encoding="utf-8")
            print("LFM2.5 server integration passed: models, ChatML, raw completion, SSE, tools, tool round-trip")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
