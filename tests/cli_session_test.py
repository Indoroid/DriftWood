"""End-to-end check of the meitte-cli --session loop over its stdin/stdout line protocol.

A failed request must not end the session unless the engine reports it fatal: the loop used to
match error text and exit on anything outside a short list. Usage: cli_session_test.py CLI MODEL."""
import json
import os
import subprocess
import sys
import tempfile

cli, model = sys.argv[1], sys.argv[2]
failures = 0


def check(ok, name):
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failures += 0 if ok else 1


# A template with no reasoning delimiters: a reasoning budget is then a request the engine refuses.
with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False) as f:
    f.write("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}")
    template = f.name

commands = [
    {"cmd": "generate", "id": 1, "prompt": "hello", "n_predict": 4},
    {"cmd": "generate", "id": 2, "prompt": "and then", "n_predict": 4},
    # Thinking off drops the budget, so this one generates.
    {"cmd": "generate", "id": 3, "prompt": "and then", "n_predict": 4, "think": False},
]
try:
    proc = subprocess.run(
        [cli, "-m", model, "--session", "--ctx-size", "128", "--chat-template-file", template,
         "--reasoning-budget", "4"],
        input="".join(json.dumps(c) + "\n" for c in commands).encode(), capture_output=True, timeout=300)
finally:
    os.unlink(template)

# The protocol stream must be UTF-8 (the JSON escaper replaces invalid bytes). stderr carries raw token
# pieces, which a random model can split mid-character, so it is only decoded for diagnostics.
stderr = proc.stderr.decode("utf-8", errors="replace")
try:
    stdout = proc.stdout.decode("utf-8")
except UnicodeDecodeError as error:
    check(False, f"protocol stream is UTF-8 ({error})")
    stdout = proc.stdout.decode("utf-8", errors="replace")

events = {}
for line in stdout.splitlines():
    tag, _, body = line.partition(" ")
    if tag in ("BMOE_ERROR", "BMOE_DONE") and body:
        try:
            payload = json.loads(body)
        except ValueError:
            check(False, f"protocol line is valid JSON: {line[:200]!r}")
            continue
        events.setdefault(payload.get("id"), []).append((tag, payload))

first = events.get(1, [])
check(any(t == "BMOE_ERROR" and not p["fatal"] for t, p in first), "refused request reports fatal:false")
second = events.get(2, [])
check(any(t == "BMOE_ERROR" and not p["fatal"] for t, p in second), "session keeps serving after a refused request")
third = events.get(3, [])
check(any(t == "BMOE_DONE" and p.get("finish") in ("stop", "length") for t, p in third),
      "a later request completes with an engine finish reason")
check(proc.returncode == 0, "session exits cleanly at end of input")
if failures:
    print(f"exit code {proc.returncode}\n--- stdout ---\n{stdout[-4000:]}\n--- stderr ---\n{stderr[-4000:]}")
sys.exit(1 if failures else 0)
