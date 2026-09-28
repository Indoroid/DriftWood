# Session mode

A fresh `meitte-cli` process per prompt re-pays two fixed costs every time: the model load (tens
of seconds for a >RAM model) and the expert-cache warm-up ramp (the LRU cache starts empty, so
the first tokens miss often and read far more from flash than the steady state — see
[benchmarks.md](benchmarks.md)). Streaming was meant to avoid exactly this kind of repeated work.

Session mode amortises both. `Session` (`core/include/bmoe/session.h`) loads the model, discovers
the MoE expert tensors, and initialises the expert source **once**; each `generate()` then runs a
prompt against that resident state, and the expert LRU cache **survives between calls**, so a
second prompt starts warm.

```cpp
std::string err;
auto s = Session::open(cfg, err);      // model load + capture + source.init — once
s->generate({.prompt = "..."});        // prefill + decode; cache filled
s->generate({.prompt = "..."});        // starts warm — no reload, no cold ramp
```

`run()` (`runtime.h`) is a thin one-shot wrapper over `Session` — open, one generate, close — so
the byte-identity gates exercise the same machinery an interactive session uses. Gates **S1/S2**
assert that a second, warm generate produces output identical to the cold one-shot reference:
warming the cache changes latency, never the bytes.

## Independent prompts vs multi-turn chat

`GenerateRequest::clear_kv` selects between two modes:

- **`true` (new chat / independent prompt):** the KV cache and the engine-held conversation are
  dropped before the prompt runs. The expert cache stays warm. This is the one-shot path — `run()`
  always uses it, and the byte-identity gates exercise it.
- **`false` (continue):** the prompt continues the current conversation. In chat mode the Session
  owns the conversation (`chat_history`) and re-renders the model's chat template over the *whole*
  history each turn, so the caller sends only the new user message — not the running transcript.

`meitte-cli --session --kv-preserve` makes subsequent commands choose `clear_kv=false` by default;
an individual JSON command can still send `"clear_kv":true` to reset. `meitte-server --kv-preserve`
does the same for one incremental server-wide conversation. It is intentionally not a multi-client
session store: a caller that needs isolation must start a fresh chat with `clear_kv:true`.

**KV prefix reuse.** Re-rendering the full history would re-tokenize the entire conversation, but
most of it is already decoded into the KV. Each turn the engine diffs the freshly rendered tokens
against `kv_tokens` (the tokens currently in the KV, in order), keeps the common prefix, removes the
divergent tail with `llama_memory_seq_rm`, and prefills only the suffix. So a follow-up turn pays
for its own tokens, not a full re-prefill — which matters because prefill is the slow phase on
device. `BMOE_DONE.n_prompt` reports the tokens actually prefilled this turn; `n_past` is the total
context length after it.

**KV layout.** `RunConfig::kv_unified` is passed unchanged through `SessionConfig` to the public
`llama_context_params::kv_unified` field. Both frontends expose it as `--kv-unified` and
`--no-kv-unified`; the latter is the default. The setting also reaches an MTP draft context because
that context starts from the same libllama parameters. This controls libllama's cache layout. It
does not create independent server conversations or change Meitte's one-sequence ownership model.

**Fallbacks and costs.** SWA-style memory (e.g. Gemma) can refuse a partial `seq_rm`; the engine
then clears the KV and re-prefills the whole prompt for that turn (correct, just slower). With
thinking **on**, the template strips the previous turn's reasoning on re-render, so the rendered
prefix diverges at the last answer and up to one answer's worth of tokens is re-prefilled per turn;
with thinking **off** (the app default) the re-fed suffix is just the new user turn plus a few
wrapper tokens.

**Reasoning is returned, not discarded.** On a thinking model the Session parses the reasoning span
out of the raw stream and carries it in its own field (`TokenMetrics`/`RunResult::reasoning`,
`delta_reasoning` on `BMOE_PROGRESS`, `reasoning` on `BMOE_DONE`) rather than dropping it. The answer text stays free of
it either way, so the byte-identity gates are unaffected; a caller that wants to show the thinking
reads the separate field. The parser wiring lives in `core/src/engine/chat_parse.cpp`
(see [seam.md](seam.md)).

## Cancel

`Session::cancel()` is thread-safe and interrupts an in-flight `generate()` at the next decode
boundary via llama's abort callback (installed unconditionally at open, so it works in serial and
overlap alike). It leaves the model and cache intact; the returned `RunResult` has `cancelled =
true`. In chat mode a cancel **rolls the turn back** to the reused prefix (dropping this turn's KV
and un-appending the user message) so prior turns stay usable and the conversation can continue.
A failed turn rolls back the same way, and so does the session sampler's RNG. When the KV cannot
keep the reused prefix (recurrent or hybrid memory refuses a partial removal), the rollback clears
the KV together with its token records and the next turn rebuilds the conversation from the
transcript — the KV and the engine's record of it never disagree.
Cancel is distinct from a fatal streaming error, which is sticky and ends the session.
Dense streaming drains interrupted reads before the next turn, clears the active matrix
window and keeps the fixed resident set; the next turn uses the same tensor addresses.

## Context capacity and opt-in recovery

`n_ctx`, `n_batch`, and `n_ubatch` are baked into the llama context at `open()`, before any prompt is
known. The frontends default to a 2048-token logical batch and a 512-token physical batch; both are
capped to the configured context. Use `--batch-size` to set the prefill chunk and `--ubatch-size` to
trade prefill throughput for resident compute-buffer memory. By default, a request that would overflow
`n_ctx` is rejected without tearing the session down.

The core `ContextPolicy` can opt into recovery. It first grows to an explicit `max_ctx`, then can
summarize the oldest complete user turn, then can remove that turn. `Auto` reserves the request's
output budget before prefill; `On` waits until capacity is reached. Summarization and trimming are
lossy, never remove the newest turn, and do not remove a turn that contains a media marker. A failed
or cancelled turn restores the transcript and clears any partial rebuilt KV so the next request can
prefill safely. RoPE scaling controls token positions; it does not allocate a larger context.

Both frontends expose the core policy directly:

```text
--dynamic-ctx off|on|auto
--dynamic-min-ctx N
--dynamic-max-ctx N
--summarize-history off|on|auto
--trim-old true|false|auto
```

The bounds must satisfy `dynamic-min-ctx <= ctx-size <= dynamic-max-ctx`. `--ctx-size` is the opening
context. Growth recreates the context with the same RoPE/YaRN parameters until it reaches
`--dynamic-max-ctx`, so that maximum is the final RoPE/YaRN-opened context. `on` reacts when the current
context reaches capacity; `auto` reserves the request's output allowance before prefill.
Summarization and trimming operate on chat history, so they need chat templating or a server chat
request. All recovery controls remain off by default, and bounds alone do not enable growth.

The frontends read `MEITTE_DYNAMIC_CTX`, `MEITTE_DYNAMIC_MIN_CTX`,
`MEITTE_DYNAMIC_MAX_CTX`, `MEITTE_SUMMARIZE_HISTORY`, and `MEITTE_TRIM_OLD` only when the
matching flag is absent. The former `--dyn-min-ctx`, `--dyn-max-ctx`, `--context-summarize`,
and `--context-trim` names remain compatibility aliases.

## C and Python FFI

Set `-DBMOE_BUILD_SHARED=ON` to build and install `libmeitte` (`meitte.dll` on Windows). Its
versioned C API in `core/include/bmoe/meitte.h` uses opaque sessions and results, caller-owned
input spans, synchronous borrowed token callbacks, cancellation, and explicit destruction. Each
configuration and request has a size tag; older prefixes use the defaults for later fields.

[`examples/python/ctypes_smoke.py`](../examples/python/ctypes_smoke.py) first checks the C error
ownership path without a model. Pass `--model /path/to/model.gguf` to also exercise callback and
result ownership, or `--cancel-after 1` to check callback cancellation.

## CLI and app

`meitte-cli --session` exposes this over a line protocol (requests on stdin, `BMOE_*` responses on
stdout — see [telemetry.md](telemetry.md)). A generate request may send either the legacy `prompt`
or a complete `messages` transcript, plus `think`, `reasoning_effort`, and JSON-valued
`chat_template_kwargs`; these request controls do not reopen the session. `--reasoning-budget N`
limits only the template's reasoning span (`0` closes it at once), not the final-answer
`n_predict` allowance. `--reasoning-preserve` and `--no-reasoning-preserve` pass the model
template's `preserve_reasoning` setting when it supports that policy. The server also accepts a
request-local `reasoning_budget_tokens` or `thinking_budget_tokens` value.
The server's `--alias NAME` sets the model identifier advertised by `/v1/models` and returned in
completion metadata. Without it, the model filename remains the identifier.
The Android example runs one such process per model:
the first prompt loads the model, later prompts reuse the warm process, and the session is freed on
an explicit **Unload** or after an idle timeout. Changing the model or any streaming setting
reopens the session; changing only the prompt, `n_predict`, or the thinking toggle does not.
