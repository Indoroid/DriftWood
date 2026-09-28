# Changelog

## [Unreleased]

- Build: `BMOE_BUILD_TESTS` no longer pulls in `cli/`. Core tests build against `libmeitte` alone
  with `BMOE_BUILD_CLI=OFF`; the CLI protocol, server API and CLI/server parity tests are host tests
  added only with `BMOE_BUILD_CLI=ON`.
- Tests: session-level n-gram speculation (drafting, cancel/retry, reset) joins the transaction test.
- Refuse a second expert-overlap session while another owns the process-wide ggml-cpu expert-ready
  hook. The second session used to take the hook over, so the first session's compute stopped
  waiting for its own expert reads. Dense overlap already had this guard.
- C ABI: add `meitte_config_init()` and `meitte_request_init()`, which fill defaults without writing
  past the caller's struct size. Fix a padding hazard: the `meitte_config` layout before the dense
  fields ended in tail padding that `dense_stream` now occupies, so callers compiled against it had
  uninitialized bytes read as `dense_stream`; that exact `struct_size` now means the older layout.
  `reserved0` names the current tail padding so the next field starts past every earlier `sizeof`.
  `tests/library_abi_test.c` reproduces old callers' layouts against the current library.
- Make `Session::perplexity()` leave the session in the documented new-chat state on every return
  path. It cleared the KV and `kv_tokens` but kept the chat history, retained media, draft context
  and the scored text in the KV, so a following `clear_kv=false` turn decoded after the scored text.
  Multiple-choice scoring now accepts choices longer than eight tokens (only the first is scored).
- Make reasoning-token accounting match the documented contract: with a reasoning budget, tokens in
  the reasoning span (up to the budget plus the forced end sequence) no longer consume `n_predict`,
  which previously counted every token. Without a budget, `n_predict` still bounds the whole
  generation. `RunSummary::n_reasoning` reports the split; `TokenMetrics::steps` includes the
  allowance. The span is observed with llama.cpp's own reasoning-budget state machine.
- Fix Session rollback when the KV cache refuses a partial removal (recurrent/hybrid memory): the
  KV was cleared while `kv_tokens` kept the old prefix, so the next turn decoded after a prefix the
  context no longer held. KV truncation now goes through one path that clears the KV together with
  its token records, media position and draft context, and the next turn rebuilds the conversation.
- Roll back every failed turn after prefill starts, not only cancelled ones: decode, overlap I/O and
  MTP failures used to leave the partial prefill in the KV and the user message in the history.
- Trim a speculative turn's recorded media position to the emitted end, matching the trimmed KV.
- Make `GenerateRequest::override_sampling` request-local. The override used to replace the
  session sampler for every later request.
- Restore the session sampler's RNG when a turn rolls back, so a retried turn samples as if the
  failed one had not run.
- Split the engine session into focused units behind one shared `Session::Impl`
  (`session_open`, `session_generate`, `session_perplexity`, `session_context`), with the
  llama.h adapters in `llama_glue`, chat request rendering in `chat_render`, and per-phase
  accounting in `run_tally.h`. `Session::open()` now runs as named phases.
- Split `router_hook.cpp` into capture/dispatch (`router_hook`), the lossy routing policies
  (`router_policy`) and prediction/route-ahead (`router_predict`).
- Move the remaining OS-specific code in `core/` into the `io/` platform layer: FFmpeg
  capture goes through `pio::run_capture`, the readiness spin-wait through `pio::cpu_relax`.
- Replace the architecture-name check for Cohere2MoE / HY V3 expert layouts with a recipe flag
  (`MoeRecipe::fused_gate_up_alt`) and a unit-tested `resolve_moe_recipe`.
- Contexts rebuilt for context growth or summarization now use the session's full abort
  predicate, so a fatal streaming I/O error stops them as it stops the original context.
- Remove unused code: `RouterHook::captured_weights()` and `DenseWeights::file_mapping_in_use()`.

## [0.24.0] - 2026-09-24

- Add bounded CPU dense-matrix streaming for zero-expert GGUF models, with serial
  mainline execution and optional weight-ready overlap on the synced llama.cpp variant.
- Add dense prefill/decode I/O, wait, residency, fault and RSS telemetry, plus Q4_0
  and F16 byte-identity gates.
- Drop the unused mmap pages after rebinding streamed matrices, reducing duplicate weight residency.
- Keep a dense session reusable after a cancelled read by draining its I/O workers and
  clearing the interrupted window before the next turn; synchronize matrix-ready waits to
  prevent lost wakeups.
- Expose dense streaming options through `meitte-server` as well as `meitte-cli`.
- Reject a second dense overlap session while the process-wide CPU callback is in use.

## [0.23.0] - 2026-09-23

- Add expert streaming recipes for Cohere2MoE, Hunyuan MoE, HY V3, HY V4,
  MiniMax M3, and MiMo2 while retaining the existing recipes.
- Select split or fused expert weights from GGUF tensor names for Cohere2MoE
  and HY V3.
- Document DeepSeek V4 Flash Vision Exp's existing `deepseek4` text architecture
  and separate vision projector, and the pinned llama.cpp Inkling limitation.
