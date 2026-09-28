# Changelog

## [Unreleased]

- Docs: `docs/limitations.md` replaces "hybrid models re-prefill every continued turn" with the
  measured cause. Qwen3.6-35B-A3B's template drops the empty reasoning block from earlier assistant
  turns, so the common prefix ends inside the cache and recurrent state cannot drop the tail;
  Qwen3.8-27B's template keeps it and reuses the prefix. `--reasoning-preserve` avoids the full
  re-prefill on the former. No engine change.
- Fix continued chat turns and rollbacks on recurrent and hybrid models with speculation on
  (`--mtp`, `--ngram`) decoding from the recurrent state of a different position. Speculation
  creates the context with `n_rs_seq = draft_max` snapshots, which cover the last decoded ubatch
  only, but llama.cpp accepts any removal of up to `n_rs_seq` positions. A turn whose cached tail
  diverged by at most `draft_max` tokens (for example `--draft 8` on Qwen3.6-35B-A3B, whose
  template drops the empty reasoning block from history) kept a wrong state; turn 2 answered "Blue"
  instead of "Red". The engine now uses the snapshots only to trim inside the last verify batch
  and clears and rebuilds for every other removal. `tests/hybrid_rollback_test.cpp` runs when
  `BMOE_TEST_HYBRID_MODEL` is configured. Recurrent snapshots for plain chat rollback were
  evaluated and not adopted; `docs/limitations.md` records the measurements.
- Expert-stream and dense-weight load failures now reach the caller with their cause.
  `DenseWeights::init` and `ExpertStreamSource::init` return the reason instead of printing it, so
  `--dense-weights ahwb` off Android reports "expert stream source init failed: dense-weights init
  failed: --dense-weights ahwb needs reclaim-exempt memory …" in `RunResult::error`, the server
  and the C ABI; they used to get only "expert stream source init failed". Gate G16 checks it.
- `--moe-stream` on a model without experts now fails with "expert streaming (--moe-stream) needs
  a model with experts … use dense streaming (--dense-stream)". `Session::open` checks the GGUF
  expert count before the recipe lookup, which used to ask for a new recipe in
  `arch_registry.cpp`. `dense_gates` checks the message on the tiny dense models.
- `--dense-weights` (and its aliases `--no-warm-dense`, `--dense-odirect`) without `--moe-stream`
  is now rejected by `meitte-cli` and `meitte-server`, like `--row-stream` and `--route-ahead`. The
  expert streamer applies the policy, so the run used to accept the flag and keep the plain mmap
  load. `MoeStreamConfig::dense_weights_set` records an explicit choice, because `anon` is also the
  default; `validate()` rejects it, or any non-default mode, without `moe.enabled`.
  `scripts/test-lfm25-server.py` passes `--dense-weights mmap` only with `--moe-stream`.
- Audio on a projector without audio support now fails with "audio input requires a projector
  with audio support" and names what the projector supports. The audio decoder used to run with a
  sample rate of 0 and report "invalid audio input, sample rate, or byte limit". Images on a
  projector without vision support are checked the same way. `meitte-server` now marks
  `input_audio` parts as audio, so it reports the same error instead of "failed to decode
  image/audio". `tests/media_capability_test.cpp` runs when `BMOE_TEST_MM_MODEL` and
  `BMOE_TEST_MMPROJ` are configured.
- Fix `--mtp` aborting inside llama.cpp on every MTP model: the pinned llama.cpp loads the nextn
  tensors only when `llama_model_params::load_mtp` is set, which the engine never did. Verified on
  Qwen3.6-35B-A3B (UD-IQ1_M): 82 % draft acceptance, streamed output identical to resident.
- Fix `meitte-cli --session` hanging after `{"cmd":"close"}` while the client keeps stdin open: exit
  flushed stdio under stdin's lock, which the detached reader held in `getline`.
- `meitte-cli --session` no longer ends on any failure outside two matched error texts: `BMOE_ERROR`
  `fatal` now follows the engine, so a refused or rolled-back request leaves the session serving.
  `BMOE_DONE` adds `finish`.
- `meitte-server` derives `finish_reason` from the engine (it compared token counts, which a
  reasoning budget breaks) and answers rejected requests with HTTP 400 instead of 500.
- `meitte-server --progress` lines now carry `dense_window_resident_frac` like `meitte-cli`'s: the
  two frontends kept separate copies of the progress emitter, which had drifted. The shared
  frontend helpers (progress line, JSON escaping, reasoning-effort normalization, template-kwargs
  parsing) now live once in `cli/frontend_util`.
- Report the turn outcome from the engine: `RunResult::finish` (stop, length, cancelled,
  context_full, error), `rejected` (refused before any state changed) and `fatal` (the session can
  no longer generate). Frontends previously inferred these from error text and token counts.
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
