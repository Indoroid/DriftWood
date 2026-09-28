# Limitations and prior art

## Prior art

BigMoeOnEdge is an engineering package, not a new technique. The ideas it combines:

- **AirLLM** — layer-by-layer streaming of >RAM models from disk.
- **Apple, "LLM in a flash"** — flash-aware weight streaming, windowing, sparsity-driven
  loading.
- **FlexGen** — offloading and I/O-bound throughput scheduling for large models.
- **PowerInfer / EdgeMoE** — hot/cold expert locality and expert-granularity residency on
  the edge.

The serial expert and dense streaming paths use llama.cpp's public API. Optional CPU overlap
uses expert and matrix readiness callbacks on the synced fork. See [seam.md § 4](seam.md).

## Limitations

- **Dense streaming is a whole-matrix path validated for text.** It accepts GGUF models with zero experts
  when capture finds native file-backed `MUL_MAT` matrix leaves. It reads the streamed matrix set
  on each token, even if only a few rows would suffice. A larger fixed set lowers reads at the cost
  of more RAM. On the 14 GiB host, Qwen3.8 27B Q4_0 completed three-token serial and
  overlap text runs with a 2 GiB fixed set and 1 GiB window; a full mmap decode had previously
  exhausted memory. A two-token NVMe-counter A/B on the same host measured 12.7 GiB of device reads
  per generated token and 0.223 tokens/s with a 2 GiB fixed set. With a 10 GiB fixed set, it read
  4.7 GiB per token at 0.474 tokens/s; peak RSS was 11.9 GiB and the process did not swap. A cold
  `llama_model_load` read 13.6 GiB through the model mapping before dense capture. These are short
  tuning runs, not sustained benchmarks. No multimodal projector was available for this model. The
  remedy for larger gathered tensors is a proven row-only capture policy, not a model-name exception.
- **Dense mode does not use an MTP draft context.** The trunk capture does not classify the
  separate next-token prediction graph or guarantee its matrix lifetimes; `--mtp` therefore fails
  at open with a clear error. N-gram drafting has no separate graph.
- **Overlap is limited to one active session per process, per hook.** The synced CPU expert-ready
  and weight-ready hooks each have one process-wide callback and user pointer. A second expert
  overlap session, or a second dense overlap session, fails at open; serial sessions remain
  available beside it. A per-context callback upstream would remove this limit.
- **Dense window and auto budget are conservative.** A two-adjacent-layer window is required by
  one-layer lookahead, and initialization reports the minimum when the configured window is too
  small. The automatic fixed set reserves 7 GiB plus the I/O window from `MemAvailable`; actual
  context, batch and OS pressure vary, so peak RSS and faults must be checked for each deployment.
- **Unified KV is a libllama layout option, not server concurrency.** `--kv-unified` works in the
  core, CLI, and server and is passed directly to libllama. Meitte still owns one sequence and the
  server still processes one conversation at a time, so the flag does not provide independent
  concurrent conversations. `--no-kv-unified` remains the default.
- **MTP cannot continue from media embeddings.** A request with current or preserved media and an
  MTP draft source fails before prefill. Upstream `common_speculative_process` handles token batches
  but does not forward `batch.embd` to the draft context. N-gram drafting works because it needs only
  confirmed text IDs. The remedy is an upstream mixed token/embedding speculative API, not a local
  copy of its draft driver. Checked against upstream master 6c7a87f (2026-09-28): `llama.h` now
  has `llama_batch_ext` (mixed token and embedding batches) and `llama_process`, but
  `common_speculative_process` still takes a plain `llama_batch`, so a pin update alone would not
  lift this. Lifting it needs that API plus a CPU multimodal + MTP test.
- **Video input is bounded sampled prefill.** It requires an mtmd vision projector, an mtmd build
  with video enabled, and FFmpeg/ffprobe. The default is 1 frame/s and 32 frames. Live video and
  soundtrack extraction are not supported. The Unix-only FFmpeg audio fallback is also absent on
  Windows; formats decoded directly by mtmd still work there.
- **Context summarization and trimming lose information.** They are opt-in core policies, protect
  the newest and media-bearing turns, and report every action in `RunResult::context_events`.
  The CLI and server expose them, but keep them off by default. Without an enabled recovery policy,
  overflow remains a non-fatal error.
- **Computed row indices add synchronization.** `--row-stream` isolates `I32` graph producers when
  a qualifying table uses computed indices. This is generic and preserves the mainline graph, but
  its cost has not been measured on device. The option remains off by default.

- **Two settings make output non-reproducible.** Every other knob is deterministic given a
  configuration: `--n-expert-used` changes the output, but changes it the same way on every run.
  [`--drop-cold-experts`](expert-dropping.md) and
  [`--expert-substitute`](cache-aware-substitution.md) decide per routing from live cache state,
  so the same prompt and the same flags can decode differently run to run, and the byte-identity
  gates cannot cover their output — only their machinery. Both off by default in the CLI, and
  neither can be priced by a wide scoring batch: use `--ppl --ppl-step`.
- **n=1 only.** The expert sparsity exists only for single-token decode, so streaming is
  incompatible with speculative decoding or batching. Prefill streams the union of the
  prompt's routed experts (still far below the full bank, but larger than one token's).
- **CPU execution only.** Models, streamed experts, projectors, and draft contexts use CPU buffers.
  GPU offload and iOS targets are outside the current plan.
- **Shared experts stay resident.** Architectures with an always-on shared expert (e.g.
  `gemma4`, `deepseek4`, `glm-dsa`, `glm5next`, `nemotron_h_moe`, `cohere2moe`, `hy_v3`,
  `hy_v4`, and `minimax-m3`) stream the routed experts but
  keep the shared expert — and any dense layers — resident (in the page cache, or in the
  engine's own buffers under `--dense-weights anon`),
  so the streamed fraction (and the memory saving) is smaller than for a purely routed model
  like `qwen3moe`. The same applies to architectures whose first blocks are dense by design
  (`lfm2moe` has a `leading_dense_block_count`): those blocks name no expert tensors, so they
  are never streamed.
- **Mixed expert layouts within one GGUF are unsupported.** llama.cpp accepts fused gate/up
  or separate gate/up expert tensors for `cohere2moe` and `hy_v3`. Meitte selects one layout
  per model file and rejects a file that mixes them across layers. The pinned converter emits
  split tensors for these families; supporting mixed files would require per-layer recipes.
- **Inkling has no loader in the pinned llama.cpp.** The [draft upstream PR](https://github.com/ggml-org/llama.cpp/pull/25731)
  adds an Inkling loader, converter, and new GGML attention operator, none of which are in
  this submodule. Meitte cannot load or stream an Inkling GGUF until those llama.cpp changes
  land in a compatible dependency; a streaming recipe can then be validated against its
  expert tensor layout.
- **A resident tensor can be larger than RAM, and then it is only ever mmap'd.** `qwen4exp`
  (Qwen3.8-Flash-Next) carries a 51B n-gram embedding table (`per_layer_token_embd`, ~28.8 GB at
  IQ4_NL) that the graph reads sixteen rows at a time through `get_rows`. It is not indexed by
  expert, so the streamer does not bind it, and it is bigger than any phone's memory, so no dense
  policy can make it resident: such a tensor stays mmap'd whatever `--dense-weights` asks, and
  the engine says so at load. The streamed fraction of this architecture is therefore unusually
  low, and its per-token cost on that table is page faults on kilobyte reads rather than
  streamed expert bytes. That cost has not been measured on a device yet.
- **Streaming does not help a model that fits.** The engine's reason to exist is a model
  larger than RAM. Registering an architecture says the layout streams losslessly, not that
  streaming is the fast way to run every model using it — a small MoE that fits in memory is
  faster loaded resident, and the registry rows are about coverage, not a recommendation.
- **Repack must stay off.** Loading uses `use_extra_bufts=false`; you cannot combine
  streaming with weight repacking.
- **macOS reads uncached, and `o_direct` says so — but it is not `O_DIRECT`.** A direct request on
  Apple is served by `fcntl(F_NOCACHE)` on each descriptor: the kernel stops caching that file's
  pages, which is the property the design wants. It is a caching hint, not an I/O mode — no
  alignment contract, no DMA promise — so reads keep ordinary `pread` semantics and the raw-read
  ceiling can sit below Linux `O_DIRECT`'s. Measured on one 16 GB Apple-silicon Mac with the model
  on an external volume (256-token protocol, buffered vs `F_NOCACHE`, interleaved A B B A A B):
  decode 0.94 vs 0.61 tok/s (+56 %, arm ranges non-overlapping), with the byte stream
  bit-identical between arms (40 822.8 MiB read by both) and cache-hit equal — the gain is not
  fewer reads but the buffered arm's page-cache pollution doubling the compute residual
  (1.22 → 0.60 s/tok) while stall stays flat (0.43 → 0.46 s/tok). The `o_direct` field records
  the open's real outcome on every platform, so a refused or downgraded run reports `0`.
- **No iOS target.** The core is portable C++ and the streaming path has no Android dependency, but
  there is no Xcode project here and iOS does not run command-line binaries, so there is no
  supported way to run or benchmark the engine on an iPhone or iPad.
- **Windows throughput.** The cache's reserve-then-commit-per-slice path is heavier on
  Windows than the POSIX lazy-commit path. The gates run on Windows; the throughput
  targets are stated for Android/Linux.
- **Depends on a ggml scheduling behaviour** (documented in [seam.md](seam.md)) that is
  not a stability-guaranteed contract. Re-verified by the gates on each submodule bump.
- **C ABI struct growth.** `meitte_default_config()` and `meitte_default_request()` return their
  struct by value, so a caller compiled against a shorter header reserves less than a newer library
  writes. `meitte_config_init()` / `meitte_request_init()` take the caller's size and are the safe
  path; `meitte_request` must not grow until hosts use them (the bundled `ctypes_smoke.py` still
  uses the by-value form). `meitte_media` is an array element with no size field, so its layout is
  frozen: new media metadata needs a new request field that states its element size. The layout
  published before the dense-stream fields left `dense_stream` in its tail padding; the library
  treats that exact `struct_size` as the older layout (`tests/library_abi_test.c`).
- **The C ABI covers a subset of the C++ Session.** It carries a prompt, media, an output limit and
  `clear_kv`, and returns text, reasoning text, an error and a cancelled flag. Structured messages
  and content parts, tools and tool calls, thinking controls, reasoning effort and budget, request
  sampling, template kwargs, finish reasons, context events, summary metrics, perplexity, cache
  controls and capability queries exist only in C++ (`bmoe/session.h`); hosts that need them use
  the C++ API or the CLI/server protocols for now.
- **Hybrid and recurrent models rebuild the KV after a rolled-back turn.** Trigger: a cancelled or
  failed chat turn on a model with recurrent state (`qwen35`, `qwen35moe` and other hybrid
  attention/SSM models). Cause: llama.cpp's recurrent memory cannot remove positions from its
  state, so the rollback clears the KV with its records and the next turn prefills the whole
  transcript in one pass. Evidence: Qwen3.8-27B, turn 3 cancelled after 5 tokens: the retry
  prefilled 83 tokens instead of 26, and its text starts like the uninterrupted turn but is not
  bit-identical over the turn, because one prefill rounds differently from the incremental path.
  Affected: prefill time of the retry; the output stays coherent.
  Evaluated and not adopted: recurrent-state snapshots for chat sessions
  (`llama_context_params::n_rs_seq >= 1`). A snapshot holds the state after one of the last
  `n_rs_seq` tokens of the last decoded ubatch, and a one-token decode step writes only the newest
  slot. A rollback to the start of a turn crosses the prefill and every decode step, so no snapshot
  holds the state before the turn. llama.cpp still accepts any removal of up to `n_rs_seq` positions
  and restores the wrong slot. Measured with libllama on both models (7 positions removed across
  four ubatches, `n_rs_seq = 8`, then one decode): the logits differ from a directly built state by
  up to 7.2 (Qwen3.6-35B-A3B) and 10.1 (Qwen3.8-27B) and the top token changes; a removal inside
  one ubatch differs by 0.53 and 0.36, the same as splitting the prefill (0.48, 0.41). Cost: one
  snapshot is 62.8 MiB on Qwen3.6-35B-A3B and 149.6 MiB on Qwen3.8-27B (`n_rs_seq = 8`: 565 and
  1347 MiB of recurrent state). Speculation requests `n_rs_seq = draft_max` for its verify
  rollback, so the engine uses the snapshots only to trim inside the last verify batch; every other
  removal on such a context takes the clear-and-rebuild path (`tests/hybrid_rollback_test.cpp`).
  Possible remedies: a partial-state checkpoint of the turn start through the public
  `llama_state_seq_get_data_ext`/`llama_state_seq_set_data_ext` with
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (one recurrent state per checkpoint, sizes as above), or an
  upstream `llama_memory_seq_rm` that refuses a snapshot rollback deeper than the last ubatch.
- **Hybrid and recurrent models re-prefill every continued turn.** Prefix reuse trims the KV back to
  the common prefix, and recurrent state cannot drop a partial tail, so on such models (qwen35moe
  among them) the trim is refused, the KV is cleared with its records, and the whole transcript is
  prefilled again. The output is exact (verified with cancel and retry on Qwen3.6-35B-A3B); the
  cost is prefill time per turn.
- **Raw-mode KV continuation is physical only.** With the chat template off, `clear_kv=false`
  decodes the new prompt after whatever the KV holds (llama.cpp infers the positions), but the
  turn's position bookkeeping starts at 0: `RunSummary::n_past`, the context-capacity check and
  speculative batch positions ignore earlier raw turns, and a rollback clears the whole KV. Chat
  mode tracks the KV prefix exactly; raw continuation should be treated as best effort.

## Not goals

- Distributing a model across devices (a different axis).
- Beating a model that already fits in RAM — if it fits, run it resident.
