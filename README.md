# Meitte

Meitte runs large GGUF models locally when the weights do not fit in RAM. It builds on llama.cpp
and streams either the experts selected by a Mixture-of-Experts (MoE) model or whole matrices of a
dense model from local storage, while keeping the native GGUF layout intact.

Meitte provides both a command-line runner and an OpenAI-compatible local server. It also supports
multimodal models, persistent conversations, KV-cache configuration, custom chat templates, and
runtime telemetry.

## What It Solves

An MoE model may contain far more parameters than it uses for any one token. Conventional loading
still requires the full expert set to reside in memory. That is a poor fit for local systems with
limited RAM, even when their storage is fast enough to supply the few selected experts on demand.

Meitte treats local storage as another level of the memory hierarchy. It is intended for CPU-first
local inference where an MoE model exceeds available RAM but its active experts can be read and
reused efficiently.

This is not a way to make every model fast on every device. If the whole model fits comfortably in
RAM, normal llama.cpp loading is usually simpler. Meitte is useful when expert residency is the
constraint and storage latency is an acceptable tradeoff.

## How It Works

1. The model is loaded through llama.cpp's public APIs with its native GGUF tensor layout.
2. Meitte discovers the expert tensor offsets and captures the live expert tensors during prompt
   evaluation.
3. When llama.cpp's router selects experts for a token, Meitte reads only those expert slices from
   storage into a bounded cache and rebinds the matching tensor data.
4. Frequently used experts remain cached; dense weights use the configured mmap, warm, or anonymous
   residency policy.
5. llama.cpp evaluates the resulting graph normally, so model loading, tokenization, sampling, and
   model architecture support stay on the upstream path.

The streaming path does not repack GGUF files or invent a second model format. Repacking increases
storage and breaks the native-layout invariant that makes direct expert reads possible. Tensor
overrides, when used, are limited to fully resident tensors rather than the streamed expert path.

## Capabilities

- `meitte-cli` for local interactive and one-shot inference.
- `meitte-server` for OpenAI-compatible completions and chat-completions endpoints.
- Optional `libmeitte` C ABI for FFI callers; a Python `ctypes` smoke example is included.
- MoE expert streaming with direct I/O where the platform supports it, bounded expert caching, and
  optional cache-aware routing.
- CPU dense-matrix streaming for zero-expert models, with a fixed resident set and bounded layer
  window; optional matrix-read overlap uses the synced llama.cpp CPU readiness callback.
- Optional model-mapping release after a safe streamed load. This restores concurrent unbuffered
  read scaling on Windows and also fixes tied output heads under anonymous dense-weight placement.
- Image, audio, and sampled-video prompts with an `--mmproj` projector for supported models.
- Persistent server sessions, custom Jinja chat templates, and configurable KV-cache types and layout.
- Reasoning controls, reasoning budgets, and preserved reasoning/KV state where the selected model
  and template support them.
- Per-token progress, CSV metrics, route traces, and compute diagnostics for measuring the I/O and
  cache tradeoffs instead of guessing at them.

## Supported Architectures

The following GGUF `general.architecture` values are supported for expert streaming. Run
`meitte-cli --list-archs` to print the list from the built binary.

| Architecture ID | Model family |
| --- | --- |
| `qwen3moe` | Qwen3 MoE |
| `qwen2moe` | Qwen2 MoE |
| `qwen35moe` | Qwen3.5 MoE |
| `glm-dsa` | GLM-5.2 and GLM-5.3 |
| `glm5next` | GLM-5.3 Flash |
| `nemotron_h_moe` | Nemotron 3, 3.5, and Nemotron-H MoE |
| `gemma4` | Gemma 4 MoE |
| `gpt-oss` | OpenAI gpt-oss |
| `lfm2moe` | Liquid AI LFM2 and LFM2.5 MoE |
| `deepseek4` | DeepSeek V4 Flash |
| `bailingmoe3` | Ling 3.0 |
| `qwen4exp` | Qwen3.8 Flash-Next / Qwen4 preview |
| `cohere2moe` | Cohere North MoE |
| `hunyuan-moe` | Hunyuan MoE |
| `inkling` | Thinking Machines Inkling |
| `hy_v3` | Hunyuan HY V3 |
| `hy_v4` | Hunyuan HY V4 |
| `minimax-m3` | MiniMax M3 |
| `mimo2` | MiMo V2 Flash and V2.5 series |

DeepSeek V4 Flash Vision Exp uses the existing `deepseek4` text architecture and a
separate `deepseek4v` vision projector. Cohere2MoE and HY V3 accept split or fused
gate/up expert tensors; Meitte selects the layout from the GGUF.

The registry is intentionally explicit: a model is supported only when its expert tensor layout is
known to preserve the native GGUF streaming invariant. See `docs/adding-a-model.md` for adding a
new architecture safely.

A model without experts runs from the plain mmap load; there is no dense streaming mode.

## Quick Start

Initialize the llama.cpp submodule and build the host binaries:

```bash
git submodule update --init --recursive
scripts/build-host.sh
```

Run a streamed model from the CLI:

```bash
build/cli/meitte-cli \
  -m /path/to/model.gguf \
  --moe-stream --cache-mb auto \
  --chatml -p "Explain what a mixture-of-experts model is."
```

Start an OpenAI-compatible server:

```bash
build/cli/meitte-server \
  -m /path/to/model.gguf \
  --alias local-model --kv-unified \
  --moe-stream --cache-mb auto \
  --host 127.0.0.1 --port 8080
```

`--alias NAME` sets the model ID returned by `/v1/models` and completion responses. KV layout is
passed directly to libllama: `--kv-unified` enables its unified cache, while
`--no-kv-unified` selects the default separate-cache layout.

Context recovery is opt-in. For example,
`--dynamic-min-ctx 2048 --ctx-size 4096 --dynamic-ctx auto --dynamic-max-ctx 8192` grows the opening context
only when a request needs it, up to the final context opened with the configured RoPE/YaRN values.
`--summarize-history auto` and `--trim-old auto` add lossy recovery for older complete chat
turns.

For a supported multimodal model, supply its projector:

```bash
build/cli/meitte-server \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --moe-stream --cache-mb auto --port 8080
```

The model architecture, available storage bandwidth, dense-weight policy, context size, and expert
cache budget all affect results. Start with `--cache-mb auto`, collect telemetry with `--progress`,
then tune from measured cache hit rate and I/O time.

On Windows, add `--release-mmap` with `--dense-weights anon` (or `ahwb` on Android) to release the
GGUF mapping after load and reopen the expert-read lanes. The core checks observed weight pointers
and declines the release when any still use the mapping. The option is off by default and is
available in `meitte-cli`, `meitte-server`, the C++ configuration, and the append-only C ABI.

## Build And Test

```bash
scripts/build-host.sh
cd build
ctest --output-on-failure
```

The `bmoe_moe_gates` tests verify that streamed experts produce byte-identical results to resident
experts. Run them after changing the streamer, the llama.cpp seam, or model recipes.
The `dense_*` tests check stable backing and byte-identical logits for serial and overlap paths.

## Documentation

- [Architecture](docs/architecture.md): component boundaries and the loading/evaluation path.
- [llama.cpp seam](docs/seam.md): the public API boundary and supported integration points.
- [Session behavior](docs/session.md): persistent context, templates, KV cache, and reasoning.
- [Multimodal usage](docs/multimodal.md): projector setup and request formats.
- [Telemetry](docs/telemetry.md): progress output, CSV fields, and diagnostic traces.
- [Adding a model](docs/adding-a-model.md): architecture recipes and validation expectations.
- [Limitations](docs/limitations.md): workload and platform constraints.
- [BigMoeOnEdge sync ledger](docs/bigmoe-sync.md): the upstream delta ported into Meitte and the
  Meitte adapters that expose it.

## Credits

Meitte is built on [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) by
[Helldez](https://github.com/Helldez). The original project established the expert-streaming design
and llama.cpp integration that this project preserves.

Meitte also relies on [llama.cpp](https://github.com/ggml-org/llama.cpp), GGUF, and the research and
engineering work behind storage-backed LLM inference, including AirLLM, LLM in a Flash, FlexGen,
PowerInfer, EdgeMoE, and flash-moe.

## License

Apache-2.0. See [LICENSE](LICENSE).
