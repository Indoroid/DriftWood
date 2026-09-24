# Changelog

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
