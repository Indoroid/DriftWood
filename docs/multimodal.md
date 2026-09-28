# Multimodal input

Driftwood can combine a text GGUF with a llama.cpp mtmd projector. The projector handles image,
audio, or sampled-video preprocessing and prefill; token generation and MoE expert streaming continue through the
normal text-model path.

## meitte-cli

```bash
build/cli/meitte-cli -m model.gguf --mmproj mmproj.gguf \
  --image photo.png -p "Describe this image."
```

- `--image PATH`, `--audio PATH`, and the format-detecting `--media PATH` alias are repeatable.
- Projector work is CPU-only. `--mmproj-offload` is rejected.
- The projector must support the input: audio needs an audio-capable projector, images and video a
  vision-capable one. Otherwise the request fails before decoding with an error that names the
  missing capability (for example, audio on a vision-only projector).
- `--image-min-tokens` and `--image-max-tokens` override dynamic image-token bounds; `-1`
  retains projector metadata.
- `--mtmd-batch-max-tokens N` limits projector output per prefill batch.
- File-backed media is one-shot only. `--session` rejects these flags rather than ignoring them.
- `--media clip.mp4` uses upstream mtmd video support. The core samples at 1 frame per second and
  accepts at most 32 frames by default. FFmpeg and ffprobe must be on `PATH`; embedders can set
  `MultimodalConfig::ffmpeg_bin_dir`, `video_fps`, and `video_max_frames`.

## meitte-server

```bash
build/cli/meitte-server -m model.gguf --mmproj mmproj.gguf --port 8080
```

`POST /v1/chat/completions` accepts ordered OpenAI content parts:

```json
{
  "model": "model.gguf",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe: "},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

Audio uses `{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}`. Image URLs must
be base64 `data:` URLs: the local server never fetches remote resources. A decoded media item is
limited to 32 MiB, all decoded media in one request to 64 MiB, and the HTTP body defaults to a
64 MiB cap configurable with `--max-request-mb`.

## Current boundaries

- A request containing media starts from a clear KV cache. It can then be continued with text-only
  turns when the caller preserves the KV; a continuation cannot add or replace media, so reset with
  `clear_kv: true` before sending another image or audio input.
- `meitte-server --kv-preserve` retains one incremental conversation. After the image request, send
  only the next user message to `POST /v1/chat/completions`; `{"clear_kv":true}` begins a new
  conversation. This state is server-wide, not isolated per HTTP client.
- N-gram speculation works after media prefill and uses only the trailing text token history.
  MTP is rejected because upstream's speculative helper does not process embedding batches.
- Route, compute, and I/O tracing include multimodal prefill batches and use their decoder positions.
  Their append-only `media_kind` column is `0` for text/decode, `1` for image, `2` for audio,
  and `3` for video. Run summaries report media preparation (decode and tokenization) separately
  from projector embedding work; neither value includes target-model decode.
- Video is sampled input only. Live streams, soundtrack extraction, and frame resampling beyond the
  upstream helper are not implemented. Input and decoded media are bounded by the configured byte limit.
- On Unix, an audio buffer that mtmd cannot decode is converted to mono float PCM by a bounded,
  cancellable FFmpeg child process. Windows accepts formats decoded directly by mtmd but does not use
  this fallback yet.

These combinations fail with explicit errors. Text-only requests remain available while the
projector is loaded.
