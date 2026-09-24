// A persistent engine session: load the model and warm the expert cache ONCE, then serve
// many prompts against that resident state.
//
// run() (runtime.h) is the one-shot convenience — it opens a Session, generates once, and
// closes. Session exists for the interactive case: a fresh process per prompt re-pays the
// model load (tens of seconds) and the whole expert-cache warm-up ramp every time, which is
// exactly what streaming was meant to avoid. Keeping the process alive amortises both:
//
//   auto s = Session::open(cfg, err);          // model load + capture + source.init, once
//   s->generate({.prompt = "..."});            // prefill + decode; cache survives
//   s->generate({.prompt = "..."});            // starts warm — no reload, no cold ramp
//
// This header is pure policy (no llama.cpp types) — the native state lives behind a pimpl,
// so the CLI and tests link it without pulling in the backend headers.
#pragma once

#include "bmoe/config.h"
#include "bmoe/metrics.h"
#include "bmoe/runtime.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace meitte {

class IRouteTraceSink;
class IComputeTraceSink;
class IIoTraceSink;

// Everything fixed for the model's lifetime — set once at open(). n_ctx, n_batch and n_ubatch are
// baked into the llama context at creation and cannot change per prompt, so size them for the
// longest prompt+generation the session will serve.
struct SessionConfig {
    std::string model_path;
    int n_threads = 4;
    int n_ctx = 2048;
    ContextPolicy context;
    int n_batch = 2048; // prefill chunk capacity; capped at n_ctx in open()
    // Widest graph actually computed at once. 0 = follow n_batch. Sizing this down trades prefill
    // throughput for resident compute buffers, which on this engine compete with the expert cache.
    // See RunConfig::n_ubatch.
    int n_ubatch = 512;
    bool chatml = false;
    std::string chat_template;
    KvCacheType cache_type_k = KvCacheType::F16;
    KvCacheType cache_type_v = KvCacheType::F16;
    bool kv_unified = false;
    FlashAttentionMode flash_attention = FlashAttentionMode::Auto;
    RopeConfig rope;
    std::vector<TensorBufferOverride> tensor_buffer_overrides;
    // Active-expert (top-k) override applied at load via a kv_override on the arch-prefixed
    // expert_used_count key. 0 = use the model's own count. See RunConfig::n_expert_used.
    int n_expert_used = 0;
    // Compute-trace granularity: false = a barrier per graph node, true = per layer boundary.
    // Only read when a compute-trace sink is attached. See RunConfig::compute_trace_layers.
    bool compute_trace_layers = false;
    SamplingConfig sampling; // fixed for the session; greedy by default (temp <= 0)
    MoeStreamConfig moe;
    RunConfig::DenseStreamConfig dense_stream;
    // Self-speculative decoding, and which source drafts. Fixed for the session: it decides whether
    // open() builds the wider verify batch, and — for the MTP source only — the draft context.
    // See RunConfig::spec.
    SpecConfig spec;
    MultimodalConfig multimodal;
};

// The RunConfig → SessionConfig mapping, in one place. Both entry points that open a session from a
// RunConfig — run() and the CLI's interactive loop — need it, and they used to spell it out field by
// field. Two copies of a mapping is one copy too many: adding a field to RunConfig must not depend on
// remembering to touch both. Session::open() caps the configured batch sizes to the context.
SessionConfig session_config_from(const RunConfig & cfg);

// How a GenerateRequest::think=false request can be honoured on THIS model. Decided once at
// open() by rendering the model's own chat template, never from a list of model names.
//
// Turning thinking off is a request to the template, and a template is free to ignore it: the
// flag reaches the jinja context either way, so a model whose template never reads it reasons on
// regardless and nothing reports that the setting was dropped. Probing says which case a model is
// in, so a caller can stop offering a control that does nothing.
enum class ThinkControl {
    // Passing the flag is all there is to do — either the template reads it, or the model does not
    // reason at all and there is nothing to suppress. Both leave the engine with nothing to add.
    Template,
    Prefill, // the template ignores it; the turn starts past the reasoning section instead
    None,    // the model reasons on every turn and cannot be asked not to
};

// Stable lowercase name ("template", "prefill", "none") for logs and the telemetry protocol.
const char * think_control_name(ThinkControl c);

// Per-prompt request. clear_kv=true (the default) makes each prompt independent while the
// expert cache stays warm; clear_kv=false continues the KV cache for multi-turn chat.
enum class ChatContentKind {
    Text,
    Media,
};

struct ChatContentPart {
    ChatContentKind kind = ChatContentKind::Text;
    std::string text;
    // Media parts reference GenerateRequest::media by index. Media references must appear in
    // monotonically increasing order across the transcript so mtmd's marker order matches its
    // bitmap/audio array exactly.
    size_t media_index = 0;
};

struct ChatMessage {
    std::string role;
    std::string content;
    // Optional structured content. When non-empty, Session renders these parts instead of `content`,
    // replacing Media entries with the loaded projector's marker. This preserves OpenAI-style
    // interleaving such as text -> image -> text without leaking mtmd types through the public API.
    std::vector<ChatContentPart> content_parts;
    std::string reasoning_content;
    std::string tool_name;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
};

struct ChatTool {
    std::string name;
    std::string description;
    std::string parameters_json;
};

enum class ChatToolChoice {
    Auto,
    Required,
    None,
};

enum class MediaKind { Auto = 0, Image = 1, Audio = 2, Video = 3 };

struct MediaInput {
    std::vector<std::uint8_t> bytes;
    std::string name;
    MediaKind kind = MediaKind::Auto;
};

struct GenerateRequest {
    std::string prompt;
    // Optional complete chat transcript. When non-empty, the model chat template receives these
    // messages verbatim instead of wrapping prompt as one user turn. This keeps HTTP requests
    // stateless while preserving system and assistant messages supplied by OpenAI-style clients.
    std::vector<ChatMessage> messages;
    // Raw image, audio, or sampled-video bytes. Prompt must contain one projector marker per item; when callers
    // supply ordinary prompt/messages DriftWood inserts those markers into the user content.
    std::vector<MediaInput> media;
    std::vector<ChatTool> tools;
    ChatToolChoice tool_choice = ChatToolChoice::Auto;
    bool parallel_tool_calls = false;
    // Apply the model's chat template. SessionConfig::chatml still controls whether template
    // support is initialised at open(); this request switch lets a persistent HTTP session serve
    // both raw /v1/completions and templated /v1/chat/completions without loading the model twice.
    bool chatml = true;
    int n_predict = 32;
    bool think = true;
    std::string reasoning_effort;
    // -1 leaves reasoning unlimited. A non-negative value limits tokens in the template's
    // reasoning span without consuming the answer's n_predict allowance.
    int reasoning_budget_tokens = -1;
    // Values must be serialized JSON literals (for example, "true" or "\"fast\"") because
    // llama.cpp parses each value before exposing it to the Jinja context.
    std::map<std::string, std::string> chat_template_kwargs;
    bool clear_kv = true;
    // HTTP callers can override sampling per request without reopening the model. Ordinary callers
    // leave this false and retain the SessionConfig sampling chain.
    bool override_sampling = false;
    SamplingConfig sampling;
    // Populate TokenMetrics::text / ::reasoning on every token. Building them means parsing the
    // WHOLE generation so far — the chat parser cannot resume — so it is O(n) per token and O(n²)
    // over a turn, and in chat mode it also allocates a copy of everything generated. A UI that
    // renders a running answer needs it; a consumer that only concatenates `piece` (the CLI's
    // default output, and every benchmark run) does not, and should turn it off. Default on so an
    // embedder that does not know about this flag keeps the old behaviour.
    bool render_text = true;
};

// Teacher-forced quality measurement. The text is fixed, so every cell scores the SAME token
// sequence and no cell can fork onto its own trajectory — which is what makes the number a scale
// rather than a coin flip. Comparing generated text instead does not work: measured on this
// engine, capping one layer's routing to 1 expert of 8 left the greedy output byte-identical while
// capping it to 7 changed it, because greedy only moves when a perturbation happens to cross an
// argmax boundary, and whether it does is unrelated to how large the perturbation was.
struct PplResult {
    bool ok = false;
    std::string error;
    double nll = 0.0; // mean negative log-likelihood per scored token, in nats
    double ppl = 0.0; // exp(nll)
    int n_scored = 0; // tokens that contributed (the first `skip` and the last are not scored)
    // Scored positions where the model's most likely token WAS the one that follows. Perplexity is
    // the sensitive metric and this is the legible one: it says how often the model would have
    // written the text itself, which a reader can picture without knowing what a nat is.
    int n_top1 = 0;
    int n_tokens = 0; // tokens in the text
    // Log-probability of each PplRequest::choices entry (its first token) at the position after
    // the whole text, in the request's order. Empty unless choices were given. This is what a
    // multiple-choice benchmark compares: the text is the question with its options, and the
    // choices are the answer letters.
    std::vector<double> choice_logp;
    double seconds = 0.0;
    // What the routing policy actually did during the scoring pass. A lossy setting that reports
    // 0 touched slots was inert, and the perplexity it produced is the baseline's — which is
    // exactly how a measurement quietly says nothing.
    long long experts_routed = 0;
    long long experts_dropped = 0;
    long long experts_reranked = 0; // slots the substitution examined (its own denominator)
    long long experts_substituted = 0;
};

struct PplRequest {
    std::string text;
    // Tokens at the start of the text whose prediction is made from too little context to be
    // informative. They are evaluated (the KV needs them) but not scored.
    int skip = 8;
    // Score with the routing policies armed as they are during DECODE. A perplexity pass is one
    // wide batch, which the engine would otherwise label prefill — and the lossy routing policies
    // are decode-only by default, so the measurement would run with the very policy it is meant
    // to price switched off.
    bool as_decode = true;
    // Score one token per decode instead of the whole text in one batch. A wide batch routes
    // every position of a layer before any of it is read, so the cache holds whatever the previous
    // layers left and a policy that consults residency (dropping, substitution) barely fires: on a
    // 2000 MiB budget it touched 1-3% of the slots it examined, against 30% and more in
    // generation. Stepping reproduces the decode regime exactly — one token, a warm cache shaped
    // by the previous tokens of the same text — at the cost of one decode per token. The first
    // `skip` tokens are fed as one prefill batch, and are not scored either way.
    bool step = false;
    // Strings whose first token's log-probability is reported at the end of the text. The text is
    // fed in full (its last token too), so the reported distribution is the one a generation
    // would sample its first token from.
    std::vector<std::string> choices;
};

class Session {
public:
    ~Session();

    // Load the model, discover the MoE expert tensors, and initialise the expert source.
    // Returns nullptr and sets `error` on failure. The returned session owns all native
    // state and must outlive every generate() call.
    //
    // The trace sinks (all nullable) turn on their diagnostic for every generate() on this session
    // and must outlive it. None is ever on for a benchmark run — each perturbs what it measures.
    //   * route_trace   — per-step, per-layer routing; ignored when streaming is off (no routing to
    //                     observe). See bmoe/route_trace.h.
    //   * compute_trace — per-node compute and fault attribution. Works with or without streaming,
    //                     so a dense mmap baseline can be compared against a streamed run.
    //   * io_trace      — per-read flash latency/size; ignored when streaming is off (no reads).
    // See bmoe/decode_trace.h for the latter two.
    static std::unique_ptr<Session> open(const SessionConfig & cfg,
                                         std::string & error,
                                         IRouteTraceSink * route_trace = nullptr,
                                         IComputeTraceSink * compute_trace = nullptr,
                                         IIoTraceSink * io_trace = nullptr);

    // Buffer-type names currently registered by llama.cpp. This does not load a model and is used
    // by both binaries' --list-buffer-types option before they require --model.
    static std::vector<std::string> available_tensor_buffer_types();

    // Generate one response. Serialized: one generation at a time per session. `on_token`
    // and `sink` receive the same per-token metrics as run(). Cache state carries over from
    // the previous call. If cancel() fired, returns ok=true with cancelled=true.
    RunResult generate(const GenerateRequest & req,
                       const std::function<void(const TokenMetrics &)> & on_token = nullptr,
                       IMetricsSink * sink = nullptr);

    // Request the in-flight generation to stop. Thread-safe; may be called from another
    // thread while generate() runs. Takes effect at the next decode boundary via the abort
    // callback and leaves the model/cache intact for the next generate().
    void cancel();

    double load_seconds() const;      // model load + streaming setup, measured once at open()
    const std::string & arch() const; // model architecture ("qwen3moe", "gemma4", …)
    int n_ctx() const;

    // Experts this model actually routes per token, after any n_expert_used override — the width a
    // caller needs to interpret MoeStreamConfig::drop_cold_frac, whose threshold is a fraction of
    // 1/top-k. 0 when the model is not MoE or the count could not be read.
    int n_expert_used() const;

    // Which thinking-off mechanism this model supports (probed at open()). Report it to the user
    // rather than leaving a Thinking toggle that silently does nothing. Always Template when chat
    // mode is off, where no template is rendered and the question does not arise.
    ThinkControl think_control() const;

    // Set the expert-cache budget in MiB and evict down to it now. PRECONDITION: no generate() in
    // flight — call it between generations (e.g. from an app's memory-pressure callback). A no-op
    // when the cache is off. The only way the budget moves after init sizes it.
    void set_cache_budget_mb(int mib);

    // Score a fixed text under teacher forcing. Clears the KV; leaves no conversation state behind.
    PplResult perplexity(const PplRequest & req);

    // Owned copy of the last computed logits row for exact identity checks. Empty if no row exists.
    // Call between evaluations; the next decode replaces the backend row.
    std::vector<float> copy_logits() const;

private:
    Session();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace meitte
