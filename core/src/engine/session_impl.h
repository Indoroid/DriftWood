#pragma once

// Session::Impl — the native state behind the Session pimpl, shared by the session units:
//   session.cpp            lifetime, accessors, cancel, abort predicate, trace framing
//   session_open.cpp       model load, context creation, capture and streamer binding
//   session_generate.cpp   prompt rendering, prefill, decode/speculation loop, summary
//   session_perplexity.cpp teacher-forced scoring
//   session_context.cpp    context growth, summarization and the MTP draft context
//
// Internal header: it includes llama.cpp's `common` layer (not the stable public API), so it must not
// be pulled into core/include/bmoe/. See docs/seam.md.

#include "bmoe/decode_trace.h"
#include "bmoe/recipe.h"
#include "bmoe/route_trace.h"
#include "bmoe/session.h"
#include "run_tally.h"
#include "../io/mapping_release.h"
#include "../moe/dense_weights.h"
#include "../moe/expert_stream_source.h"
#include "../moe/gguf_offsets.h"
#include "../moe/router_hook.h"
#include "../multimodal/mtmd_runtime.h"

#include "llama.h"

#include "chat.h"
#include "speculative.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace meitte {

namespace detail {

using ContextPtr = std::unique_ptr<llama_context, void (*)(llama_context *)>;

// Create the MTP draft context: the same model as the target, with ctx_type = MTP so llama.cpp
// builds the nextn graph. `target` is the target context's parameters. Returns nullptr on failure.
llama_context *
create_mtp_context(llama_model * model, const llama_context_params & target, const SpecConfig & spec, int n_threads);

// Create llama.cpp's speculative driver over a target/draft context pair. Returns nullptr on failure.
common_speculative * create_mtp_driver(const SpecConfig & spec, llama_context * target, llama_context * draft);

// Remove positions [keep, end) of sequence 0. False when the memory cannot keep that prefix — the
// recurrent and hybrid caches refuse a partial removal without a snapshot — and then the memory is
// unchanged. Every partial KV removal of the session goes through here (see session_testing.h).
bool kv_remove_tail(llama_context * ctx, llama_pos keep);

} // namespace detail

// All native state lives here, behind the pimpl. Built once by Session::open(); every
// generate() reuses the loaded model, the live context and the warm expert cache.
struct Session::Impl {
    SessionConfig cfg;
    std::string arch;
    double load_seconds = 0.0;

    // Ownership order matters at teardown: the source's I/O pool holds fds into the mmap'd
    // file and its buffers back the rebound expert tensors, so it must be shut down before
    // the context and model are freed. The destructor does that explicitly.
    std::unique_ptr<llama_model, void (*)(llama_model *)> model{nullptr, llama_model_free};
    detail::ContextPtr ctx{nullptr, llama_free};
    MtmdRuntime mtmd;
    llama_context_params context_params{};
    std::unique_ptr<RouterHook> hook; // heap: its address is baked into cparams.cb_eval_user_data
    ExpertStreamSource source;
    std::atomic<bool> capture_abort{false};

    // The MTP draft source (SpecConfig::source == mtp). A SECOND context over the SAME model,
    // created with ctx_type = MTP so llama.cpp builds the nextn graph instead of the trunk one. It
    // carries the same eval callback as the target — the hook is per-context, not per-model — so the
    // MTP block's expert layer is captured and streamed by the one source both contexts share.
    //
    // The n-gram source has no equivalent state: it drafts from the token history alone, which is
    // why it costs no context, no memory and no expert read. Everything below this pair is shared by
    // both sources, because the verify half of the loop does not care who drafted.
    detail::ContextPtr ctx_dft{nullptr, llama_free};
    common_speculative_ptr mtp;
    // Serves both roles on the speculative path, never both at once: a prefill chunk (up to
    // n_batch tokens, logits on the last) and a verify pass (1 + draft_max tokens, logits on all).
    llama_batch mtp_batch{};
    bool mtp_batch_owned = false; // whether mtp_batch holds a llama_batch_init allocation to free
    std::vector<llama_token> draft_buf;
    detail::SpecCounters spec_totals;

    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    int n_expert_used = 0; // effective routing width, after any override (0 = not MoE)
    int n_layer = 0;
    // Trained MTP heads in this gguf (0 = no nextn block). The MTP block occupies layer indices
    // [n_layer, n_layer + n_layer_nextn), contiguous with the trunk and using the same tensor
    // naming, so widening the streamer's layer bound by it is all the streamer needs to reach it.
    int n_layer_nextn = 0;
    bool chat_on = false;
    common_chat_templates_ptr chat_tmpls;
    // How a think=false request can be honoured on this model; probed once at open(). Template
    // (the fail-open default) means the flag alone does the job and generate() adds nothing.
    ThinkControl think_ctl = ThinkControl::Template;
    bool backend_inited = false;
    // The placeholders must outlive the model, whose teardown still releases the original ranges.
    pio::MappingPlaceholders mapping_placeholders;

    // Sampling chain, built once at open() only when sampling is requested (temp > 0); null on the
    // greedy default, where the decode loop stays on the argmax fast path. See open()/generate().
    llama_sampler * smpl = nullptr;

    // Multi-turn chat state (chat mode only). chat_history is the running conversation the
    // template is re-rendered over each turn; kv_tokens mirrors the tokens currently decoded
    // into the context's KV (seq 0), in order, so the next turn can reuse the common prefix
    // and prefill only the diverging suffix instead of re-running the whole conversation.
    std::vector<common_chat_msg> chat_history;
    std::vector<llama_token> kv_tokens;
    // Text tokens plus k_media_kv_sentinel entries form a logical transcript used to prove that a
    // preserved multimodal KV is extended rather than rewritten. kv_n_past stays physical: mtmd's
    // image spans occupy many decoder positions, never one sentinel position.
    bool kv_has_media = false;
    std::vector<MediaInput> retained_media;
    llama_pos kv_n_past = 0;
    size_t kv_last_generation_start = 0;

    // Route trace (diagnostics): null unless requested AND streaming is on — there is no routing
    // to trace otherwise.
    IRouteTraceSink * route_trace = nullptr;

    // Which generate() this is, 0-based. It labels a trace's rows, and it labels every token's
    // metrics — a multi-turn CSV is unreadable without it, and the two-turn A/B (a fast turn, an
    // idle, then the turn that pays for it) is exactly what this engine is measured by.
    int turn = 0;

    // Decode traces (diagnostics): null unless requested. The compute trace needs no streaming —
    // it measures the graph, which a dense mmap run has too; the I/O trace needs the streamer,
    // since there are no engine-issued reads without it. See bmoe/decode_trace.h.
    IComputeTraceSink * compute_trace = nullptr;
    IIoTraceSink * io_trace = nullptr;
    std::vector<IoTraceRow> io_rows_scratch;

    // The frame the I/O rows are stamped with at trace_flush(); the other traces carry their own.
    int trace_phase = 0;
    int trace_step = 0;
    uint8_t trace_media_kind = 0;

    // Stated once to a metrics sink, before the first token: the model and configuration every row
    // it writes was produced under. info_sent guards a session's many generate() calls from
    // interleaving preambles between turns.
    RunInfo info;
    bool info_sent = false;

    std::atomic<bool> cancel_requested{false};

    ~Impl();

    // The abort callback of every context this session decodes on. It checks an explicit cancel()
    // request (any mode), the dense capture abort, and a fatal streaming I/O error.
    static bool abort_requested(void * impl);
    // Give a new context the session's thread count and abort callback.
    void attach_context(llama_context * c);

    // Frame one decode: tell the hook the batch phase and open the trace rows. Traces are handed to
    // their sinks by trace_flush() once the decode has returned — never from inside the callback,
    // which runs on a compute thread mid-graph. `base_pos` is the context position of the batch's
    // first token, so prefill rows carry real step numbers.
    void trace_begin(int base_pos, int n_tokens, int phase, MediaKind media_kind = MediaKind::Auto);
    // `media_batch` maps the rows of an mtmd embedding batch back to its real positions.
    void trace_flush(const llama_batch * media_batch = nullptr);

    // Summarize history[first, end) into `output` on a temporary context. False on failure.
    bool summarize(const std::vector<common_chat_msg> & history,
                   size_t first,
                   size_t end,
                   std::string & output,
                   std::string & error);
    // Replace the target (and draft) context with ones of `size` positions. The KV is dropped.
    bool resize_context(int size, std::string & error);
    // The context size a growth step targets: at least `need`, at least double, at most max_ctx.
    int grown_context_size(int64_t need) const;

    // Keep target KV positions [0, keep_pos) and the first `keep_tokens` of kv_tokens, and move the
    // draft context to the same position. The physical KV and its logical records must never
    // disagree: when either context cannot keep the prefix, both are cleared together with
    // kv_tokens, kv_n_past and kv_last_generation_start. The next turn then rebuilds the conversation
    // from chat_history (and from retained_media for a media transcript, as after a context resize).
    // Returns false in that case.
    //
    // `in_last_decode` says that every removed position was decoded by the last llama_decode call
    // (the trim after a speculative verify batch). Only such a removal may use recurrent-state
    // snapshots; see the definition.
    bool truncate_kv(llama_pos keep_pos, size_t keep_tokens, bool in_last_decode);
    // Drop the KV, the conversation and the retained media: the state of a new chat.
    void reset_conversation();

    // Session::open() phases, in call order (session_open.cpp). Each returns false and sets `error`
    // on failure; the caller then destroys the Impl, which tears down whatever the phase built.
    bool load_model(LazyGgufMeta & meta, std::string & error);
    bool init_chat_templates(std::string & error);
    bool create_contexts(bool install_eval_callback, std::string & error);
    bool bind_expert_stream(const MoeRecipe & recipe,
                            int n_layer_streamed,
                            LazyGgufMeta & meta,
                            IRouteTraceSink * route_sink,
                            IIoTraceSink * io_sink,
                            std::string & error);
    void release_model_mapping(const GgufOffsets & offs, const std::vector<std::string> & unaccounted);
    void attach_decode_traces(IComputeTraceSink * compute_sink);
    void build_run_info(LazyGgufMeta & meta);
    void warn_about_load() const;
};

} // namespace meitte
