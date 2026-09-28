#include "session_impl.h"
#include "tensor_overrides.h"

#ifdef BMOE_HAVE_WEIGHT_READY_HOOK
#include "ggml-cpu.h"
#endif

#include <algorithm>

namespace meitte {

#ifdef BMOE_HAVE_WEIGHT_READY_HOOK
std::atomic<bool> detail::dense_weight_hook_active{false};
#endif

Session::Impl::~Impl() {
    // Deterministic teardown order: stop the I/O pool (it holds fds into the mmap and its
    // buffers back the rebound expert tensors), then the context (its eval callback points
    // at the hook), then the hook, then unmap the model, then release the backend.
    source.shutdown();
#ifdef BMOE_HAVE_WEIGHT_READY_HOOK
    if (weight_hook_registered) {
        ggml_cpu_set_weight_ready_hook(nullptr, nullptr);
        detail::dense_weight_hook_active.store(false, std::memory_order_release);
    }
#endif
    dense_stream.shutdown();
    dense_fixed.shutdown();
    if (smpl) llama_sampler_free(smpl); // independent of ctx/model; free before them
    // The speculative driver holds both contexts and detaches the backend samplers it
    // installed on the draft one, so it goes before either context is freed.
    mtp.reset();
    if (mtp_batch_owned) llama_batch_free(mtp_batch);
    ctx_dft.reset();
    mtmd.reset();
    ctx.reset();
    hook.reset();
    model.reset();
    mapping_placeholders.release();
    if (backend_inited) llama_backend_free();
}

bool Session::Impl::abort_requested(void * impl) {
    auto * p = static_cast<Impl *>(impl);
    return p->capture_abort.load(std::memory_order_acquire) || p->cancel_requested.load(std::memory_order_relaxed) ||
           p->hook->fatal() || p->source.fatal() || p->dense_stream.fatal();
}

void Session::Impl::attach_context(llama_context * c) {
    llama_set_n_threads(c, cfg.n_threads, cfg.n_threads);
    llama_set_abort_callback(c, &Impl::abort_requested, this);
}

void Session::Impl::trace_begin(int base_pos, int n_tokens, int phase, MediaKind media_kind) {
    // Not a trace concern, but the same per-decode frame: the drop policy is decode-only
    // unless armed for prefill, so it has to be told which phase this batch is.
    hook->set_batch_phase(phase);
    hook->begin_graph();
    if (route_trace) hook->begin_trace_batch(base_pos, n_tokens, phase, turn, static_cast<uint8_t>(media_kind));
    // A node is computed once for the whole batch, not per token, so a prefill chunk's graph is
    // attributed to its last position rather than pretending to split across the chunk.
    if (compute_trace)
        hook->begin_compute_batch(base_pos + n_tokens - 1, phase, turn, static_cast<uint8_t>(media_kind));
    trace_phase = phase;
    trace_step = base_pos + n_tokens - 1;
    trace_media_kind = static_cast<uint8_t>(media_kind);
}

void Session::Impl::trace_flush(const llama_batch * media_batch) {
    // Close the compute trace's dangling interval FIRST: at layer granularity the "post" row
    // is charged the wall since the last boundary, and everything trace_flush does before the
    // close would be billed to the LM head.
    if (compute_trace) hook->end_compute_batch();
    if (route_trace) {
        hook->end_trace_batch();
        std::vector<RouteTraceRow> & rows = hook->trace_rows();
        if (media_batch)
            for (auto & row : rows) {
                int index = row.step - (trace_step - media_batch->n_tokens + 1);
                row.step = index >= 0 && index < media_batch->n_tokens ? media_batch->pos[index] : -1;
            }
        if (!rows.empty()) route_trace->on_rows(rows.data(), rows.size());
        rows.clear();
    }
    if (compute_trace) {
        std::vector<ComputeTraceRow> & rows = hook->compute_rows();
        if (media_batch)
            for (auto & row : rows)
                row.step = media_batch->pos[media_batch->n_tokens - 1];
        if (!rows.empty()) compute_trace->on_rows(rows.data(), rows.size());
        rows.clear();
    }
    if (io_trace) {
        // The reads carry no frame of their own — a lane does not know which token it serves —
        // so stamp them with the decode they were drained after.
        source.take_io_trace_rows(io_rows_scratch);
        for (IoTraceRow & r : io_rows_scratch) {
            r.turn = turn;
            r.phase = trace_phase;
            r.media_kind = trace_media_kind;
            r.step = media_batch ? media_batch->pos[media_batch->n_tokens - 1] : trace_step;
        }
        if (!io_rows_scratch.empty()) io_trace->on_rows(io_rows_scratch.data(), io_rows_scratch.size());
        io_rows_scratch.clear();
    }
}

Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

std::vector<std::string> Session::available_tensor_buffer_types() {
    return meitte::available_tensor_buffer_types();
}

double Session::load_seconds() const {
    return impl_->load_seconds;
}
const std::string & Session::arch() const {
    return impl_->arch;
}
int Session::n_ctx() const {
    return impl_->cfg.n_ctx;
}
int Session::n_expert_used() const {
    return impl_->n_expert_used;
}
std::vector<float> Session::copy_logits() const {
    const float * row = impl_->ctx ? llama_get_logits_ith(impl_->ctx.get(), -1) : nullptr;
    return row ? std::vector<float>(row, row + impl_->n_vocab) : std::vector<float>();
}
ThinkControl Session::think_control() const {
    return impl_->think_ctl;
}
void Session::set_cache_budget_mb(int mib) {
    impl_->source.set_cache_budget((size_t) std::max(0, mib) * 1024ull * 1024ull);
}

void Session::cancel() {
    impl_->cancel_requested.store(true, std::memory_order_relaxed);
    impl_->dense_stream.notify_cancel();
}

} // namespace meitte
