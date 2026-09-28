// RouterHook's look-ahead machinery: next-layer routing prediction and its accuracy probe
// (docs/expert-prediction.md), the prediction worker, and route-ahead (docs/route-ahead.md).
#include "router_hook.h"
#include "router_tensor.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>

namespace meitte {

using namespace routing;

// ── expert-prediction accuracy probe ────────────────────────────────────────────────────
//
// Observes only: it reads the gate matrix and the gate input, ranks experts from them, and
// compares the ranking with what the router went on to select. Nothing it computes reaches
// load_layer, the cache, or the graph, so a probed run routes and reads exactly as an unprobed
// one — it is only slower.

void RouterHook::set_predict_log(bool on) {
    predict_log_ = on;
    predict_reset();
}

void RouterHook::set_predict_prefetch(bool on, int spec_max) {
    pred_spec_max_ = spec_max >= 0 ? spec_max : 0;
    predict_prefetch_ = on;
    predict_reset();
    if (on && !pred_worker_.joinable()) pred_worker_ = std::thread([this] { predict_worker_main(); });
    if (!on) predict_worker_stop();
}

void RouterHook::set_route_ahead(int n) {
    route_ahead_ = n > 0 ? n : 0;
    const int nl = n_layer_ > 0 ? n_layer_ : 0;
    ra_pred_.assign((size_t) nl, std::vector<int32_t>{});
    ra_gate_q_.assign((size_t) nl, std::vector<int8_t>{});
    ra_gate_s_.assign((size_t) nl, std::vector<float>{});
    ra_overridden_ = ra_passthrough_ = ra_slots_ = ra_hits_ = 0;
    ra_tripped_ = false;
    ra_gemv_ns_.store(0);
    ra_gemv_jobs_.store(0);
    ra_issue_ns_ = 0;
    // The pointer stashes are normally sized by predict_reset(); route-ahead can be armed with
    // both probe and prefetch off, so make sure they exist rather than assume who ran first.
    if (gate_w_.size() != (size_t) nl) gate_w_.assign((size_t) nl, nullptr);
    if (h_t_.size() != (size_t) nl) h_t_.assign((size_t) nl, nullptr);
    // The GEMV runs on the shared prediction worker (predict_prefetch is mutually exclusive, so
    // the job slot is never contended); own the thread's lifecycle the same way it does.
    if (route_ahead_ > 0 && !pred_worker_.joinable()) pred_worker_ = std::thread([this] { predict_worker_main(); });
    if (route_ahead_ == 0 && !predict_prefetch_) predict_worker_stop();
}

void RouterHook::predict_worker_stop() {
    if (!pred_worker_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(pred_mtx_);
        pred_stop_ = true;
    }
    pred_cv_.notify_all();
    pred_worker_.join();
    pred_stop_ = false;
}

void RouterHook::predict_reset() {
    const int n = n_layer_ > 0 ? n_layer_ : 0;
    gate_w_.assign(n, nullptr);
    h_t_.assign(n, nullptr);
    pred_stale_.assign(n, std::vector<int32_t>{});
    pred_stale2_.assign(n, std::vector<int32_t>{});
    pred_self_.assign(n, std::vector<int32_t>{});
    ps_stale_.assign(n, PredictorStats{});
    ps_prev_.assign(n, PredictorStats{});
    ps_self_.assign(n, PredictorStats{});
    agg_stale_ = agg_stale2_ = agg_prev_ = agg_self_ = PredictorStats{};
    predict_unscored_ = 0;
    wd_slots_ = wd_hits_ = wd_rout_ = 0;
    wd_tripped_ = false;
    std::lock_guard<std::mutex> lk(pred_mtx_);
    pred_job_pending_ = false;
    pred_result_ = PredictResult{};
}

static bool gate_scores(const ggml_tensor * w, const std::vector<float> & h, std::vector<float> & out);

// The prediction worker: everything it touches is either job-local or a read-only weight leaf, so
// it needs no coordination with the graph, the source, or the LRU — the eval thread snapshots its
// inputs and consumes its outputs, and this thread only computes.
void RouterHook::predict_worker_main() {
    for (;;) {
        PredictJob job;
        {
            std::unique_lock<std::mutex> lk(pred_mtx_);
            pred_cv_.wait(lk, [&] { return pred_stop_ || pred_job_pending_; });
            if (pred_stop_) return;
            job = std::move(pred_job_);
            pred_job_pending_ = false;
        }
        std::vector<float> scores;
        // Time the GEMV itself (the ranking that follows is O(k*n_expert) and rides along): this
        // is the feature's own CPU bill, invisible in wall time wherever a spare core exists.
        const auto tg0 = std::chrono::steady_clock::now();
        // Commit jobs read the int8 mirror — a quarter of the bytes, which on a phone IS the cost:
        // the float GEMV there is bound by pulling ~4 MB of gate matrix per layer across a memory
        // bus the decode is already saturating. The probe's jobs always keep the exact weights,
        // because its whole purpose is to be a control. The mirror is built here, on this worker,
        // the first time a layer is predicted for.
        //
        // aarch64 only, and measured both ways: on x86 the float path is fully vectorized and DRAM
        // bandwidth is ample, so the int8 detour costs more than it saves (0.38 vs 0.26 ms per
        // GEMV on the host, same config). The bytes only matter where they are scarce.
        bool scored = false;
#if defined(__aarch64__)
        if (job.commit && job.nl >= 0 && quantize_gate(job.nl)) scored = gate_scores_q(job.nl, job.row, scores);
#endif
        if (!scored) scored = job.gate && gate_scores(job.gate, job.row, scores);
        if (job.commit) {
            ra_gemv_ns_.fetch_add(
                (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tg0)
                    .count());
            ra_gemv_jobs_.fetch_add(1);
        }
        if (!scored) continue;
        PredictResult r;
        r.seq = job.seq;
        r.nl = job.nl;
        r.commit = job.commit;
        if (job.commit) {
            // Rank exactly the routing's width, not the probe's max: rank_top_k is a partial
            // selection whose cost grows with the square of the width, so ranking 32 to use 8
            // was four times the work for entries nothing reads — measurable on a phone, where
            // this runs 40 times a token beside a decode competing for the same cores.
            rank_top_k(scores, job.nu, r.ids);
            // The issue list rides in r.spec: of the top-nu the commit will route, only the
            // experts the drop policy would actually READ. Same arithmetic as build_spec_lists
            // (softmax over the top-nu as a proxy for the routing weights, threshold at
            // drop_frac/nu, the top expert always kept) — an expert predicted below the drop
            // threshold is left cold ON PURPOSE, so the commit-time drop discards it unread
            // exactly as it does the router's own cold tail. Measured on device before this
            // filter existed: uncapped early reads made every committed expert resident, the
            // residency-gated drop stopped biting at all, and a run paid 3x the flash and half
            // the tok/s for a perfect prefetch of bytes the baseline never read (78 -> 244
            // MB/token at depth 4, drop 0.75). With drop off the threshold is 0 and the list
            // is simply the committed top-nu.
            const int nu = job.nu < (int) r.ids.size() ? job.nu : (int) r.ids.size();
            if (nu > 0) {
                float w[predict_max_k];
                float mx = scores[(size_t) r.ids[0]];
                for (int k = 1; k < nu; ++k)
                    if (scores[(size_t) r.ids[k]] > mx) mx = scores[(size_t) r.ids[k]];
                float sum = 0.0f;
                for (int k = 0; k < nu; ++k) {
                    w[k] = std::exp(scores[(size_t) r.ids[k]] - mx);
                    sum += w[k];
                }
                const float thr = job.drop_frac > 0.0f ? job.drop_frac / (float) nu : 0.0f;
                for (int k = 0; k < nu; ++k)
                    if (k == 0 || job.drop_frac <= 0.0f || (sum > 0.0f && w[k] / sum >= thr))
                        r.spec.push_back(r.ids[(size_t) k]);
            }
        } else {
            build_spec_lists(scores, job.nu, job.drop_frac, job.spec_max, job.resident, r.spec, r.keep);
        }
        r.ready = true;
        std::lock_guard<std::mutex> lk(pred_mtx_);
        pred_result_ = std::move(r);
    }
}

// Copy one token's row out of a 2-D activation as float, honouring the strides (the row may be a
// view). Returns false for a dtype the probe does not read, which is how an unsupported model
// declines the measurement instead of reporting a wrong one.
static bool row_to_float(const ggml_tensor * t, int j, std::vector<float> & out) {
    const int64_t n = t->ne[0];
    if (n <= 0) return false;
    const char * row = (const char *) t->data + (size_t) j * t->nb[1];
    out.assign((size_t) n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        for (int64_t d = 0; d < n; ++d)
            out[(size_t) d] = *(const float *) (row + (size_t) d * t->nb[0]);
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        for (int64_t d = 0; d < n; ++d)
            out[(size_t) d] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) (row + (size_t) d * t->nb[0]));
        return true;
    }
    return false;
}

// The router's own arithmetic: one score per expert, from a gate matrix [n_embd, n_expert] and a
// gate input row. This is the whole prediction — the same GEMV the graph will do a layer later,
// done early on an input that has not finished changing.
//
// The F16 path matters more than it looks. ggml_fp16_to_fp32 is an exported function, not an
// inlinable conversion, so the obvious loop pays a function call per weight element — 21M calls
// per token on a 40-layer, 256-expert model, which measured as ~35-45 ms per GEMV pass and
// dwarfed everything else this feature does. On aarch64 the compiler converts __fp16 natively
// (one instruction, vectorizable), so that path is used wherever it exists.
// The four-accumulator shape is not a micro-optimisation, it is what makes this loop vectorize at
// all: floating-point addition is not associative, so a compiler may not re-order a single-
// accumulator reduction, and the obvious `acc += p[d] * h[d]` compiles to one scalar multiply-add
// per cycle however high the optimisation level. Splitting the reduction into independent partial
// sums gives the vectorizer permission it cannot take on its own. Measured on the host: 0.77 ->
// 0.20 ms per gate pass on a 256-expert model — and this is the whole feature's per-layer tax, so
// on a 4-thread phone (where it competes with the decode for cores AND for DRAM bandwidth) it was
// the difference between a win and a rout. The partial sums change the summation order, so the
// last bits of a score may differ from the graph's own gate; the ranking is unaffected in every
// case that is not already a numerical tie, and the watchdog is what would catch it if it were.
static bool gate_scores(const ggml_tensor * w, const std::vector<float> & h, std::vector<float> & out) {
    const int64_t nd = w->ne[0], ne = w->ne[1];
    if (nd != (int64_t) h.size() || ne <= 0) return false;
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) return false;
    out.assign((size_t) ne, 0.0f);
    const float * hp = h.data();
    const int64_t nd4 = nd & ~(int64_t) 3;
    for (int64_t e = 0; e < ne; ++e) {
        const char * row = (const char *) w->data + (size_t) e * w->nb[1];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if (w->type == GGML_TYPE_F32) {
            const float * p = (const float *) row;
            for (int64_t d = 0; d < nd4; d += 4) {
                a0 += p[d + 0] * hp[d + 0];
                a1 += p[d + 1] * hp[d + 1];
                a2 += p[d + 2] * hp[d + 2];
                a3 += p[d + 3] * hp[d + 3];
            }
            for (int64_t d = nd4; d < nd; ++d)
                a0 += p[d] * hp[d];
        } else {
#if defined(__aarch64__)
            const __fp16 * p = (const __fp16 *) row;
#else
            const ggml_fp16_t * q = (const ggml_fp16_t *) row;
#endif
            for (int64_t d = 0; d < nd4; d += 4) {
#if defined(__aarch64__)
                a0 += (float) p[d + 0] * hp[d + 0];
                a1 += (float) p[d + 1] * hp[d + 1];
                a2 += (float) p[d + 2] * hp[d + 2];
                a3 += (float) p[d + 3] * hp[d + 3];
#else
                a0 += ggml_fp16_to_fp32(q[d + 0]) * hp[d + 0];
                a1 += ggml_fp16_to_fp32(q[d + 1]) * hp[d + 1];
                a2 += ggml_fp16_to_fp32(q[d + 2]) * hp[d + 2];
                a3 += ggml_fp16_to_fp32(q[d + 3]) * hp[d + 3];
#endif
            }
            for (int64_t d = nd4; d < nd; ++d)
#if defined(__aarch64__)
                a0 += (float) p[d] * hp[d];
#else
                a0 += ggml_fp16_to_fp32(q[d]) * hp[d];
#endif
        }
        out[(size_t) e] = (a0 + a1) + (a2 + a3);
    }
    return true;
}

// Ranking on RAW scores is exact wherever the gating function is monotonic — softmax, sigmoid and
// sqrt-softplus all are, so the order of the logits is the order of the probabilities the router
// sorts. It is NOT exact where the architecture adds a per-expert selection bias or masks whole
// expert groups before the argsort; there the ranking here is an approximation, and the fresh-gate
// control is what reveals it (it will sit below 100%) instead of the loss being blamed on staleness.
void RouterHook::rank_top_k(const std::vector<float> & scores, int k, std::vector<int32_t> & out) {
    const int n = (int) scores.size();
    const int want = k < n ? k : n;
    out.clear();
    if (want <= 0) return;
    out.reserve((size_t) want);
    // Partial selection rather than a sort: `want` is at most predict_max_k and n at most a few
    // hundred, so this is cheaper than ordering the tail nobody reads, and it needs no scratch.
    for (int i = 0; i < want; ++i) {
        int best = -1;
        for (int e = 0; e < n; ++e) {
            bool used = false;
            for (int32_t taken : out)
                if (taken == e) {
                    used = true;
                    break;
                }
            if (used) continue;
            if (best < 0 || scores[(size_t) e] > scores[(size_t) best]) best = e; // ties to the lower index
        }
        if (best < 0) break;
        out.push_back((int32_t) best);
    }
}

// Build the int8 mirror of layer il's gate matrix (see the header for why). One symmetric scale
// per expert ROW keeps each row's dynamic range: the rows are what get compared against one
// another, and a single matrix-wide scale would flatten a quiet row into noise. Runs once per
// layer, on the prediction worker, reading a weight leaf nothing mutates.
bool RouterHook::quantize_gate(int il) {
    if (il < 0 || il >= (int) ra_gate_q_.size()) return false;
    if (!ra_gate_q_[il].empty()) return true; // already mirrored
    const ggml_tensor * w = gate_w_[il];
    if (!w || !w->data) return false;
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) return false;
    const int64_t nd = w->ne[0], ne = w->ne[1];
    if (nd <= 0 || ne <= 0) return false;

    std::vector<int8_t> q((size_t) nd * (size_t) ne);
    std::vector<float> s((size_t) ne, 0.0f);
    std::vector<float> row((size_t) nd);
    for (int64_t e = 0; e < ne; ++e) {
        const char * src = (const char *) w->data + (size_t) e * w->nb[1];
        float amax = 0.0f;
        for (int64_t d = 0; d < nd; ++d) {
            float v;
            if (w->type == GGML_TYPE_F32) {
                v = ((const float *) src)[d];
            } else {
#if defined(__aarch64__)
                v = (float) ((const __fp16 *) src)[d];
#else
                v = ggml_fp16_to_fp32(((const ggml_fp16_t *) src)[d]);
#endif
            }
            row[(size_t) d] = v;
            const float a = v < 0.0f ? -v : v;
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        s[(size_t) e] = scale;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        int8_t * dst = q.data() + (size_t) e * (size_t) nd;
        for (int64_t d = 0; d < nd; ++d) {
            const float t = row[(size_t) d] * inv;
            const int v = (int) (t < 0.0f ? t - 0.5f : t + 0.5f);
            dst[d] = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
        }
    }
    ra_gate_q_[il] = std::move(q);
    ra_gate_s_[il] = std::move(s);
    return true;
}

bool RouterHook::gate_scores_q(int il, const std::vector<float> & h, std::vector<float> & out) {
    if (il < 0 || il >= (int) ra_gate_q_.size() || ra_gate_q_[il].empty()) return false;
    const ggml_tensor * w = gate_w_[il];
    if (!w) return false;
    const int64_t nd = w->ne[0], ne = w->ne[1];
    if (nd != (int64_t) h.size() || (size_t) nd * (size_t) ne != ra_gate_q_[il].size()) return false;

    // Quantize the activation row: one pass over n_embd, amortised across all n_expert rows.
    float amax = 0.0f;
    for (int64_t d = 0; d < nd; ++d) {
        const float a = h[(size_t) d] < 0.0f ? -h[(size_t) d] : h[(size_t) d];
        if (a > amax) amax = a;
    }
    if (!(amax > 0.0f)) return false; // a zero row says nothing; let the caller fall back
    const float hs = amax / 127.0f;
    std::vector<int8_t> & qh = ra_qh_;
    qh.resize((size_t) nd);
    {
        const float inv = 1.0f / hs;
        for (int64_t d = 0; d < nd; ++d) {
            const float t = h[(size_t) d] * inv;
            const int v = (int) (t < 0.0f ? t - 0.5f : t + 0.5f);
            qh[(size_t) d] = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
        }
    }

    const int8_t * q = ra_gate_q_[il].data();
    const float * sc = ra_gate_s_[il].data();
    const int8_t * hp = qh.data();
    const int64_t nd4 = nd & ~(int64_t) 3;
    out.assign((size_t) ne, 0.0f);
    for (int64_t e = 0; e < ne; ++e) {
        const int8_t * p = q + (size_t) e * (size_t) nd;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int64_t d = 0; d < nd4; d += 4) {
            a0 += (int32_t) p[d + 0] * (int32_t) hp[d + 0];
            a1 += (int32_t) p[d + 1] * (int32_t) hp[d + 1];
            a2 += (int32_t) p[d + 2] * (int32_t) hp[d + 2];
            a3 += (int32_t) p[d + 3] * (int32_t) hp[d + 3];
        }
        for (int64_t d = nd4; d < nd; ++d)
            a0 += (int32_t) p[d] * (int32_t) hp[d];
        out[(size_t) e] = (float) ((a0 + a1) + (a2 + a3)) * sc[e] * hs;
    }
    return true;
}

// At the topk of layer il, BEFORE the commit overwrites the ids: sample the watchdog against the
// router's own choice, copy the gate-input row, and submit the ranking job for layer il+N to the
// worker. This is the prefetch's barrier-less read (the row was stashed in the ask pass and is
// only PROBABLY still intact by now — ggml's planner may reuse the buffer), made safe for a
// committed consumer by two things: the sampled fresh-gate control below, which disarms the whole
// policy out loud if the read stops reproducing the router; and the passthrough default, which
// keeps any unproven layer on the router's own routing.
void RouterHook::route_ahead_submit(ggml_tensor * ids, int il, int nu, int nt) {
    if (ra_tripped_ || il < 0 || il >= n_layer_ || nu <= 0 || nu > predict_max_k || nt <= 0) return;
    // Invalidate this token's slot for the target layer FIRST, before any early return below can
    // skip it: what sits there is the PREVIOUS token's ranking for that layer, and a submit that
    // declines (an unstashed gate, an unreadable row) would otherwise leave it in place for
    // apply_route_ahead to commit — routing a token by a prediction made from another token's
    // hidden state. Passthrough is the correct behaviour for a layer this token cannot predict.
    const int target = il + route_ahead_;
    if (target < n_layer_) ra_pred_[target].clear();
    ggml_tensor * h = h_t_[il];
    if (!h || !h->data || h->ne[1] <= 0) return;
    const int j = (int) h->ne[1] - 1; // the batch's last token, the row every predictor here uses

    // Watchdog: the layer's OWN gate on the just-copied row must reproduce the ids the router just
    // chose — read from the tensor now, before apply_route_ahead rewrites them. Same cadence and
    // thresholds as the prefetch's; the difference is what a trip protects against.
    if (++wd_rout_ % wd_interval == 0) {
        // The sampled control is an exact float GEMV on the EVAL thread — cheap because it is
        // sampled, but it lives in the compute residual, so it carries its own meter.
        const auto tw0 = std::chrono::steady_clock::now();
        if (gate_w_[il] && row_to_float(h, j, pred_row_) && gate_scores(gate_w_[il], pred_row_, pred_scores_)) {
            std::vector<int32_t> ctrl;
            rank_top_k(pred_scores_, nu, ctrl);
            int hits = 0;
            for (int k = 0; k < nu; ++k) {
                const int32_t actual = *id_at(ids, nt - 1, k);
                for (int32_t p : ctrl)
                    if (p == actual) {
                        ++hits;
                        break;
                    }
            }
            wd_slots_ += nu;
            wd_hits_ += hits;
            if (wd_slots_ >= wd_min_slots && (double) wd_hits_ < wd_min_frac * (double) wd_slots_) {
                ra_tripped_ = true;
                std::fprintf(stderr,
                             "bmoe: route-ahead disarmed — control at %.1f%% over %lld slots says the "
                             "barrier-less gate-input read is not trustworthy on this graph; routing "
                             "stays the router's own from here\n",
                             100.0 * wd_hits_ / wd_slots_, wd_slots_);
            }
        }
        ra_wd_ns_ +=
            (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw0)
                .count();
        if (ra_tripped_) return;
    }

    const int nl = target;
    if (nl >= n_layer_) return;
    const ggml_tensor * wn = gate_w_[nl];
    if (!wn || !wn->data) return; // target matrix not stashed yet (first decode token): passthrough
    if (!row_to_float(h, j, pred_row_)) return;
    {
        std::lock_guard<std::mutex> lk(pred_mtx_);
        pred_job_.seq = ++pred_seq_;
        pred_job_.nl = nl;
        pred_job_.nu = nu;
        pred_job_.commit = true;
        pred_job_.drop_frac = drop_frac_; // the issue list is drop-aware; the commit is not
        pred_job_.gate = wn;
        pred_job_.row = pred_row_;
        pred_job_pending_ = true;
    }
    pred_cv_.notify_one();
}

// Pop a finished commit ranking, right after layer il's load was handed to the source — the same
// post-load window every speculation issues in. A result for a layer still ahead of il gets its
// reads started HERE: that is the route-ahead window paying out, N-1 layers of compute before the
// committing topk even runs. Only the latest submission is accepted; a ranking the previous token
// never collected fails the seq check and dies here rather than routing anything.
void RouterHook::route_ahead_collect(int il) {
    if (route_ahead_ <= 0 || ra_tripped_ || batch_phase_ != 1 || !source_) return;
    PredictResult r;
    {
        // Never waited for, only popped — measured, not a stylistic choice: a bounded cv wait here
        // (to force every committed layer's reads to start at the predicting layer) bought I/O
        // 0.060→0.040 s/token on the host and paid +0.045 s/token of eval-thread wall for it,
        // because a cv wakeup costs on the order of a millisecond, i.e. the demand read it was
        // trying to save. The depth knob is the honest lever instead: at N>=2 the ranking is
        // simply ready a whole layer before its topk, and the next callback issues it with no one
        // waiting on anyone.
        std::lock_guard<std::mutex> lk(pred_mtx_);
        if (!pred_result_.ready || !pred_result_.commit || pred_result_.seq != pred_seq_) return;
        r = std::move(pred_result_);
        pred_result_.ready = false;
    }
    if (r.nl <= 0 || r.nl >= n_layer_ || r.ids.empty()) return;
    ra_pred_[r.nl] = std::move(r.ids);
    // r.spec is the drop-aware issue list the worker built alongside the ranking: the committed
    // experts the drop policy would actually read. Issue only those; the rest stay cold and die
    // at the commit's drop exactly as the router's own cold tail does.
    if (r.nl > il) route_ahead_issue_for(r.nl, r.spec);
}

// At the topk of layer il, before anything reads the ids: replace the router's selection with the
// ranking made N layers back. This runs BEFORE the weight chain's get_rows consumes the ids, so
// the graph itself computes the substituted experts' weights from this layer's true logits — the
// committed routing carries real, renormalized probabilities, not copies of the prediction's.
// Rewriting here also means the gather below, the trace, the drop policy and load_layer all see
// the committed ids: the whole pipeline treats them as THE routing, which is the experiment.
void RouterHook::apply_route_ahead(ggml_tensor * ids, int il, int nu, int nt) {
    if (ra_tripped_ || il < 0 || il >= n_layer_ || nu <= 0 || nu > predict_max_k) return;
    if (il < route_ahead_) return; // structurally no prediction reaches these layers; not a miss
    std::vector<int32_t> & pred = ra_pred_[il];
    // One-token decode rows only: the prediction was made from a single hidden-state row. A wider
    // decode batch has no per-token prediction to substitute, so it keeps the router's choice.
    if (nt != 1 || (int) pred.size() < nu) {
        ++ra_passthrough_;
        pred.clear();
        return;
    }
    // The agreement between the committed selection and the router's own choice IS the measured
    // perturbation this flag trades on; score it before the overwrite destroys the evidence.
    int hits = 0;
    for (int k = 0; k < nu; ++k) {
        const int32_t actual = *id_at(ids, 0, k);
        for (int p = 0; p < nu; ++p)
            if (pred[(size_t) p] == actual) {
                ++hits;
                break;
            }
    }
    ra_slots_ += nu;
    ra_hits_ += hits;
    ++ra_overridden_;
    for (int k = 0; k < nu; ++k)
        *id_at(ids, 0, k) = pred[(size_t) k];
    pred.clear();
}

// Start reading layer nl's experts ahead of its topk. `cand` is the worker's drop-aware issue
// list: committed experts the drop policy would actually read, confidence-ordered. Within that
// list no spec_max cap applies — these are not guesses, and a miss not issued here is the same
// read paid on demand later, minus the window. Residents are retained instead (protecting a
// slice that is certainly about to be routed costs zero bytes). Runs on the eval thread, from a
// collect point that sits after some layer's own load was handed to the source — the same
// post-load issue window every other speculation uses, so the demand reads in flight keep their
// head-of-line position.
void RouterHook::route_ahead_issue_for(int nl, const std::vector<int32_t> & cand) {
    if (route_ahead_ <= 0 || batch_phase_ != 1 || !source_) return;
    if (nl <= 0 || nl >= n_layer_ || cand.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    ra_ids_.assign(cand.begin(), cand.end());
    // Settle landed speculation first, or a slice already read for an earlier token would classify
    // as a miss and be read again — the exact double-spend settle_spec exists to prevent.
    source_->settle_spec();
    ra_res_.assign(ra_ids_.size(), (uint8_t) 0);
    source_->query_residency(nl, ra_ids_.data(), (int) ra_ids_.size(), ra_res_.data());
    ra_keep_.clear();
    ra_spec_.clear();
    for (size_t i = 0; i < ra_ids_.size(); ++i)
        (ra_res_[i] != route_miss ? ra_keep_ : ra_spec_).push_back(ra_ids_[i]);
    if (!ra_keep_.empty()) source_->retain(nl, ra_keep_.data(), (int) ra_keep_.size());
    if (!ra_spec_.empty()) source_->prefetch(nl, ra_spec_.data(), (int) ra_spec_.size());
    ra_issue_ns_ +=
        (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

// Score one prediction against one routing. Only the prediction's first k entries count: a
// prefetch would fetch k experts, so crediting a hit found deeper in the ranking would measure a
// predictor nobody could build.
void RouterHook::score_prediction(
    const std::vector<int32_t> & pred, const int32_t * actual, int k, PredictorStats & layer, PredictorStats & agg) {
    const int np = (int) pred.size() < k ? (int) pred.size() : k;
    int hits = 0;
    for (int i = 0; i < k; ++i)
        for (int p = 0; p < np; ++p)
            if (pred[(size_t) p] == actual[i]) {
                ++hits;
                break;
            }
    const long long exact = hits == k ? 1 : 0;
    layer.rows += 1;
    layer.slots += k;
    layer.hits += hits;
    layer.exact += exact;
    agg.rows += 1;
    agg.slots += k;
    agg.hits += hits;
    agg.exact += exact;
}

// At the gate matmul of layer il: learn its router matrix, rank its own logits (the control), and
// predict layer il+1 from the input this layer is about to consume.
void RouterHook::predict_at_logits(ggml_tensor * t, int il) {
    if (il < 0 || il >= n_layer_) return;

    // The gate matmul's sources are the router matrix and the gate input. Taking the matrix from
    // the graph instead of looking it up by tensor name is what keeps the probe architecture-neutral
    // — and it is a weight LEAF, so unlike a graph intermediate the pointer stays valid into the
    // layers this token has not reached yet, which is exactly what predicting forward needs.
    ggml_tensor * w = t->src[0];
    ggml_tensor * h = t->src[1];
    if (w && w->op == GGML_OP_NONE && w->data && ggml_is_contiguous(w)) gate_w_[il] = w;

    const int nt = (int) t->ne[1];
    if (nt <= 0) return;
    const int j = nt - 1; // the batch's last token, the one prev_ids_ also records
    const int nl = il + 1;

    pred_self_[il].clear();
    if (nl < n_layer_) pred_stale_[nl].clear();

    // Both rankings below run the same code over the same gate input; only the matrix differs. That
    // is what makes the control a control — it shares the row read, the GEMV and the ranking with
    // the prediction under test, so a bug in any of them surfaces as a control below 100% instead
    // of quietly depressing the number the experiment is about.
    if (!h || !h->data || !row_to_float(h, j, pred_row_)) return;

    // Fresh-gate control: this layer's own matrix on its own input, with no staleness at all. It has
    // to reproduce the routing the graph is about to compute from those very two tensors, so it is
    // also the check that this GEMV agrees with llama.cpp's. Probe-only — the prefetch has no use
    // for a prediction of the layer it is already standing in, and skipping it halves the tax.
    if (predict_log_ && gate_w_[il] && gate_scores(gate_w_[il], pred_row_, pred_scores_))
        rank_top_k(pred_scores_, predict_max_k, pred_self_[il]);

    // Stale-gate: the NEXT layer's matrix on this layer's input — the prediction under test. Silent
    // on the first token of a run, whose next layer has not been seen yet; score_layer counts those.
    if (nl < n_layer_ && gate_w_[nl] && gate_scores(gate_w_[nl], pred_row_, pred_scores_))
        rank_top_k(pred_scores_, predict_max_k, pred_stale_[nl]);

    // Stale-2: the matrix two layers ahead on the same input — the accuracy the ASYNC prefetch
    // actually runs at (see predict_at_topk for why its horizon is two). Probe-only bookkeeping;
    // the aggregate is what prices the extra layer of staleness on a given model.
    const int n2 = il + 2;
    if (predict_log_ && n2 < n_layer_) {
        pred_stale2_[n2].clear();
        if (gate_w_[n2] && gate_scores(gate_w_[n2], pred_row_, pred_scores_))
            rank_top_k(pred_scores_, predict_max_k, pred_stale2_[n2]);
    }
}

// From one full score vector to the two lists the prefetch acts on. The softmax over the top-nu
// predicted scores approximates the routing weights the real gate will produce (exact for
// softmax-gated models, a fair proxy elsewhere); a candidate below the drop threshold is only ever
// read on demand — if it misses, the policy drops it unread, so speculating it would spend the
// exact I/O the policy exists to save. The top prediction is always kept, mirroring the policy's
// own pin of the top-weighted expert.
//
// The split by residency is the cost model in one line: an absent predicted expert is worth AT
// MOST spec_max flash reads (head-of-line stall is all a prefetch can remove — the tail is
// already hidden behind the expert matmul, and speculating a full top-8 was measured to read 33%
// more flash per token and to triple major faults against a full cache); a RESIDENT predicted
// expert costs nothing to keep and is merely retained, protecting it from an eviction that would
// turn a predicted hit back into a read.
void RouterHook::build_spec_lists(const std::vector<float> & scores,
                                  int nu,
                                  float drop_frac,
                                  int spec_max,
                                  const std::vector<uint8_t> & resident,
                                  std::vector<int32_t> & spec,
                                  std::vector<int32_t> & keep) {
    spec.clear();
    keep.clear();
    std::vector<int32_t> pred;
    rank_top_k(scores, nu, pred);
    if ((int) pred.size() < nu) return;

    float w[predict_max_k];
    float mx = scores[(size_t) pred[0]];
    for (int k = 1; k < nu; ++k)
        if (scores[(size_t) pred[k]] > mx) mx = scores[(size_t) pred[k]];
    float sum = 0.0f;
    for (int k = 0; k < nu; ++k) {
        w[k] = std::exp(scores[(size_t) pred[k]] - mx);
        sum += w[k];
    }
    const float thr = drop_frac > 0.0f ? drop_frac / (float) nu : 0.0f;
    for (int k = 0; k < nu; ++k) {
        if (k != 0 && drop_frac > 0.0f && !(sum > 0.0f && w[k] / sum >= thr)) continue;
        const int32_t e = pred[k];
        if (e >= 0 && (size_t) e < resident.size() && resident[(size_t) e] != route_miss)
            keep.push_back(e);
        else if ((int) spec.size() < spec_max)
            spec.push_back(e);
    }
}

// At the topk of layer il, with the routing ids in hand: sample the watchdog, then submit the
// prediction job for layer il+2 built from THIS layer's gate input.
void RouterHook::predict_at_topk(ggml_tensor * t, int il, int nu, int nt) {
    if (!predict_prefetch_ || wd_tripped_ || batch_phase_ != 1 || !source_) return;
    if (il < 0 || il >= n_layer_ || nu <= 0 || nu > predict_max_k || nt <= 0) return;
    ggml_tensor * h = h_t_[il];
    if (!h || !h->data || h->ne[1] <= 0) return;
    const int j = (int) h->ne[1] - 1; // the batch's last token

    // Watchdog sample: the layer's OWN gate on the row just read must reproduce the ids the router
    // just chose. This is the barrier-less read validating itself — if ggml's planner reused the
    // buffer between the gate matmul and here, or the architecture does not select by raw-logit
    // ranking, the control collapses and the prefetch disarms instead of speculating on noise.
    if (++wd_rout_ % wd_interval == 0 && gate_w_[il] && row_to_float(h, j, pred_row_) &&
        gate_scores(gate_w_[il], pred_row_, pred_scores_)) {
        std::vector<int32_t> ctrl;
        rank_top_k(pred_scores_, nu, ctrl);
        const int32_t * actual = gathered_.data() + (size_t) (nt - 1) * nu;
        int hits = 0;
        for (int k = 0; k < nu; ++k)
            for (int32_t p : ctrl)
                if (p == actual[k]) {
                    ++hits;
                    break;
                }
        wd_slots_ += nu;
        wd_hits_ += hits;
        if (wd_slots_ >= wd_min_slots && (double) wd_hits_ < wd_min_frac * (double) wd_slots_) {
            wd_tripped_ = true;
            std::fprintf(stderr,
                         "bmoe: predict-prefetch disarmed — control at %.1f%% over %lld slots says the "
                         "barrier-less gate-input read is not trustworthy on this graph\n",
                         100.0 * wd_hits_ / wd_slots_, wd_slots_);
            return;
        }
    }

    // Submit il+2: copy the row and snapshot residency on this thread (both are cheap and neither
    // is safe to read from the worker), and let the worker do the arithmetic.
    const int nl = il + 2;
    if (nl >= n_layer_) return;
    const ggml_tensor * wn = gate_w_[nl];
    if (!wn || !wn->data) return;
    const int ne = (int) wn->ne[1];
    if (ne <= 0 || !row_to_float(h, j, pred_row_)) return;

    spec_ids_.resize((size_t) ne); // scratch: the identity id list for the residency snapshot
    for (int e = 0; e < ne; ++e)
        spec_ids_[(size_t) e] = e;
    pred_res_.assign((size_t) ne, (uint8_t) 0);
    source_->query_residency(nl, spec_ids_.data(), ne, pred_res_.data());

    {
        std::lock_guard<std::mutex> lk(pred_mtx_);
        pred_job_.seq = ++pred_seq_;
        pred_job_.nl = nl;
        pred_job_.nu = nu;
        pred_job_.drop_frac = drop_frac_;
        pred_job_.spec_max = pred_spec_max_;
        pred_job_.gate = wn;
        pred_job_.row = pred_row_;
        pred_job_.resident = pred_res_;
        pred_job_pending_ = true;
    }
    pred_cv_.notify_one();
}

// Collect and act on a finished prediction, right after layer il's experts were handed to the
// source — the one moment a queued read has a whole layer's worth of idle-lane time before the
// next quiesce, whichever load path (plain topk, deferred drop, or the drop fallback) just ran.
// The result present here targets il+1 (it was submitted one layer back, so the worker had a full
// layer to make the deadline); one that is not ready is skipped, never waited for.
void RouterHook::predict_after_load(int il) {
    if (!predict_prefetch_ || wd_tripped_ || batch_phase_ != 1 || !source_) return;
    const int nl = il + 1;
    if (nl <= 0 || nl >= n_layer_) return;
    PredictResult r;
    {
        std::lock_guard<std::mutex> lk(pred_mtx_);
        if (!pred_result_.ready || pred_result_.nl != nl) return;
        r = std::move(pred_result_);
        pred_result_.ready = false;
    }
    // Retention first: it is free, and an entry protected before the eviction that a subsequent
    // prefetch commit might force is protected in time.
    if (!r.keep.empty()) source_->retain(nl, r.keep.data(), (int) r.keep.size());
    if (!r.spec.empty()) source_->prefetch(nl, r.spec.data(), (int) r.spec.size());
}

// At the topk of layer il: compare every predictor with what the router actually chose. Must run
// before prev_ids_[il] is overwritten, or the previous-token predictor would be handed the answer.
void RouterHook::score_layer(int il, const int32_t * actual, int nu) {
    if (nu <= 0 || nu > predict_max_k) { // a routing wider than the probe ranks; declined, not missed
        ++predict_unscored_;
        return;
    }
    if (pred_stale_[il].empty())
        ++predict_unscored_;
    else
        score_prediction(pred_stale_[il], actual, nu, ps_stale_[il], agg_stale_);
    if (!pred_self_[il].empty()) score_prediction(pred_self_[il], actual, nu, ps_self_[il], agg_self_);
    if (!prev_ids_[il].empty()) score_prediction(prev_ids_[il], actual, nu, ps_prev_[il], agg_prev_);
    if (!pred_stale2_[il].empty()) {
        // Aggregate only: the two-layer horizon exists to price the async prefetch's staleness,
        // and one number does that; a per-layer table of it would double the report for no call
        // anyone makes differently.
        PredictorStats discard;
        score_prediction(pred_stale2_[il], actual, nu, discard, agg_stale2_);
        pred_stale2_[il].clear();
    }
    pred_stale_[il].clear();
    pred_self_[il].clear();
}

} // namespace meitte
