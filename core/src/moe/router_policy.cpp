// The lossy routing policies of RouterHook: cache-aware dropping (docs/expert-dropping.md) and
// cache-aware substitution (docs/cache-aware-substitution.md). Both edit a routing node in place.
#include "router_hook.h"
#include "router_tensor.h"

#include "ggml.h"

#include <cmath>

namespace meitte {

using namespace routing;

void RouterHook::set_drop_policy(float frac, bool renorm, bool in_prefill) {
    drop_frac_ = frac > 0.0f ? frac : 0.0f;
    drop_renorm_ = renorm;
    drop_prefill_ = in_prefill;
    term_variant_.assign(n_layer_ > 0 ? n_layer_ : 0, (int8_t) -1);
    drop_ = PendingDrop{};
    chain_last_ = -1;
    experts_routed_ = experts_dropped_ = 0;
}

void RouterHook::set_expert_substitute(float lambda) {
    sub_lambda_ = lambda > 0.0f ? lambda : 0.0f;
    experts_reranked_ = experts_substituted_ = 0;
}

// Substitution is decode-only, unconditionally: a prefill batch runs against a cold cache, where
// nearly everything is a miss and there is little resident to prefer, and it is compute-bound
// anyway, so the reads it would save are not the reads that cost.
bool RouterHook::substitute_armed() const {
    return sub_lambda_ > 0.0f && source_ != nullptr && batch_phase_ == 1;
}

// Is dropping live for the batch being decoded? Needs a source to ask about residency, a non-zero
// threshold, and — unless armed for prefill — a decode batch: with a cold cache the same threshold
// discards several times the weight mass for a phase that is not I/O-bound anyway.
bool RouterHook::drop_armed() const {
    return drop_frac_ > 0.0f && source_ != nullptr && (batch_phase_ == 1 || drop_prefill_);
}

// Apply the policy to the layer held in drop_, then load only what survives.
//
// `wt` is the terminal node of the weight chain: the weights as the expert matmul will apply them.
// Both edits happen here, before any node consumes them:
//   - the dropped slot's weight is zeroed (and, with renorm, the survivors are scaled back up so
//     the routing keeps the total mass it had);
//   - the dropped slot's ID is repointed at the routing's top-weighted expert. That second edit is
//     not cosmetic. An expert we decline to read may sit in a reserved-but-uncommitted slot, and
//     mul_mat_id would still touch it; pointing the slot at an expert that is certainly resident
//     makes the kernel read valid memory and multiply it by exactly zero. It costs a duplicate
//     matmul, which is the right trade on a decode bound by flash rather than arithmetic.
// The top-weighted expert is never dropped, so a routing always keeps at least one live expert
// whatever the threshold — the guarantee does not rest on frac <= 1 alone.
void RouterHook::apply_drop(ggml_tensor * wt) {
    PendingDrop & D = drop_;
    const int nu = D.nu, nt = D.nt;
    gather_weights(wt, nu, nt, drop_w_);

    // Classify against the cache BEFORE anything is loaded; settle landed prefetches first, or an
    // expert a prefetch correctly guessed would look like a miss and be dropped for nothing.
    source_->settle_spec();
    drop_res_.assign(drop_ids_.size(), (uint8_t) 0);
    source_->query_residency(D.layer, drop_ids_.data(), (int) drop_ids_.size(), drop_res_.data());

    const float thr = drop_frac_ / (float) nu; // frac of the uniform share each of k experts would get
    const bool tracing = trace_on_ && pending_.layer == D.layer;
    drop_mask_.assign((size_t) nu * nt, (uint8_t) 0);

    for (int j = 0; j < nt; ++j) {
        const size_t row = (size_t) j * nu;
        int best = 0;
        float total = 0.0f;
        for (int k = 0; k < nu; ++k) {
            total += drop_w_[row + k];
            if (drop_w_[row + k] > drop_w_[row + best]) best = k;
        }
        const int32_t best_id = drop_ids_[row + best];

        float kept = 0.0f;
        int n_dropped = 0;
        for (int k = 0; k < nu; ++k) {
            const size_t idx = row + k;
            const bool drop = k != best && drop_res_[idx] == route_miss && drop_w_[idx] < thr;
            if (!drop) {
                kept += drop_w_[idx];
                continue;
            }
            *weight_at(wt, nu, nt, j, k) = 0.0f;
            *id_at(D.ids, j, k) = best_id;
            drop_ids_[idx] = best_id;
            drop_mask_[idx] = 1;
            ++n_dropped;
        }
        experts_dropped_ += n_dropped;

        // Restore the routing's total mass. Without this the layer's expert output is scaled down
        // by whatever was discarded, which perturbs the residual stream in a direction the model
        // never sees in training — a systematic shrink, unlike the one missing contribution.
        if (drop_renorm_ && n_dropped > 0 && kept > 0.0f) {
            const float g = total / kept;
            for (int k = 0; k < nu; ++k)
                if (!drop_mask_[row + k]) *weight_at(wt, nu, nt, j, k) *= g;
        }
    }

    if (tracing) pending_.dropped = drop_mask_;
    if (!source_->load_layer(D.layer, drop_ids_.data(), (int) drop_ids_.size()))
        fatal_.store(true, std::memory_order_release);
    D.deferred = false;
    predict_after_load(D.layer);
    route_ahead_collect(D.layer);
}

// Finish with the layer whose topk we last saw: record which node ended its weight chain, so the
// next graph can decide there, and make sure nothing was left waiting on a node that never came.
void RouterHook::close_drop_layer() {
    PendingDrop & D = drop_;
    if (D.layer < 0) return;
    if (D.layer < (int) term_variant_.size() && term_variant_[D.layer] < 0 && chain_last_ >= 0)
        term_variant_[D.layer] = chain_last_;
    if (D.deferred && source_) {
        // The node we learned as terminal did not appear this time, so the deferral was never
        // honoured and this layer's matmul has already run against slots nothing loaded. Load the
        // routing now to keep the cache's accounting straight, and — more importantly — FORGET the
        // terminal node, so the next graph re-learns it and loads at the topk node meanwhile.
        // Deferring again on the same stale guess would repeat the fault every single token; one
        // bad layer in one token is recoverable, a standing bet against a graph that moved is not.
        if (!source_->load_layer(D.layer, drop_ids_.data(), (int) drop_ids_.size()))
            fatal_.store(true, std::memory_order_release);
        if (D.layer < (int) term_variant_.size()) term_variant_[D.layer] = -1;
        D.deferred = false;
        predict_after_load(D.layer);
        route_ahead_collect(D.layer);
    }
    D.layer = -1;
}

// The prediction's GEMV against the int8 mirror. The ACTIVATION is quantized too, once per call,
// so the inner loop is int8 x int8 accumulated in int32 — the shape ARM's dotprod instructions
// exist for (this build targets armv8.2-a+dotprod), and the shape a compiler will vectorize
// without needing permission to reorder a float reduction. Both effects point the same way on a
// phone: a quarter of the bytes off the memory bus, and integer MACs instead of float ones.
// Ranking only needs the ORDER of the scores, and both scales are positive, so the per-expert
// scale is applied at the end where it costs one multiply per expert instead of one per element.
// Cache-aware substitution: re-rank one layer's routing toward the experts already resident.
//
// The scores come from the tensor the graph itself sorted — the source of the argsort / top_k
// that produced these ids (ffn_moe_probs, or its biased / group-masked variant where the
// architecture has one). Two things make it the right one. It is EXACT: whatever gating function,
// selection bias or group mask the model applies is already in it, so the re-ranking starts from
// the router's real order rather than an approximation of it. And it is ALIVE: the graph reads it
// again after the top-k (ggml_get_rows gathers the weights from it), so the allocator cannot have
// recycled its buffer by the time this callback runs. The gate INPUT does not have that property —
// on Gemma 4 it is a private intermediate of the logits matmul, freed the moment that matmul is
// done, and re-scoring from it read another node's bytes (measured: two layers' inputs at the
// same address).
//
// When that tensor is a softmax, the scores are taken in log space: log-softmax differs from the
// logits by a per-token constant, so its range IS the logit range and the margin means what it
// does in the paper below. Anything else (sigmoid probabilities, raw logits, biased probabilities)
// is ranked on the values as they are.
//
// Each resident expert then gets its score raised by lambda × (max − min) of that token's scores,
// and the top-k is taken again. The margin is a fraction of the token's own range, so the same
// lambda means the same thing on a model whose router scores span 2 and one whose scores span 20 —
// that is what keeps it free of per-model tuning and of calibration state. An expert only wins a
// slot if it is resident AND within the margin of the one it displaces, so a confident routing is
// left alone and a near-tie is resolved in favour of the one already in RAM.
//
// The weights are NOT touched. The graph gathers them from the router's own scores for whatever
// ids end up here, so a substituted expert is applied with the weight the router would have given
// it — the routing changes, the arithmetic that follows does not.
//
// The mechanism is the one Skliar et al. describe for DRAM-cached experts (arXiv:2412.00099);
// here the cache is RAM in front of flash, and the range is the token's own rather than a running
// average, so nothing has to warm up.
void RouterHook::apply_substitute(ggml_tensor * ids, int il, int nu, int nt) {
    if (il < 0 || il >= n_layer_ || nu <= 0 || nt <= 0) return;

    // Walk from the ids to what was sorted: a view of an argsort, or a top_k node, depending on
    // the ggml the graph was built with. Anything else (a cast, a scale) is an architecture this
    // does not understand, and it declines rather than guess.
    const ggml_tensor * sc = ids;
    while (sc && (sc->op == GGML_OP_VIEW || sc->op == GGML_OP_ARGSORT || sc->op == GGML_OP_TOP_K))
        sc = sc->src[0];
    if (!sc || !sc->data || sc->type != GGML_TYPE_F32 || sc->ne[1] != nt) return;
    const int ne = (int) sc->ne[0];
    if (ne <= nu) return; // every expert is already selected: nothing to substitute FROM
    const bool log_space = sc->op == GGML_OP_SOFT_MAX;

    // Residency is a property of the layer, not of a token, so it is queried once for the whole
    // batch — but the scores are per token, so every row is re-ranked on its own.
    if ((int) sub_all_.size() != ne) {
        sub_all_.resize((size_t) ne);
        for (int e = 0; e < ne; ++e)
            sub_all_[(size_t) e] = e;
    }
    sub_res_.assign((size_t) ne, (uint8_t) 0);
    source_->settle_spec();
    source_->query_residency(il, sub_all_.data(), ne, sub_res_.data());

    sub_scores_.resize((size_t) ne);
    for (int j = 0; j < nt; ++j) {
        const float * row = (const float *) ((const char *) sc->data + (size_t) j * sc->nb[1]);
        // A masked-out expert (group routing) sits at -inf: it stays there, and it is not part of
        // the range.
        float lo = 0.0f, hi = 0.0f;
        bool any = false;
        for (int e = 0; e < ne; ++e) {
            float v = row[e];
            if (log_space) v = v > 0.0f ? std::log(v) : -INFINITY;
            sub_scores_[(size_t) e] = v;
            if (!std::isfinite(v)) continue;
            if (!any || v < lo) lo = v;
            if (!any || v > hi) hi = v;
            any = true;
        }
        const float boost = any ? sub_lambda_ * (hi - lo) : 0.0f;
        if (!(boost > 0.0f)) continue;
        for (int e = 0; e < ne; ++e)
            if (sub_res_[(size_t) e] != route_miss && std::isfinite(sub_scores_[(size_t) e]))
                sub_scores_[(size_t) e] += boost;

        rank_top_k(sub_scores_, nu, sub_pick_);
        if ((int) sub_pick_.size() != nu) continue;
        experts_reranked_ += nu;

        // Compare as SETS. The graph's ids and this ranking can list the same experts in a
        // different order (a numerical tie resolved differently), and rewriting the ids for that
        // would change nothing the matmul computes while counting a substitution that did not
        // happen. Only a routing whose membership moved is written.
        int changed = 0;
        for (int k = 0; k < nu; ++k) {
            const int32_t e = sub_pick_[(size_t) k];
            bool had = false;
            for (int q = 0; q < nu && !had; ++q)
                had = *id_at(ids, j, q) == e;
            if (!had) ++changed;
        }
        if (changed == 0) continue;
        experts_substituted_ += changed;
        for (int k = 0; k < nu; ++k)
            *id_at(ids, j, k) = sub_pick_[(size_t) k];
    }
}

} // namespace meitte
