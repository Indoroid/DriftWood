#include "router_hook.h"
#include "router_tensor.h"
#include "dense_stream.h"

#include "ggml.h"
#include "../io/platform_io.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>

namespace meitte {

using namespace routing;

RouterHook::RouterHook(const MoeRecipe & recipe, int n_layer) : recipe_(recipe), n_layer_(n_layer) {
    const int n = n_layer_ > 0 ? n_layer_ : 0;
    captured_.assign(n, LayerExperts{});
    prev_ids_.assign(n, std::vector<int32_t>{});
}

// Match a tensor name of the form "blk.<il>.<suffix>.weight" against the recipe's expert
// tensor suffixes. Returns the slot index (its position in recipe.exps_suffix) and layer,
// or -1. The exact ".weight" tail comparison is load-bearing: a companion tensor like
// "ffn_down_exps.scale" prefix-matches its suffix but fails the tail strcmp, so per-expert
// scales — and every other non-".weight" tensor — are left mmap-resident, not streamed.
static int match_expert(const char * name, const MoeRecipe & r, int & il_out) {
    int il = -1;
    int consumed = 0;
    if (std::sscanf(name, "blk.%d.%n", &il, &consumed) != 1 || il < 0) return -1;
    const char * rest = name + consumed; // "<suffix>.weight"
    for (int p = 0; p < MoeRecipe::max_exps; ++p) {
        const char * suffix = r.exps_suffix[p];
        if (!suffix) continue; // unused slot (fewer than max_exps expert tensors)
        const size_t sl = std::strlen(suffix);
        if (std::strncmp(rest, suffix, sl) == 0 && std::strcmp(rest + sl, ".weight") == 0) {
            il_out = il;
            return p;
        }
    }
    return -1;
}

// Every routing node this hook looks for — the topk, the weight chain, the gate matmul — is named
// "ffn_moe_<something>". A graph has thousands of nodes and the ask pass sees all of them, so this
// one comparison is what keeps the rest of the matching off the hot path entirely. `name` is a
// fixed-size array inside the tensor, so reading eight bytes of it is always in bounds.
static inline bool is_moe_node(const char * name) {
    return std::memcmp(name, "ffn_moe_", 8) == 0;
}

// Parse the "<il>" that llama.cpp appends after the '-' on a per-layer node name. Digits only, to
// the end of the string — the same strictness node_layer applies, and stricter than the "%d" this
// replaces, which would have read "12abc" as layer 12. Returns -1 on anything else.
static inline int parse_layer_tail(const char * p) {
    if (*p < '0' || *p > '9') return -1;
    int il = 0;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9') return -1;
        il = il * 10 + (*p - '0');
        if (il > (1 << 20)) return -1; // a runaway digit run is not a layer index
    }
    return il;
}

// "<prefix><il>" → il, or -1 if the name does not have that shape. Used for the two singular
// routing nodes (the topk and the gate matmul), whose prefixes carry their own length.
static inline int match_layer_node(const char * name, std::string_view prefix) {
    return std::memcmp(name, prefix.data(), prefix.size()) == 0 ? parse_layer_tail(name + prefix.size()) : -1;
}

// The router-weight nodes build_moe_ffn emits per layer. The chain is ffn_moe_weights →
// (_softmax | _norm) → (_scaled), each an optional refinement of the previous, so the LAST one
// offered for a layer carries the weight actually applied to that layer's expert outputs.
// Last-wins is what keeps this architecture-independent: no table of which gating each model
// uses. The '-' check is load-bearing — it rejects "ffn_moe_weights_sum-3", which prefix-matches
// "ffn_moe_weights" but is a row sum, not a weight.
//
// It also makes the four names mutually exclusive, which is why the match can be reported as an
// INDEX rather than a name: that index is how a layer's terminal node is remembered, so learning it
// costs no allocation and comparing against it is an integer test — on a path that runs for every
// weight node of every layer of every token whenever the drop policy is armed.
static constexpr std::string_view kWeightNodes[] = {
    "ffn_moe_weights",
    "ffn_moe_weights_softmax",
    "ffn_moe_weights_norm",
    "ffn_moe_weights_scaled",
};

static int match_weights(const char * name, int & il_out) {
    for (int v = 0; v < (int) (sizeof(kWeightNodes) / sizeof(kWeightNodes[0])); ++v) {
        const std::string_view n = kWeightNodes[v];
        if (std::memcmp(name, n.data(), n.size()) != 0 || name[n.size()] != '-') continue;
        const int il = parse_layer_tail(name + n.size() + 1);
        if (il < 0) continue;
        il_out = il;
        return v;
    }
    return -1;
}

// Is `t` readable the moment the node that consumes it is offered to the callback? A graph input is
// (llama.cpp fills it before the graph runs), and so is a pure view of one — a view shares its
// source's memory, so no computation stands between the two. Anything else is produced by a node
// that may not have run yet, and reading it would be reading uninitialized memory.
static bool index_materialized(const ggml_tensor * t) {
    for (int guard = 0; t && guard < 8; ++guard) {
        switch (t->op) {
        case GGML_OP_NONE:
            return t->type == GGML_TYPE_I32; // an index we can actually read as row numbers
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            t = t->src[0];
            break;
        default:
            return false;
        }
    }
    return false;
}

std::unordered_set<std::string> RouterHook::row_gathered_weights() const {
    std::unordered_set<std::string> out;
    for (const std::string & n : row_gathered_)
        if (!row_disqualified_.count(n)) out.insert(n);
    return out;
}

void RouterHook::begin_capture() {
    capturing_ = true;
    for (auto & L : captured_)
        L = LayerExperts{};
    captured_weight_objects_.clear();
    captured_weight_seen_.clear();
    matrix_weights_.clear();
    early_matrix_weights_.clear();
    layer_seen_.assign((size_t) std::max(0, n_layer_), false);
    row_gathered_.clear();
    row_disqualified_.clear();
    row_computed_index_.clear();
    row_computed_indices_ = false;
}
void RouterHook::end_capture() {
    capturing_ = false;
    for (const std::string & name : row_computed_index_)
        if (!row_disqualified_.count(name)) {
            row_computed_indices_ = true;
            break;
        }
}

RouterHook::~RouterHook() {
    predict_worker_stop();
}

void RouterHook::set_trace(bool on) {
    trace_on_ = on;
    pending_ = PendingLayer{};
    trace_rows_.clear();
}

void RouterHook::begin_trace_batch(int base_pos, int n_tokens, int phase, int turn, uint8_t media_kind) {
    trace_base_pos_ = base_pos;
    trace_batch_n_ = n_tokens > 0 ? n_tokens : 1;
    trace_phase_ = phase;
    trace_turn_ = turn;
    trace_media_kind_ = media_kind;
    pending_ = PendingLayer{};
    trace_rows_.clear();
}

void RouterHook::end_trace_batch() {
    flush_pending();
}

// Turn the pending layer's (ids, weights, residency) into one row per routed expert.
void RouterHook::flush_pending() {
    PendingLayer & P = pending_;
    if (P.layer < 0 || P.nu <= 0 || P.nt <= 0) return;

    // Which context position is this layer's token j? Normally the batch's j-th. But before the
    // LAST layer's FFN, llama.cpp gathers only the tokens whose logits were asked for
    // (inp_out_ids; see e.g. models/qwen3moe.cpp "il == n_layer - 1"), so that layer routes
    // fewer tokens than the batch — during prefill, one. Our decode loops ask for logits on the
    // final token only, so a single-token row set is that final token, not the batch's first.
    // Attributing it to base_pos would silently misplace the last layer's whole prefill.
    // Anything else (n_outputs > 1) we cannot map, and say so with a negative step rather than
    // guess; the CLI's greedy loops never produce it.
    const int base = trace_base_pos_, span = trace_batch_n_, nt = P.nt;
    auto position_of = [base, span, nt](int j) -> int {
        if (nt == span) return base + j;
        if (nt == 1) return base + span - 1;
        return -1;
    };

    // A layer with no weight node seen (fused graph, unknown gating) reports NaN rather than 0 —
    // see route_trace.h. Anything else means the gather and the ids disagree, so distrust both.
    const bool have_w = P.weights.size() == (size_t) P.nu * P.nt;
    const uint64_t ebytes = source_ ? source_->expert_bytes(P.layer) : 0;

    // A missing expert is read ONCE per decode however many of the batch's tokens route it
    // (load_layer dedups), so only its first row carries the byte cost. Residency stays per-row:
    // every one of those routings did face a cold cache.
    charged_.clear();
    for (int j = 0; j < P.nt; ++j) {
        for (int k = 0; k < P.nu; ++k) {
            const size_t idx = (size_t) j * P.nu + k;
            RouteTraceRow r;
            r.turn = trace_turn_;
            r.phase = trace_phase_;
            r.media_kind = trace_media_kind_;
            r.step = position_of(j);
            r.layer = P.layer;
            r.slot = k;
            r.expert = P.ids[idx];
            r.weight = have_w ? P.weights[idx] : std::numeric_limits<float>::quiet_NaN();
            r.residency = idx < P.residency.size() ? P.residency[idx] : (uint8_t) 0;
            r.dropped = idx < P.dropped.size() ? P.dropped[idx] : (uint8_t) 0;
            // A dropped routing is never read, so it is neither charged nor allowed to claim the
            // charge for its expert: if another token of the batch routes the same expert and keeps
            // it, that routing pays the read.
            if (!r.dropped && r.residency == route_miss && charged_.insert(r.expert).second) r.expert_bytes = ebytes;
            trace_rows_.push_back(r);
        }
    }
    P.layer = -1;
    P.nu = P.nt = 0;
}

bool RouterHook::c_eval(ggml_tensor * t, bool ask, void * user_data) {
    return static_cast<RouterHook *>(user_data)->on_eval(t, ask);
}

// Layer id from a node name. llama.cpp suffixes per-layer nodes with "-<il>"; anything else
// (embeddings, the output head, the KQ mask) belongs to no layer and reports -1. Kept generic on
// purpose: the trace must not carry a table of which node names a given architecture emits.
static int node_layer(const char * name) {
    const char * dash = std::strrchr(name, '-');
    if (!dash || !dash[1]) return -1;
    for (const char * p = dash + 1; *p; ++p)
        if (*p < '0' || *p > '9') return -1;
    return std::atoi(dash + 1);
}

void RouterHook::set_compute_trace(bool on, bool per_layer) {
    ctrace_on_ = on;
    ctrace_layers_ = per_layer;
    compute_rows_.clear();
}

void RouterHook::begin_compute_batch(int step, int phase, int turn, uint8_t media_kind) {
    ctrace_step_ = step;
    ctrace_phase_ = phase;
    ctrace_turn_ = turn;
    ctrace_media_kind_ = media_kind;
    ctrace_seq_ = 0;
    ctrace_ask_layer_ = -1;
    ctrace_obs_layer_ = -1;
    // The first node of the graph is charged from here, so the mark must be taken as close to
    // llama_decode as the caller can manage — anything between them lands on node 0.
    ctrace_mark_ = std::chrono::steady_clock::now();
    ctrace_faults_ = pio::major_faults();
}

// Layer granularity: emit the closing row for the segment `interval_layer`, charged the wall
// and faults since the previous boundary. Shared by the observe path and end_compute_batch.
void RouterHook::ctrace_close_segment(int interval_layer, const char * tail_name) {
    const auto now = std::chrono::steady_clock::now();
    const uint64_t faults = pio::major_faults();
    ComputeTraceRow r;
    r.turn = ctrace_turn_;
    r.phase = ctrace_phase_;
    r.media_kind = ctrace_media_kind_;
    r.step = ctrace_step_;
    r.seq = ctrace_seq_++;
    r.layer = interval_layer;
    r.op = "LAYER";
    r.name = tail_name ? tail_name : (interval_layer < 0 ? "pre" : "blk." + std::to_string(interval_layer));
    r.wall_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(now - ctrace_mark_).count();
    r.majflt = faults - ctrace_faults_;
    compute_rows_.push_back(std::move(r));
    ctrace_mark_ = now;
    ctrace_faults_ = faults;
}

void RouterHook::end_compute_batch() {
    // Node granularity has no dangling interval: the last node was itself isolated and observed.
    // The tail row absorbs whatever ran between the last boundary and this call — keep the call
    // adjacent to llama_decode's return or the decode epilogue is billed to the LM head.
    if (!ctrace_on_ || !ctrace_layers_) return;
    ctrace_close_segment(-1, "post");
}

bool RouterHook::on_eval(ggml_tensor * t, bool ask) {
    // ── compute trace: close the previous node's interval, open the next ──
    // Ordering matters: this runs before every other job below, so the timestamp is as close to the
    // boundary as possible and the streamer's own work (load_layer, the residency query) lands
    // inside the routing node's interval where it belongs — that IS what routing costs here.
    if (ctrace_on_ && !ask) {
        if (ctrace_layers_) {
            // Only isolated nodes reach this branch: layer boundaries, plus the routing nodes the
            // streamer isolates anyway (a barrier that exists untraced too, so it is free to use).
            // The interval since the previous boundary belongs to the segment we are LEAVING —
            // attributing it to this node's layer would misfile nearly a whole layer, since a
            // boundary node is the first node of the next one.
            ctrace_close_segment(ctrace_obs_layer_, nullptr);
            const int nl = node_layer(t->name);
            if (nl >= 0) ctrace_obs_layer_ = nl; // layerless nodes (reshapes, views) don't move the cursor
        } else {
            const auto now = std::chrono::steady_clock::now();
            const uint64_t faults = pio::major_faults();
            ComputeTraceRow r;
            r.turn = ctrace_turn_;
            r.phase = ctrace_phase_;
            r.media_kind = ctrace_media_kind_;
            r.step = ctrace_step_;
            r.seq = ctrace_seq_++;
            r.layer = node_layer(t->name);
            r.op = ggml_op_name(t->op);
            r.name = t->name;
            r.wall_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(now - ctrace_mark_).count();
            r.majflt = faults - ctrace_faults_;
            compute_rows_.push_back(std::move(r));
            ctrace_mark_ = now;
            ctrace_faults_ = faults;
        }
    }

    // ── capture: harvest expert weight tensors from every node's sources ──
    if (capturing_) {
        if (ask) {
            const int node_il = node_layer(t->name);
            const bool first_in_layer = node_il >= 0 && node_il < n_layer_ && !layer_seen_[(size_t) node_il];
            if (node_il >= 0 && node_il < n_layer_) layer_seen_[(size_t) node_il] = true;
            if (t->op == GGML_OP_MUL_MAT) {
                const ggml_tensor * weight = t->src[0];
                while (weight && weight->op != GGML_OP_NONE && weight->src[0])
                    weight = weight->src[0];
                if (weight && weight->op == GGML_OP_NONE && weight->name[0]) {
                    matrix_weights_.insert(weight->name);
                    int weight_il = -1;
                    if (std::sscanf(weight->name, "blk.%d.", &weight_il) == 1 && weight_il >= 0 &&
                        weight_il < n_layer_ && (first_in_layer || !layer_seen_[(size_t) weight_il]))
                        early_matrix_weights_.insert(weight->name);
                }
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                ggml_tensor * src = t->src[s];
                if (!src || src->name[0] == '\0') continue;
                int il = -1;
                const int p = match_expert(src->name, recipe_, il);
                if (p >= 0) {
                    // An expert-named tensor: streamed, never a dense-rebind candidate.
                    if (il >= 0 && il < n_layer_ && src->ne[2] > 0) { // expert dim is dim-2
                        LayerExperts & L = captured_[il];
                        L.bound = true;
                        L.proj[p].tensor = src;
                        L.proj[p].nb2 = (uint64_t) src->nb[2];
                    }
                    continue;
                }
                // A weight LEAF (op NONE, named) that is not an expert: the persistent dense weights
                // --dense-weights anon may rebind. Graph inputs and KV tensors share this shape but are
                // filtered out downstream by the gguf tensor set, so recording them here is harmless.
                if (src->op != GGML_OP_NONE) continue;
                if (captured_weight_seen_.insert(src).second) captured_weight_objects_.push_back(src);
                // …and HOW this node used it, which is what decides whether its residency can be
                // reduced to the rows the graph asks for. Only the table position of an I32 row
                // gather counts. Computed indices request a barrier after capture.
                if (t->op == GGML_OP_GET_ROWS && s == 0 && t->src[1] && t->src[1]->type == GGML_TYPE_I32) {
                    row_gathered_.insert(src->name);
                    if (!index_materialized(t->src[1])) row_computed_index_.insert(src->name);
                } else
                    row_disqualified_.insert(src->name);
            }
        }
        return false; // capture never isolates a node
    }

    if (dense_stream_) {
        const int dense_il = node_layer(t->name);
        if (ask) {
            if (dense_il >= 0 && dense_il < n_layer_ && dense_il != dense_ask_layer_) {
                dense_ask_layer_ = dense_il;
                dense_boundaries_.insert(t);
                return true;
            }
        } else if (dense_boundaries_.erase(t)) {
            if (!dense_stream_->enter_layer(dense_il) && dense_stream_->fatal())
                fatal_.store(true, std::memory_order_release);
        }
    }

    // ── row-gathered dense tables: put the rows in place before the node reads them ──
    //
    // The gather itself needs no barrier because the rows are loaded before it runs. A computed
    // index producer is isolated separately so its output is ready when this ask callback reads it.
    // The other arm restores the mmap if a later graph uses a shape that capture did not see.
    if (row_source_ && ask) {
        if (t->op == GGML_OP_GET_ROWS) {
            ggml_tensor * table = t->src[0];
            ggml_tensor * ids = t->src[1];
            if (table && row_source_->serves(table)) {
                if (ids && ids->data && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) &&
                    ggml_nelements(ids) <= INT32_MAX) {
                    if (!row_source_->gather(table, (const int32_t *) ids->data, (int) ggml_nelements(ids)))
                        fatal_.store(true, std::memory_order_release);
                } else if (!row_source_->materialize(table)) {
                    fatal_.store(true, std::memory_order_release); // an index we cannot read: take the whole table
                }
            }
        } else {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                ggml_tensor * src = t->src[s];
                if (src && row_source_->serves(src) && !row_source_->materialize(src))
                    fatal_.store(true, std::memory_order_release);
            }
        }
    }

    // ── stream: the routing nodes get the single-node barrier so we see the selected ids ──
    // One prefix test decides whether any of the three matchers below can possibly fire. The vast
    // majority of a graph's nodes fail it and leave having done nothing else.
    const bool moe_node = is_moe_node(t->name);
    const int il = moe_node ? match_layer_node(t->name, "ffn_moe_topk-") : -1;
    const bool is_topk = il >= 0;
    // The weight nodes are asked for by a traced run, and by the drop policy, which decides on the
    // weights the matmul will actually apply. Each extra ask is another barrier — a handful per MoE
    // layer, on tensors of a few floats — so neither is on by default.
    int wl = -1;
    const bool want_weights = trace_on_ || drop_armed();
    const int weight_variant = (moe_node && want_weights) ? match_weights(t->name, wl) : -1;
    const bool is_weights = weight_variant >= 0;
    // Which of them is worth a BARRIER is a narrower question than which of them we recognise. The
    // chain's earlier nodes exist only so the terminal one can be identified; once a layer's
    // terminal is known, both consumers — the drop policy's decision point and the trace's
    // last-wins gather — read that node and no other. Isolating the rest costs a graph split and a
    // full compute-thread synchronization each, per layer, per token, on tensors of a few floats.
    //
    // Ask for the whole chain only while the layer is still learning its shape (term_variant_ < 0),
    // which is the first graph of a run and any graph after close_drop_layer forgets a terminal
    // that stopped appearing. That re-widening is what keeps this self-correcting: a graph that
    // moves is detected exactly as before, because the deferral still fails to be honoured.
    const bool weights_iso = is_weights && (wl < 0 || wl >= (int) term_variant_.size() || term_variant_[wl] < 0 ||
                                            term_variant_[wl] == weight_variant);
    // The gate matmul. The probe ISOLATES it (a barrier per layer — a probed run is not a
    // benchmark run); the prefetch only wants its source pointers, which the ask pass hands over
    // for free, so it asks for nothing and validates its barrier-less read with the watchdog
    // instead. Decode-only either way. The "-" in the pattern is what keeps
    // "ffn_moe_logits_biased-<il>" — a different node — from matching.
    const int gl = (moe_node && (predict_on() || route_ahead_ > 0) && source_ && batch_phase_ == 1)
                       ? match_layer_node(t->name, "ffn_moe_logits-")
                       : -1;
    const bool is_logits = gl >= 0;
    if (ask && is_logits && gl < n_layer_) {
        ggml_tensor * w = t->src[0];
        if (w && w->op == GGML_OP_NONE && w->data && ggml_is_contiguous(w)) gate_w_[gl] = w;
        h_t_[gl] = t->src[1];
    }
    // The compute trace wants every node isolated — or, at layer granularity, only the first
    // node of each layer: the cursor advances on the ask stream (every node passes through
    // here), so one isolation request per layer transition. Layerless names (embeddings, the
    // head, mid-layer reshapes) never move the cursor, so they cannot fake a boundary. The
    // streamer only wants the routing ones.
    if (ask) {
        bool ctrace_iso = false;
        if (ctrace_on_) {
            if (!ctrace_layers_) {
                ctrace_iso = true;
            } else {
                const int nl = node_layer(t->name);
                if (nl >= 0 && nl != ctrace_ask_layer_) {
                    ctrace_iso = true;
                    ctrace_ask_layer_ = nl;
                }
            }
        }
        // Route-ahead deliberately does NOT isolate the gate matmul: it wants the node's source
        // pointers (free, from this ask pass) and reads the row barrier-less at the topk, exactly
        // as the prefetch does — the isolated variant measured ~+0.04 s/token of pure barrier and
        // GEMV tax on the host. What makes that safe for a COMMITTED consumer is the watchdog in
        // route_ahead_submit plus the passthrough default, not a barrier.
        // Integer producers must finish before a row gather reads their indices. This broad
        // barrier uses graph types, not model names. Measure its cost on computed-index tables.
        return ctrace_iso || is_topk || weights_iso || (is_logits && predict_log_) ||
               (row_source_ && row_computed_indices_ && t->type == GGML_TYPE_I32);
    }

    // The probe attaches to the gate matmul rather than to the topk node because this is where the
    // router's own two inputs are reachable: a graph intermediate cannot be found by name later, and
    // its pointer does not survive to the next graph. It writes nothing the graph will read.
    if (is_logits && predict_on()) predict_at_logits(t, gl);

    // Weights follow their layer's topk, so the pending record is already open; keep the last
    // one offered (match_weights explains why) and let the flush read it. This runs BEFORE the drop
    // policy edits the same tensor, so the trace records the routing the router produced, not the
    // one the policy left behind — `dropped` is what says which is which.
    if (is_weights && t->data && t->type == GGML_TYPE_F32 && pending_.layer == wl && pending_.nu > 0)
        gather_weights(t, pending_.nu, pending_.nt, pending_.weights);

    // Learn which node ends this layer's weight chain, and — once known — use it as the point where
    // the routing is decided: everything the drop policy needs is final here, and nothing has
    // consumed it yet. The learning pass and the deferral are the same walk, so a layer whose chain
    // shape the hook has not seen yet simply keeps the undropped behaviour.
    if (is_weights && t->data && t->type == GGML_TYPE_F32 && drop_.layer == wl) {
        chain_last_ = (int8_t) weight_variant;
        if (drop_.deferred && wl >= 0 && wl < (int) term_variant_.size() && term_variant_[wl] == weight_variant)
            apply_drop(t);
    }

    if (source_ && is_topk && t->data && t->type == GGML_TYPE_I32) {
        // selected_experts is [n_expert_used, n_tokens] but a VIEW of the full argsort
        // [n_expert, n_tokens]: its row stride is nb[1] (= n_expert*4), not
        // n_expert_used*4. Gather respecting the strides — a flat read would grab token
        // 0's sorted tail as token 1's experts, corrupting the KV cache.
        // The previous layer's weight chain has been fully offered by now.
        close_drop_layer();

        const int nu = (int) t->ne[0], nt = (int) t->ne[1];
        // Route-ahead, in submission order: first the watchdog sample and the ranking job for
        // layer il+N (both need the ROUTER's ids, still untouched here), then the commit — BEFORE
        // the ids are gathered, so everything downstream — the trace, the drop policy, load_layer,
        // the weight chain the graph is about to run — sees the committed routing, not the
        // router's, and nothing disagrees about which experts this layer used.
        //
        // Substitution runs FIRST, before route-ahead and before the gather, for the same reason:
        // everything downstream — the trace, the drop policy, load_layer, the weight chain — must
        // see the routing that will actually run, and this is the last point where the ids are
        // still ours to change.
        if (substitute_armed()) apply_substitute(t, il, nu, nt);

        if (route_ahead_ > 0 && batch_phase_ == 1) {
            // Collect FIRST: the ranking racing this topk is keyed to the seq of the job the
            // PREVIOUS topk submitted, and the submit below bumps that seq — collecting after it
            // would stale-fail nearly every legitimate result (measured: half the layers fell to
            // passthrough exactly that way, surviving only when a slow load let an earlier
            // collect site catch them).
            route_ahead_collect(il);
            route_ahead_submit(t, il, nu, nt);
            apply_route_ahead(t, il, nu, nt);
        }

        gathered_.clear();
        for (int j = 0; j < nt; ++j)
            for (int k = 0; k < nu; ++k)
                gathered_.push_back(
                    *(const int32_t *) ((const char *) t->data + (size_t) j * t->nb[1] + (size_t) k * t->nb[0]));

        if (trace_on_) {
            flush_pending(); // the previous layer's whole weight chain has been offered by now
            pending_.layer = il;
            pending_.nu = nu;
            pending_.nt = nt;
            pending_.ids = gathered_;
            pending_.weights.clear();
            pending_.dropped.clear();
            // Classify against the cache BEFORE load_layer makes these experts resident —
            // afterwards everything reads as a hit. Settle landed prefetches first, or an expert
            // a prefetch correctly guessed would be recorded as a miss.
            source_->settle_spec();
            pending_.residency.assign(gathered_.size(), (uint8_t) 0);
            source_->query_residency(il, gathered_.data(), (int) gathered_.size(), pending_.residency.data());
        }

        // Count what the ROUTER selected, here rather than inside apply_drop: a layer that is not
        // deferred yet (the first graph, or a phase the policy is not armed for) still routed these
        // experts, and a denominator that skipped them would report the drop rate as a fraction of
        // the wrong thing.
        if (drop_frac_ > 0.0f) experts_routed_ += (long long) gathered_.size();

        // Watchdog sample + submit the il+2 prediction job. Before the load flow on purpose: the
        // snapshot must precede load_layer, whose staging would otherwise count this layer's own
        // reads into the residency picture the job carries.
        predict_at_topk(t, il, nu, nt);

        // Open the layer for the drop policy. Deferring the load is only safe once this layer's
        // terminal weight node is known — otherwise there is no callback left to decide in, and the
        // expert matmul would run against slots nothing loaded. First graph of a run: load here.
        const bool defer = drop_armed() && il >= 0 && il < (int) term_variant_.size() && term_variant_[il] >= 0;
        drop_.layer = il;
        drop_.nu = nu;
        drop_.nt = nt;
        drop_.ids = t;
        drop_.deferred = defer;
        chain_last_ = -1;
        if (defer) {
            drop_ids_ = gathered_;
        } else {
            if (!source_->load_layer(il, gathered_.data(), (int) gathered_.size()))
                fatal_.store(true, std::memory_order_release);
            // Deferred layers speculate from apply_drop instead — after THEIR load, for the same
            // reason this call sits after the one above: the load's quiesce would cancel it.
            predict_after_load(il);
            route_ahead_collect(il);
        }

        // Score the predictors against the routing the router just produced — before the record
        // below moves on, or the previous-token predictor would be graded on the answer itself.
        // The ids gathered above are still the router's own choice: the drop policy edits them at
        // a later node, so a probed run measures routing, not what the cache left of it.
        if (predict_log_ && batch_phase_ == 1 && il >= 0 && il < n_layer_ && nt > 0)
            score_layer(il, gathered_.data() + (size_t) (nt - 1) * nu, nu);

        // Temporal prefetch: hint the next K layers with what the PREVIOUS token routed there,
        // to be read on idle lanes while this layer computes.
        if (prefetch_layers_ > 0 && il >= 0 && il < n_layer_) {
            for (int k = 1; k <= prefetch_layers_; ++k) {
                const int tl = il + k;
                if (tl < n_layer_ && !prev_ids_[tl].empty())
                    source_->prefetch(tl, prev_ids_[tl].data(), (int) prev_ids_[tl].size());
            }
        }

        // Record this layer's routing (the last token's row during prefill) for the next token to
        // predict from. Both consumers read it: the prefetch above, and the probe's previous-token
        // predictor — which is that same bet, scored instead of acted on.
        if ((prefetch_layers_ > 0 || predict_log_) && il >= 0 && il < n_layer_) {
            std::vector<int32_t> & rec = prev_ids_[il];
            rec.clear();
            const int last = nt - 1;
            for (int k = 0; k < nu; ++k)
                rec.push_back(
                    *(const int32_t *) ((const char *) t->data + (size_t) last * t->nb[1] + (size_t) k * t->nb[0]));
        }
    }
    return true;
}

} // namespace meitte
