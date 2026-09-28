// Session::open(): load the model with the layout the streamer requires, create the contexts, then
// capture and bind whichever streaming policy the configuration asks for. Every phase is an Impl
// method, so a failure anywhere returns nullptr and the Impl destructor tears down what was built.
#include "session_impl.h"
#include "llama_glue.h"
#include "tensor_overrides.h"
#include "thinking_control.h"
#include "bmoe/version.h"
#include "../io/platform_io.h"

#include "ggml.h"
#ifdef BMOE_HAVE_WEIGHT_READY_HOOK
#include "ggml-cpu.h"
#endif

#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace meitte {

using namespace detail;

namespace {

bool fail(std::string & error, std::string message) {
    error = std::move(message);
    return false;
}

// Reject session shapes that no model could serve, before anything is loaded. Empty = valid.
std::string check_session_config(const SessionConfig & c) {
    if (c.n_ctx <= 0) return "n_ctx must be positive";
    if (c.n_batch <= 0) return "n_batch must be positive";
    if (c.n_ubatch < 0) return "n_ubatch must be >= 0";
    const auto & ds = c.dense_stream;
    if (ds.enabled && (c.moe.enabled || ds.resident_mb < 0 || ds.window_mb <= 0 || ds.io_lanes < 1 || ds.io_lanes > 2 ||
                       (ds.two_wave && !ds.overlap) || c.spec.is_mtp() || !c.tensor_buffer_overrides.empty()))
        return "invalid dense streaming configuration";
#ifndef BMOE_HAVE_WEIGHT_READY_HOOK
    if (ds.enabled && ds.overlap) return "dense overlap requires the CPU weight-ready hook";
#endif
    if (c.context.min_ctx < 0 || c.context.max_ctx < 0 || c.context.min_ctx > c.n_ctx ||
        (c.context.max_ctx && c.context.max_ctx < c.n_ctx) ||
        (c.context.grow != ContextMode::Off && !c.context.max_ctx))
        return "invalid dynamic context bounds: require dyn-min-ctx <= ctx-size <= dyn-max-ctx";
    return {};
}

// The expert recipe for `arch`, resolved against the file's own layout (see resolve_moe_recipe).
bool resolve_stream_recipe(const std::string & arch,
                           int n_layers,
                           const std::string & model_path,
                           LazyGgufMeta & meta,
                           MoeRecipe & out,
                           std::string & error) {
    const MoeRecipe * recipe = find_moe_recipe(arch.c_str());
    if (!recipe) {
        error = "no MoE recipe for architecture '" + arch +
                "' — add one in core/src/moe/arch_registry.cpp (see docs/adding-a-model.md)";
        return false;
    }
    if (!recipe->fused_gate_up_alt) {
        out = *recipe;
        return true;
    }
    // Read the GGUF names before capture, so the hook can find every routed weight.
    const GgufOffsets & offs = meta.offsets();
    if (!offs.ok) {
        error = "cannot read gguf offsets: " + model_path;
        return false;
    }
    if (!resolve_moe_recipe(
            *recipe, n_layers, [&](const std::string & name) { return offs.off_by_name.count(name) != 0; }, out)) {
        error = "mixed fused and split MoE expert layouts are unsupported";
        return false;
    }
    return true;
}

llama_context_params make_context_params(const SessionConfig & cfg) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg.n_ctx;
    cparams.n_batch = std::min<uint32_t>((uint32_t) cfg.n_batch, cparams.n_ctx);
    // The graph is reserved for the widest ubatch, so this is what sets the resident compute
    // buffers — the memory this engine is always short of. 0 keeps the historical behaviour
    // (one graph as wide as the batch); a smaller value chunks prefill to buy that memory back.
    cparams.n_ubatch = std::min<uint32_t>((uint32_t) cfg.n_ubatch, cparams.n_batch);
    cparams.type_k = to_ggml_type(cfg.cache_type_k);
    cparams.type_v = to_ggml_type(cfg.cache_type_v);
    cparams.kv_unified = cfg.kv_unified;
    cparams.flash_attn_type = to_flash_attn(cfg.flash_attention);
    // These map one-to-one to llama.cpp's public context parameters. Do not derive a scaling
    // method from the model name: upstream resolves Auto and LongRope from the GGUF metadata and
    // factor tensors, which keeps this adapter correct as architectures are added upstream.
    cparams.rope_scaling_type = to_llama_rope_scaling(cfg.rope.scaling);
    cparams.rope_freq_base = cfg.rope.freq_base;
    cparams.rope_freq_scale = cfg.rope.freq_scale;
    cparams.yarn_ext_factor = cfg.rope.yarn_ext_factor;
    cparams.yarn_attn_factor = cfg.rope.yarn_attn_factor;
    cparams.yarn_beta_fast = cfg.rope.yarn_beta_fast;
    cparams.yarn_beta_slow = cfg.rope.yarn_beta_slow;
    cparams.yarn_orig_ctx = (uint32_t) cfg.rope.yarn_orig_ctx;
    // Rejecting a draft means rewinding the KV to the last accepted position. With recurrent-state
    // snapshots the rewind is a cheap restore; without them llama.cpp has to fall back to replaying
    // the sequence, which would hand back exactly the decode the speculation just saved.
    if (cfg.spec.enabled()) cparams.n_rs_seq = (uint32_t) cfg.spec.draft_max;
    return cparams;
}

std::string context_failure(const char * what, const SessionConfig & cfg) {
    return std::string("failed to create ") + what + " (cache-type-k=" + kv_cache_type_name(cfg.cache_type_k) +
           ", cache-type-v=" + kv_cache_type_name(cfg.cache_type_v) +
           ", flash-attn=" + flash_attention_mode_name(cfg.flash_attention) + ")";
}

// One decode of the BOS token. Any valid token builds the same graph (the weight structure is
// prompt-independent), so the warm-up the capture needs costs one position.
llama_token capture_token(const llama_vocab * vocab) {
    const llama_token token = llama_vocab_bos(vocab);
    return token < 0 ? 0 : token;
}

// The names of the expert weight tensors the streamer rebinds. "Dense" is defined by subtraction —
// everything the model has that is NOT one of these.
std::unordered_set<std::string> expert_tensor_names(const std::vector<LayerExperts> & layers) {
    std::unordered_set<std::string> names;
    for (const LayerExperts & L : layers) {
        if (!L.bound) continue;
        for (int p = 0; p < MoeRecipe::max_exps; ++p)
            if (L.proj[p].tensor) names.insert(L.proj[p].tensor->name);
    }
    return names;
}

// Each layer's bytes that the streamer does NOT manage: everything under blk.<il>. except the
// expert weight tensors it rebinds — attention, norms, the router, and any per-expert scale left
// mmap-resident. This is what the layer costs to page in, and it is a static property of the
// file: nothing about decoding changes it, which is why the route trace states it once in the
// static block instead of pretending to measure it per step.
std::vector<uint64_t>
dense_bytes_per_layer(const GgufOffsets & offs, const std::vector<LayerExperts> & layers, int n_layer) {
    const std::unordered_set<std::string> streamed = expert_tensor_names(layers);
    std::vector<uint64_t> out((size_t) std::max(0, n_layer), 0);
    for (const auto & kv : offs.size_by_name) {
        int il = -1;
        if (std::sscanf(kv.first.c_str(), "blk.%d.", &il) != 1) continue;
        if (il < 0 || il >= n_layer || streamed.count(kv.first)) continue;
        out[(size_t) il] += kv.second;
    }
    return out;
}

// Fill each captured layer's file offsets from the gguf and check that every recipe tensor was
// captured with the layout the file records. Returns the expert count, or 0 and sets `error`.
int bind_layer_offsets(const MoeRecipe & recipe,
                       const GgufOffsets & offs,
                       std::vector<LayerExperts> & layers,
                       std::string & error) {
    int n_expert = 0;
    int n_bound = 0;
    for (LayerExperts & L : layers) {
        if (!L.bound) continue;
        ++n_bound;
        for (int p = 0; p < MoeRecipe::max_exps; ++p) {
            if (!recipe.exps_suffix[p]) continue; // slot unused by this architecture
            ggml_tensor * t = L.proj[p].tensor;
            if (!t) {
                error = std::string("captured MoE layer is missing expert tensor '") + recipe.exps_suffix[p] + "'";
                return 0;
            }
            auto it = offs.off_by_name.find(t->name);
            if (it == offs.off_by_name.end()) {
                error = std::string("no gguf offset for tensor ") + t->name;
                return 0;
            }
            auto size = offs.size_by_name.find(t->name);
            if (size == offs.size_by_name.end()) {
                error = std::string("no gguf size for tensor ") + t->name;
                return 0;
            }
            if (t->ne[2] <= 0 || t->ne[2] > std::numeric_limits<int>::max() ||
                (uint64_t) t->nb[2] > std::numeric_limits<uint64_t>::max() / (uint64_t) t->ne[2] ||
                (uint64_t) t->nb[2] * (uint64_t) t->ne[2] != size->second) {
                error = std::string("gguf layout does not match captured expert tensor ") + t->name;
                return 0;
            }
            L.proj[p].file_off = it->second;
            L.proj[p].file_idx = offs.file_by_name.at(t->name); // same parse as the offset, so present
            const int ne2 = (int) t->ne[2];
            if (n_expert == 0)
                n_expert = ne2;
            else if (ne2 != n_expert) {
                error = std::string("inconsistent expert count: tensor ") + t->name + " has " + std::to_string(ne2) +
                        ", expected " + std::to_string(n_expert);
                return 0;
            }
        }
    }
    if (n_bound == 0) {
        error = "no MoE expert tensors captured — is this a MoE model?";
        return 0;
    }
    return n_expert;
}

const char * dense_weights_name(DenseWeightsMode mode) {
    switch (mode) {
    case DenseWeightsMode::Mmap:
        return "mmap";
    case DenseWeightsMode::Anonymous:
        return "anon";
    case DenseWeightsMode::Pinned:
        return "ahwb";
    case DenseWeightsMode::Warmed:
        break;
    }
    return "warm";
}

} // namespace

bool Session::Impl::load_model(LazyGgufMeta & meta, std::string & error) {
    // Load with the layout the streamer requires: file-backed mmap, no repack (a repacked
    // q4_K buffer would break the rebind), experts on CPU.
    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    mparams.use_extra_bufts = false;
    mparams.n_gpu_layers = 0;

    // These are placement requests interpreted by llama.cpp while it loads a fully resident
    // model. The strings belong to cfg and the C array stays alive through the load call.
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    if (!resolve_tensor_buffer_overrides(cfg.tensor_buffer_overrides, tensor_buft_overrides, error)) return false;
    if (!tensor_buft_overrides.empty()) mparams.tensor_buft_overrides = tensor_buft_overrides.data();

    // Optional active-expert override: reduce the model's top-k routing (e.g. 8 -> 6) to cut
    // per-token compute and — under streaming — flash I/O, at a quality cost. Applied purely
    // through llama.cpp's public kv_overrides on the arch-prefixed expert_used_count key: the
    // graph then routes to fewer experts and the whole streaming path adapts automatically
    // (the router hook reads the top-k width from the graph). The array must outlive the load
    // call below. See docs/adding-a-model.md — no llama.cpp patch, no per-arch constants.
    llama_model_kv_override kv_overrides[2];
    std::memset(kv_overrides, 0, sizeof(kv_overrides)); // second entry stays the key[0]==0 terminator
    if (cfg.n_expert_used > 0) {
        const GgufModelInfo & info = meta.info();
        if (!info.ok) return fail(error, "cannot read gguf metadata: " + cfg.model_path);
        if (info.arch.empty()) return fail(error, "gguf has no general.architecture; cannot set n_expert_used");
        if (info.n_expert <= 0)
            return fail(error, "n_expert_used was set but the model is not MoE (no " + info.arch + ".expert_count)");
        if (cfg.n_expert_used > info.n_expert)
            return fail(error, "n_expert_used=" + std::to_string(cfg.n_expert_used) +
                                   " exceeds the model's expert count (" + std::to_string(info.n_expert) + ")");
        const std::string key = info.arch + ".expert_used_count";
        kv_overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::snprintf(kv_overrides[0].key, sizeof(kv_overrides[0].key), "%s", key.c_str());
        kv_overrides[0].val_i64 = cfg.n_expert_used;
        mparams.kv_overrides = kv_overrides;
    }

    model.reset(llama_model_load_from_file(cfg.model_path.c_str(), mparams));
    if (!model) return fail(error, "failed to load model: " + cfg.model_path);

    vocab = llama_model_get_vocab(model.get());
    n_vocab = llama_vocab_n_tokens(vocab);
    n_layer = llama_model_n_layer(model.get());
    // Excluded from n_layer by llama.cpp, so ask for it separately. Zero on every model without a
    // trained MTP head — which is most ggufs, including quants of MTP-capable models that were
    // converted without the nextn tensors.
    n_layer_nextn = llama_model_n_layer_nextn(model.get());
    if (cfg.spec.is_mtp() && n_layer_nextn <= 0)
        return fail(error, "--mtp needs a model with a trained MTP head, and this gguf has no nextn block "
                           "(nextn_predict_layers is absent or zero). Qwen3.5/3.6 carry one in their "
                           "ordinary quantisations; other architectures have none. --ngram speculates on any "
                           "model, since it drafts from the text rather than from the weights. See "
                           "docs/mtp.md and docs/ngram.md.");

    char arch_buf[128] = {0};
    llama_model_meta_val_str(model.get(), "general.architecture", arch_buf, sizeof(arch_buf));
    arch = arch_buf;

    // The effective routing width, resolved once: an override IS the applied width, otherwise the
    // model's own count. Exposed so a UI can interpret drop_cold_frac, and stated in the traces.
    n_expert_used = cfg.n_expert_used;
    if (n_expert_used <= 0) {
        const GgufModelInfo & mi = meta.info();
        n_expert_used = mi.ok ? mi.n_expert_used : 0;
    }
    return true;
}

bool Session::Impl::init_chat_templates(std::string & error) {
    // Chat templates are model-bound: initialise once here, apply per prompt in generate().
    if (!cfg.chatml) return true;
    if (!cfg.chat_template.empty() && !common_chat_verify_template(cfg.chat_template, true))
        return fail(error, "invalid custom chat template");
    try {
        chat_tmpls = common_chat_templates_init(model.get(), cfg.chat_template);
        chat_on = true;
        // Which "thinking off" mechanism this template supports is a property of the model, so
        // it is settled once here rather than re-derived on every turn.
        think_ctl = detail::probe_think_control(chat_tmpls.get());
    } catch (const std::exception & e) {
        if (!cfg.chat_template.empty()) return fail(error, std::string("custom chat template failed: ") + e.what());
        std::fprintf(stderr, "bmoe: chat template unavailable (%s); using raw prompts\n", e.what());
        chat_on = false;
    }
    return true;
}

bool Session::Impl::create_contexts(bool install_eval_callback, std::string & error) {
    llama_context_params cparams = make_context_params(cfg);
    if (install_eval_callback) {
        cparams.cb_eval = &RouterHook::c_eval;
        cparams.cb_eval_user_data = hook.get();
    }
    context_params = cparams;

    ctx.reset(llama_init_from_model(model.get(), cparams));
    if (!ctx) return fail(error, context_failure("context", cfg));
    llama_set_n_threads(ctx.get(), cfg.n_threads, cfg.n_threads);

    if (cfg.multimodal.enabled() && !mtmd.init(cfg.multimodal, model.get(), cfg.n_threads, error)) return false;

    // The MTP draft context: same model, same eval callback, but ctx_type = MTP so llama.cpp builds
    // the nextn graph.
    if (cfg.spec.is_mtp()) {
        ctx_dft.reset(create_mtp_context(model.get(), cparams, cfg.spec, cfg.n_threads));
        if (!ctx_dft) return fail(error, context_failure("the MTP draft context", cfg));
    }

    smpl = make_sampler_chain(cfg.sampling);

    // One abort callback for the session's whole life, checking independent predicates: an
    // explicit cancel() request (any mode) and a fatal streaming I/O error (overlap only).
    // Installing it unconditionally is what lets cancel() interrupt a serial decode too.
    llama_set_abort_callback(ctx.get(), &Impl::abort_requested, this);

    // Built before the capture warm-up on purpose: the driver's constructor turns on nextn
    // extraction for both contexts, which is what the target graph looks like for the rest of the
    // session. Capturing the graph it actually runs beats capturing the one it briefly had.
    if (cfg.spec.is_mtp()) {
        mtp.reset(create_mtp_driver(cfg.spec, ctx.get(), ctx_dft.get()));
        if (!mtp) return fail(error, "failed to initialise MTP speculative decoding");
    }

    // The wide verify batch belongs to the loop, not to a source: whoever drafted, the target is
    // handed 1 + draft_max positions in one decode. Allocated for any speculation, which is what
    // lets the n-gram source reuse the whole verify half without a draft context.
    if (cfg.spec.enabled()) {
        // One batch for the session, wide enough for the larger of its two roles.
        mtp_batch = llama_batch_init(std::max(cfg.n_batch, cfg.spec.draft_max + 1), /*embd*/ 0, /*n_seq_max*/ 1);
        mtp_batch_owned = true;
        draft_buf.reserve((size_t) cfg.spec.draft_max);
    }
    return true;
}

bool Session::Impl::bind_dense_stream(LazyGgufMeta & meta, std::string & error) {
    const GgufOffsets & offs = meta.offsets();
    if (!offs.ok) return fail(error, "cannot read gguf offsets: " + cfg.model_path);
    if (meta.info().n_expert > 0) return fail(error, "dense streaming requires a model with zero experts");

    // The scheduler offers every node to the ask callback before it computes the graph. Abort
    // the capture graph at compute start so a >RAM model never makes a full warm-up pass.
    hook->begin_capture();
    capture_abort.store(true, std::memory_order_release);
    llama_token token = capture_token(vocab);
    const int capture_rc = llama_decode(ctx.get(), llama_batch_get_one(&token, 1));
    capture_abort.store(false, std::memory_order_release);
    hook->end_capture();
    if (capture_rc == 0 || hook->captured_weight_objects().empty())
        return fail(error, "dense graph capture did not abort before compute with weights captured");
    pio::ProcessMemory capture_memory;
    if (pio::process_memory(&capture_memory))
        std::fprintf(stderr, "bmoe: dense capture — %zu weight leaves, RSS %llu MiB before weight reads\n",
                     hook->captured_weight_objects().size(), (unsigned long long) (capture_memory.rss_bytes >> 20));
    llama_memory_clear(llama_get_memory(ctx.get()), true);

    // A matrix streams when the graph uses it as a per-layer matrix after the early (pinned) span.
    auto streamable_layer = [&](const std::string & name) {
        int layer = -1;
        if (std::sscanf(name.c_str(), "blk.%d.", &layer) != 1 || layer < 0 || layer >= n_layer ||
            !hook->captured_matrix_weights().count(name) || hook->early_matrix_weights().count(name))
            return -1;
        return layer;
    };

    std::vector<DenseTensorRef> fixed, candidates;
    std::map<std::tuple<int, uint64_t, uint64_t>, size_t> seen;
    for (ggml_tensor * tensor : hook->captured_weight_objects()) {
        if (!tensor) continue;
        const std::string name = tensor->name;
        auto off = offs.off_by_name.find(name);
        auto size = offs.size_by_name.find(name);
        auto type = offs.type_by_name.find(name);
        if (off == offs.off_by_name.end() || size == offs.size_by_name.end() || type == offs.type_by_name.end())
            continue; // graph input, not a GGUF weight
        if (!tensor->data || !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != size->second ||
            (int) tensor->type != type->second)
            return fail(error, "dense GGUF type, size, strides or backing mismatch: " + name);
        const int shard = offs.file_by_name.at(name);
        const auto key = std::make_tuple(shard, off->second, size->second);
        auto found = seen.find(key);
        if (found != seen.end()) {
            DenseTensorRef & owner = fixed[found->second];
            if (owner.tensor->type != tensor->type)
                return fail(error, "dense alias type mismatch for GGUF file range: " + name);
            owner.aliases.push_back(tensor);
            if (streamable_layer(name) != owner.layer)
                owner.layer = -1; // a tied output or gather keeps its shared range resident
            continue;
        }
        DenseTensorRef ref;
        ref.tensor = tensor;
        ref.file_idx = shard;
        ref.file_off = off->second;
        ref.size = size->second;
        ref.layer = streamable_layer(name);
        seen.emplace(key, fixed.size());
        fixed.push_back(std::move(ref));
    }
    if (fixed.empty()) return fail(error, "dense capture found no GGUF weight leaves");

    // Alias groups are classified together. The fixed set includes controls, embeddings and
    // output weights; the remaining budget pins the earliest matrices across every token.
    uint64_t compulsory = 0;
    for (const DenseTensorRef & ref : fixed)
        if (ref.layer < 0) compulsory += ref.size;
    const uint64_t window = (uint64_t) cfg.dense_stream.window_mb << 20;
    const uint64_t available = pio::mem_available_bytes();
    const uint64_t auto_budget = available > (7ull << 30) + window ? available - (7ull << 30) - window : 0;
    const uint64_t budget = cfg.dense_stream.resident_mb ? (uint64_t) cfg.dense_stream.resident_mb << 20 : auto_budget;
    if (compulsory > budget)
        return fail(error, "dense resident budget is smaller than embeddings, output and control tensors (need " +
                               std::to_string((compulsory + (1 << 20) - 1) >> 20) + " MiB)");
    std::stable_sort(fixed.begin(), fixed.end(),
                     [](const DenseTensorRef & a, const DenseTensorRef & b) { return a.layer < b.layer; });
    uint64_t pinned_bytes = compulsory;
    std::vector<DenseTensorRef> pinned;
    for (DenseTensorRef & ref : fixed) {
        if (ref.layer < 0 || pinned_bytes + ref.size <= budget) {
            if (ref.layer >= 0) pinned_bytes += ref.size;
            pinned.push_back(std::move(ref));
        } else
            candidates.push_back(std::move(ref));
    }
    if (candidates.empty()) return fail(error, "dense stream selected no matrices; lower resident budget or use mmap");
    std::vector<const void *> mapped_addresses;
    for (const DenseTensorRef & ref : candidates) {
        mapped_addresses.push_back(ref.tensor->data);
        for (const ggml_tensor * alias : ref.aliases)
            if (alias) mapped_addresses.push_back(alias->data);
    }
    if (pio::addresses_in_file_mappings(offs.shard_paths, mapped_addresses) != mapped_addresses.size())
        return fail(error, "dense stream requires native file-backed GGUF matrix pointers");
    const size_t n_streamed = candidates.size();
    // The anonymous owner reads the fixed set once. The bounded streamer rebinds the rest to
    // stable reserved addresses and commits pages only while their layer is active.
    if (!dense_fixed.init(DenseWeightsMode::Anonymous, offs.shard_paths, 4096, {}, std::move(pinned)))
        return fail(error, "dense fixed-resident load failed");
    if (!dense_stream.init(std::move(candidates), offs.shard_paths, 4096, window, cfg.dense_stream.io_lanes,
                           cfg.dense_stream.overlap, cfg.dense_stream.two_wave, &cancel_requested))
        return fail(error, "dense stream setup failed");
    hook->set_dense_stream(&dense_stream);
#ifdef BMOE_HAVE_WEIGHT_READY_HOOK
    if (cfg.dense_stream.overlap) {
        bool expected = false;
        if (!dense_weight_hook_active.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
            return fail(error, "only one dense overlap session can use the process-wide CPU hook at a time");
        ggml_cpu_set_weight_ready_hook(
            [](const ggml_tensor * src0, void * ud) -> bool {
                return static_cast<DenseStream *>(ud)->weight_ready(src0);
            },
            &dense_stream);
        weight_hook_registered = true;
    }
#endif
    std::fprintf(stderr, "bmoe: dense stream — fixed %llu MiB, window %llu MiB, %zu streamed matrices, direct I/O %s\n",
                 (unsigned long long) (pinned_bytes >> 20), (unsigned long long) (window >> 20), n_streamed,
                 dense_stream.direct() ? "on" : "off");
    return true;
}

bool Session::Impl::bind_expert_stream(const MoeRecipe & recipe,
                                       int n_layer_streamed,
                                       LazyGgufMeta & meta,
                                       IRouteTraceSink * route_sink,
                                       IIoTraceSink * io_sink,
                                       std::string & error) {
    // Capture warm-up: one mmap-resident decode so the eval-callback can harvest the expert
    // tensor pointers from the graph. KV is wiped afterwards.
    hook->begin_capture();
    llama_token warm_tok = capture_token(vocab);
    if (cfg.spec.is_mtp()) {
        batch_fill(mtp_batch, &warm_tok, 1, /*pos0*/ 0, /*all_logits*/ true);
        if (llama_decode(ctx.get(), mtp_batch) != 0) return fail(error, "capture warm-up decode failed");
        // The MTP graph is built only by a decode on the draft context, and process() is what
        // issues it. Without this pass the head's expert layer never reaches the eval callback,
        // so it would stay unbound and silently mmap-resident — the one thing streaming exists
        // to avoid on a model that does not fit.
        if (!common_speculative_process(mtp.get(), mtp_batch))
            return fail(error, "MTP capture warm-up failed: the draft context could not process the batch");
    } else {
        if (llama_decode(ctx.get(), llama_batch_get_one(&warm_tok, 1)) != 0)
            return fail(error, "capture warm-up decode failed");
    }
    hook->end_capture();

    const GgufOffsets & offs = meta.offsets();
    if (!offs.ok) return fail(error, "cannot read gguf offsets: " + cfg.model_path);

    std::vector<LayerExperts> layers = hook->captured();
    const int n_expert = bind_layer_offsets(recipe, offs, layers, error);
    if (n_expert == 0) return false;

    // Computed before init consumes `layers`: the dense split needs the captured expert
    // tensor names and the gguf sizes together.
    std::vector<uint64_t> dense_bytes;
    if (route_sink) dense_bytes = dense_bytes_per_layer(offs, layers, n_layer_streamed);

    // Hand the streamer the dense (non-expert) model weights: every captured weight leaf that IS
    // a gguf tensor (dropping graph inputs and KV, which share the leaf shape) and is NOT one of
    // the streamed experts. Built before init consumes `layers`. Anonymous/Pinned read these into
    // their own buffers; every mode needs the list to hold back a tensor too large to be
    // resident at all (qwen4exp's n-gram table), which must also leave the warm sweep and the
    // residency sensor — see DenseWeights::hold_back_oversized.
    std::vector<std::string> unaccounted;
    {
        const std::unordered_set<std::string> expert_names = expert_tensor_names(layers);
        // The subset the graph only row-gathers, when the run asked for the row policy. Their
        // membership is the hook's verdict on what the graph did, and this is a filter over the
        // dense list rather than a second list, so a table cannot be both resident and streamed.
        const std::unordered_set<std::string> row_names =
            cfg.moe.row_stream ? hook->row_gathered_weights() : std::unordered_set<std::string>();
        std::vector<DenseTensorRef> dense, rows;
        std::unordered_map<uint64_t, size_t> dense_at;
        for (ggml_tensor * tensor : hook->captured_weight_objects()) {
            if (!tensor) continue;
            const std::string name = tensor->name;
            if (expert_names.count(name)) continue;
            auto off = offs.off_by_name.find(name);
            auto sz = offs.size_by_name.find(name);
            if (off == offs.off_by_name.end() || sz == offs.size_by_name.end()) continue; // not a file tensor
            const int file_idx = offs.file_by_name.at(name);
            const uint64_t key = ((uint64_t) (uint32_t) file_idx << 48) ^ off->second;
            auto seen = dense_at.find(key);
            if (seen != dense_at.end()) {
                dense[seen->second].aliases.push_back(tensor);
                continue;
            }
            DenseTensorRef d;
            d.tensor = tensor;
            d.file_off = off->second;
            d.size = sz->second;
            d.file_idx = file_idx;
            dense_at.emplace(key, dense.size());
            dense.push_back(d);
            if (row_names.count(name)) rows.push_back(d);
        }
        for (const DenseTensorRef & tensor : dense)
            if (!tensor.aliases.empty())
                std::fprintf(stderr, "bmoe: '%s' is bound %zu times over one range (tied head) — all rebound\n",
                             tensor.tensor->name, tensor.aliases.size() + 1);
        if (cfg.moe.release_mmap) {
            std::unordered_set<std::string> owned;
            owned.reserve(dense.size());
            for (const DenseTensorRef & tensor : dense)
                if (tensor.tensor) owned.insert(tensor.tensor->name);
            for (const auto & offset : offs.off_by_name)
                if (!expert_names.count(offset.first) && !owned.count(offset.first))
                    unaccounted.push_back(offset.first);
        }
        const uint64_t row_budget = (uint64_t) std::max(0, cfg.moe.row_stream_mb) * 1024ull * 1024ull;
        source.set_dense_tensors(std::move(dense));
        source.set_row_tensors(std::move(rows), row_budget);
    }

    if (!source.init(offs.shard_paths, n_expert, std::move(layers), cfg.moe))
        return fail(error, "expert stream source init failed");
    hook->set_source(&source);
    // The row policy exists only once dense_.init has taken the tables over; null means nothing
    // qualified or the takeover declined, and the hook then costs exactly nothing per node.
    hook->set_row_source(source.row_source());

    if (cfg.moe.release_mmap) release_model_mapping(offs, unaccounted);

    if (route_sink) {
        route_trace = route_sink;
        hook->set_trace(true);

        RouteTraceStatic st;
        st.model = cfg.model_path;
        st.arch = arch;
        // The streamed span, not the trunk: with speculation on, the MTP block is a layer the
        // streamer manages like any other, and a trace that stopped at the trunk would drop it.
        st.n_layer = n_layer_streamed;
        st.n_expert = n_expert;
        st.n_expert_used = n_expert_used; // the effective top-k
        st.dense_bytes_per_layer = std::move(dense_bytes);
        st.expert_bytes_per_layer.resize((size_t) n_layer_streamed);
        for (int il = 0; il < n_layer_streamed; ++il)
            st.expert_bytes_per_layer[(size_t) il] = source.expert_bytes(il);
        route_sink->on_static(st);
    }

    if (io_sink) {
        io_trace = io_sink;
        source.set_io_trace(true);
    }

    if (cfg.moe.overlap) {
#ifdef BMOE_HAVE_EXPERT_READY_HOOK
        if (!source.enable_overlap_hook())
            return fail(error, "only one expert overlap session can use the process-wide CPU hook at a time");
#else
        return fail(error, "--overlap requires the bmoe llama.cpp fork (expert-ready hook not compiled in)");
#endif
    }

    llama_memory_clear(llama_get_memory(ctx.get()), true); // discard warm-up KV
    if (ctx_dft) llama_memory_clear(llama_get_memory(ctx_dft.get()), true);
    return true;
}

void Session::Impl::release_model_mapping(const GgufOffsets & offs, const std::vector<std::string> & unaccounted) {
    // Release only when every captured weight has left the model mapping. The check covers the
    // graph objects that capture observed. MTP can use uncaptured file tensors, so it also needs
    // complete GGUF accounting before this opt-in operation is safe.
    std::vector<const void *> weight_addresses;
    weight_addresses.reserve(hook->captured_weight_objects().size());
    for (const ggml_tensor * tensor : hook->captured_weight_objects())
        if (tensor && tensor->data) weight_addresses.push_back(tensor->data);
    const size_t still_mapped = pio::addresses_in_file_mappings(offs.shard_paths, weight_addresses);
    if (still_mapped) {
        std::fprintf(stderr, "bmoe: release-mmap skipped: %zu weight(s) still read the model's mapping\n",
                     still_mapped);
        return;
    }
    if (!unaccounted.empty() && cfg.spec.is_mtp()) {
        std::fprintf(stderr, "bmoe: release-mmap skipped: %zu file tensor(s) no policy owns (first: %s)\n",
                     unaccounted.size(), unaccounted.front().c_str());
        return;
    }
    const pio::MappingReleaseReport report = pio::release_file_mappings(offs.shard_paths, &mapping_placeholders);
    if (report.supported && (report.views_unmapped || report.sections_closed) && !source.reopen_readers())
        std::fprintf(stderr, "bmoe: release-mmap: reader reopen failed; reads stay serialized\n");
    if (report.supported)
        std::fprintf(stderr, "bmoe: release-mmap: %d view(s) unmapped (%llu MiB), %d section(s) closed%s%s%s\n",
                     report.views_unmapped, (unsigned long long) (report.bytes >> 20), report.sections_closed,
                     report.plugs_missed ? ", handle slot not reclaimed" : "", report.error.empty() ? "" : "; ",
                     report.error.c_str());
}

void Session::Impl::attach_decode_traces(IComputeTraceSink * compute_sink) {
    // Outside the streaming phase on purpose: the compute trace measures the graph, which exists with
    // or without the streamer, so a dense mmap baseline can be traced and compared. The I/O trace
    // was armed by bind_expert_stream() (it needs the source) and only reports here.
    if (!compute_sink && !io_trace) return;
    DecodeTraceStatic st;
    st.model = cfg.model_path;
    st.arch = arch;
    st.n_layer = n_layer;
    st.n_threads = cfg.n_threads;
    st.io_threads = cfg.moe.enabled ? cfg.moe.io_threads : 0;
    st.o_direct = cfg.moe.enabled && source.stats().o_direct; // the open's outcome, not the request
    st.overlap = cfg.moe.enabled && cfg.moe.overlap;
    if (compute_sink) {
        compute_trace = compute_sink;
        hook->set_compute_trace(true, cfg.compute_trace_layers);
        compute_sink->on_static(st);
    }
    if (io_trace) io_trace->on_static(st);
}

void Session::Impl::build_run_info(LazyGgufMeta & meta) {
    // What this session IS, for any metrics sink that will describe what it DOES. Built last, where
    // every fact is resolved: cache_mb in particular is what the streamer settled on, which under
    // auto-sizing is a number no flag ever mentioned.
    RunInfo & ri = info;
    const std::string & p = cfg.model_path;
    const size_t slash = p.find_last_of("/\\");
    ri.model = slash == std::string::npos ? p : p.substr(slash + 1);
    ri.arch = arch;
    ri.n_layer = n_layer;
    ri.engine_version = version();
    ri.n_threads = cfg.n_threads;
    ri.n_ctx = cfg.n_ctx;
    ri.n_batch = cfg.n_batch;
    ri.n_ubatch = cfg.n_ubatch;
    ri.chatml = cfg.chatml;
    ri.cache_type_k = kv_cache_type_name(cfg.cache_type_k);
    ri.cache_type_v = kv_cache_type_name(cfg.cache_type_v);
    ri.flash_attention = flash_attention_mode_name(cfg.flash_attention);
    ri.rope_scaling = rope_scaling_mode_name(cfg.rope.scaling);
    ri.rope_freq_base = cfg.rope.freq_base;
    ri.rope_freq_scale = cfg.rope.freq_scale;
    ri.yarn_ext_factor = cfg.rope.yarn_ext_factor;
    ri.yarn_attn_factor = cfg.rope.yarn_attn_factor;
    ri.yarn_beta_fast = cfg.rope.yarn_beta_fast;
    ri.yarn_beta_slow = cfg.rope.yarn_beta_slow;
    ri.yarn_orig_ctx = cfg.rope.yarn_orig_ctx;
    ri.custom_chat_template = !cfg.chat_template.empty();
    ri.compute_trace_layers = cfg.compute_trace_layers;
    ri.spec = cfg.spec.is_mtp() ? "mtp" : cfg.spec.is_ngram() ? "ngram" : "off";
    ri.spec_draft_max = cfg.spec.enabled() ? cfg.spec.draft_max : 0;
    ri.mtp_p_min = cfg.spec.is_mtp() ? cfg.spec.draft_p_min : 0.0f;
    ri.ngram_min_match = cfg.spec.is_ngram() ? cfg.spec.ngram_min_match : 0;
    ri.temp = cfg.sampling.temp;
    ri.top_k = cfg.sampling.top_k;
    ri.top_p = cfg.sampling.top_p;
    ri.seed = cfg.sampling.seed;
    const MoeStreamConfig & moe = cfg.moe;
    ri.moe_stream = moe.enabled;
    ri.cache_auto = moe.cache_auto;
    ri.cache_floor_mb = moe.cache_floor_mb;
    ri.cache_ceil_mb = moe.cache_ceil_mb;
    ri.force_cache = moe.force_cache;
    ri.load_all = moe.enabled && moe.load_all;
    ri.io_threads = moe.enabled ? moe.io_threads : 0;
    ri.overlap = moe.enabled && moe.overlap;
    ri.io_two_wave = moe.enabled && moe.io_two_wave;
    ri.prefetch_layers = moe.enabled ? moe.prefetch_layers : 0;
    ri.route_ahead = moe.enabled ? moe.route_ahead : 0;
    ri.predict_prefetch = moe.enabled && moe.predict_prefetch;
    ri.predict_log = moe.enabled && moe.predict_log;
    ri.predict_spec_max = moe.enabled ? moe.predict_spec_max : 0;
    ri.prefetch_sync = moe.enabled && moe.prefetch_sync;
    ri.drop_cold_frac = moe.enabled ? moe.drop_cold_frac : 0.0f;
    ri.drop_renorm = moe.drop_renorm;
    ri.drop_prefill = moe.drop_prefill;
    ri.substitute_lambda = moe.enabled ? moe.substitute_lambda : 0.0f;
    const RunConfig::DenseStreamConfig & ds = cfg.dense_stream;
    ri.dense_stream = ds.enabled;
    ri.dense_resident_mb = ds.resident_mb;
    ri.dense_window_mb = ds.window_mb;
    ri.dense_io_lanes = ds.enabled ? ds.io_lanes : 0;
    ri.dense_overlap = ds.enabled && ds.overlap;
    ri.dense_two_wave = ds.enabled && ds.two_wave;
    ri.dense_direct = ds.enabled && dense_stream.direct();
    // The CSV keeps the two familiar flags, derived from the resolved dense-weights policy.
    ri.dense_weights = dense_weights_name(moe.dense_weights);
    if (moe.enabled) {
        const IExpertSource::Stats st = source.stats();
        ri.cache_mb = (int) (st.cache_budget_bytes / (1024ull * 1024ull));
        // o_direct from the same sample: whether the shard readers actually got cache bypass
        // (O_DIRECT honoured, or F_NOCACHE applied on Apple), never what the flag asked for.
        ri.o_direct = st.o_direct;
    }
    // The EFFECTIVE top-k — a run whose top-k is unknown cannot be compared against one whose top-k
    // differs, which is most of the point.
    ri.n_expert_used = n_expert_used;
    const GgufModelInfo & mi = meta.info();
    if (mi.ok) ri.n_expert = mi.n_expert;
    // Last, because it needs both numbers above: the budget the streamer settled on and the
    // width that was actually applied. Rounded UP — a cycle reported as smaller than it is
    // would put a budget on the wrong side of the cliff in exactly the borderline case.
    ri.cache_cycle_mb =
        (int) ((source.worst_cycle_bytes(ri.n_expert_used) + 1024ull * 1024ull - 1) / (1024ull * 1024ull));
}

void Session::Impl::warn_about_load() const {
    // Cache-aware dropping on a narrow routing: say so once, at load.
    //
    // The threshold is a fraction of the uniform share 1/top-k, so what it removes scales with how
    // wide the routing is. At top-k 8 (where it was measured) it trims a long tail; at top-k 2 the
    // same fraction can discard the whole minority expert on every miss, which is closer to halving
    // the routing than to trimming it. The engine has no way to know whether that is acceptable for
    // a given model, so it states the fact rather than clamping the flag — a silent adjustment
    // would be worse than a loud caveat.
    if (cfg.moe.enabled && cfg.moe.drop_cold_frac > 0.0f) {
        const int k = n_expert_used;
        if (k > 0 && k <= MoeStreamConfig::drop_low_topk_warn)
            std::fprintf(stderr,
                         "bmoe: WARNING drop-cold-experts=%.2f with top-k %d — the threshold is %.1f%% of the "
                         "routing here, against 12.5%% at the top-k 8 this was measured on. Expect it to discard "
                         "much more, and check output quality on your own task.\n",
                         (double) cfg.moe.drop_cold_frac, k, 100.0 * cfg.moe.drop_cold_frac / k);
    }

    // A cache budget below one token's worst-case cycle: say so once, at load.
    //
    // The number is computable from the model's shape alone, so this costs nothing and is available
    // at the only moment it can still be acted on. It is a warning, not a rejection: the budget is
    // legal, the run is byte-correct, and cache_min_mb already rejects the band it was drawn for.
    // What this catches is the case that number cannot see — a budget comfortably above the fixed
    // floor and still under THIS model's cycle, which --n-expert-used widens without touching the
    // budget. Below the cycle no entry survives to the next token, so the run pays management and
    // RAM for a cache that cannot hit. The engine states the fact; what to do about it is
    // docs/cache-sizing.md's business, not a knob the engine should editorialize about.
    if (info.cache_mb > 0 && info.cache_cycle_mb > 0 && info.cache_mb < info.cache_cycle_mb) {
        std::fprintf(stderr,
                     "bmoe: WARNING expert cache %d MiB is below this model's worst-case token cycle of %d MiB "
                     "at top-k %d — no entry can survive to the next token, so expect a hit rate near zero.\n",
                     info.cache_mb, info.cache_cycle_mb, info.n_expert_used);
    }
}

std::unique_ptr<Session> Session::open(const SessionConfig & input_cfg,
                                       std::string & error,
                                       IRouteTraceSink * route_trace,
                                       IComputeTraceSink * compute_trace,
                                       IIoTraceSink * io_trace) {
    if (std::string invalid = check_session_config(input_cfg); !invalid.empty()) {
        error = std::move(invalid);
        return nullptr;
    }

    SessionConfig cfg = input_cfg;
    cfg.n_batch = std::min(cfg.n_batch, cfg.n_ctx);
    cfg.n_ubatch = cfg.n_ubatch > 0 ? std::min(cfg.n_ubatch, cfg.n_batch) : cfg.n_batch;

    // Create the session first so its Impl destructor owns backend teardown from this point
    // on: any failure below returns nullptr, destroying `self`, which frees the backend once.
    std::unique_ptr<Session> self(new Session());
    Impl & im = *self->impl_;
    im.cfg = cfg;

    // llama_backend_init/free are process-global and NOT reference counted. In the pinned llama.cpp
    // init is idempotent (time and f16 tables, backend registry) and free only releases the
    // quantization tables that ggml_quantize_chunk rebuilds on demand, so one session's teardown
    // does not disturb another (tests/multi_session_test.cpp). Re-check both on a submodule bump.
    llama_backend_init();
    im.backend_inited = true;

    const auto t_load0 = Clock::now();
    LazyGgufMeta meta(cfg.model_path);
    if (!im.load_model(meta, error)) return nullptr;

    // The MTP block lives at layer index n_layer and routes experts of its own, so with speculation
    // on the hook and the streamer must span the trunk PLUS the head. The index space is contiguous
    // and the tensor naming identical, so this bound is the only thing standing between the streamer
    // and the MTP experts — left at n_layer they are silently skipped and stay mmap-resident.
    const int n_layer_streamed = im.n_layer + (cfg.spec.is_mtp() ? im.n_layer_nextn : 0);
    MoeRecipe recipe{};
    if (cfg.moe.enabled && !resolve_stream_recipe(im.arch, n_layer_streamed, cfg.model_path, meta, recipe, error))
        return nullptr;

    if (!im.init_chat_templates(error)) return nullptr;

    im.hook = std::make_unique<RouterHook>(recipe, n_layer_streamed);
    im.hook->set_prefetch_layers(cfg.moe.prefetch_layers);
    im.hook->set_drop_policy(cfg.moe.drop_cold_frac, cfg.moe.drop_renorm, cfg.moe.drop_prefill);
    im.hook->set_expert_substitute(cfg.moe.substitute_lambda);
    im.hook->set_predict_log(cfg.moe.predict_log);
    im.hook->set_predict_prefetch(cfg.moe.predict_prefetch, cfg.moe.predict_spec_max);
    im.hook->set_route_ahead(cfg.moe.route_ahead);

    // The streamer needs the callback to see routing; the compute trace needs it to time nodes.
    // Installing it for the trace alone is what lets a NON-streamed run be measured — the dense
    // mmap baseline the streamed numbers are argued against.
    if (!im.create_contexts(cfg.moe.enabled || cfg.dense_stream.enabled || compute_trace, error)) return nullptr;
    if (cfg.dense_stream.enabled && !im.bind_dense_stream(meta, error)) return nullptr;
    if (cfg.moe.enabled && !im.bind_expert_stream(recipe, n_layer_streamed, meta, route_trace, io_trace, error))
        return nullptr;
    im.attach_decode_traces(compute_trace);
    im.build_run_info(meta);
    im.warn_about_load();

    im.load_seconds = secs(t_load0, Clock::now());
    return self;
}

} // namespace meitte
