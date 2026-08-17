#include "bmoe/config.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace bmoe {

namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

const char * kv_cache_type_name(KvCacheType type) {
    switch (type) {
    case KvCacheType::F32: return "f32";
    case KvCacheType::F16: return "f16";
    case KvCacheType::BF16: return "bf16";
    case KvCacheType::Q8_0: return "q8_0";
    case KvCacheType::Q5_0: return "q5_0";
    case KvCacheType::Q5_1: return "q5_1";
    case KvCacheType::Q4_0: return "q4_0";
    case KvCacheType::Q4_1: return "q4_1";
    case KvCacheType::IQ4_NL: return "iq4_nl";
    }
    return "f16";
}

const char * flash_attention_mode_name(FlashAttentionMode mode) {
    switch (mode) {
    case FlashAttentionMode::Auto: return "auto";
    case FlashAttentionMode::Enabled: return "on";
    case FlashAttentionMode::Disabled: return "off";
    }
    return "auto";
}

bool parse_kv_cache_type(const std::string & value, KvCacheType & out) {
    const std::string v = lower(value);
    KvCacheType parsed;
    if (v == "f32") parsed = KvCacheType::F32;
    else if (v == "f16") parsed = KvCacheType::F16;
    else if (v == "bf16") parsed = KvCacheType::BF16;
    else if (v == "q8_0") parsed = KvCacheType::Q8_0;
    else if (v == "q5_0") parsed = KvCacheType::Q5_0;
    else if (v == "q5_1") parsed = KvCacheType::Q5_1;
    else if (v == "q4_0") parsed = KvCacheType::Q4_0;
    else if (v == "q4_1") parsed = KvCacheType::Q4_1;
    else if (v == "iq4_nl") parsed = KvCacheType::IQ4_NL;
    else return false;
    out = parsed;
    return true;
}

bool parse_flash_attention_mode(const std::string & value, FlashAttentionMode & out) {
    const std::string v = lower(value);
    FlashAttentionMode parsed;
    if (v == "auto") parsed = FlashAttentionMode::Auto;
    else if (v == "on") parsed = FlashAttentionMode::Enabled;
    else if (v == "off") parsed = FlashAttentionMode::Disabled;
    else return false;
    out = parsed;
    return true;
}

ValidationResult validate(const RunConfig & cfg) {
    ValidationResult r;
    auto fail = [&](std::string msg) {
        r.ok = false;
        r.error = std::move(msg);
        return r;
    };

    if (cfg.model_path.empty()) {
        return fail("model_path is required");
    }
    if (cfg.n_predict <= 0) {
        return fail("n_predict must be positive");
    }
    if (cfg.n_threads <= 0) {
        return fail("n_threads must be positive");
    }
    if (cfg.n_ctx <= 0) {
        return fail("n_ctx must be positive");
    }
    if (cfg.n_batch <= 0) {
        return fail("n_batch must be positive");
    }
    if (cfg.n_ubatch < 0) {
        return fail("n_ubatch must be >= 0 (0 = as wide as n_batch)");
    }
    // Lower bound only: 0 means "use the model default". The upper bound (<= the model's
    // real expert count) needs the loaded gguf, so it is deferred to run() where the model
    // is available — same rationale as the streaming checks that stay out of this pure path.
    if (cfg.n_expert_used < 0) {
        return fail("n_expert_used must be >= 0 (0 = model default)");
    }
    if (!cfg.media_paths.empty() && !cfg.multimodal.enabled()) {
        return fail("media input requires --mmproj");
    }
    if (cfg.multimodal.batch_max_tokens <= 0) {
        return fail("multimodal.batch_max_tokens must be positive");
    }
    if (cfg.multimodal.image_min_tokens < -1 || cfg.multimodal.image_max_tokens < -1) {
        return fail("multimodal image token limits must be -1 (metadata default) or >= 0");
    }
    if (cfg.multimodal.image_min_tokens >= 0 && cfg.multimodal.image_max_tokens >= 0 &&
        cfg.multimodal.image_min_tokens > cfg.multimodal.image_max_tokens) {
        return fail("multimodal.image_min_tokens must not exceed image_max_tokens");
    }
    if (!cfg.think && !cfg.reasoning_effort.empty() && lower(cfg.reasoning_effort) != "none") {
        return fail("reasoning_effort requires thinking to be enabled (use 'none' to disable reasoning)");
    }
    if (cfg.reasoning_effort.size() > 64) return fail("reasoning_effort must be at most 64 bytes");
    if (cfg.cache_type_v != KvCacheType::F32 && cfg.cache_type_v != KvCacheType::F16 &&
        cfg.cache_type_v != KvCacheType::BF16 && cfg.flash_attention == FlashAttentionMode::Disabled) {
        return fail("quantized V cache requires Flash Attention; use auto/on or choose an unquantized V cache type");
    }

    // Sampling ranges are enforced only when sampling is actually on (temp > 0). With temp <= 0
    // the engine takes the argmax path and the other knobs are inert, so a caller that leaves them
    // at any value still describes a valid greedy run — today's default stays valid untouched.
    if (cfg.sampling.temp > 0.0f) {
        if (cfg.sampling.top_k < 0) {
            return fail("sampling.top_k must be >= 0 (0 disables the top-k stage)");
        }
        if (cfg.sampling.top_p <= 0.0f || cfg.sampling.top_p > 1.0f) {
            return fail("sampling.top_p must be in (0, 1]");
        }
    }

    // Verification is exact only because greedy acceptance compares argmax against argmax. Under a
    // sampling chain the accepted prefix would depend on which draws happened to agree, which is a
    // different distribution from the one the caller asked for. Rejected rather than silently
    // ignored: a caller who asked for both was promised something the engine cannot give.
    if (cfg.spec.enabled() && cfg.sampling.temp > 0.0f) {
        return fail("speculative decoding requires greedy decoding (sampling.temp <= 0): verification is "
                    "token-identical to single-token decode only under argmax.");
    }
    // The floor is 1 (draft one token, verify two positions); at 0 there is nothing to verify and
    // the loop degenerates into plain decode with the speculative scaffolding attached. The ceiling
    // is an evidence boundary, not a physical one — see SpecConfig::draft_max.
    if (cfg.spec.enabled() && (cfg.spec.draft_max < 1 || cfg.spec.draft_max > SpecConfig::draft_max_limit)) {
        return fail("spec.draft_max must be in [1, " + std::to_string(SpecConfig::draft_max_limit) + "]");
    }
    // A probability, so [0,1]. 1 would mean "only draft what the head is certain of", which is a
    // legitimate (if extreme) setting; above 1 nothing would ever be drafted and the flag would be
    // an expensive way to decode one token at a time.
    if (cfg.spec.is_mtp() && (cfg.spec.draft_p_min < 0.0f || cfg.spec.draft_p_min > 1.0f)) {
        return fail("spec.draft_p_min must be in [0, 1]");
    }
    // Both of these are source-specific knobs. Accepting one against the other source would silently
    // do nothing, which is exactly the class of "the flag was ignored" bug this validation exists to
    // prevent — the caller asked for a behaviour the chosen source does not have.
    if (cfg.spec.is_ngram() && cfg.spec.draft_p_min != 0.0f) {
        return fail("spec.draft_p_min is an MTP knob (the head's own confidence in its proposal) and has "
                    "no meaning for the n-gram source, which has no probabilities — its confidence gate "
                    "is spec.ngram_min_match.");
    }
    if (cfg.spec.is_ngram() &&
        (cfg.spec.ngram_max_match < 1 || cfg.spec.ngram_max_match > SpecConfig::ngram_match_limit)) {
        return fail("spec.ngram_max_match must be in [1, " + std::to_string(SpecConfig::ngram_match_limit) + "]");
    }
    if (cfg.spec.is_ngram() && (cfg.spec.ngram_min_match < 1 || cfg.spec.ngram_min_match > cfg.spec.ngram_max_match)) {
        return fail(
            "spec.ngram_min_match must be in [1, spec.ngram_max_match=" + std::to_string(cfg.spec.ngram_max_match) +
            "]: a floor above the longest suffix considered would never draft anything.");
    }
    // The verify pass is 1 + draft_max positions and its whole point is that they are computed
    // TOGETHER. A narrower graph splits it back into single-token passes, which spends the draft
    // and keeps none of the amortisation — the feature would cost time and buy nothing. Rejected
    // rather than silently degraded: nothing in the output would show that it happened. 0 means
    // "as wide as n_batch".
    const int effective_batch = std::min(cfg.n_batch, cfg.n_ctx);
    const int effective_ubatch = cfg.n_ubatch > 0 ? std::min(cfg.n_ubatch, effective_batch) : effective_batch;
    if (cfg.spec.enabled() && effective_ubatch < cfg.spec.draft_max + 1) {
        return fail("effective n_ubatch=" + std::to_string(effective_ubatch) + " is narrower than the verify batch (" +
                    std::to_string(cfg.spec.draft_max + 1) +
                    " positions): the graph would be split back into single-token passes and speculation "
                    "would draft at a cost with nothing to show for it. Raise n_batch/n_ubatch or "
                    "lower spec.draft_max.");
    }

    // overlap is meaningless without streaming (it gates the streamer's own reads). The
    // hook-availability check is deferred to run(): validate() stays pure (no native).
    if (cfg.moe.overlap && !cfg.moe.enabled) {
        return fail("moe.overlap requires moe.enabled");
    }

    // The probe rides on the routing nodes the streamer already isolates. Off the streaming path
    // there is nothing for it to attach to — and nothing to learn either: routing does not depend
    // on how the weights got into memory, so a dense run would only reproduce the same numbers
    // more slowly.
    if (cfg.moe.predict_log && !cfg.moe.enabled) {
        return fail("moe.predict_log requires moe.enabled");
    }
    if (cfg.moe.predict_prefetch && !cfg.moe.enabled) {
        return fail("moe.predict_prefetch requires moe.enabled");
    }
    if (cfg.moe.route_ahead > 0 && !cfg.moe.enabled) {
        return fail("moe.route_ahead requires moe.enabled: it attaches to the routing nodes the "
                    "streamer isolates, and off the streaming path there is nothing to commit to");
    }

    if (cfg.moe.enabled) {
        const MoeStreamConfig & m = cfg.moe;
        if (m.io_threads < 1 || m.io_threads > MoeStreamConfig::io_threads_max) {
            return fail("moe.io_threads must be in [1, " + std::to_string(MoeStreamConfig::io_threads_max) + "]");
        }
        if (m.cache_mb < 0) {
            return fail("moe.cache_mb must be >= 0");
        }
        if (m.cache_mb > 0 && m.cache_mb < MoeStreamConfig::cache_min_mb && !m.force_cache) {
            return fail("moe.cache_mb=" + std::to_string(m.cache_mb) + " is in the pathological band (< " +
                        std::to_string(MoeStreamConfig::cache_min_mb) +
                        " MiB): a cache smaller than one token's routed working set thrashes and is "
                        "slower than no cache. Use 0 to disable the cache, a value >= " +
                        std::to_string(MoeStreamConfig::cache_min_mb) + ", or set force_cache to override.");
        }
        if (m.prefetch_layers < 0 || m.prefetch_layers > MoeStreamConfig::prefetch_layers_max) {
            return fail("moe.prefetch_layers must be in [0, " + std::to_string(MoeStreamConfig::prefetch_layers_max) +
                        "]");
        }
        if (m.cache_auto && m.cache_mb > 0) {
            return fail("moe.cache_auto and an explicit moe.cache_mb are mutually exclusive: choose "
                        "auto-sizing (cache_mb = 0, cache_auto) or a fixed budget (cache_mb > 0).");
        }
        if (m.cache_floor_mb < 0) {
            return fail("moe.cache_floor_mb must be >= 0");
        }
        if (m.cache_ceil_mb < 0) {
            return fail("moe.cache_ceil_mb must be >= 0 (0 = no explicit ceiling)");
        }
        // "The LRU cache is on" means a fixed budget OR auto-sizing (which sizes a real LRU cache).
        const bool cache_on = m.cache_mb > 0 || m.cache_auto;
        if (m.prefetch_layers > 0 && !cache_on) {
            return fail("moe.prefetch_layers requires the LRU cache (cache_mb > 0 or cache_auto): "
                        "speculative reads land in the per-layer cache buffers, which do not exist "
                        "with the cache off.");
        }
        if (m.predict_prefetch && !cache_on) {
            return fail("moe.predict_prefetch requires the LRU cache (cache_mb > 0 or cache_auto): "
                        "speculative reads land in the per-layer cache buffers, which do not exist "
                        "with the cache off.");
        }
        if (m.io_two_wave && !m.overlap) {
            return fail("moe.io_two_wave requires moe.overlap: without overlap the caller drains the "
                        "batch synchronously and there is no lane idling on the publish to wake early.");
        }
        if (m.io_two_wave && !cache_on) {
            return fail("moe.io_two_wave requires the LRU cache (cache_mb > 0 or cache_auto): the wave "
                        "split exists to move per-expert page commits off the publish path, and the "
                        "shared-slot path has none.");
        }
        if (m.predict_spec_max < 0 || m.predict_spec_max > 8) {
            return fail("moe.predict_spec_max must be in [0, 8] (0 = retention only, no speculation)");
        }
        if (m.predict_prefetch && m.prefetch_layers > 0) {
            return fail("moe.predict_prefetch and moe.prefetch_layers are mutually exclusive: they "
                        "are two predictors for the same speculative read lanes, and running both "
                        "doubles the speculated bytes for the same future.");
        }
        if (m.route_ahead < 0 || m.route_ahead > MoeStreamConfig::route_ahead_max) {
            return fail("moe.route_ahead must be in [0, " + std::to_string(MoeStreamConfig::route_ahead_max) +
                        "] (0 = off)");
        }
        if (m.route_ahead > 0 && (m.predict_log || m.predict_prefetch || m.prefetch_layers > 0)) {
            return fail("moe.route_ahead excludes predict_log, predict_prefetch and prefetch_layers: "
                        "the probe would grade a predictor against a routing rewritten from that same "
                        "predictor, and a speculative prefetch would bet lanes on a future route-ahead "
                        "has already fixed.");
        }
        // Measured, not theorised: with a draft source on, a verify decode is several positions wide
        // and route-ahead declines every one of them — a run at draft 3 committed 0 routings and
        // passed 249 through. It still pays for itself, though: the prediction GEMVs run, and its
        // early reads become ordinary speculation that can now miss (81% useful against 100% when it
        // commits). Cost with no commitment is worse than either feature alone, so the pair is
        // refused rather than silently charged for. Making it work means committing the whole verify
        // batch to one selection, which costs draft acceptance; see docs/route-ahead.md.
        if (m.route_ahead > 0 && cfg.spec.enabled()) {
            return fail("moe.route_ahead and self-speculative decoding are mutually exclusive: a "
                        "verify decode is several positions wide, so route-ahead declines to commit "
                        "on every one of them while still paying for its prediction and its early "
                        "reads. Turn off --mtp/--ngram, or turn off --route-ahead.");
        }
        if (m.drop_cold_frac > 0.0f && !cache_on) {
            return fail("moe.drop_cold_frac requires the LRU cache (cache_mb > 0 or cache_auto): with the "
                        "cache off every expert is a miss, so the policy stops being cache-aware and "
                        "degenerates into an unconditional weight cut — which is what n_expert_used already "
                        "does, without pretending to consult residency.");
        }
        // Written as a negated inclusive range so NaN (every comparison false) is rejected too,
        // instead of slipping past both this check and the cache_on one above.
        if (!(m.drop_cold_frac >= 0.0f && m.drop_cold_frac <= 1.0f)) {
            return fail("moe.drop_cold_frac must be in [0, 1] (0 = off). Above 1.0 the threshold can "
                        "exceed the largest weight in a routing, which would discard every expert of a "
                        "layer; 1.0 is the uniform share 1/n_expert_used and the useful maximum.");
        }
    }

    return r;
}

} // namespace bmoe
