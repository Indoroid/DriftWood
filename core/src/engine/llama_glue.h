#pragma once

// Small adapters between the bmoe policy types and llama.cpp's public C API: enum mapping, batch
// filling, tokenization, logits helpers and the opt-in sampler chain. The session code uses these
// in several places, so they live here once. Only llama.h and ggml.h are used: nothing in this
// unit depends on llama.cpp's non-stable `common` layer.
//
// Internal header: it includes llama.cpp headers, so it must not be pulled into core/include/bmoe/.

#include "bmoe/config.h"

#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace meitte::detail {

using Clock = std::chrono::steady_clock;

inline double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Keep the policy header free of llama.cpp types. RopeScalingMode::Auto maps to the upstream
// UNSPECIFIED value: llama_context then selects the RoPE method stored in the GGUF.
llama_rope_scaling_type to_llama_rope_scaling(RopeScalingMode mode);
ggml_type to_ggml_type(KvCacheType type);
llama_flash_attn_type to_flash_attn(FlashAttentionMode mode);

// Fill an explicitly-allocated batch with `n` tokens at consecutive positions on sequence 0.
//
// The engine otherwise decodes through llama_batch_get_one, which leaves pos/seq_id/logits null and
// lets llama.cpp infer them. Speculation cannot: the driver reads the batch's sequence ids, and a
// verify pass needs logits at EVERY position, not just the last. So every batch on the speculative
// path is spelled out — including prefill, which the driver must see to keep the draft context's
// KV in step with the target's.
void batch_fill(llama_batch & b, const llama_token * toks, int n, llama_pos pos0, bool all_logits);

llama_token argmax(const float * logits, int n_vocab);

// The log-softmax normaliser of one logits row, computed in a numerically safe order.
struct LogSoftmax {
    float max = 0.0f;
    double log_sum = 0.0; // log(sum(exp(logit - max)))

    LogSoftmax(const float * logits, int n_vocab);
    double logp(const float * logits, llama_token token) const { return (double) (logits[token] - max) - log_sum; }
};

std::string token_piece(const llama_vocab * vocab, llama_token token);

// Tokenize `text` into `out`, growing the buffer once when llama.cpp reports it too small. Returns
// the token count, or a negative value when tokenization fails.
int tokenize(const llama_vocab * vocab,
             std::string_view text,
             bool add_special,
             bool parse_special,
             std::vector<llama_token> & out);

// mtmd replaces its marker with embedding positions, so raw token indices no longer line up with
// the KV after a media turn. A sentinel lets continuation prove the rendered transcript is unchanged
// through each media span without pretending an image is one text token.
constexpr llama_token k_media_kv_sentinel = std::numeric_limits<llama_token>::min();

// Tokenize a prompt that contains media markers. Each marker becomes one k_media_kv_sentinel.
bool tokenize_media_logical(const llama_vocab * vocab,
                            const std::string & prompt,
                            const char * marker,
                            std::vector<llama_token> & out);

// Opt-in sampling. temp <= 0 returns nullptr and the decode loop stays on argmax — the deterministic
// default the byte-identity gates rely on. temp > 0 builds the standard chain, using only the public
// llama_sampler_* API (hard rule 1): common_sampler lives in the non-stable common layer. The caller
// owns the result and frees it with llama_sampler_free().
llama_sampler * make_sampler_chain(const SamplingConfig & sampling);

} // namespace meitte::detail
