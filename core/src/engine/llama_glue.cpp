#include "llama_glue.h"

#include <cmath>

namespace meitte::detail {

static_assert(SamplingConfig{}.seed == LLAMA_DEFAULT_SEED,
              "SamplingConfig::seed default must mirror LLAMA_DEFAULT_SEED");

llama_rope_scaling_type to_llama_rope_scaling(RopeScalingMode mode) {
    switch (mode) {
    case RopeScalingMode::Auto:
        return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    case RopeScalingMode::None:
        return LLAMA_ROPE_SCALING_TYPE_NONE;
    case RopeScalingMode::Linear:
        return LLAMA_ROPE_SCALING_TYPE_LINEAR;
    case RopeScalingMode::Yarn:
        return LLAMA_ROPE_SCALING_TYPE_YARN;
    case RopeScalingMode::LongRope:
        return LLAMA_ROPE_SCALING_TYPE_LONGROPE;
    }
    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

ggml_type to_ggml_type(KvCacheType type) {
    switch (type) {
    case KvCacheType::F32:
        return GGML_TYPE_F32;
    case KvCacheType::F16:
        return GGML_TYPE_F16;
    case KvCacheType::BF16:
        return GGML_TYPE_BF16;
    case KvCacheType::Q8_0:
        return GGML_TYPE_Q8_0;
    case KvCacheType::Q5_0:
        return GGML_TYPE_Q5_0;
    case KvCacheType::Q5_1:
        return GGML_TYPE_Q5_1;
    case KvCacheType::Q4_0:
        return GGML_TYPE_Q4_0;
    case KvCacheType::Q4_1:
        return GGML_TYPE_Q4_1;
    case KvCacheType::IQ4_NL:
        return GGML_TYPE_IQ4_NL;
    }
    return GGML_TYPE_F16;
}

llama_flash_attn_type to_flash_attn(FlashAttentionMode mode) {
    switch (mode) {
    case FlashAttentionMode::Auto:
        return LLAMA_FLASH_ATTN_TYPE_AUTO;
    case FlashAttentionMode::Enabled:
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    case FlashAttentionMode::Disabled:
        return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
}

void batch_fill(llama_batch & b, const llama_token * toks, int n, llama_pos pos0, bool all_logits) {
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i];
        b.pos[i] = pos0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (int8_t) (all_logits || i == n - 1);
    }
}

llama_token argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_v = logits[0];
    for (int v = 1; v < n_vocab; ++v)
        if (logits[v] > best_v) {
            best_v = logits[v];
            best = v;
        }
    return best;
}

LogSoftmax::LogSoftmax(const float * logits, int n_vocab) {
    max = logits[0];
    for (int v = 1; v < n_vocab; ++v)
        if (logits[v] > max) max = logits[v];
    double sum = 0.0;
    for (int v = 0; v < n_vocab; ++v)
        sum += std::exp((double) (logits[v] - max));
    log_sum = std::log(sum);
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char small[256];
    int n = llama_token_to_piece(vocab, token, small, sizeof(small), 0, true);
    if (n > 0) return std::string(small, n);
    if (n >= 0) return {};

    std::string piece((size_t) -n, '\0');
    n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
    if (n <= 0) return {};
    piece.resize((size_t) n);
    return piece;
}

int tokenize(const llama_vocab * vocab,
             std::string_view text,
             bool add_special,
             bool parse_special,
             std::vector<llama_token> & out) {
    out.resize(text.size() + 8);
    int n =
        llama_tokenize(vocab, text.data(), (int) text.size(), out.data(), (int) out.size(), add_special, parse_special);
    if (n < 0) {
        out.resize((size_t) -n);
        n = llama_tokenize(vocab, text.data(), (int) text.size(), out.data(), (int) out.size(), add_special,
                           parse_special);
    }
    out.resize(n < 0 ? 0 : (size_t) n);
    return n;
}

bool tokenize_media_logical(const llama_vocab * vocab,
                            const std::string & prompt,
                            const char * marker,
                            std::vector<llama_token> & out) {
    if (!marker || !*marker) return false;
    out.clear();
    size_t begin = 0;
    const std::string media_marker(marker);
    std::vector<llama_token> part;
    for (;;) {
        const size_t marker_at = prompt.find(media_marker, begin);
        const size_t end = marker_at == std::string::npos ? prompt.size() : marker_at;
        const std::string_view text(prompt.data() + begin, end - begin);
        if (tokenize(vocab, text, /*add_special*/ false, /*parse_special*/ true, part) < 0) return false;
        out.insert(out.end(), part.begin(), part.end());
        if (marker_at == std::string::npos) break;
        out.push_back(k_media_kv_sentinel);
        begin = marker_at + media_marker.size();
    }
    if (llama_vocab_get_add_bos(vocab)) out.insert(out.begin(), llama_vocab_bos(vocab));
    if (llama_vocab_get_add_eos(vocab)) out.push_back(llama_vocab_eos(vocab));
    return true;
}

llama_sampler * make_sampler_chain(const SamplingConfig & sampling) {
    if (sampling.temp <= 0.0f) return nullptr;
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(sampling.top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(sampling.top_p, /*min_keep*/ 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sampling.temp));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sampling.seed));
    return chain;
}

} // namespace meitte::detail
