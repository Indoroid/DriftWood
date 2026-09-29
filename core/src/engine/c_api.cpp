#include "bmoe/meitte.h"
#include "bmoe/session.h"
#include "bmoe/version.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>

struct meitte_session {
    std::unique_ptr<meitte::Session> value;
    uint64_t media_max_bytes = 0;
};
struct meitte_result {
    meitte::RunResult value;
};

namespace {

bool has_field(size_t struct_size, size_t offset, size_t field_size) {
    return struct_size >= offset && struct_size - offset >= field_size;
}

#define MEITTE_COPY_FIELD(type, output, input, size, field)                                                            \
    if (has_field((size), offsetof(type, field), sizeof((input)->field))) (output).field = (input)->field

constexpr size_t round_up(size_t n, size_t align) {
    return (n + align - 1) / align * align;
}

// sizeof(meitte_config) before the dense-stream fields. That layout ended at release_mmap, and its
// sizeof included 4 bytes of tail padding, which dense_stream now occupies. A caller compiled against
// that header passes this size with uninitialized padding, so the size means "no dense fields".
constexpr size_t k_config_size_before_dense = round_up(offsetof(meitte_config, dense_stream), alignof(meitte_config));

size_t config_content_size(size_t struct_size) {
    return struct_size == k_config_size_before_dense ? offsetof(meitte_config, dense_stream) : struct_size;
}

} // namespace

meitte_config meitte_default_config() {
    meitte_config out{};
    out.struct_size = sizeof(out);
    out.abi_version = MEITTE_ABI_VERSION;
    out.context_size = 2048;
    out.threads = 4;
    out.chat = 1;
    out.video_fps = 1.0f;
    out.video_max_frames = 32;
    out.media_max_bytes = 64ull * 1024 * 1024;
    return out;
}

meitte_request meitte_default_request() {
    meitte_request out{};
    out.struct_size = sizeof(out);
    out.abi_version = MEITTE_ABI_VERSION;
    out.max_tokens = 128;
    out.clear_kv = 1;
    return out;
}

namespace {

// Copy the defaults into the caller's struct, never past the size the caller owns.
template <typename T> void init_prefix(T * out, size_t size, const T & defaults) {
    if (!out || size < offsetof(T, abi_version) + sizeof(uint32_t)) return;
    const size_t n = size < sizeof(T) ? size : sizeof(T);
    std::memcpy(out, &defaults, n);
    out->struct_size = n;
}

} // namespace

void meitte_config_init(meitte_config * config, size_t size) {
    init_prefix(config, size, meitte_default_config());
}

void meitte_request_init(meitte_request * request, size_t size) {
    init_prefix(request, size, meitte_default_request());
}

uint32_t meitte_abi_version() {
    return MEITTE_ABI_VERSION;
}

const char * meitte_version() {
    return meitte::version();
}

meitte_session * meitte_open(const meitte_config * cfg, char * error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    try {
        if (!cfg || !has_field(cfg->struct_size, offsetof(meitte_config, model_path), sizeof(cfg->model_path)) ||
            cfg->abi_version != MEITTE_ABI_VERSION)
            throw std::invalid_argument("invalid config size or ABI version");
        meitte_config input = meitte_default_config();
        const size_t config_size = config_content_size(cfg->struct_size);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, model_path);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, projector_path);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, context_size);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, threads);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, streaming);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, cache_mb);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, chat);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, kv_unified);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, speculation);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, context_grow);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, context_summarize);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, context_trim);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, context_min);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, context_max);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, video_fps);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, video_max_frames);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, ffmpeg_bin_dir);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, media_max_bytes);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, release_mmap);
        MEITTE_COPY_FIELD(meitte_config, input, cfg, config_size, dense_stream);
        if (!input.model_path || !input.model_path[0]) throw std::invalid_argument("model path is required");
        // Dense streaming was removed. The fields keep the struct layout; a caller that still asks
        // for the mode gets an error instead of a silent mmap run.
        if (input.dense_stream) throw std::invalid_argument("dense streaming was removed; load dense models with mmap");
        if (input.speculation < 0 || input.speculation > 2) throw std::invalid_argument("invalid speculation source");
        auto context_mode = [](int32_t value) {
            if (value < 0 || value > 2) throw std::invalid_argument("invalid context mode");
            return static_cast<meitte::ContextMode>(value);
        };
        meitte::RunConfig rc;
        rc.model_path = input.model_path;
        rc.n_ctx = input.context_size;
        rc.n_threads = input.threads;
        rc.chatml = input.chat != 0;
        rc.kv_unified = input.kv_unified != 0;
        rc.moe.enabled = input.streaming != 0;
        rc.moe.cache_mb = input.cache_mb;
        rc.moe.release_mmap = input.release_mmap != 0;
        rc.context.grow = context_mode(input.context_grow);
        rc.context.summarize = context_mode(input.context_summarize);
        rc.context.trim = context_mode(input.context_trim);
        rc.context.min_ctx = input.context_min;
        rc.context.max_ctx = input.context_max;
        if (input.projector_path) {
            rc.multimodal.mmproj_path = input.projector_path;
            rc.multimodal.video_fps = input.video_fps;
            rc.multimodal.video_max_frames = input.video_max_frames;
            rc.multimodal.media_max_bytes = input.media_max_bytes;
            if (input.ffmpeg_bin_dir) rc.multimodal.ffmpeg_bin_dir = input.ffmpeg_bin_dir;
        }
        rc.spec.source = input.speculation == 1   ? meitte::DraftSource::ngram
                         : input.speculation == 2 ? meitte::DraftSource::mtp
                                                  : meitte::DraftSource::none;
        const auto valid = meitte::validate(rc);
        if (!valid) throw std::invalid_argument(valid.error);
        std::string detail;
        auto out = std::make_unique<meitte_session>();
        out->value = meitte::Session::open(meitte::session_config_from(rc), detail);
        if (!out->value) throw std::runtime_error(detail);
        out->media_max_bytes = rc.multimodal.media_max_bytes;
        return out.release();
    } catch (const std::exception & ex) {
        if (error && capacity) std::snprintf(error, capacity, "%s", ex.what());
    } catch (...) {
        if (error && capacity) std::snprintf(error, capacity, "%s", "unknown library error");
    }
    return nullptr;
}

meitte_result *
meitte_generate(meitte_session * session, const meitte_request * req, meitte_token_callback callback, void * user) {
    try {
        auto out = std::make_unique<meitte_result>();
        try {
            if (!req || !has_field(req->struct_size, offsetof(meitte_request, prompt), sizeof(req->prompt)) ||
                req->abi_version != MEITTE_ABI_VERSION)
                throw std::invalid_argument("invalid session, request size, or ABI version");
            meitte_request input = meitte_default_request();
            MEITTE_COPY_FIELD(meitte_request, input, req, req->struct_size, prompt);
            MEITTE_COPY_FIELD(meitte_request, input, req, req->struct_size, max_tokens);
            MEITTE_COPY_FIELD(meitte_request, input, req, req->struct_size, clear_kv);
            MEITTE_COPY_FIELD(meitte_request, input, req, req->struct_size, media);
            MEITTE_COPY_FIELD(meitte_request, input, req, req->struct_size, media_count);
            if (!session || !input.prompt || input.max_tokens <= 0 || (input.media_count && !input.media))
                throw std::invalid_argument("invalid prompt, output limit, or media array");
            meitte::GenerateRequest gr;
            gr.prompt = input.prompt;
            gr.n_predict = input.max_tokens;
            gr.clear_kv = input.clear_kv != 0;
            size_t total = 0;
            for (size_t i = 0; i < input.media_count; ++i) {
                const meitte_media & item = input.media[i];
                if (!item.data || !item.size || item.kind < 0 || item.kind > 3 ||
                    item.size > session->media_max_bytes || total > session->media_max_bytes - item.size)
                    throw std::invalid_argument("invalid media or media byte limit exceeded");
                total += item.size;
                gr.media.push_back({{item.data, item.data + item.size},
                                    item.name ? item.name : "",
                                    static_cast<meitte::MediaKind>(item.kind)});
            }
            out->value = session->value->generate(gr, [&](const meitte::TokenMetrics & metric) {
                if (callback && callback(metric.piece.data(), metric.piece.size(), user)) session->value->cancel();
            });
        } catch (const std::exception & ex) {
            out->value.error = ex.what();
        } catch (...) {
            out->value.error = "unknown library error";
        }
        return out.release();
    } catch (...) {
        return nullptr;
    }
}

#undef MEITTE_COPY_FIELD

void meitte_cancel(meitte_session * session) {
    if (session) session->value->cancel();
}

void meitte_close(meitte_session * session) {
    delete session;
}
int meitte_result_ok(const meitte_result * r) {
    return r && r->value.ok;
}
int meitte_result_cancelled(const meitte_result * r) {
    return r && r->value.cancelled;
}
const char * meitte_result_error(const meitte_result * r) {
    return r ? r->value.error.c_str() : "allocation failed";
}
const char * meitte_result_text(const meitte_result * r) {
    return r ? r->value.generated_text.c_str() : "";
}
const char * meitte_result_reasoning(const meitte_result * r) {
    return r ? r->value.reasoning_text.c_str() : "";
}
void meitte_result_free(meitte_result * r) {
    delete r;
}
