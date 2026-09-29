#include "mtmd_runtime.h"
#include "audio_convert.h"
#include "media_io.h"
#include "mtmd-helper.h"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <algorithm>

namespace meitte {
bool MtmdRuntime::init(const MultimodalConfig & cfg, const llama_model * model, int n_threads, std::string & error) {
    reset();
    if (!cfg.enabled()) return true;
    if (cfg.offload) {
        error = "projector offload is not supported: Meitte is CPU-only";
        return false;
    }
    if (!std::isfinite(cfg.video_fps) || cfg.video_fps <= 0 || cfg.video_max_frames <= 0 || !cfg.media_max_bytes ||
        cfg.media_max_bytes > std::numeric_limits<size_t>::max()) {
        error = "media frame rate, frame limit, and byte limit must be positive";
        return false;
    }
    cfg_ = cfg;
    auto params = mtmd_context_params_default();
    params.use_gpu = false;
    params.print_timings = false;
    params.n_threads = n_threads;
    params.warmup = cfg.warmup;
    params.image_min_tokens = cfg.image_min_tokens;
    params.image_max_tokens = cfg.image_max_tokens;
    params.batch_max_tokens = cfg.batch_max_tokens;
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;
    ctx_.reset(mtmd_init_from_file(cfg.mmproj_path.c_str(), model, params));
    if (!ctx_) error = "failed to load multimodal projector: " + cfg.mmproj_path;
    return ctx_ != nullptr;
}
void MtmdRuntime::reset() {
    ctx_.reset();
}
const char * MtmdRuntime::marker() const {
    return ctx_ ? mtmd_get_marker(ctx_.get()) : mtmd_default_marker();
}
std::string MtmdRuntime::projector_capability_error(bool audio) const {
    const bool vision_ok = mtmd_support_vision(ctx_.get());
    const bool audio_ok = mtmd_support_audio(ctx_.get());
    if (audio ? audio_ok : vision_ok) return {};
    const char * has = vision_ok ? "vision only" : audio_ok ? "audio only" : "neither vision nor audio";
    return std::string(audio ? "audio input requires a projector with audio support"
                             : "image input requires a projector with vision support") +
           "; the loaded projector supports " + has;
}

namespace {
struct VideoInput {
    mtmd_helper::video_ptr video;
    const MultimodalConfig * cfg;
    std::function<bool()> cancelled;
    std::shared_ptr<uint64_t> decoded_bytes;
    int frames = 0;
    static int next(size_t, void * opaque, mtmd_bitmap ** bitmap, char ** text) {
        auto & self = *static_cast<VideoInput *>(opaque);
        if (self.cancelled && self.cancelled()) return -2;
        int rc = mtmd_helper_video_read_next(self.video.get(), bitmap, text);
        if (rc != 0) return rc;
        size_t bytes = *bitmap ? mtmd_bitmap_get_n_bytes(*bitmap) : 0;
        if ((*bitmap && ++self.frames > self.cfg->video_max_frames) ||
            bytes > self.cfg->media_max_bytes - *self.decoded_bytes) {
            if (*bitmap) mtmd_bitmap_free(*bitmap);
            std::free(*text);
            *bitmap = nullptr;
            *text = nullptr;
            return -2;
        }
        *self.decoded_bytes += bytes;
        return 0;
    }
};
} // namespace

bool MtmdRuntime::prepare(const std::string & prompt,
                          const std::vector<MediaInput> & media,
                          Prepared & out,
                          std::string & error,
                          const std::function<bool()> & cancelled) {
    out.chunks.reset();
    out.bitmaps.entries.clear();
    out.video_owners.clear();
    out.chunk_media_kinds.clear();
    out.n_tokens = 0;
    out.n_pos = 0;
    out.text_tail.clear();
    if (!ctx_ || media.empty()) {
        error = "media preparation requires a projector and media";
        return false;
    }
    // Tokenization consumes lazy video callbacks. Keep their owners at stable addresses.
    uint64_t input_bytes = 0;
    auto decoded_bytes = std::make_shared<uint64_t>(0);
    std::vector<MediaKind> bitmap_kinds;
    bitmap_kinds.reserve(media.size());
    for (const auto & input : media) {
        if (cancelled && cancelled()) {
            error = "media preparation cancelled";
            return false;
        }
        if (input.bytes.empty() || input.bytes.size() > cfg_.media_max_bytes - input_bytes) {
            error = "empty media or media input byte limit exceeded";
            return false;
        }
        input_bytes += input.bytes.size();
        const MediaKind kind = input.kind == MediaKind::Auto ? media_kind_from_name(input.name) : input.kind;
        if (kind == MediaKind::Video) {
            if (!mtmd_support_vision(ctx_.get()) || !mtmd_helper_support_video(ctx_.get())) {
                error = "video requires a vision projector and an mtmd build with video support";
                return false;
            }
            auto options = mtmd_helper_video_init_params_default();
            options.fps_target = cfg_.video_fps;
            options.ffmpeg_bin_dir = cfg_.ffmpeg_bin_dir.empty() ? nullptr : cfg_.ffmpeg_bin_dir.c_str();
            auto video = std::make_shared<VideoInput>();
            video->video.reset(
                mtmd_helper_video_init_from_buf(ctx_.get(), input.bytes.data(), input.bytes.size(), options));
            if (!video->video) {
                error = "video decoding failed; check FFmpeg and ffprobe availability";
                return false;
            }
            video->cfg = &cfg_;
            video->cancelled = cancelled;
            video->decoded_bytes = decoded_bytes;
            auto * bitmap = mtmd_bitmap_init_lazy(ctx_.get(), input.name.c_str(), video.get(), VideoInput::next);
            if (!bitmap) {
                error = "video bitmap allocation failed";
                return false;
            }
            out.bitmaps.entries.emplace_back(bitmap);
            bitmap_kinds.push_back(kind);
            out.video_owners.push_back(std::move(video));
        } else {
            // Reject a declared kind the projector cannot encode before any decoder runs. Otherwise
            // the audio decoder gets a sample rate of 0 and reports an invalid input.
            if (kind == MediaKind::Audio || kind == MediaKind::Image) {
                error = projector_capability_error(kind == MediaKind::Audio);
                if (!error.empty()) return false;
            }
            auto decoded = mtmd_helper_bitmap_init_from_buf(ctx_.get(), input.bytes.data(), input.bytes.size(), false,
                                                            mtmd_helper_init_opt_default());
            if (decoded.video_ctx) mtmd_helper_video_free(decoded.video_ctx);
            if (!decoded.bitmap && kind == MediaKind::Audio) {
                std::vector<float> audio;
                if (!decode_audio_ffmpeg(input.bytes, cfg_.ffmpeg_bin_dir, mtmd_get_audio_sample_rate(ctx_.get()),
                                         cfg_.media_max_bytes - *decoded_bytes, cancelled, audio, error))
                    return false;
                decoded.bitmap = mtmd_bitmap_init_from_audio(audio.size(), audio.data());
            }
            if (!decoded.bitmap) {
                error = "failed to decode image/audio: " + input.name;
                return false;
            }
            out.bitmaps.entries.emplace_back(decoded.bitmap);
            // An Auto input is known only after decoding. The bitmap is owned by out.bitmaps now.
            error = projector_capability_error(mtmd_bitmap_is_audio(decoded.bitmap));
            if (!error.empty()) return false;
            bitmap_kinds.push_back(kind);
            size_t size = mtmd_bitmap_get_n_bytes(decoded.bitmap);
            if (size > cfg_.media_max_bytes - *decoded_bytes) {
                error = "decoded media byte limit exceeded";
                return false;
            }
            *decoded_bytes += size;
        }
    }
    out.chunks.reset(mtmd_input_chunks_init());
    if (!out.chunks) {
        error = "media chunk allocation failed";
        return false;
    }
    mtmd_input_text text{};
    text.text = prompt.data();
    text.text_len = prompt.size();
    text.add_special = true;
    text.parse_special = true;
    auto bitmaps = out.bitmaps.c_ptr();
    int rc = mtmd_tokenize(ctx_.get(), out.chunks.get(), &text, bitmaps.data(), bitmaps.size());
    if (rc != 0) {
        if (error.empty()) error = "media tokenization failed (marker mismatch or invalid media)";
        out.chunks.reset();
        out.bitmaps.entries.clear();
        return false;
    }
    // Keep bitmaps and video handles alive until their prepared chunks have been evaluated.
    out.n_tokens = mtmd_helper_get_n_tokens(out.chunks.get());
    out.n_pos = mtmd_helper_get_n_pos(out.chunks.get());
    size_t bitmap_index = 0;
    out.chunk_media_kinds.assign(mtmd_input_chunks_size(out.chunks.get()), MediaKind::Auto);
    for (size_t i = 0; i < mtmd_input_chunks_size(out.chunks.get()); ++i) {
        auto * chunk = mtmd_input_chunks_get(out.chunks.get(), i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            out.text_tail.clear();
            // Upstream emits media chunks in marker order. Keep Auto if a future helper changes it.
            out.chunk_media_kinds[i] =
                bitmap_index < bitmap_kinds.size() ? bitmap_kinds[bitmap_index] : MediaKind::Auto;
            ++bitmap_index;
        } else {
            size_t n = 0;
            const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &n);
            if (n) out.text_tail.insert(out.text_tail.end(), ids, ids + n);
        }
    }
    return true;
}

MtmdPrefillResult MtmdRuntime::evaluate(llama_context * lctx,
                                        Prepared & prepared,
                                        int n_batch,
                                        const DecodeObserver & observer,
                                        const std::function<bool()> & cancelled) {
    MtmdPrefillResult out;
    if (!prepared.chunks || n_batch <= 0) {
        out.error = "invalid prepared media or batch size";
        return out;
    }
    llama_pos pos = 0;
    const size_t count = mtmd_input_chunks_size(prepared.chunks.get());
    for (size_t i = 0; i < count; ++i) {
        if (cancelled && cancelled()) {
            out.error = "media prefill cancelled";
            return out;
        }
        const auto * chunk = mtmd_input_chunks_get(prepared.chunks.get(), i);
        int rc = 0;
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t size = 0;
            const auto * tokens = mtmd_input_chunk_get_tokens_text(chunk, &size);
            auto batch = llama_batch_init(n_batch, 0, 1);
            for (size_t offset = 0; offset < size && rc == 0;) {
                batch.n_tokens = (int) std::min<size_t>(n_batch, size - offset);
                for (int j = 0; j < batch.n_tokens; ++j) {
                    batch.token[j] = tokens[offset + j];
                    batch.pos[j] = pos + j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j] = i + 1 == count && offset + j + 1 == size;
                }
                if (observer.before) observer.before(pos, batch.n_tokens, MediaKind::Auto);
                rc = llama_decode(lctx, batch);
                if (rc == 0 && observer.after) observer.after(batch);
                pos += batch.n_tokens;
                offset += batch.n_tokens;
                if (cancelled && cancelled()) rc = 1;
            }
            llama_batch_free(batch);
        } else {
            const auto projector_start = std::chrono::steady_clock::now();
            rc = mtmd_encode_chunk(ctx_.get(), chunk);
            out.projector_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - projector_start).count();
            if (rc == 0) {
                struct Callback {
                    const DecodeObserver & observer;
                    const std::function<bool()> & cancelled;
                    MediaKind media_kind;
                    int remaining;
                    int batch_size;
                    int offset;
                    static int32_t after(const mtmd_helper_embd_batch * embd, void * opaque) {
                        auto & self = *static_cast<Callback *>(opaque);
                        // The observer reads n_tokens and pos only. pos is section-major, so its first
                        // n_tokens entries are the decoder positions of this sub-batch.
                        llama_batch batch{};
                        batch.n_tokens = embd->n_tokens;
                        batch.embd = const_cast<float *>(embd->embd);
                        batch.pos = const_cast<llama_pos *>(embd->pos);
                        if (self.observer.after) self.observer.after(batch);
                        self.remaining -= batch.n_tokens;
                        self.offset += batch.n_tokens;
                        if (self.cancelled && self.cancelled()) return 1;
                        if (self.remaining > 0 && self.observer.before)
                            self.observer.before(self.offset, std::min(self.remaining, self.batch_size),
                                                 self.media_kind);
                        return 0;
                    }
                } callback{observer,
                           cancelled,
                           i < prepared.chunk_media_kinds.size() ? prepared.chunk_media_kinds[i] : MediaKind::Auto,
                           (int) mtmd_input_chunk_get_n_tokens(chunk),
                           n_batch,
                           0};
                // Trace rows start at a local ordinal. The observer maps them to batch.pos after decode.
                if (observer.before) observer.before(0, std::min(callback.remaining, n_batch), callback.media_kind);
                rc = mtmd_helper_decode_image_chunk(ctx_.get(), lctx, chunk, mtmd_get_output_embd(ctx_.get()), pos, 0,
                                                    n_batch, &pos, Callback::after, &callback);
            }
        }
        if (rc != 0) {
            out.error = "multimodal prefill decode failed";
            return out;
        }
    }
    out.n_tokens = prepared.n_tokens;
    out.n_past = pos;
    out.text_tail = prepared.text_tail;
    out.ok = true;
    return out;
}

MtmdPrefillResult MtmdRuntime::prefill(llama_context * lctx,
                                       const std::string & prompt,
                                       const std::vector<MediaInput> & media,
                                       int n_batch,
                                       llama_pos max_prompt_pos,
                                       const std::function<bool()> & cancelled) {
    MtmdPrefillResult out;
    Prepared prepared;
    if (!prepare(prompt, media, prepared, out.error, cancelled)) return out;
    if (prepared.n_pos <= 0 || prepared.n_pos > max_prompt_pos || prepared.n_tokens > (size_t) max_prompt_pos) {
        out.error = "multimodal prompt exceeds context capacity; reopen with a larger context";
        return out;
    }
    return evaluate(lctx, prepared, n_batch, {}, cancelled);
}
} // namespace meitte
