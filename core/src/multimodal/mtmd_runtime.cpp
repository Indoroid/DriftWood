#include "mtmd_runtime.h"

#include "mtmd-helper.h"

namespace bmoe {

bool MtmdRuntime::init(const MultimodalConfig & cfg,
                       const llama_model * model,
                       int n_threads,
                       std::string & error) {
    reset();
    if (!cfg.enabled()) return true;

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = cfg.offload;
    params.print_timings = false;
    params.n_threads = n_threads;
    params.warmup = cfg.warmup;
    params.image_min_tokens = cfg.image_min_tokens;
    params.image_max_tokens = cfg.image_max_tokens;
    params.batch_max_tokens = cfg.batch_max_tokens;

    // The projector graph is independent from the text-model graph. DriftWood's RouterHook stays
    // installed on the llama_context; mtmd later feeds media embeddings through that same context.
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;

    ctx_.reset(mtmd_init_from_file(cfg.mmproj_path.c_str(), model, params));
    if (!ctx_) {
        error = "failed to load multimodal projector: " + cfg.mmproj_path;
        return false;
    }
    return true;
}

void MtmdRuntime::reset() {
    ctx_.reset();
}

const char * MtmdRuntime::marker() const {
    return ctx_ ? mtmd_get_marker(ctx_.get()) : mtmd_default_marker();
}

MtmdPrefillResult MtmdRuntime::prefill(llama_context * lctx,
                                       const std::string & prompt,
                                       const std::vector<MediaInput> & media,
                                       int n_batch,
                                       llama_pos max_prompt_pos) {
    MtmdPrefillResult out;
    if (!ctx_) {
        out.error = "multimodal projector is not loaded";
        return out;
    }
    if (media.empty()) {
        out.error = "multimodal prefill called without media";
        return out;
    }

    mtmd::bitmaps bitmaps;
    bitmaps.entries.reserve(media.size());
    for (const MediaInput & input : media) {
        if (input.bytes.empty()) {
            out.error = "empty media input: " + input.name;
            return out;
        }
        mtmd_helper_bitmap_wrapper decoded = mtmd_helper_bitmap_init_from_buf(
            ctx_.get(), input.bytes.data(), input.bytes.size(), /*placeholder*/ false);
        if (!decoded.bitmap) {
            out.error = "failed to decode media: " + input.name;
            return out;
        }
        // Buffer input supports image/audio only, so video_ctx is not expected/owned here.
        bitmaps.entries.emplace_back(decoded.bitmap);
    }

    std::vector<const mtmd_bitmap *> bitmap_ptrs = bitmaps.c_ptr();
    mtmd::input_chunks_ptr chunks{mtmd_input_chunks_init()};
    if (!chunks) {
        out.error = "failed to allocate multimodal input chunks";
        return out;
    }

    mtmd_input_text text{};
    text.text = prompt.data();
    text.text_len = prompt.size();
    text.add_special = true;
    text.parse_special = true;

    const int32_t rc = mtmd_tokenize(ctx_.get(), chunks.get(), &text,
                                     bitmap_ptrs.data(), bitmap_ptrs.size());
    if (rc == 1) {
        out.error = "number of media markers does not match supplied media";
        return out;
    }
    if (rc != 0) {
        out.error = "multimodal preprocessing/tokenization failed (mtmd code " + std::to_string(rc) + ")";
        return out;
    }

    out.n_tokens = mtmd_helper_get_n_tokens(chunks.get());
    const llama_pos n_pos = mtmd_helper_get_n_pos(chunks.get());
    if (n_pos <= 0) {
        out.error = "multimodal prompt produced no decoder positions";
        return out;
    }
    if (n_pos > max_prompt_pos) {
        out.error = "multimodal prompt positions (" + std::to_string(n_pos) +
                    ") exceed available context before generation (" + std::to_string(max_prompt_pos) + ")";
        return out;
    }

    llama_pos new_n_past = 0;
    const int32_t eval_rc = mtmd_helper_eval_chunks(ctx_.get(), lctx, chunks.get(),
                                                    /*n_past*/ 0, /*seq_id*/ 0,
                                                    n_batch, /*logits_last*/ true,
                                                    &new_n_past);
    if (eval_rc != 0) {
        out.error = "multimodal prefill decode failed (mtmd code " + std::to_string(eval_rc) + ")";
        return out;
    }

    out.n_past = new_n_past;
    out.ok = true;
    return out;
}

} // namespace bmoe
