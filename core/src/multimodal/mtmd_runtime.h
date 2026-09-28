#pragma once

#include "bmoe/config.h"
#include "bmoe/session.h"
#include "llama.h"
#include "mtmd.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace meitte {

struct MtmdPrefillResult {
    bool ok = false;
    std::string error;
    size_t n_tokens = 0;
    llama_pos n_past = 0;
    double projector_seconds = 0.0;
    std::vector<llama_token> text_tail;
};

class MtmdRuntime {
public:
    bool init(const MultimodalConfig & cfg, const llama_model * model, int n_threads, std::string & error);
    void reset();
    bool enabled() const { return ctx_ != nullptr; }
    const char * marker() const;

    struct Prepared {
        std::vector<std::shared_ptr<void>> video_owners;
        mtmd::bitmaps bitmaps;
        mtmd::input_chunks_ptr chunks{nullptr};
        std::vector<MediaKind> chunk_media_kinds;
        size_t n_tokens = 0;
        llama_pos n_pos = 0;
        std::vector<llama_token> text_tail;
    };

    bool prepare(const std::string & prompt,
                 const std::vector<MediaInput> & media,
                 Prepared & out,
                 std::string & error,
                 const std::function<bool()> & cancelled = {});
    struct DecodeObserver {
        std::function<void(int, int, MediaKind)> before;
        std::function<void(const llama_batch &)> after;
    };
    MtmdPrefillResult evaluate(llama_context * lctx,
                               Prepared & prepared,
                               int n_batch,
                               const DecodeObserver & observer = {},
                               const std::function<bool()> & cancelled = {});

    MtmdPrefillResult prefill(llama_context * lctx,
                              const std::string & prompt,
                              const std::vector<MediaInput> & media,
                              int n_batch,
                              llama_pos max_prompt_pos,
                              const std::function<bool()> & cancelled = {});

private:
    // Empty when the projector can encode the input; otherwise the error that names the missing
    // capability. Requires ctx_.
    std::string projector_capability_error(bool audio) const;

    mtmd::context_ptr ctx_{nullptr};
    MultimodalConfig cfg_;
};

} // namespace meitte
