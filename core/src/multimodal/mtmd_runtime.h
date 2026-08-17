#pragma once

#include "bmoe/config.h"
#include "bmoe/session.h"
#include "llama.h"
#include "mtmd.h"

#include <cstddef>
#include <string>
#include <vector>

namespace bmoe {

struct MtmdPrefillResult {
    bool ok = false;
    std::string error;
    size_t n_tokens = 0;
    llama_pos n_past = 0;
};

class MtmdRuntime {
public:
    bool init(const MultimodalConfig & cfg,
              const llama_model * model,
              int n_threads,
              std::string & error);
    void reset();
    bool enabled() const { return ctx_ != nullptr; }
    const char * marker() const;

    MtmdPrefillResult prefill(llama_context * lctx,
                              const std::string & prompt,
                              const std::vector<MediaInput> & media,
                              int n_batch,
                              llama_pos max_prompt_pos);

private:
    mtmd::context_ptr ctx_{nullptr};
};

} // namespace bmoe
