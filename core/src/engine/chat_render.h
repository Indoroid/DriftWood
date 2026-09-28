#pragma once

// Translation between the public chat request types (bmoe/session.h) and llama.cpp's chat-template
// layer. Kept apart from the generation loop so the request rules — media ordering, tool mapping,
// template kwargs — read in one place and can be tested without a live context.
//
// Internal header: includes llama.cpp's `common` (not the stable public API), so it must not be
// pulled into core/include/bmoe/. See docs/seam.md.

#include "bmoe/session.h"

#include "chat.h"
#include "sampling.h"

#include <memory>
#include <string>
#include <vector>

namespace meitte::detail {

// Check the request's template kwargs and reasoning effort. Returns an empty string when valid.
std::string check_template_request(const GenerateRequest & req);

// Convert req.messages into chat messages. Structured media parts become `marker`, and must
// reference req.media in order (mtmd binds media to markers purely by encounter order). Without
// structured parts, all of req.media attaches as `media_prefix` to the last user message — the
// backward-compatible form. `media_enabled` tells whether a projector is loaded.
bool render_request_messages(const GenerateRequest & req,
                             const char * marker,
                             bool media_enabled,
                             const std::string & media_prefix,
                             std::vector<common_chat_msg> & out,
                             std::string & error);

// Template inputs that render `history` for this request: tools, tool choice, thinking and kwargs.
common_chat_templates_inputs make_template_inputs(const GenerateRequest & req,
                                                  const std::vector<common_chat_msg> & history);

std::vector<ToolCall> to_tool_calls(const std::vector<common_chat_tool_call> & calls);

// The oldest complete user turn, as history[first, end). Tool exchanges stay attached to their
// initiating turn. False when the history has no turn older than the newest one, which is never
// compacted. `protected_content` is set when the turn holds a system message or a media marker.
bool oldest_complete_turn(const std::vector<common_chat_msg> & history,
                          const char * marker,
                          size_t & first,
                          size_t & end,
                          bool & protected_content);

// One turn's reasoning budget. llama.cpp owns the mechanism: its common_sampler enforces the budget
// and forces the template's end sequence. A second instance of llama.cpp's reasoning-budget state
// machine, with an unlimited budget, observes which generated tokens fall inside the reasoning span,
// because the pinned common_sampler does not expose its own state. Meitte owns the accounting
// policy: reasoning tokens do not consume the answer's n_predict allowance (see GenerateRequest).
class ReasoningBudget {
public:
    // `prompt` is the rendered prompt, whose trailing reasoning opener (if a template emits one
    // before decoding) counts as already inside the span. False when llama.cpp cannot build it.
    bool init(const llama_model * model,
              const llama_vocab * vocab,
              const SamplingConfig & sampling,
              int budget_tokens,
              const common_chat_params & chat_params,
              const std::string & prompt);

    common_sampler * sampler() const { return sampler_.get(); }

    // Reasoning tokens the turn may generate outside the answer allowance: the budget plus the
    // forced end sequence. Reasoning past it (a model that reopens its span) is charged to n_predict,
    // so a turn always ends within n_predict + allowance() tokens.
    int allowance() const { return allowance_; }

    // Accept a generated token into the sampler and the observer. True when the token belongs to the
    // reasoning span, its opening and closing delimiters included.
    bool accept(llama_token token);

private:
    common_sampler_ptr sampler_;
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> observer_{nullptr, llama_sampler_free};
    int allowance_ = 0;
};

} // namespace meitte::detail
