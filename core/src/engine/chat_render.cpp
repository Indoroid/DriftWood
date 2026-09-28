#include "chat_render.h"

#include "common.h"
#include "reasoning-budget.h"
#include <nlohmann/json.hpp>

#include <cstdint>

namespace meitte::detail {

std::string check_template_request(const GenerateRequest & req) {
    for (const auto & [key, value] : req.chat_template_kwargs) {
        if (key.empty()) return "chat_template_kwargs contains an empty key";
        if (key.size() > 128) return "chat_template_kwargs keys must be 1..128 bytes";
        if (nlohmann::json::parse(value, nullptr, false).is_discarded())
            return "chat_template_kwargs['" + key + "'] is not valid JSON";
    }
    if (req.reasoning_effort.size() > 64) return "reasoning_effort must be at most 64 bytes";
    return {};
}

bool render_request_messages(const GenerateRequest & req,
                             const char * marker,
                             bool media_enabled,
                             const std::string & media_prefix,
                             std::vector<common_chat_msg> & out,
                             std::string & error) {
    const bool has_media = !req.media.empty();
    bool has_structured_media = false;
    for (const ChatMessage & src : req.messages)
        for (const ChatContentPart & part : src.content_parts)
            has_structured_media |= part.kind == ChatContentKind::Media;

    size_t media_user_index = req.messages.size();
    if (has_media && !has_structured_media) {
        // Backward-compatible API: callers that provide media bytes but no structured
        // parts get the old behaviour — all markers attach to the last user message.
        for (size_t i = req.messages.size(); i > 0; --i) {
            if (req.messages[i - 1].role == "user") {
                media_user_index = i - 1;
                break;
            }
        }
        if (media_user_index == req.messages.size()) {
            error = "multimodal chat request has no user message to attach media to";
            return false;
        }
    }

    out.clear();
    out.reserve(req.messages.size());
    size_t next_media_index = 0;
    for (size_t msg_i = 0; msg_i < req.messages.size(); ++msg_i) {
        const ChatMessage & src = req.messages[msg_i];
        common_chat_msg msg;
        msg.role = src.role;
        if (!src.content_parts.empty()) {
            for (const ChatContentPart & part : src.content_parts) {
                if (part.kind == ChatContentKind::Text) {
                    msg.content += part.text;
                    continue;
                }
                if (!media_enabled) {
                    error = "media content part requires a loaded --mmproj";
                    return false;
                }
                if (part.media_index >= req.media.size()) {
                    error = "media content part references a missing media input";
                    return false;
                }
                if (part.media_index != next_media_index) {
                    error = "media content parts must reference media inputs in request order";
                    return false;
                }
                msg.content += marker;
                ++next_media_index;
            }
        } else {
            msg.content = (has_media && !has_structured_media && msg_i == media_user_index) ? media_prefix + src.content
                                                                                            : src.content;
        }
        msg.reasoning_content = src.reasoning_content;
        msg.tool_name = src.tool_name;
        msg.tool_call_id = src.tool_call_id;
        for (const ToolCall & src_call : src.tool_calls) {
            common_chat_tool_call call;
            call.id = src_call.id;
            call.name = src_call.name;
            call.arguments = src_call.arguments;
            msg.tool_calls.push_back(std::move(call));
        }
        out.push_back(std::move(msg));
    }
    if (has_structured_media && next_media_index != req.media.size()) {
        error = "structured chat did not reference every supplied media input";
        return false;
    }
    return true;
}

common_chat_templates_inputs make_template_inputs(const GenerateRequest & req,
                                                  const std::vector<common_chat_msg> & history) {
    common_chat_templates_inputs inputs;
    inputs.messages = history; // the full conversation, not just this turn
    for (const ChatTool & src : req.tools) {
        common_chat_tool tool;
        tool.name = src.name;
        tool.description = src.description;
        tool.parameters = src.parameters_json;
        inputs.tools.push_back(std::move(tool));
    }
    switch (req.tool_choice) {
    case ChatToolChoice::Required:
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        break;
    case ChatToolChoice::None:
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        break;
    case ChatToolChoice::Auto:
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        break;
    }
    inputs.parallel_tool_calls = req.parallel_tool_calls;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = req.think;
    // AUTO is what bakes reasoning-stripping into the generated parser grammar. It is set
    // here, before apply — the field defaults to NONE, which produces a content-only
    // grammar that leaves <think> markers in the answer no matter how the parse is wired.
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    inputs.chat_template_kwargs = req.chat_template_kwargs;
    inputs.chat_template_kwargs.erase("enable_thinking");
    inputs.chat_template_kwargs.erase("reasoning_effort");
    if (req.think && !req.reasoning_effort.empty())
        inputs.chat_template_kwargs["reasoning_effort"] = nlohmann::json(req.reasoning_effort).dump();
    return inputs;
}

std::vector<ToolCall> to_tool_calls(const std::vector<common_chat_tool_call> & calls) {
    std::vector<ToolCall> out;
    out.reserve(calls.size());
    for (const common_chat_tool_call & src : calls)
        out.push_back({src.id, src.name, src.arguments});
    return out;
}

bool oldest_complete_turn(const std::vector<common_chat_msg> & history,
                          const char * marker,
                          size_t & first,
                          size_t & end,
                          bool & protected_content) {
    first = end = history.size();
    for (size_t i = 0; i < history.size(); ++i) {
        if (history[i].role != "user") continue;
        if (first == history.size())
            first = i;
        else {
            end = i;
            break;
        }
    }
    if (end == history.size()) return false; // The newest turn is protected.
    protected_content = false;
    for (size_t i = first; i < end; ++i)
        protected_content |= history[i].role == "system" || history[i].content.find(marker) != std::string::npos;
    return true;
}

bool ReasoningBudget::init(const llama_model * model,
                           const llama_vocab * vocab,
                           const SamplingConfig & sampling,
                           int budget_tokens,
                           const common_chat_params & chat_params,
                           const std::string & prompt) {
    common_params_sampling params;
    params.seed = sampling.seed;
    params.top_k = sampling.top_k;
    params.top_p = sampling.top_p;
    params.min_keep = 1;
    params.temp = sampling.temp;
    params.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE};
    params.reasoning_budget_tokens = budget_tokens;
    params.generation_prompt = chat_params.generation_prompt;
    const size_t last_thinking_start = prompt.rfind(chat_params.thinking_start_tag);
    size_t last_thinking_end = std::string::npos;
    for (const std::string & tag : chat_params.thinking_end_tags) {
        const size_t at = prompt.rfind(tag);
        if (at != std::string::npos && (last_thinking_end == std::string::npos || at > last_thinking_end))
            last_thinking_end = at;
    }
    const bool thinking_prefilled = last_thinking_start != std::string::npos &&
                                    (last_thinking_end == std::string::npos || last_thinking_end < last_thinking_start);
    // Some Jinja templates emit this opener before decoding, while common_sampler only sees
    // generation_prompt. Arm it explicitly below instead of accepting it twice during init.
    if (thinking_prefilled) params.generation_prompt.clear();
    params.reasoning_budget_start = common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
    for (const std::string & tag : chat_params.thinking_end_tags)
        params.reasoning_budget_end.push_back(common_tokenize(vocab, tag, false, true));
    params.reasoning_budget_forced = params.reasoning_budget_end.front();
    sampler_.reset(common_sampler_init(model, params));
    observer_.reset(common_reasoning_budget_init(vocab, {params.reasoning_budget_start}, params.reasoning_budget_end,
                                                 params.reasoning_budget_forced, INT32_MAX));
    if (!sampler_ || !observer_) return false;
    if (thinking_prefilled)
        for (const llama_token token : params.reasoning_budget_start) {
            common_sampler_accept(sampler_.get(), token, /*is_generated*/ true);
            llama_sampler_accept(observer_.get(), token);
        }
    allowance_ = budget_tokens + (int) params.reasoning_budget_forced.size();
    return true;
}

bool ReasoningBudget::accept(llama_token token) {
    auto in_span = [](common_reasoning_budget_state s) {
        return s == REASONING_BUDGET_COUNTING || s == REASONING_BUDGET_WAITING_UTF8 || s == REASONING_BUDGET_FORCING;
    };
    const bool before = in_span(common_reasoning_budget_get_state(observer_.get()));
    llama_sampler_accept(observer_.get(), token);
    const bool after = in_span(common_reasoning_budget_get_state(observer_.get()));
    common_sampler_accept(sampler_.get(), token, /*is_generated*/ true);
    // A token that closes the span was inside before it; one that opens the span is inside after.
    return before || after;
}

} // namespace meitte::detail
