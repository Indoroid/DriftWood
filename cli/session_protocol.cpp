#include "session_protocol.h"
#include "frontend_util.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>

#include <nlohmann/json.hpp>

namespace meitte {
namespace {

using json = nlohmann::json;

bool parse_session_messages(const json & value, std::vector<ChatMessage> & out, std::string & error) {
    if (!value.is_array() || value.empty()) {
        error = "messages must be a non-empty array";
        return false;
    }
    for (const json & item : value) {
        if (!item.is_object() || !item.contains("role") || !item["role"].is_string()) {
            error = "each message needs a string role";
            return false;
        }
        ChatMessage message;
        message.role = item["role"].get<std::string>();
        if (item.contains("content")) {
            if (!item["content"].is_null() && !item["content"].is_string()) {
                error = "message content must be a string or null";
                return false;
            }
            if (item["content"].is_string()) message.content = item["content"].get<std::string>();
        }
        if (item.contains("reasoning_content")) {
            if (!item["reasoning_content"].is_string()) {
                error = "reasoning_content must be a string";
                return false;
            }
            message.reasoning_content = item["reasoning_content"].get<std::string>();
        }
        if (item.contains("name")) {
            if (!item["name"].is_string()) {
                error = "name must be a string";
                return false;
            }
            message.tool_name = item["name"].get<std::string>();
        }
        if (item.contains("tool_call_id")) {
            if (!item["tool_call_id"].is_string()) {
                error = "tool_call_id must be a string";
                return false;
            }
            message.tool_call_id = item["tool_call_id"].get<std::string>();
        }
        if (item.contains("tool_calls")) {
            if (!item["tool_calls"].is_array()) {
                error = "tool_calls must be an array";
                return false;
            }
            for (const json & item_call : item["tool_calls"]) {
                if (!item_call.is_object() || !item_call.contains("function") || !item_call["function"].is_object()) {
                    error = "each tool call needs a function object";
                    return false;
                }
                const json & function = item_call["function"];
                if (!function.contains("name") || !function["name"].is_string() || !function.contains("arguments") ||
                    !function["arguments"].is_string() || (item_call.contains("id") && !item_call["id"].is_string())) {
                    error = "each tool call function needs string id, name, and arguments";
                    return false;
                }
                ToolCall call;
                if (item_call.contains("id")) call.id = item_call["id"].get<std::string>();
                call.name = function["name"].get<std::string>();
                call.arguments = function["arguments"].get<std::string>();
                message.tool_calls.push_back(std::move(call));
            }
        }
        if (message.role.empty() || (!item.contains("content") && message.tool_calls.empty())) {
            error = "each message needs a role and content or tool_calls";
            return false;
        }
        out.push_back(std::move(message));
    }
    return true;
}

bool parse_session_generate(const json & root, const RunConfig & cfg, SessionCommand & out, std::string & error) {
    const bool has_prompt = root.contains("prompt");
    const bool has_messages = root.contains("messages");
    if (has_prompt == has_messages) {
        error = "generate requires exactly one of prompt or messages";
        return false;
    }
    if (has_prompt) {
        if (!root["prompt"].is_string() || root["prompt"].get<std::string>().empty()) {
            error = "prompt must be a non-empty string";
            return false;
        }
        out.prompt = root["prompt"].get<std::string>();
    } else if (!parse_session_messages(root["messages"], out.messages, error)) {
        return false;
    }
    if (root.contains("n_predict")) {
        if (!root["n_predict"].is_number_integer() || root["n_predict"].get<long long>() < 1 ||
            root["n_predict"].get<long long>() > std::numeric_limits<int>::max()) {
            error = "n_predict must be a positive integer";
            return false;
        }
        out.n_predict = root["n_predict"].get<int>();
    } else {
        out.n_predict = cfg.n_predict;
    }
    std::optional<bool> explicit_think;
    if (root.contains("think")) {
        if (!root["think"].is_boolean()) {
            error = "think must be a boolean";
            return false;
        }
        explicit_think = root["think"].get<bool>();
    }
    std::optional<std::string> explicit_effort;
    std::optional<bool> generic_think;
    if (root.contains("chat_template_kwargs") &&
        !frontend::parse_template_kwargs(root["chat_template_kwargs"], out.chat_template_kwargs, generic_think,
                                         explicit_effort, error))
        return false;
    if (explicit_think && generic_think && *explicit_think != *generic_think) {
        error = "conflicting thinking controls: think and chat_template_kwargs.enable_thinking disagree";
        return false;
    }
    const bool request_think_control = explicit_think.has_value() || generic_think.has_value();
    out.think = explicit_think.value_or(generic_think.value_or(cfg.think));
    std::optional<std::string> typed_effort;
    if (root.contains("reasoning_effort")) {
        if (!root["reasoning_effort"].is_string() || root["reasoning_effort"].get<std::string>().empty()) {
            error = "reasoning_effort must be a non-empty string";
            return false;
        }
        typed_effort = frontend::normalize_reasoning_effort(root["reasoning_effort"].get<std::string>());
    }
    if (typed_effort && explicit_effort && *typed_effort != *explicit_effort) {
        error = "conflicting reasoning_effort controls";
        return false;
    }
    if (typed_effort) explicit_effort = std::move(typed_effort);
    out.reasoning_effort = explicit_effort.value_or(cfg.reasoning_effort);
    if (out.reasoning_effort.size() > 64) {
        error = "reasoning_effort must be at most 64 bytes";
        return false;
    }
    if (out.reasoning_effort == "none") {
        out.think = false;
        out.reasoning_effort.clear();
    } else if (explicit_effort && !request_think_control) {
        out.think = true;
    } else if (!out.think) {
        out.reasoning_effort.clear();
    }
    if (root.contains("clear_kv")) {
        if (!root["clear_kv"].is_boolean()) {
            error = "clear_kv must be a boolean";
            return false;
        }
        out.clear_kv = root["clear_kv"].get<bool>();
    }
    return true;
}

} // namespace

bool parse_session_command(const std::string & line, const RunConfig & cfg, SessionCommand & out, std::string & error) {
    try {
        const json root = json::parse(line, nullptr, false);
        if (root.is_discarded() || !root.is_object()) {
            error = "command must be a JSON object";
            return false;
        }
        if (!root.contains("cmd") || !root["cmd"].is_string()) {
            error = "cmd must be a string";
            return false;
        }
        if (root.contains("id")) {
            if (!root["id"].is_number_integer() || root["id"].get<long long>() < std::numeric_limits<int>::min() ||
                root["id"].get<long long>() > std::numeric_limits<int>::max()) {
                error = "id must be an integer";
                return false;
            }
            out.id = root["id"].get<int>();
        }
        const std::string cmd = root["cmd"].get<std::string>();
        if (cmd == "close") {
            out.kind = SessionCommand::kClose;
            return true;
        }
        if (cmd == "cancel") {
            out.kind = SessionCommand::kCancel;
            return true;
        }
        if (cmd != "generate") {
            error = "unknown command: " + cmd;
            return false;
        }
        out.kind = SessionCommand::kGenerate;
        return parse_session_generate(root, cfg, out, error);
    } catch (const json::exception &) {
        error = "command contains a field with an invalid type";
        return false;
    }
}

} // namespace meitte
