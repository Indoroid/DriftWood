// bmoe-server — HTTP server mode for BigMoeOnEdge.
//
// Loads a model once (like --session) and serves inferences over HTTP on a configurable
// port. Exposes an OpenAI-compatible REST API:
//
//   POST /v1/completions       text completion (raw prompt)
//   POST /v1/chat/completions  chat completion (message array -> chat template)
//   GET  /v1/models            model metadata
//
// Streaming via server-sent events (stream=true) mirrors the --progress protocol.
// The expert cache and model stay loaded between requests — the same amortisation
// the --session mode provides.
//
// Usage: bmoe-server -m <model.gguf> [--port N] [--host ADDR] [options]
//
// All bmoe-cli streaming/flags work the same way (--moe-stream, --cache-mb, etc.)
// except --prompt and --session.
#include "bmoe/config.h"
#include "bmoe/decode_trace.h"
#include "bmoe/metrics.h"
#include "bmoe/recipe.h"
#include "bmoe/route_trace.h"
#include "bmoe/runtime.h"
#include "bmoe/session.h"
#include "bmoe/version.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

using namespace bmoe;
using json = nlohmann::json;

// ── Socket helpers ───────────────────────────────────────────────────────────

// ── Minimal JSON utilities ───────────────────────────────────────────────────

static std::string json_escape(const std::string & s) {
    std::string quoted = json(s).dump(-1, ' ', false, json::error_handler_t::replace);
    return quoted.size() >= 2 ? quoted.substr(1, quoted.size() - 2) : std::string{};
}

static bool read_text_file(const std::string & path, std::string & out, std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot read " + path;
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return true;
}

static std::string normalize_reasoning_effort(std::string value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower == "low" || lower == "medium" || lower == "high" || lower == "none") return lower;
    return value;
}

struct ProgressDelta {
    std::string reasoning;
    std::string text;
};

static bool is_extension(const std::string & full, const std::string & previous) {
    return full.size() >= previous.size() && full.compare(0, previous.size(), previous) == 0;
}

// Byte-compatible with bmoe-cli --progress so existing telemetry consumers can observe an HTTP
// generation without learning another format. One ProgressDelta lives per request.
static void emit_progress_line(const TokenMetrics & m, ProgressDelta & state) {
    if (m.read_bytes || m.io_ms > 0.0)
        std::printf("BMOE_LOAD {\"mb\":%.2f,\"ms\":%.1f}\n", m.read_bytes / (1024.0 * 1024.0), m.io_ms);
    const bool extension = is_extension(m.reasoning, state.reasoning) && is_extension(m.text, state.text);
    const std::string reasoning = extension ? m.reasoning.substr(state.reasoning.size()) : m.reasoning;
    const std::string text = extension ? m.text.substr(state.text.size()) : m.text;
    std::printf("BMOE_PROGRESS {\"step\":%d,\"steps\":%d,\"wall_ms\":%.1f,\"io_ms\":%.1f,"
                "\"compute_ms\":%.1f,\"mgmt_ms\":%.1f,\"stall_ms\":%.1f,\"read_mb\":%.2f,"
                "\"cache_hit_pct\":%.1f,\"majflt\":%llu,\"cpu_ms\":%.1f,\"dense_resident_frac\":%.3f,"
                "%s\"delta_reasoning\":\"%s\",\"delta_text\":\"%s\"}\n",
                m.step, m.steps, m.wall_ms, m.io_ms, m.compute_ms, m.mgmt_ms, m.stall_ms,
                m.read_bytes / (1024.0 * 1024.0), m.cache_hit_pct, (unsigned long long) m.majflt, m.cpu_ms,
                m.dense_resident_frac, extension ? "" : "\"reset\":1,", json_escape(reasoning).c_str(),
                json_escape(text).c_str());
    state.reasoning = m.reasoning;
    state.text = m.text;
    std::fflush(stdout);
}

static size_t json_find_key(const std::string & json, const char * key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return std::string::npos;
    size_t c = json.find(':', k + pat.size());
    if (c == std::string::npos) return std::string::npos;
    return c + 1;
}

static std::string json_extract_string(const std::string & json, const char * key, const std::string & dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    if (p >= json.size() || json[p] != '"') return dflt;
    ++p;
    std::string raw;
    for (; p < json.size(); ++p) {
        if (json[p] == '\\' && p + 1 < json.size()) {
            raw += json[p];
            raw += json[p + 1];
            ++p;
        } else if (json[p] == '"') {
            break;
        } else {
            raw += json[p];
        }
    }
    // Unescape
    std::string out;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            switch (raw[++i]) {
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            default:
                out += raw[i];
                break;
            }
        } else {
            out += raw[i];
        }
    }
    return out;
}

static int json_extract_int(const std::string & json, const char * key, int dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    return std::atoi(json.c_str() + p);
}

static double json_extract_double(const std::string & json, const char * key, double dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    return std::atof(json.c_str() + p);
}

static bool json_extract_bool(const std::string & json, const char * key, bool dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    return json.compare(p, 4, "true") == 0;
}

// Extract the last user message content from a chat messages array.
// Handles both string content ("content":"text") and array content
// ("content":[{"type":"text","text":"..."}] as used by the OpenAI SDK / pi).
static std::string extract_last_user_message(const std::string & body) {
    size_t msgs = body.find("\"messages\"");
    if (msgs == std::string::npos) return "";

    std::string last_content;
    size_t pos = msgs;
    while (true) {
        size_t role_pos = body.find("\"role\"", pos);
        if (role_pos == std::string::npos) break;
        size_t role_val = role_pos + 7; // skip "role"
        while (role_val < body.size() &&
               (body[role_val] == ' ' || body[role_val] == ':' || body[role_val] == '\t' || body[role_val] == '\n'))
            ++role_val;

        bool is_user = body.compare(role_val, 6, "\"user\"") == 0;

        size_t content_pos = body.find("\"content\"", role_pos + 1);
        if (content_pos == std::string::npos) break;

        // Make sure this content belongs to THIS role entry (not a later one)
        size_t next_role = body.find("\"role\"", role_pos + 1);
        if (next_role != std::string::npos && content_pos > next_role) break;

        size_t cp = body.find(':', content_pos + 9);
        if (cp != std::string::npos) {
            ++cp;
            while (cp < body.size() && (body[cp] == ' ' || body[cp] == '\t' || body[cp] == '\n'))
                ++cp;
            if (cp < body.size()) {
                if (body[cp] == '"') {
                    // String content
                    ++cp;
                    std::string raw;
                    for (; cp < body.size(); ++cp) {
                        if (body[cp] == '\\' && cp + 1 < body.size()) {
                            raw += body[cp];
                            raw += body[cp + 1];
                            ++cp;
                        } else if (body[cp] == '"') {
                            break;
                        } else {
                            raw += body[cp];
                        }
                    }
                    std::string content;
                    for (size_t i = 0; i < raw.size(); ++i) {
                        if (raw[i] == '\\' && i + 1 < raw.size()) {
                            switch (raw[++i]) {
                            case 'n':
                                content += '\n';
                                break;
                            case 'r':
                                content += '\r';
                                break;
                            case 't':
                                content += '\t';
                                break;
                            case '"':
                                content += '"';
                                break;
                            case '\\':
                                content += '\\';
                                break;
                            default:
                                content += raw[i];
                                break;
                            }
                        } else {
                            content += raw[i];
                        }
                    }
                    if (is_user) last_content = content;
                } else if (body[cp] == '[') {
                    // Array content — find the "text" field inside
                    size_t text_pos = body.find("\"text\"", cp + 1);
                    if (text_pos != std::string::npos && (next_role == std::string::npos || text_pos < next_role)) {
                        size_t text_colon = body.find(':', text_pos + 6);
                        if (text_colon != std::string::npos) {
                            size_t text_start = text_colon + 1;
                            while (text_start < body.size() &&
                                   (body[text_start] == ' ' || body[text_start] == '\t' || body[text_start] == '\n'))
                                ++text_start;
                            if (text_start < body.size() && body[text_start] == '"') {
                                ++text_start;
                                std::string raw;
                                for (; text_start < body.size(); ++text_start) {
                                    if (body[text_start] == '\\' && text_start + 1 < body.size()) {
                                        raw += body[text_start];
                                        raw += body[text_start + 1];
                                        ++text_start;
                                    } else if (body[text_start] == '"') {
                                        break;
                                    } else {
                                        raw += body[text_start];
                                    }
                                }
                                std::string content;
                                for (size_t i = 0; i < raw.size(); ++i) {
                                    if (raw[i] == '\\' && i + 1 < raw.size()) {
                                        switch (raw[++i]) {
                                        case 'n':
                                            content += '\n';
                                            break;
                                        case 'r':
                                            content += '\r';
                                            break;
                                        case 't':
                                            content += '\t';
                                            break;
                                        case '"':
                                            content += '"';
                                            break;
                                        case '\\':
                                            content += '\\';
                                            break;
                                        default:
                                            content += raw[i];
                                            break;
                                        }
                                    } else {
                                        content += raw[i];
                                    }
                                }
                                if (is_user) last_content = content;
                            }
                        }
                    }
                }
            }
        }

        pos = role_pos + 7;
    }
    return last_content;
}

struct ApiCompletionRequest {
    std::string prompt;
    std::vector<ChatMessage> messages;
    std::vector<ChatTool> tools;
    ChatToolChoice tool_choice = ChatToolChoice::Auto;
    bool parallel_tool_calls = false;
    int n_predict = 128;
    bool stream = false;
    SamplingConfig sampling;
    std::optional<bool> think;
    std::string reasoning_effort;
    std::optional<int> reasoning_budget_tokens;
    std::map<std::string, std::string> chat_template_kwargs;
};

static bool parse_message_content(const json & value, std::string & out) {
    if (value.is_null()) {
        out.clear();
        return true;
    }
    if (value.is_string()) {
        out = value.get<std::string>();
        return true;
    }
    if (!value.is_array()) return false;

    out.clear();
    for (const json & part : value) {
        if (!part.is_object()) return false;
        const std::string type = part.value("type", "text");
        if (type != "text" || !part.contains("text") || !part["text"].is_string()) continue;
        out += part["text"].get<std::string>();
    }
    return true;
}

static bool parse_chat_template_kwargs(const json & value, ApiCompletionRequest & out, std::string & error) {
    if (!value.is_object()) {
        error = "chat_template_kwargs must be an object";
        return false;
    }
    std::optional<bool> generic_think;
    std::optional<std::string> generic_effort;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (it.key().empty() || it.key().size() > 128) {
            error = "chat_template_kwargs keys must be 1..128 bytes";
            return false;
        }
        if (it.key() == "enable_thinking") {
            if (!it.value().is_boolean()) {
                error = "chat_template_kwargs.enable_thinking must be a boolean";
                return false;
            }
            generic_think = it.value().get<bool>();
        } else if (it.key() == "reasoning_effort") {
            if (!it.value().is_string()) {
                error = "chat_template_kwargs.reasoning_effort must be a string";
                return false;
            }
            generic_effort = normalize_reasoning_effort(it.value().get<std::string>());
            if (generic_effort->empty()) {
                error = "chat_template_kwargs.reasoning_effort must be non-empty";
                return false;
            }
        } else {
            out.chat_template_kwargs[it.key()] = it.value().dump();
        }
    }
    if (generic_think) {
        if (out.think && *out.think != *generic_think) {
            error = "Conflicting thinking controls: think and chat_template_kwargs.enable_thinking disagree";
            return false;
        }
        out.think = *generic_think;
    }
    if (generic_effort) {
        if (!out.reasoning_effort.empty() && out.reasoning_effort != *generic_effort) {
            error = "Conflicting reasoning_effort controls";
            return false;
        }
        out.reasoning_effort = *generic_effort;
    }
    if (out.reasoning_effort.size() > 64) {
        error = "reasoning_effort must be at most 64 bytes";
        return false;
    }
    return true;
}

static bool parse_completion_request(const std::string & body,
                                     bool chat,
                                     const SamplingConfig & defaults,
                                     int default_n_predict,
                                     ApiCompletionRequest & out,
                                     std::string & error) {
    json root = json::parse(body, nullptr, /*allow_exceptions*/ false);
    if (root.is_discarded() || !root.is_object()) {
        error = "Request body must be a JSON object";
        return false;
    }

    out = {};
    out.sampling = defaults;
    out.n_predict = default_n_predict;
    if (root.contains("think")) {
        if (!root["think"].is_boolean()) {
            error = "think must be a boolean";
            return false;
        }
        out.think = root["think"].get<bool>();
    }
    if (root.contains("reasoning_effort")) {
        if (!root["reasoning_effort"].is_string() || root["reasoning_effort"].get<std::string>().empty()) {
            error = "reasoning_effort must be a non-empty string";
            return false;
        }
        out.reasoning_effort = normalize_reasoning_effort(root["reasoning_effort"].get<std::string>());
    }
    for (const char * name : {"reasoning_budget_tokens", "thinking_budget_tokens"}) {
        if (!root.contains(name)) continue;
        const json & value = root[name];
        if (!value.is_number_integer() || value.get<long long>() < -1 ||
            value.get<long long>() > std::numeric_limits<int>::max()) {
            error = std::string(name) + " must be an integer in -1.." +
                    std::to_string(std::numeric_limits<int>::max());
            return false;
        }
        const int budget = value.get<int>();
        if (out.reasoning_budget_tokens && *out.reasoning_budget_tokens != budget) {
            error = "Conflicting reasoning budget controls";
            return false;
        }
        out.reasoning_budget_tokens = budget;
    }
    if (root.contains("chat_template_kwargs") &&
        !parse_chat_template_kwargs(root["chat_template_kwargs"], out, error))
        return false;
    if (out.reasoning_effort == "none") {
        out.think = false;
        out.reasoning_effort.clear();
    } else if (out.think && !*out.think) {
        out.reasoning_effort.clear();
    }
    if (chat) {
        if (!root.contains("messages") || !root["messages"].is_array() || root["messages"].empty()) {
            error = "messages must be a non-empty array";
            return false;
        }
        for (const json & item : root["messages"]) {
            if (!item.is_object() || !item.contains("role") || !item["role"].is_string()) {
                error = "Each message needs a string role";
                return false;
            }
            ChatMessage msg;
            msg.role = item["role"].get<std::string>();
            if (item.contains("content") && !parse_message_content(item["content"], msg.content)) {
                error = "Message content must be a string or text-content array";
                return false;
            }
            msg.reasoning_content = item.value("reasoning_content", "");
            msg.tool_name = item.value("name", "");
            msg.tool_call_id = item.value("tool_call_id", "");
            if (item.contains("tool_calls")) {
                if (!item["tool_calls"].is_array()) {
                    error = "tool_calls must be an array";
                    return false;
                }
                for (const json & value : item["tool_calls"]) {
                    if (!value.is_object() || !value.contains("function") || !value["function"].is_object()) {
                        error = "Each tool call needs a function object";
                        return false;
                    }
                    const json & function = value["function"];
                    if (!function.contains("name") || !function["name"].is_string() ||
                        !function.contains("arguments") || !function["arguments"].is_string()) {
                        error = "Each tool call function needs string name and arguments";
                        return false;
                    }
                    ToolCall call;
                    call.id = value.value("id", "");
                    call.name = function["name"].get<std::string>();
                    call.arguments = function["arguments"].get<std::string>();
                    msg.tool_calls.push_back(std::move(call));
                }
            }
            if (msg.role.empty()) {
                error = "Message role cannot be empty";
                return false;
            }
            if (!item.contains("content") && msg.tool_calls.empty()) {
                error = "Message needs content or tool_calls";
                return false;
            }
            out.messages.push_back(std::move(msg));
        }
        for (auto it = out.messages.rbegin(); it != out.messages.rend(); ++it) {
            if (it->role == "user") {
                out.prompt = it->content; // raw fallback if this model has no chat template
                break;
            }
        }
        if (out.prompt.empty()) {
            error = "messages must contain a non-empty user message";
            return false;
        }

        if (root.contains("tools")) {
            if (!root["tools"].is_array()) {
                error = "tools must be an array";
                return false;
            }
            for (const json & item : root["tools"]) {
                if (!item.is_object() || item.value("type", "function") != "function" || !item.contains("function") ||
                    !item["function"].is_object()) {
                    error = "Each tool must contain a function object";
                    return false;
                }
                const json & function = item["function"];
                if (!function.contains("name") || !function["name"].is_string()) {
                    error = "Each tool function needs a string name";
                    return false;
                }
                ChatTool tool;
                tool.name = function["name"].get<std::string>();
                tool.description = function.value("description", "");
                tool.parameters_json = function.value("parameters", json::object()).dump();
                out.tools.push_back(std::move(tool));
            }
        }
        if (root.contains("tool_choice")) {
            if (!root["tool_choice"].is_string()) {
                error = "tool_choice must be auto, required, or none";
                return false;
            }
            const std::string choice = root["tool_choice"].get<std::string>();
            if (choice == "auto")
                out.tool_choice = ChatToolChoice::Auto;
            else if (choice == "required")
                out.tool_choice = ChatToolChoice::Required;
            else if (choice == "none")
                out.tool_choice = ChatToolChoice::None;
            else {
                error = "tool_choice must be auto, required, or none";
                return false;
            }
        }
        if (root.contains("parallel_tool_calls")) {
            if (!root["parallel_tool_calls"].is_boolean()) {
                error = "parallel_tool_calls must be a boolean";
                return false;
            }
            out.parallel_tool_calls = root["parallel_tool_calls"].get<bool>();
        }
    } else {
        if (!root.contains("prompt") || !root["prompt"].is_string()) {
            error = "prompt must be a string";
            return false;
        }
        out.prompt = root["prompt"].get<std::string>();
        if (out.prompt.empty()) {
            error = "prompt cannot be empty";
            return false;
        }
    }

    const char * token_key = root.contains("max_tokens") ? "max_tokens" : "max_completion_tokens";
    if (root.contains(token_key)) {
        if (!root[token_key].is_number_integer()) {
            error = std::string(token_key) + " must be an integer";
            return false;
        }
        out.n_predict = root[token_key].get<int>();
    }
    if (out.n_predict < 1) {
        error = std::string(token_key) + " must be at least 1";
        return false;
    }
    if (root.contains("stream")) {
        if (!root["stream"].is_boolean()) {
            error = "stream must be a boolean";
            return false;
        }
        out.stream = root["stream"].get<bool>();
    }
    if (root.contains("temperature")) {
        if (!root["temperature"].is_number()) {
            error = "temperature must be numeric";
            return false;
        }
        out.sampling.temp = root["temperature"].get<float>();
    }
    if (root.contains("top_p")) {
        if (!root["top_p"].is_number()) {
            error = "top_p must be numeric";
            return false;
        }
        out.sampling.top_p = root["top_p"].get<float>();
    }
    if (out.sampling.temp < 0.0f || out.sampling.temp > 2.0f) {
        error = "temperature must be between 0 and 2";
        return false;
    }
    if (out.sampling.top_p <= 0.0f || out.sampling.top_p > 1.0f) {
        error = "top_p must be in (0, 1]";
        return false;
    }
    return true;
}

// ── HTTP primitives ──────────────────────────────────────────────────────────

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::string body;
    std::string content_type;
    bool keep_alive = false;
    size_t content_length = 0;
};

static bool parse_http_request(const std::string & raw, HttpRequest & req) {
    size_t eol = raw.find("\r\n");
    if (eol == std::string::npos) return false;

    std::string reqline = raw.substr(0, eol);
    size_t sp1 = reqline.find(' ');
    if (sp1 == std::string::npos) return false;
    size_t sp2 = reqline.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return false;

    req.method = reqline.substr(0, sp1);

    std::string full_path = reqline.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t qm = full_path.find('?');
    if (qm != std::string::npos) {
        req.path = full_path.substr(0, qm);
        req.query = full_path.substr(qm + 1);
    } else {
        req.path = full_path;
    }

    // Parse headers
    size_t hdr_start = eol + 2;
    while (true) {
        size_t hdr_end = raw.find("\r\n", hdr_start);
        if (hdr_end == std::string::npos || hdr_end == hdr_start) break;
        std::string hdr = raw.substr(hdr_start, hdr_end - hdr_start);
        size_t colon = hdr.find(':');
        if (colon != std::string::npos) {
            std::string key = hdr.substr(0, colon);
            std::string val = hdr.substr(colon + 1);
            val.erase(0, val.find_first_not_of(" \t"));

            std::string lkey;
            lkey.resize(key.size());
            std::transform(key.begin(), key.end(), lkey.begin(), [](unsigned char c) { return std::tolower(c); });

            if (lkey == "content-type") req.content_type = val;
            if (lkey == "connection") {
                std::string lv;
                lv.resize(val.size());
                std::transform(val.begin(), val.end(), lv.begin(), [](unsigned char c) { return std::tolower(c); });
                req.keep_alive = (lv == "keep-alive");
            }
            if (lkey == "content-length") req.content_length = (size_t) std::atoll(val.c_str());
        }
        hdr_start = hdr_end + 2;
    }

    // Body follows the blank line (\r\n\r\n ends the headers)
    size_t body_start = raw.find("\r\n\r\n");
    if (body_start != std::string::npos) body_start += 4;
    if (body_start < raw.size() && req.content_length > 0) {
        req.body = raw.substr(body_start, req.content_length);
    }

    return true;
}

// Write all bytes to a socket.
static void http_write(int fd, const std::string & s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = write(fd, s.data() + off, s.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return;
        }
        off += (size_t) n;
    }
}

// Send a complete HTTP response.
static void send_response(int fd,
                          int status,
                          const char * status_text,
                          const std::string & content_type,
                          const std::string & body,
                          bool keep_alive) {
    char buf[128];
    std::string resp;
    std::snprintf(buf, sizeof(buf), "HTTP/1.1 %d %s\r\n", status, status_text);
    resp += buf;
    resp += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    resp += "Content-Type: " + content_type + "\r\n";
    resp += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    resp += "Access-Control-Allow-Origin: *\r\n";
    resp += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    resp += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
    resp += "\r\n";
    resp += body;
    http_write(fd, resp);
}

// Send SSE response headers (no Content-Length — streamed body).
static void send_sse_headers(int fd) {
    // OpenAI-compatible SDKs (OpenAI/JS, OpenAI/Python) use fetch() and expect
    // Transfer-Encoding: chunked for streaming. Connection: close without
    // chunked encoding causes the SDK to read the entire body before parsing,
    // which deadlocks on single-token streams.
    std::string resp = "HTTP/1.1 200 OK\r\n"
                       "Connection: close\r\n"
                       "Transfer-Encoding: chunked\r\n"
                       "Content-Type: text/event-stream\r\n"
                       "Cache-Control: no-cache\r\n"
                       "Access-Control-Allow-Origin: *\r\n"
                       "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                       "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                       "\r\n";
    http_write(fd, resp);
}

// Send an SSE data chunk with proper HTTP chunked transfer encoding.
static void send_sse(int fd, const std::string & data) {
    std::string chunk = "data: " + data + "\n\n";
    char size_buf[16];
    std::snprintf(size_buf, sizeof(size_buf), "%zx\r\n", chunk.size());
    http_write(fd, size_buf);
    http_write(fd, chunk);
    http_write(fd, "\r\n");
}

// Send the terminating zero-size chunk.
static void send_sse_done(int fd) {
    send_sse(fd, "[DONE]");
    http_write(fd, "0\r\n\r\n");
}

static void send_json_error(int fd, int status, const char * msg, bool ka) {
    const char * type = status == 400 ? "invalid_request_error" : "api_error";
    std::string body = "{\"error\":{\"message\":\"" + json_escape(msg) + "\",\"type\":\"" + type + "\"}}";
    const char * text = status >= 500   ? "Internal Server Error"
                        : status == 400 ? "Bad Request"
                        : status == 404 ? "Not Found"
                                        : "Error";
    send_response(fd, status, text, "application/json", body, ka);
}

// ── Server state ─────────────────────────────────────────────────────────────

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    int max_connections = 32;
    bool default_think = true;
    std::string default_reasoning_effort;
    std::string default_system_prompt;
    bool completion_chatml = false;
    bool progress = false;
};

struct ServerState {
    std::unique_ptr<Session> session;
    SessionConfig session_cfg;
    ServerConfig srv_cfg;
    IMetricsSink * metrics = nullptr;
    int default_n_predict = 128;
    std::atomic<bool> running{true};
};

// ── Request handlers ─────────────────────────────────────────────────────────

static std::string make_stream_delta(
    bool chat, const std::string & id, const std::string & object, long created, const std::string & piece) {
    json choice = {{"index", 0}, {"finish_reason", nullptr}};
    if (chat)
        choice["delta"] = {{"content", piece}};
    else {
        choice["text"] = piece;
        choice["logprobs"] = nullptr;
    }
    return json({{"id", id},
                 {"object", object},
                 {"created", created},
                 {"model", "bmoe"},
                 {"choices", json::array({std::move(choice)})}})
        .dump(-1, ' ', false, json::error_handler_t::replace);
}

static std::string response_tool_call_id(const ToolCall & call, size_t index, const std::string & request_tag) {
    if (!call.id.empty()) return call.id;
    return "call_" + request_tag + "_" + std::to_string(index);
}

static std::atomic<unsigned long long> response_sequence{0};

static void handle_completions(int fd, const HttpRequest & req, ServerState & state, bool chat);

static void handle_request(int fd, const HttpRequest & req, ServerState & state) {
    const bool ka = req.keep_alive;

    // CORS preflight
    if (req.method == "OPTIONS") {
        send_response(fd, 204, "No Content", "text/plain", "", ka);
        return;
    }

    // GET /
    if (req.method == "GET" && (req.path == "/" || req.path == "")) {
        std::string body = "{\"name\":\"bmoe-server\","
                           "\"version\":\"" BMOE_VERSION "\","
                           "\"description\":\"BigMoeOnEdge streaming inference server\"}";
        send_response(fd, 200, "OK", "application/json", body, ka);
        return;
    }

    // GET /v1/models
    if (req.method == "GET" && req.path == "/v1/models") {
        if (!state.session) {
            send_json_error(fd, 500, "Model not loaded", ka);
            return;
        }
        std::string model_id = "model";
        const std::string & mp = state.session_cfg.model_path;
        size_t slash = mp.rfind('/');
        size_t bslash = mp.rfind('\\');
        size_t sep = (slash != std::string::npos) ? slash : bslash;
        if (sep != std::string::npos && sep + 1 < mp.size()) model_id = mp.substr(sep + 1);

        std::string body = "{\"object\":\"list\",\"data\":[{"
                           "\"id\":\"" +
                           json_escape(model_id) +
                           "\","
                           "\"object\":\"model\","
                           "\"created\":0,"
                           "\"owned_by\":\"bmoe\","
                           "\"meta\":{"
                           "\"arch\":\"" +
                           json_escape(state.session->arch()) +
                           "\","
                           "\"n_ctx\":" +
                           std::to_string(state.session->n_ctx()) +
                           ","
                           "\"n_expert_used\":" +
                           std::to_string(state.session->n_expert_used()) + "}}]}";
        send_response(fd, 200, "OK", "application/json", body, ka);
        return;
    }

    // POST /v1/chat/completions
    if (req.method == "POST" && req.path == "/v1/chat/completions") {
        handle_completions(fd, req, state, true);
        return;
    }

    // POST /v1/completions
    if (req.method == "POST" && req.path == "/v1/completions") {
        handle_completions(fd, req, state, false);
        return;
    }

    send_json_error(fd, 404, "Not found", ka);
}

static void handle_completions(int fd, const HttpRequest & req, ServerState & state, bool chat) {
    if (!state.session) {
        send_json_error(fd, 500, "Model not loaded", false);
        return;
    }

    ApiCompletionRequest api;
    std::string parse_error;
    if (!parse_completion_request(req.body, chat, state.session_cfg.sampling, state.default_n_predict, api,
                                  parse_error)) {
        send_json_error(fd, 400, parse_error.c_str(), false);
        return;
    }

    // Build generate request
    const bool buffer_for_tools = chat && !api.tools.empty();
    GenerateRequest greq;
    greq.prompt = std::move(api.prompt);
    greq.messages = std::move(api.messages);
    if (chat && !state.srv_cfg.default_system_prompt.empty()) {
        const bool has_system = std::any_of(greq.messages.begin(), greq.messages.end(), [](const ChatMessage & message) {
            return message.role == "system";
        });
        if (!has_system) greq.messages.insert(greq.messages.begin(), {"system", state.srv_cfg.default_system_prompt});
    }
    greq.tools = std::move(api.tools);
    greq.tool_choice = api.tool_choice;
    greq.parallel_tool_calls = api.parallel_tool_calls;
    greq.chatml = chat || state.srv_cfg.completion_chatml;
    greq.n_predict = api.n_predict;
    // OpenAI streaming needs only piece deltas. The optional CLI-compatible progress protocol needs
    // parsed cumulative text/reasoning, accepting its documented O(n²) rendering cost when enabled.
    greq.render_text = state.srv_cfg.progress;
    greq.think = api.think.value_or(!api.reasoning_effort.empty() ? true : state.srv_cfg.default_think);
    greq.reasoning_effort = api.reasoning_effort.empty() ? state.srv_cfg.default_reasoning_effort
                                                          : api.reasoning_effort;
    if (!greq.think) greq.reasoning_effort.clear();
    greq.reasoning_budget_tokens = greq.think ? api.reasoning_budget_tokens.value_or(-1) : -1;
    greq.chat_template_kwargs = std::move(api.chat_template_kwargs);
    greq.override_sampling = true;
    greq.sampling = api.sampling;
    long created = static_cast<long>(std::time(nullptr));
    const std::string request_tag =
        std::to_string(created) + "_" + std::to_string(response_sequence.fetch_add(1, std::memory_order_relaxed));
    ProgressDelta progress;
    auto progress_token = [&](const TokenMetrics & metrics) {
        if (state.srv_cfg.progress) emit_progress_line(metrics, progress);
    };
    std::function<void(const TokenMetrics &)> progress_callback;
    if (state.srv_cfg.progress) progress_callback = progress_token;

    if (!api.stream) {
        auto result = state.session->generate(greq, progress_callback, state.metrics);
        if (!result) {
            send_json_error(fd, 500, result.error.c_str(), false);
            return;
        }

        std::string id_prefix = chat ? "chatcmpl" : "cmpl";
        std::string object = chat ? "chat.completion" : "text_completion";

        json choice;
        if (chat) {
            json message = {{"role", "assistant"}, {"content", result.generated_text}};
            if (!result.reasoning_text.empty()) message["reasoning_content"] = result.reasoning_text;
            if (!result.tool_calls.empty()) {
                message["tool_calls"] = json::array();
                for (size_t i = 0; i < result.tool_calls.size(); ++i) {
                    const ToolCall & call = result.tool_calls[i];
                    message["tool_calls"].push_back(
                        {{"id", response_tool_call_id(call, i, request_tag)},
                         {"type", "function"},
                         {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
                }
            }
            choice = {{"index", 0},
                      {"message", std::move(message)},
                      {"finish_reason", result.tool_calls.empty() ? "stop" : "tool_calls"}};
        } else {
            choice = {{"text", result.generated_text}, {"index", 0}, {"finish_reason", "stop"}, {"logprobs", nullptr}};
        }
        const json body = {{"id", id_prefix + "-" + request_tag},
                           {"object", object},
                           {"created", created},
                           {"model", "bmoe"},
                           {"choices", json::array({std::move(choice)})},
                           {"usage",
                            {{"prompt_tokens", result.summary.n_prompt},
                             {"completion_tokens", result.summary.n_generated},
                             {"total_tokens", result.summary.n_prompt + result.summary.n_generated}}}};
        send_response(fd, 200, "OK", "application/json", body.dump(-1, ' ', false, json::error_handler_t::replace),
                      false);
        return;
    }

    // ── Streaming (SSE) ─────────────────────────────────────────────────
    send_sse_headers(fd);

    std::string id_prefix = chat ? "chatcmpl" : "cmpl";
    std::string object = chat ? "chat.completion.chunk" : "text_completion";

    // For chat, send the role first
    if (chat) {
        std::string data = "{\"id\":\"" + id_prefix + "-" + request_tag +
                           "\","
                           "\"object\":\"" +
                           object +
                           "\","
                           "\"created\":" +
                           std::to_string(created) +
                           ","
                           "\"model\":\"bmoe\","
                           "\"choices\":[{"
                           "\"index\":0,"
                           "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                           "\"finish_reason\":null"
                           "}]}";
        send_sse(fd, data);
    }

    auto on_token = [&](const TokenMetrics & m) {
        progress_token(m);
        if (!buffer_for_tools)
            send_sse(fd, make_stream_delta(chat, id_prefix + "-" + request_tag, object, created, m.piece));
    };

    auto result = state.session->generate(greq, on_token, state.metrics);
    if (result) {
        if (buffer_for_tools) {
            json delta = json::object();
            if (!result.generated_text.empty()) delta["content"] = result.generated_text;
            if (!result.reasoning_text.empty()) delta["reasoning_content"] = result.reasoning_text;
            if (!result.tool_calls.empty()) {
                delta["tool_calls"] = json::array();
                for (size_t i = 0; i < result.tool_calls.size(); ++i) {
                    const ToolCall & call = result.tool_calls[i];
                    delta["tool_calls"].push_back({{"index", i},
                                                   {"id", response_tool_call_id(call, i, request_tag)},
                                                   {"type", "function"},
                                                   {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
                }
            }
            const json buffered = {
                {"id", id_prefix + "-" + request_tag},
                {"object", object},
                {"created", created},
                {"model", "bmoe"},
                {"choices", json::array({{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", nullptr}}})}};
            send_sse(fd, buffered.dump(-1, ' ', false, json::error_handler_t::replace));
        }

        json choice = {{"index", 0}, {"finish_reason", result.tool_calls.empty() ? "stop" : "tool_calls"}};
        if (chat)
            choice["delta"] = json::object();
        else {
            choice["text"] = "";
            choice["logprobs"] = nullptr;
        }
        const json data = {
            {"id", id_prefix + "-" + request_tag},
            {"object", object},
            {"created", created},
            {"model", "bmoe"},
            {"choices", json::array({std::move(choice)})},
            {"usage",
             {{"prompt_tokens", result.summary.n_prompt},
              {"completion_tokens", result.summary.n_generated},
              {"total_tokens", result.summary.n_prompt + result.summary.n_generated}}},
        };
        send_sse(fd, data.dump(-1, ' ', false, json::error_handler_t::replace));
        send_sse_done(fd);
    } else {
        const json error = {{"error", {{"message", result.error}, {"type", "api_error"}}}};
        send_sse(fd, error.dump(-1, ' ', false, json::error_handler_t::replace));
        send_sse_done(fd);
    }
}

// ── Connection handling ──────────────────────────────────────────────────────

// Read the full HTTP request from a blocking socket: headers + body.
// Returns false if the connection closed or the request was too large.
static bool read_request(int fd, std::string & raw) {
    char buf[65536];
    while (true) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false; // connection closed or error
        }
        raw.append(buf, (size_t) n);

        // Check if we have the full headers
        size_t hdr_end = raw.find("\r\n\r\n");
        if (hdr_end == std::string::npos) {
            if (raw.size() > 65536) return false; // headers too large
            continue;                             // need more data
        }

        // Parse Content-Length case-insensitively. OpenAI SDKs are conventional here, but HTTP
        // field names are case-insensitive and accepting only two spellings made valid requests
        // appear body-less.
        size_t body_start = hdr_end + 4;
        std::string headers = raw.substr(0, hdr_end);
        size_t content_length = 0;
        bool have_length = false;
        size_t line_start = 0;
        while (line_start < headers.size()) {
            size_t line_end = headers.find("\r\n", line_start);
            if (line_end == std::string::npos) line_end = headers.size();
            const std::string line = headers.substr(line_start, line_end - line_start);
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string key = line.substr(0, colon);
                std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
                if (key == "content-length") {
                    const char * begin = line.c_str() + colon + 1;
                    char * end = nullptr;
                    errno = 0;
                    const unsigned long long parsed = std::strtoull(begin, &end, 10);
                    while (end && *end == ' ')
                        ++end;
                    if (errno != 0 || end == begin || (end && *end != '\0') || parsed > 1024ull * 1024ull) return false;
                    content_length = (size_t) parsed;
                    have_length = true;
                    break;
                }
            }
            line_start = line_end + 2;
        }
        if (!have_length) return true;
        if (raw.size() - body_start >= content_length) return true;
        if (raw.size() > 1024 * 1024) return false;
    }
}

// Process one HTTP request per connection. The model/cache remain resident; only the cheap socket
// is short-lived. This also avoids dropping a pipelined request after the first parsed body.
static void process_connection(int fd, ServerState & state) {
    std::string raw;
    if (!read_request(fd, raw)) return; // connection closed

    HttpRequest req;
    if (!parse_http_request(raw, req)) {
        send_json_error(fd, 400, "Bad request", false);
        return;
    }

    req.keep_alive = false;
    handle_request(fd, req, state);
}

// ── Server lifecycle ─────────────────────────────────────────────────────────

static void print_usage(const char * argv0) {
    std::printf("usage: %s -m <model.gguf> [options]\n"
                "\n"
                "  -m, --model PATH        gguf model (required)\n"
                "      --port N            HTTP server port (default 8080)\n"
                "      --host ADDR         bind address (default 127.0.0.1; use 0.0.0.0 for\n"
                "                          remote access)\n"
                "\n"
                "  bmoe-cli parity (model/session-wide):\n"
                "  -n, --n-predict, -t, --threads, -c, --ctx-size\n"
                "  --batch-size N (default 2048), --ubatch-size N (default 512; --ubatch alias)\n"
                "  --chatml, --system-prompt TEXT, --system-prompt-file PATH\n"
                "  --no-think, --reasoning-effort VALUE\n"
                "  --chat-template TEXT, --chat-template-file PATH\n"
                "  --cache-type-k TYPE, --cache-type-v TYPE, --flash-attn auto|on|off\n"
                "  --temp, --top-k, --top-p, --seed\n"
                "  --progress, --session\n"
                "  --mtp, --ngram, --draft, --mtp-p-min, --ngram-min-match\n"
                "  --csv, --route-trace, --compute-trace, --compute-trace-layers, --io-trace\n"
                "  --moe-stream, --cache-mb, --cache-floor-mb, --cache-ceil-mb\n"
                "  --io-threads, --no-odirect, --dense-weights, --load-all, --force-cache\n"
                "  --overlap, --io-two-wave, --prefetch, --prefetch-sync\n"
                "  --drop-cold-experts, --drop-no-renorm, --drop-in-prefill\n"
                "  --route-ahead, --predict-log, --predict-prefetch, --predict-spec-max\n"
                "  --n-expert-used, --list-archs\n"
                "\n"
                "  -h, --help              show this text and exit\n"
                "      --version           print the engine version and exit\n"
                "\n"
                "API endpoints:\n"
                "  GET  /v1/models           list loaded model\n"
                "  POST /v1/completions      text completion (OpenAI-compatible)\n"
                "  POST /v1/chat/completions chat completion (OpenAI-compatible)\n"
                "\n"
                "  Both POST endpoints accept stream=true for SSE token streaming.\n"
                "\n"
                "Environment:\n"
                "  BMOE_SERVER_PORT  override --port\n"
                "  BMOE_SERVER_HOST  override --host\n"
                "  BMOE_CACHE_MB, BMOE_IO_THREADS, BMOE_OVERLAP, BMOE_PREFETCH,\n"
                "  BMOE_N_EXPERT_USED, BMOE_PREDICT_LOG and BMOE_PREDICT_PREFETCH also apply\n",
                argv0);
}

int main(int argc, char ** argv) {
    // A streaming client can disconnect between tokens. Ignore SIGPIPE so one abandoned response
    // closes that connection instead of terminating the model server process.
    std::signal(SIGPIPE, SIG_IGN);

    RunConfig cfg;
    ServerConfig srv;
    std::string csv_path;
    std::string route_trace_path;
    std::string compute_trace_path;
    std::string io_trace_path;
    bool no_think_seen = false;
    bool reasoning_effort_seen = false;
    bool system_prompt_seen = false;
    bool system_prompt_file_seen = false;
    std::string system_prompt_file;
    bool chat_template_seen = false;
    bool chat_template_file_seen = false;
    std::string chat_template_file;

    std::set<std::string> seen;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        seen.insert(a);
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(1);
            }
            return argv[++i];
        };

        if (a == "-m" || a == "--model")
            cfg.model_path = next("-m");
        else if (a == "--port")
            srv.port = std::atoi(next("--port"));
        else if (a == "--host")
            srv.host = next("--host");
        else if (a == "-p" || a == "--prompt") {
            next("-p"); // ignored in server mode
        } else if (a == "-n" || a == "--n-predict")
            cfg.n_predict = std::atoi(next("-n"));
        else if (a == "-t" || a == "--threads")
            cfg.n_threads = std::atoi(next("-t"));
        else if (a == "-c" || a == "--ctx-size")
            cfg.n_ctx = std::atoi(next("-c"));
        else if (a == "--batch-size")
            cfg.n_batch = std::atoi(next("--batch-size"));
        else if (a == "--ubatch" || a == "--ubatch-size")
            cfg.n_ubatch = std::atoi(next("--ubatch-size"));
        else if (a == "--n-expert-used")
            cfg.n_expert_used = std::atoi(next("--n-expert-used"));
        else if (a == "--temp")
            cfg.sampling.temp = (float) std::atof(next("--temp"));
        else if (a == "--top-k")
            cfg.sampling.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p")
            cfg.sampling.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--seed")
            cfg.sampling.seed = (uint32_t) std::strtoul(next("--seed"), nullptr, 10);
        else if (a == "--mtp" || a == "--ngram") {
            const DraftSource want = a == "--mtp" ? DraftSource::mtp : DraftSource::ngram;
            if (cfg.spec.enabled() && cfg.spec.source != want) {
                std::fprintf(stderr, "bmoe-server: --mtp and --ngram are exclusive; choose one.\n");
                return 2;
            }
            cfg.spec.source = want;
        } else if (a == "--draft")
            cfg.spec.draft_max = std::atoi(next("--draft"));
        else if (a == "--mtp-p-min")
            cfg.spec.draft_p_min = (float) std::atof(next("--mtp-p-min"));
        else if (a == "--ngram-min-match")
            cfg.spec.ngram_min_match = std::atoi(next("--ngram-min-match"));
        else if (a == "--chatml")
            srv.completion_chatml = true;
        else if (a == "--system-prompt") {
            if (system_prompt_file_seen) {
                std::fprintf(stderr, "bmoe-server: --system-prompt conflicts with --system-prompt-file\n");
                return 2;
            }
            cfg.system_prompt = next("--system-prompt");
            system_prompt_seen = true;
        } else if (a == "--system-prompt-file") {
            if (system_prompt_seen) {
                std::fprintf(stderr, "bmoe-server: --system-prompt conflicts with --system-prompt-file\n");
                return 2;
            }
            system_prompt_file = next("--system-prompt-file");
            system_prompt_file_seen = true;
        }
        else if (a == "--reasoning-effort") {
            cfg.reasoning_effort = next("--reasoning-effort");
            if (cfg.reasoning_effort.empty()) {
                std::fprintf(stderr, "bmoe-server: --reasoning-effort cannot be empty\n");
                return 2;
            }
            reasoning_effort_seen = true;
        } else if (a == "--chat-template") {
            if (chat_template_file_seen) {
                std::fprintf(stderr, "bmoe-server: --chat-template conflicts with --chat-template-file\n");
                return 2;
            }
            cfg.chat_template = next("--chat-template");
            chat_template_seen = true;
        } else if (a == "--chat-template-file") {
            if (chat_template_seen) {
                std::fprintf(stderr, "bmoe-server: --chat-template conflicts with --chat-template-file\n");
                return 2;
            }
            chat_template_file = next("--chat-template-file");
            chat_template_file_seen = true;
        } else if (a == "--cache-type-k" || a == "-ctk") {
            if (!parse_kv_cache_type(next("--cache-type-k"), cfg.cache_type_k)) {
                std::fprintf(stderr, "bmoe-server: invalid --cache-type-k\n");
                return 2;
            }
        } else if (a == "--cache-type-v" || a == "-ctv") {
            if (!parse_kv_cache_type(next("--cache-type-v"), cfg.cache_type_v)) {
                std::fprintf(stderr, "bmoe-server: invalid --cache-type-v\n");
                return 2;
            }
        } else if (a == "--flash-attn") {
            if (!parse_flash_attention_mode(next("--flash-attn"), cfg.flash_attention)) {
                std::fprintf(stderr, "bmoe-server: --flash-attn expects auto|on|off\n");
                return 2;
            }
        }
        else if (a == "--progress")
            srv.progress = true;
        else if (a == "--session") {
            // A server is intrinsically a persistent session; accept the CLI flag for exact parser
            // parity so one shared option vector can launch either frontend.
        } else if (a == "--no-think") {
            cfg.think = false;
            no_think_seen = true;
        } else if (a == "--csv")
            csv_path = next("--csv");
        else if (a == "--route-trace")
            route_trace_path = next("--route-trace");
        else if (a == "--compute-trace")
            compute_trace_path = next("--compute-trace");
        else if (a == "--compute-trace-layers") {
            compute_trace_path = next("--compute-trace-layers");
            cfg.compute_trace_layers = true;
        } else if (a == "--io-trace")
            io_trace_path = next("--io-trace");
        else if (a == "--moe-stream")
            cfg.moe.enabled = true;
        else if (a == "--cache-mb") {
            const std::string v = next("--cache-mb");
            if (v == "auto")
                cfg.moe.cache_auto = true;
            else
                cfg.moe.cache_mb = std::atoi(v.c_str());
        } else if (a == "--cache-floor-mb")
            cfg.moe.cache_floor_mb = std::atoi(next("--cache-floor-mb"));
        else if (a == "--cache-ceil-mb")
            cfg.moe.cache_ceil_mb = std::atoi(next("--cache-ceil-mb"));
        else if (a == "--io-threads")
            cfg.moe.io_threads = std::atoi(next("--io-threads"));
        else if (a == "--no-odirect")
            cfg.moe.o_direct = false;
        else if (a == "--dense-weights") {
            const std::string m = next("--dense-weights");
            if (m == "mmap")
                cfg.moe.dense_weights = DenseWeightsMode::Mmap;
            else if (m == "warm")
                cfg.moe.dense_weights = DenseWeightsMode::Warmed;
            else if (m == "anon")
                cfg.moe.dense_weights = DenseWeightsMode::Anonymous;
            else if (m == "ahwb")
                cfg.moe.dense_weights = DenseWeightsMode::Pinned;
            else {
                std::fprintf(stderr, "bmoe-server: --dense-weights expects mmap|warm|anon|ahwb\n");
                return 2;
            }
        }
        // Deprecated bmoe-cli aliases retained for command-line parity.
        else if (a == "--no-warm-dense")
            cfg.moe.dense_weights = DenseWeightsMode::Mmap;
        else if (a == "--dense-odirect")
            cfg.moe.dense_weights = DenseWeightsMode::Anonymous;
        else if (a == "--load-all")
            cfg.moe.load_all = true;
        else if (a == "--force-cache")
            cfg.moe.force_cache = true;
        else if (a == "--overlap")
            cfg.moe.overlap = true;
        else if (a == "--io-two-wave")
            cfg.moe.io_two_wave = true;
        else if (a == "--prefetch")
            cfg.moe.prefetch_layers = std::atoi(next("--prefetch"));
        else if (a == "--prefetch-sync")
            cfg.moe.prefetch_sync = true;
        else if (a == "--drop-cold-experts")
            cfg.moe.drop_cold_frac = (float) std::atof(next("--drop-cold-experts"));
        else if (a == "--drop-no-renorm")
            cfg.moe.drop_renorm = false;
        else if (a == "--drop-in-prefill")
            cfg.moe.drop_prefill = true;
        else if (a == "--route-ahead")
            cfg.moe.route_ahead = std::atoi(next("--route-ahead"));
        else if (a == "--predict-log")
            cfg.moe.predict_log = true;
        else if (a == "--predict-prefetch")
            cfg.moe.predict_prefetch = true;
        else if (a == "--predict-spec-max")
            cfg.moe.predict_spec_max = std::atoi(next("--predict-spec-max"));
        else if (a == "--list-archs") {
            std::printf("supported MoE architectures:\n");
            for (int k = 0; k < n_moe_recipes(); ++k)
                std::printf("  %s\n", moe_recipe_at(k)->arch);
            return 0;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--version") {
            std::printf("%s\n", bmoe::version());
            return 0;
        } else {
            std::fprintf(stderr, "bmoe-server: unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    std::string file_error;
    if (system_prompt_file_seen && !read_text_file(system_prompt_file, cfg.system_prompt, file_error)) {
        std::fprintf(stderr, "bmoe-server: %s\n", file_error.c_str());
        return 2;
    }
    if (chat_template_file_seen && !read_text_file(chat_template_file, cfg.chat_template, file_error)) {
        std::fprintf(stderr, "bmoe-server: %s\n", file_error.c_str());
        return 2;
    }
    const std::string normalized_effort = normalize_reasoning_effort(cfg.reasoning_effort);
    cfg.reasoning_effort = normalized_effort;
    if (normalized_effort == "none") {
        cfg.think = false;
        cfg.reasoning_effort.clear();
    } else if (no_think_seen && reasoning_effort_seen) {
        std::fprintf(stderr, "bmoe-server: --no-think conflicts with --reasoning-effort %s\n",
                     cfg.reasoning_effort.c_str());
        return 2;
    }
    srv.default_think = cfg.think;
    srv.default_reasoning_effort = cfg.reasoning_effort;
    srv.default_system_prompt = cfg.system_prompt;

    // Env overrides
    const char * env_port = std::getenv("BMOE_SERVER_PORT");
    if (env_port && *env_port) srv.port = std::atoi(env_port);
    const char * env_host = std::getenv("BMOE_SERVER_HOST");
    if (env_host && *env_host) srv.host = env_host;

    auto env_int = [](const char * key, int dflt) {
        const char * v = std::getenv(key);
        return v && *v ? std::atoi(v) : dflt;
    };

    // Keep server and CLI environment behavior identical. An explicit flag always wins, including
    // when its value equals the default.
    if (!seen.count("--cache-mb")) cfg.moe.cache_mb = env_int("BMOE_CACHE_MB", 0);
    if (!seen.count("--io-threads")) cfg.moe.io_threads = env_int("BMOE_IO_THREADS", 4);
    if (!seen.count("--progress")) srv.progress = env_int("BMOE_PROGRESS", 0) != 0;
    if (!seen.count("--overlap")) cfg.moe.overlap = env_int("BMOE_OVERLAP", 0) != 0;
    if (!seen.count("--prefetch")) cfg.moe.prefetch_layers = env_int("BMOE_PREFETCH", 0);
    if (!seen.count("--n-expert-used")) cfg.n_expert_used = env_int("BMOE_N_EXPERT_USED", 0);
    if (!seen.count("--predict-log")) cfg.moe.predict_log = env_int("BMOE_PREDICT_LOG", 0) != 0;
    if (!seen.count("--predict-prefetch")) cfg.moe.predict_prefetch = env_int("BMOE_PREDICT_PREFETCH", 0) != 0;

    if (cfg.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (srv.port < 1 || srv.port > 65535) {
        std::fprintf(stderr, "bmoe-server: --port must be in 1..65535\n");
        return 1;
    }

    // Load template support once. Chat requests always use it; raw completions use it only when
    // --chatml was supplied, matching bmoe-cli without requiring a second model session.
    cfg.chatml = true;

    ValidationResult vr = validate(cfg);
    if (!vr) {
        std::fprintf(stderr, "config error: %s\n", vr.error.c_str());
        return 1;
    }

    // ── Open the session ──────────────────────────────────────────────
    std::fprintf(stderr, "bmoe-server: loading model %s ...\n", cfg.model_path.c_str());

    std::unique_ptr<IMetricsSink> metrics;
    if (!csv_path.empty()) {
        metrics.reset(make_csv_metrics_sink(csv_path));
        if (!metrics) std::fprintf(stderr, "warning: could not open csv %s\n", csv_path.c_str());
    }

    std::unique_ptr<IRouteTraceSink> route_trace;
    if (!route_trace_path.empty()) {
        if (!cfg.moe.enabled) {
            std::fprintf(stderr, "warning: --route-trace needs --moe-stream; no trace will be written\n");
        } else {
            route_trace.reset(make_csv_route_trace_sink(route_trace_path));
            if (!route_trace)
                std::fprintf(stderr, "warning: could not open route trace %s\n", route_trace_path.c_str());
        }
    }

    std::unique_ptr<IComputeTraceSink> compute_trace;
    if (!compute_trace_path.empty()) {
        compute_trace.reset(make_csv_compute_trace_sink(compute_trace_path));
        if (!compute_trace)
            std::fprintf(stderr, "warning: could not open compute trace %s\n", compute_trace_path.c_str());
    }

    std::unique_ptr<IIoTraceSink> io_trace;
    if (!io_trace_path.empty()) {
        if (!cfg.moe.enabled) {
            std::fprintf(stderr, "warning: --io-trace needs --moe-stream; no trace will be written\n");
        } else {
            io_trace.reset(make_csv_io_trace_sink(io_trace_path));
            if (!io_trace) std::fprintf(stderr, "warning: could not open io trace %s\n", io_trace_path.c_str());
        }
    }

    const SessionConfig sc = session_config_from(cfg);
    std::string error;
    std::unique_ptr<Session> session = Session::open(sc, error, route_trace.get(), compute_trace.get(), io_trace.get());
    if (!session) {
        std::fprintf(stderr, "bmoe-server: failed to load model: %s\n", error.c_str());
        return 1;
    }

    std::fprintf(stderr, "bmoe-server: model loaded: arch=%s, n_ctx=%d, think_ctl=%s, n_expert_used=%d\n",
                 session->arch().c_str(), session->n_ctx(), think_control_name(session->think_control()),
                 session->n_expert_used());
    std::fprintf(stderr, "bmoe-server: listening on http://%s:%d\n", srv.host.c_str(), srv.port);

    // ── Create the listening socket ───────────────────────────────────
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "bmoe-server: socket() failed: %s\n", std::strerror(errno));
        return 1;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) srv.port);

    if (srv.host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, srv.host.c_str(), &addr.sin_addr) != 1) {
            std::fprintf(stderr, "bmoe-server: invalid host: %s\n", srv.host.c_str());
            close(listen_fd);
            return 1;
        }
    }

    if (bind(listen_fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        std::fprintf(stderr, "bmoe-server: bind(%s:%d) failed: %s\n", srv.host.c_str(), srv.port, std::strerror(errno));
        close(listen_fd);
        return 1;
    }

    if (listen(listen_fd, srv.max_connections) < 0) {
        std::fprintf(stderr, "bmoe-server: listen() failed: %s\n", std::strerror(errno));
        close(listen_fd);
        return 1;
    }

    // ── Simple single-threaded server loop ────────────────────────────
    // One connection at a time; good enough for on-device use.
    ServerState state;
    state.session = std::move(session);
    state.session_cfg = sc;
    state.srv_cfg = srv;
    state.metrics = metrics.get();
    state.default_n_predict = cfg.n_predict;

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(listen_fd, (struct sockaddr *) &client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "bmoe-server: accept() error: %s\n", std::strerror(errno));
            continue;
        }

        process_connection(client_fd, state);
        close(client_fd);
    }

    close(listen_fd);
    std::fprintf(stderr, "bmoe-server: shutting down\n");
    return 0;
}
