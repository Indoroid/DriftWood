#include "frontend_util.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iterator>

namespace meitte::frontend {

using json = nlohmann::json;

std::string json_escape(const std::string & s) {
    const std::string quoted = json(s).dump(-1, ' ', false, json::error_handler_t::replace);
    return quoted.size() >= 2 ? quoted.substr(1, quoted.size() - 2) : std::string{};
}

bool read_text_file(const std::string & path, std::string & out, std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot read " + path;
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return true;
}

std::string normalize_reasoning_effort(std::string value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "low" || lower == "medium" || lower == "high" || lower == "none") return lower;
    return value;
}

bool parse_template_kwargs(const json & value,
                           std::map<std::string, std::string> & out,
                           std::optional<bool> & generic_think,
                           std::optional<std::string> & generic_effort,
                           std::string & error) {
    if (!value.is_object()) {
        error = "chat_template_kwargs must be an object";
        return false;
    }
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
            out[it.key()] = it.value().dump();
        }
    }
    return true;
}

static bool is_extension(const std::string & full, const std::string & prev) {
    return full.size() >= prev.size() && full.compare(0, prev.size(), prev) == 0;
}

void emit_progress_line(const TokenMetrics & m, ProgressDelta & st) {
    if (m.read_bytes || m.io_ms > 0.0)
        std::printf("BMOE_LOAD {\"mb\":%.2f,\"ms\":%.1f}\n", m.read_bytes / (1024.0 * 1024.0), m.io_ms);
    const bool ext = is_extension(m.reasoning, st.reasoning) && is_extension(m.text, st.text);
    const std::string d_reason = ext ? m.reasoning.substr(st.reasoning.size()) : m.reasoning;
    const std::string d_text = ext ? m.text.substr(st.text.size()) : m.text;
    std::printf("BMOE_PROGRESS {\"step\":%d,\"steps\":%d,\"wall_ms\":%.1f,\"io_ms\":%.1f,"
                "\"compute_ms\":%.1f,\"mgmt_ms\":%.1f,\"stall_ms\":%.1f,\"read_mb\":%.2f,"
                "\"cache_hit_pct\":%.1f,\"majflt\":%llu,\"cpu_ms\":%.1f,\"dense_resident_frac\":%.3f,"
                "%s\"delta_reasoning\":\"%s\",\"delta_text\":\"%s\"}\n",
                m.step, m.steps, m.wall_ms, m.io_ms, m.compute_ms, m.mgmt_ms, m.stall_ms,
                m.read_bytes / (1024.0 * 1024.0), m.cache_hit_pct, (unsigned long long) m.majflt, m.cpu_ms,
                m.dense_resident_frac, ext ? "" : "\"reset\":1,", json_escape(d_reason).c_str(),
                json_escape(d_text).c_str());
    st.reasoning = m.reasoning;
    st.text = m.text;
    std::fflush(stdout);
}

} // namespace meitte::frontend
