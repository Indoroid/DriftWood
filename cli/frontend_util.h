#pragma once

// Helpers shared by the meitte-cli and meitte-server frontends: the BMOE_* line-protocol progress
// line, JSON string escaping, and the request-control parsing both protocols accept. One copy, so the
// two frontends cannot drift apart. Frontend code only: libmeitte never links this.

#include "bmoe/metrics.h"

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>

namespace meitte::frontend {

// `s` as the inside of a JSON string literal; invalid UTF-8 is replaced, never emitted raw.
std::string json_escape(const std::string & s);

bool read_text_file(const std::string & path, std::string & out, std::string & error);

// "low" / "medium" / "high" / "none" in any case become lowercase; other values pass through.
std::string normalize_reasoning_effort(std::string value);

// Parse a chat_template_kwargs object into serialized JSON values. enable_thinking and
// reasoning_effort are generic controls, returned separately rather than passed to the template.
bool parse_template_kwargs(const nlohmann::json & value,
                           std::map<std::string, std::string> & out,
                           std::optional<bool> & think,
                           std::optional<std::string> & effort,
                           std::string & error);

// What BMOE_PROGRESS already delivered this generation, so each line carries only the new tail.
// One state per generation: start each generation with a fresh object.
struct ProgressDelta {
    std::string reasoning;
    std::string text;
};

// One token's line-protocol output: the optional BMOE_LOAD, then BMOE_PROGRESS (docs/telemetry.md).
//
// The answer travels as a DELTA: sending the cumulative text every token made a generation of n
// tokens write, escape and parse O(n^2) bytes (#119). The common case appends the suffix since the
// last line; when a closing tag makes common_chat_parse retroactively reclassify answer text as
// reasoning, an append cannot express it, so the line carries the full snapshot with "reset":1 and
// the reader replaces instead of appending. The full final text still travels in BMOE_DONE.
void emit_progress_line(const TokenMetrics & m, ProgressDelta & state);

} // namespace meitte::frontend
