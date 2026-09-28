// The engine entry point: compose model + streaming + generation from a RunConfig.
//
// run() loads the model with the layout the streamer requires (mmap on, no weight
// repack), discovers the MoE expert tensors through a one-token capture warm-up, binds
// them to the expert source, then greedily generates n_predict tokens — reporting each
// token to the optional callback/sink and returning a RunSummary. Greedy sampling makes
// the output a deterministic function of the graph, which is what the byte-identity
// gates rely on — true of every configuration except MoeStreamConfig::drop_cold_frac, which
// decides from live cache state and so is reproducible only within a single run.
#pragma once

#include "bmoe/config.h"
#include "bmoe/metrics.h"

#include <functional>
#include <string>
#include <vector>

namespace meitte {

class IRouteTraceSink;
class IComputeTraceSink;
class IIoTraceSink;

// Provider-neutral representation used by Session and surfaced through OpenAI-compatible APIs.
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

// Why a generation ended. Decided by the engine, which is the only place that knows: a frontend
// that compares token counts cannot tell a reasoning allowance, a cancel or a full context apart.
enum class FinishReason {
    None,        // no generation ran
    Stop,        // the model emitted end-of-generation
    Length,      // the n_predict answer allowance ran out
    Cancelled,   // Session::cancel() interrupted it
    ContextFull, // the context reached capacity (context_exhausted is set too)
    Error,       // it failed; see error, rejected and fatal
};

// Stable lowercase name ("stop", "length", "cancelled", "context_full", "error", "none").
const char * finish_reason_name(FinishReason reason);

struct RunResult {
    bool ok = false;
    bool cancelled = false; // generation was interrupted by Session::cancel() (ok stays true)
    bool context_exhausted = false;
    FinishReason finish = FinishReason::None;
    // The request failed before it changed any conversation state (validation, template, capacity
    // preflight): the caller's input is at fault, and the same session serves the next request.
    bool rejected = false;
    // The failure left the session unable to generate again (a fatal streaming I/O error). Every
    // other failure rolls the turn back and leaves the session usable; see docs/session.md.
    bool fatal = false;
    std::vector<std::string> context_events;
    std::string error;
    std::string generated_text;
    // The reasoning span, when a thinking model's chat template separated it from the answer.
    // Empty otherwise (chat off, non-reasoning model, harmony no-think). Display-only; the answer
    // in generated_text already has it stripped. See TokenMetrics::reasoning.
    std::string reasoning_text;
    std::vector<ToolCall> tool_calls;
    RunSummary summary;
    explicit operator bool() const { return ok; }
};

// Run one generation. `on_token` (nullable) is invoked once per generated token before
// the next decode; `sink` (nullable) receives the same per-token metrics plus the final
// summary. The trace sinks (all nullable) are diagnostics that perturb what they measure — see
// bmoe/route_trace.h and bmoe/decode_trace.h. Blocks until generation completes or errors.
RunResult run(const RunConfig & cfg,
              const std::function<void(const TokenMetrics &)> & on_token = nullptr,
              IMetricsSink * sink = nullptr,
              IRouteTraceSink * route_trace = nullptr,
              IComputeTraceSink * compute_trace = nullptr,
              IIoTraceSink * io_trace = nullptr);

} // namespace meitte
