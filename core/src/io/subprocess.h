// Run a helper program and capture its standard output — the platform side of the engine's
// external converters (FFmpeg audio decoding). Kept in io/ with the other OS primitives, so the
// callers hold no platform conditionals.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace meitte::pio {

// Where a capture stopped. The caller words the error, because only it knows what the child is for.
enum class CaptureStatus {
    Ok,
    Unsupported,       // this platform has no implementation
    StageFailed,       // the input could not be written to its private temporary file
    PipeFailed,        // the output pipe could not be created
    SpawnInitFailed,   // the spawn attributes could not be initialized
    SpawnConfigFailed, // the child's descriptors could not be configured
    SpawnFailed,       // the program could not be started (see CaptureResult::spawn_error)
    Stopped,           // cancelled, or the deadline passed, while output was being read
    PollFailed,
    ReadFailed,
    LimitExceeded, // the output exceeded max_bytes
    WaitStopped,   // cancelled, or the deadline passed, while the exit status was awaited
    WaitFailed,
    ExitFailed, // the child did not exit normally with status 0
};

struct CaptureResult {
    CaptureStatus status = CaptureStatus::Unsupported;
    int spawn_error = 0; // errno-style code for SpawnFailed
};

// Start `argv` (argv[0] is looked up in PATH unless it has a directory) with `input` as its standard
// input and standard error discarded, and collect its standard output into `output`, at most
// `max_bytes`. The input is staged in a private temporary file, so a child that fills its output pipe
// before it reads its input cannot deadlock. `stopped` is polled; when it returns true, or `timeout`
// passes, the child is killed and reaped. The child never outlives the call.
CaptureResult run_capture(const std::vector<std::string> & argv,
                          const std::vector<uint8_t> & input,
                          size_t max_bytes,
                          std::chrono::milliseconds timeout,
                          const std::function<bool()> & stopped,
                          std::vector<uint8_t> & output);

} // namespace meitte::pio
