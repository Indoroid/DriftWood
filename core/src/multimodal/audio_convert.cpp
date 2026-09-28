#include "audio_convert.h"
#include "../io/subprocess.h"

#include <cstring>

namespace meitte {

namespace {

const char * capture_error(pio::CaptureStatus status) {
    switch (status) {
    case pio::CaptureStatus::Unsupported:
        return "FFmpeg audio conversion on Windows is not available; supply WAV, MP3, or FLAC";
    case pio::CaptureStatus::StageFailed:
        return "cannot stage audio for FFmpeg";
    case pio::CaptureStatus::PipeFailed:
        return "cannot create FFmpeg output pipe";
    case pio::CaptureStatus::SpawnInitFailed:
        return "cannot initialize FFmpeg process";
    case pio::CaptureStatus::SpawnConfigFailed:
        return "cannot configure FFmpeg process";
    case pio::CaptureStatus::SpawnFailed:
        return "cannot start FFmpeg: ";
    case pio::CaptureStatus::Stopped:
        return "audio conversion cancelled or exceeded 30 seconds";
    case pio::CaptureStatus::PollFailed:
        return "FFmpeg output poll failed";
    case pio::CaptureStatus::ReadFailed:
        return "FFmpeg output read failed";
    case pio::CaptureStatus::LimitExceeded:
        return "decoded audio byte limit exceeded";
    case pio::CaptureStatus::WaitStopped:
        return "audio conversion cancelled or timed out";
    case pio::CaptureStatus::WaitFailed:
        return "cannot collect FFmpeg exit status";
    case pio::CaptureStatus::Ok: // exited cleanly, but the output is not whole f32 samples
    case pio::CaptureStatus::ExitFailed:
        break;
    }
    return "FFmpeg could not decode the audio";
}

} // namespace

bool decode_audio_ffmpeg(const std::vector<uint8_t> & input,
                         const std::string & bin_dir,
                         int sample_rate,
                         size_t max_bytes,
                         const std::function<bool()> & cancelled,
                         std::vector<float> & output,
                         std::string & error) {
    output.clear();
    if (input.empty() || sample_rate <= 0 || max_bytes < sizeof(float)) {
        error = "invalid audio input, sample rate, or byte limit";
        return false;
    }
    const std::string executable = bin_dir.empty() ? "ffmpeg" : bin_dir + "/ffmpeg";
    // Only the input/output pipes are valid protocols. Container references cannot open files or URLs.
    const std::vector<std::string> argv{
        executable, "-nostdin", "-v", "error", "-protocol_whitelist",       "pipe", "-i",    "pipe:0", "-map", "0:a:0",
        "-vn",      "-ac",      "1",  "-ar",   std::to_string(sample_rate), "-f",   "f32le", "pipe:1"};
    std::vector<uint8_t> bytes;
    const pio::CaptureResult run = pio::run_capture(argv, input, max_bytes, std::chrono::seconds(30), cancelled, bytes);
    if (run.status != pio::CaptureStatus::Ok || bytes.empty() || bytes.size() % 4) {
        error = capture_error(run.status);
        if (run.status == pio::CaptureStatus::SpawnFailed) error += std::strerror(run.spawn_error);
        return false;
    }
    // FFmpeg writes little-endian f32 samples; assemble each from its bytes so the host byte order
    // does not matter.
    output.resize(bytes.size() / 4);
    for (size_t i = 0; i < output.size(); ++i) {
        const uint8_t * b = bytes.data() + 4 * i;
        uint32_t value = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
        static_assert(sizeof(float) == sizeof(value));
        std::memcpy(&output[i], &value, sizeof(value));
    }
    return true;
}

} // namespace meitte
