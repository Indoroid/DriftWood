// A media input the projector cannot encode must fail with an error that names the missing
// capability. Audio on a vision-only projector used to reach the audio decoder with a sample rate of
// 0 and fail as "invalid audio input, sample rate, or byte limit".
//
// Needs a real model and projector: usage media_capability_test MODEL MMPROJ. The audio is a
// generated WAV, so no media file is needed. A projector that supports audio has nothing to reject;
// the test then passes when the request succeeds.
#include "bmoe/session.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {
void put_le(std::vector<uint8_t> & out, uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}

// One second of a 440 Hz tone, 16 kHz mono PCM16.
std::vector<uint8_t> tone_wav() {
    const uint32_t rate = 16000, samples = rate, data_bytes = samples * 2;
    std::vector<uint8_t> w;
    const char * riff = "RIFF";
    w.insert(w.end(), riff, riff + 4);
    put_le(w, 36 + data_bytes, 4);
    const char * wave_fmt = "WAVEfmt ";
    w.insert(w.end(), wave_fmt, wave_fmt + 8);
    put_le(w, 16, 4);
    put_le(w, 1, 2); // PCM
    put_le(w, 1, 2); // mono
    put_le(w, rate, 4);
    put_le(w, rate * 2, 4);
    put_le(w, 2, 2);
    put_le(w, 16, 2);
    const char * data = "data";
    w.insert(w.end(), data, data + 4);
    put_le(w, data_bytes, 4);
    for (uint32_t i = 0; i < samples; ++i)
        put_le(w, uint32_t(int16_t(8000 * std::sin(2 * M_PI * 440 * i / rate))) & 0xffff, 2);
    return w;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    meitte::RunConfig config;
    config.model_path = argv[1];
    config.multimodal.mmproj_path = argv[2];
    config.chatml = true;
    config.n_ctx = 1024;
    std::string error;
    auto session = meitte::Session::open(meitte::session_config_from(config), error);
    if (!session) {
        std::fprintf(stderr, "open failed: %s\n", error.c_str());
        return 1;
    }
    meitte::GenerateRequest request;
    request.prompt = "Describe the sound.";
    request.media.push_back({tone_wav(), "tone.wav", meitte::MediaKind::Audio});
    request.n_predict = 1;
    const auto result = session->generate(request);
    if (result.ok) {
        std::printf("[PASS] the projector supports audio; nothing to reject\n");
        return 0;
    }
    if (result.error.find("audio input requires a projector with audio support") == std::string::npos) {
        std::printf("[FAIL] audio on a projector without audio: %s\n", result.error.c_str());
        return 1;
    }
    std::printf("[PASS] audio on a projector without audio: %s\n", result.error.c_str());
    return 0;
}
