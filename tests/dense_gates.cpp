#include "bmoe/session.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace meitte;

struct Output {
    std::vector<float> prefill;
    std::vector<float> decode;
    std::vector<float> generated;
    std::string text;
};

static bool run_case(const char * path, int workers, bool stream, bool overlap, bool two_wave, Output & out) {
    SessionConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx = 128;
    cfg.n_batch = 8;
    cfg.n_ubatch = 8;
    cfg.n_threads = workers;
    cfg.dense_stream.enabled = stream;
    cfg.dense_stream.resident_mb = 1;
    cfg.dense_stream.window_mb = 4;
    cfg.dense_stream.io_lanes = workers;
    cfg.dense_stream.overlap = overlap;
    cfg.dense_stream.two_wave = two_wave;
    std::string error;
    auto session = Session::open(cfg, error);
    if (!session) {
        std::fprintf(stderr, "open failed: %s\n", error.c_str());
        return false;
    }
    if (stream && overlap && workers == 1 && !two_wave) {
        std::string second_error;
        auto second = Session::open(cfg, second_error);
        if (second || second_error.find("only one dense overlap session") == std::string::npos) {
            std::fprintf(stderr, "second dense overlap session was not rejected: %s\n", second_error.c_str());
            return false;
        }
    }
    PplRequest score;
    score.text = "Hello world";
    score.skip = 0;
    score.as_decode = false;
    if (auto r = session->perplexity(score); !r.ok) {
        std::fprintf(stderr, "prefill failed: %s\n", r.error.c_str());
        return false;
    }
    out.prefill = session->copy_logits();
    score.as_decode = true;
    score.step = true;
    if (auto r = session->perplexity(score); !r.ok) {
        std::fprintf(stderr, "decode failed: %s\n", r.error.c_str());
        return false;
    }
    out.decode = session->copy_logits();
    GenerateRequest request;
    request.prompt = "Hello world";
    request.n_predict = 5;
    RunResult r = session->generate(request);
    if (!r) {
        std::fprintf(stderr, "generate failed: %s\n", r.error.c_str());
        return false;
    }
    out.generated = session->copy_logits();
    out.text = r.generated_text;
    if (stream) {
        bool requested = false;
        RunResult cancelled = session->generate(request, [&](const TokenMetrics &) {
            if (!requested) {
                requested = true;
                session->cancel();
            }
        });
        if (!requested || !cancelled.cancelled) {
            std::fprintf(stderr, "dense cancellation did not stop generation\n");
            return false;
        }
        if (auto score_after_cancel = session->perplexity(score); !score_after_cancel.ok) {
            std::fprintf(stderr, "dense perplexity did not recover after cancellation: %s\n",
                         score_after_cancel.error.c_str());
            return false;
        }
        RunResult resumed = session->generate(request);
        if (!resumed || resumed.generated_text != out.text) {
            std::fprintf(stderr, "dense session did not recover after cancellation\n");
            return false;
        }
    }
    return !out.prefill.empty() && !out.decode.empty() && !out.generated.empty();
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    for (int workers : {1, 2}) {
        Output baseline;
        if (!run_case(argv[1], workers, false, false, false, baseline)) return 1;
        for (int mode = 0; mode < 3; ++mode) {
            Output streamed;
            const bool overlap = mode != 0;
            const bool two_wave = mode == 2;
            if (!run_case(argv[1], workers, true, overlap, two_wave, streamed)) return 1;
            if (!same(baseline.prefill, streamed.prefill) || !same(baseline.decode, streamed.decode) ||
                !same(baseline.generated, streamed.generated) || baseline.text != streamed.text) {
                std::fprintf(stderr, "dense logits differ at %d worker(s), mode %d\n", workers, mode);
                return 1;
            }
        }
    }
    std::puts("dense prefill/decode logits and generated text are byte-identical");
}
