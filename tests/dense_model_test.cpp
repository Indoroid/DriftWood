// A model without experts runs from the plain mmap load, and expert streaming refuses it with an
// error that says so. The error used to ask for a MoE recipe for the architecture, which no recipe
// could fix. Usage: dense_model_test TINY_DENSE_MODEL.
#include "bmoe/session.h"

#include <cstdio>
#include <string>

using namespace meitte;

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    SessionConfig cfg;
    cfg.model_path = argv[1];
    cfg.n_ctx = 128;
    cfg.moe.enabled = true;
    std::string error;
    if (auto streamed = Session::open(cfg, error);
        streamed || error.find("without --moe-stream") == std::string::npos) {
        std::fprintf(stderr, "expert streaming on a dense model: %s\n", streamed ? "opened" : error.c_str());
        return 1;
    }
    cfg.moe.enabled = false;
    auto session = Session::open(cfg, error);
    if (!session) {
        std::fprintf(stderr, "open failed: %s\n", error.c_str());
        return 1;
    }
    GenerateRequest request;
    request.prompt = "Hello world";
    request.n_predict = 5;
    RunResult r = session->generate(request);
    if (!r || r.summary.n_generated <= 0) {
        std::fprintf(stderr, "generate failed: %s\n", r.error.c_str());
        return 1;
    }
    std::puts("dense model: expert streaming refused with the right error; mmap run generates");
    return 0;
}
