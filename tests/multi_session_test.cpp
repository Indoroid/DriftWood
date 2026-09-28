// Several Sessions in one process. llama_backend_free() is not reference counted in the pinned
// llama.cpp (it only frees quantization tables), and the CPU readiness hooks are process globals, so
// closing one session must not disturb another, and a second overlap session must be refused rather
// than take over the first one's hook.
#include "bmoe/session.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char * name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

meitte::RunConfig config(const char * model, bool stream, bool overlap) {
    meitte::RunConfig cfg;
    cfg.model_path = model;
    cfg.n_ctx = 128;
    cfg.moe.enabled = stream;
    if (stream) {
        cfg.moe.cache_mb = 0;
        cfg.moe.io_threads = 2;
        cfg.moe.overlap = overlap;
    }
    return cfg;
}

std::unique_ptr<meitte::Session> open(const meitte::RunConfig & cfg, std::string & error) {
    return meitte::Session::open(meitte::session_config_from(cfg), error);
}

std::vector<float> run(meitte::Session & s, const char * prompt, std::string * text = nullptr) {
    meitte::GenerateRequest req;
    req.prompt = prompt;
    req.chatml = false;
    req.n_predict = 6;
    auto r = s.generate(req);
    if (text) *text = r.generated_text;
    return r.ok ? s.copy_logits() : std::vector<float>();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const char * model = argv[1];
    std::string error, ref_text, text;

    std::vector<float> reference;
    {
        auto s = open(config(model, false, false), error);
        if (!s) return 1;
        reference = run(*s, "hello there", &ref_text);
    }
    check(!reference.empty(), "reference session");

    {
        auto a = open(config(model, false, false), error);
        auto b = open(config(model, true, false), error);
        check(a && b, "two sessions open at once");
        if (!a || !b) return 1;
        check(run(*a, "and then") == run(*a, "and then"), "first session is repeatable");
        run(*b, "and then");
        a.reset(); // the first session's teardown frees process-wide backend state
        check(run(*b, "hello there", &text) == reference && text == ref_text,
              "second session unaffected by closing the first");
        b.reset();
    }

#ifdef BMOE_HAVE_EXPERT_READY_HOOK
    {
        auto first = open(config(model, true, true), error);
        check(first != nullptr, "overlap session opens");
        auto second = open(config(model, true, true), error);
        check(!second && error.find("process-wide CPU hook") != std::string::npos,
              "second overlap session is refused while the hook is owned");
        check(first && run(*first, "hello there", &text) == reference && text == ref_text,
              "first overlap session still streams correctly");
        auto plain = open(config(model, true, false), error);
        check(plain && run(*plain, "hello there") == reference, "serial streaming session beside overlap");
        first.reset();
        auto third = open(config(model, true, true), error);
        check(third && run(*third, "hello there") == reference, "overlap session opens after the owner closes");
    }
#endif
    return failures ? 1 : 0;
}
