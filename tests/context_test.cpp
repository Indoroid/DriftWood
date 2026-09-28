#include "bmoe/session.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    int failures = 0;
    auto check = [&](bool ok, const char * name) {
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
        if (!ok) ++failures;
    };
    meitte::RunConfig config;
    config.model_path = argv[1];
    config.n_ctx = 32;
    config.context.grow = meitte::ContextMode::Auto;
    check(!meitte::validate(config), "growth requires an explicit maximum");
    config.context.max_ctx = 256;
    config.context.min_ctx = 128;
    check(!meitte::validate(config), "minimum cannot exceed the opening context");
    config.context.min_ctx = 0;
    std::string error;
    auto session = meitte::Session::open(meitte::session_config_from(config), error);
    check(session != nullptr, "open dynamic session");
    if (!session) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    meitte::GenerateRequest request;
    request.chatml = false;
    request.n_predict = 8;
    request.prompt = std::string(200, 'a');
    auto result = session->generate(request);
    if (!result.ok)
        std::fprintf(stderr, "growth request failed: %s\n", result.error.c_str());
    else if (session->n_ctx() <= 32 || result.context_events.empty())
        std::fprintf(stderr, "growth request used n_ctx=%d and emitted %zu context events\n", session->n_ctx(),
                     result.context_events.size());
    check(result.ok && session->n_ctx() > 32 && !result.context_events.empty(), "raw prompt grows and decodes");
    config.context = {};
    session = meitte::Session::open(meitte::session_config_from(config), error);
    if (!session) return 1;
    result = session->generate(request);
    if (result.ok || !result.context_exhausted)
        std::fprintf(stderr, "fixed request returned ok=%d, exhausted=%d, n_ctx=%d, error=%s\n", result.ok,
                     result.context_exhausted, session->n_ctx(), result.error.c_str());
    check(!result.ok && result.context_exhausted && session->n_ctx() == 32, "default overflow leaves capacity fixed");
    check(result.finish == meitte::FinishReason::ContextFull && result.rejected && !result.fatal,
          "overflow is a rejected request with a context_full finish");
    request.prompt = "a";
    result = session->generate(request);
    if (!result.ok) std::fprintf(stderr, "recovery request failed: %s\n", result.error.c_str());
    check(result.ok, "session works after an oversized request");
    request.n_predict = 0;
    result = session->generate(request);
    check(!result.ok, "session rejects an invalid output budget");

    config.n_ctx = 64;
    config.context.trim = meitte::ContextMode::Auto;
    config.chatml = true;
    config.chat_template = R"({% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{% if add_generation_prompt %}assistant: {% endif %})";
    session = meitte::Session::open(meitte::session_config_from(config), error);
    check(session != nullptr, "open trimming session");
    if (!session) return 1;
    request = {};
    request.n_predict = 1;
    request.messages = {{"user", std::string(500, 'a')}, {"assistant", "old"}, {"user", "new"}};
    result = session->generate(request);
    check(result.ok && !result.context_events.empty(), "automatic trimming removes only an old complete turn");
    return failures ? 1 : 0;
}
