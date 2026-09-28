// Reasoning-token accounting against the n_predict answer allowance (GenerateRequest). Runs the tiny
// test model under the pinned Qwen3.5 template, whose generation prompt opens the reasoning span, so
// every turn starts inside it. The random model never closes the span itself: the budget sampler
// must force the end sequence, and the answer allowance must start counting only after it.
#include "bmoe/session.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char * name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    std::ifstream file(BMOE_TMPL_QWEN3);
    std::stringstream tmpl;
    tmpl << file.rdbuf();

    meitte::RunConfig cfg;
    cfg.model_path = argv[1];
    cfg.n_ctx = 512;
    cfg.chatml = true;
    cfg.chat_template = tmpl.str();
    std::string error;
    auto session = meitte::Session::open(meitte::session_config_from(cfg), error);
    if (!session) {
        std::fprintf(stderr, "open failed: %s\n", error.c_str());
        return 1;
    }

    const int n_predict = 5;
    auto run = [&](bool think, int budget) {
        meitte::GenerateRequest req;
        req.prompt = "hi";
        req.n_predict = n_predict;
        req.think = think;
        req.reasoning_budget_tokens = budget;
        return session->generate(req);
    };

    auto off = run(false, -1);
    check(off.ok && off.summary.n_generated == n_predict && off.summary.n_reasoning == 0,
          "reasoning disabled: n_predict counts every token");
    auto unlimited = run(true, -1);
    check(unlimited.ok && unlimited.summary.n_generated == n_predict && unlimited.summary.n_reasoning == 0,
          "unlimited reasoning: n_predict bounds the whole generation");

    // Budget 0 closes the span at once: the forced end sequence is reasoning, the rest is answer.
    auto zero = run(true, 0);
    const int forced = zero.summary.n_reasoning;
    check(zero.ok && forced > 0 && zero.summary.n_generated == forced + n_predict,
          "budget 0: forced end sequence outside the answer allowance");
    auto three = run(true, 3);
    check(three.ok && three.summary.n_reasoning == 3 + forced && three.summary.n_generated == 3 + forced + n_predict,
          "finite budget: reasoning up to the budget is not charged");
    auto ten = run(true, 10);
    check(ten.ok && ten.summary.n_reasoning == 10 + forced &&
              ten.summary.n_generated - ten.summary.n_reasoning == n_predict,
          "answer token limit holds after a longer span");
    check(ten.finish == meitte::FinishReason::Length && ten.summary.n_generated > n_predict,
          "finish is length although more than n_predict tokens were generated");
    // The reasoning/answer TEXT split is not checked here: the random model's answer bytes do not
    // match the template's output grammar, so the parser falls back to the raw stream by design.

    // Cancellation inside the span rolls the turn back; the same request then runs unchanged.
    meitte::GenerateRequest req;
    req.prompt = "hi";
    req.n_predict = n_predict;
    req.reasoning_budget_tokens = 10;
    int seen = 0;
    auto cancelled = session->generate(req, [&](const meitte::TokenMetrics &) {
        if (++seen == 2) session->cancel();
    });
    check(cancelled.ok && cancelled.cancelled && seen < 10, "cancellation during reasoning");
    auto again = session->generate(req);
    check(again.ok && again.generated_text == ten.generated_text && again.reasoning_text == ten.reasoning_text &&
              again.summary.n_generated == ten.summary.n_generated,
          "turn after a cancelled reasoning span");
    return failures ? 1 : 0;
}
