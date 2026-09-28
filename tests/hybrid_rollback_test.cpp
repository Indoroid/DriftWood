// A continued chat turn on a recurrent or hybrid model must decode from the state of the kept
// prefix. With speculation on, the context has recurrent-state snapshots (n_rs_seq = draft_max),
// and llama.cpp accepts a removal of up to n_rs_seq positions even when the removed span crosses
// earlier decodes. It then restores a snapshot of a different position without an error. The turn
// continued from that wrong state.
//
// The second turn edits the first reply, so the cached tail diverges by a few tokens (fewer than
// draft_max). Its logits must be byte-identical to the same request on a cleared KV: the removal
// cannot use the snapshots, so the turn rebuilds the KV in the same prefill as the cleared request.
//
// Needs a real recurrent or hybrid model: usage hybrid_rollback_test MODEL. An attention-only model
// keeps the prefix, and a split prefill then changes the logits by rounding, so it does not apply.
#include "bmoe/session.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace meitte;

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    SessionConfig cfg;
    cfg.model_path = argv[1];
    cfg.n_ctx = 512;
    cfg.n_threads = 4;
    cfg.chatml = true;
    cfg.spec.source = DraftSource::ngram;
    cfg.spec.draft_max = SpecConfig::draft_max_limit;
    std::string error;
    auto session = Session::open(cfg, error);
    if (!session) {
        std::fprintf(stderr, "open failed: %s\n", error.c_str());
        return 1;
    }

    GenerateRequest first;
    first.messages = {{"user", "My name is Ada."}};
    first.think = false;
    first.n_predict = 3;
    first.clear_kv = true;
    RunResult r1 = session->generate(first);
    if (!r1) {
        std::fprintf(stderr, "turn 1 failed: %s\n", r1.error.c_str());
        return 1;
    }

    GenerateRequest second;
    second.messages = {{"user", "My name is Ada."}, {"assistant", "x"}, {"user", "What is my name?"}};
    second.think = false;
    second.n_predict = 1;
    second.clear_kv = false;
    RunResult continued = session->generate(second);
    const std::vector<float> continued_logits = session->copy_logits();
    second.clear_kv = true;
    RunResult rebuilt = session->generate(second);
    const std::vector<float> rebuilt_logits = session->copy_logits();
    if (!continued || !rebuilt) {
        std::fprintf(stderr, "turn 2 failed: %s\n", (continued ? rebuilt : continued).error.c_str());
        return 1;
    }
    std::printf("turn 2 prefilled %d tokens continued, %d tokens on a cleared KV\n", continued.summary.n_prompt,
                rebuilt.summary.n_prompt);
    const bool same =
        continued_logits.size() == rebuilt_logits.size() && !continued_logits.empty() &&
        std::memcmp(continued_logits.data(), rebuilt_logits.data(), continued_logits.size() * sizeof(float)) == 0;
    std::printf("[%s] continued turn logits == cleared-KV turn logits\n", same ? "PASS" : "FAIL");
    return same ? 0 : 1;
}
