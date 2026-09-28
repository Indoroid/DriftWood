// Transactional Session state: a cancelled or failed turn must leave the session exactly as it was,
// the physical KV and its logical records must never disagree, and request-local sampling must not
// leak into later requests. Runs on the tiny test model; see session_testing.h for the one failure
// that model cannot produce by itself (a refused partial KV removal).
#include "bmoe/session.h"
#include "session_testing.h"

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

const char * k_template = R"({% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{% if add_generation_prompt %}assistant: {% endif %})";

std::unique_ptr<meitte::Session> open_session(const char * model, float temp, bool ngram = false) {
    meitte::RunConfig cfg;
    cfg.model_path = model;
    cfg.n_ctx = 256;
    cfg.chatml = true;
    cfg.chat_template = k_template;
    cfg.sampling.temp = temp;
    cfg.sampling.seed = 1234;
    if (ngram) {
        cfg.spec.source = meitte::DraftSource::ngram;
        cfg.spec.draft_max = 3;
        cfg.spec.ngram_min_match = 1;
    }
    std::string error;
    auto session = meitte::Session::open(meitte::session_config_from(cfg), error);
    if (!session) std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return session;
}

meitte::GenerateRequest turn(const char * prompt, bool clear_kv, int n_predict = 6) {
    meitte::GenerateRequest req;
    req.prompt = prompt;
    req.clear_kv = clear_kv;
    req.n_predict = n_predict;
    req.render_text = false;
    return req;
}

// Generate, cancelling after the first token. The turn must report cancellation and roll back.
bool cancelled_turn(meitte::Session & s, const meitte::GenerateRequest & req) {
    auto r = s.generate(req, [&](const meitte::TokenMetrics &) { s.cancel(); });
    return r.ok && r.cancelled && r.finish == meitte::FinishReason::Cancelled && !r.rejected && !r.fatal;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const char * model = argv[1];

    // Reference: two ordinary turns, and the same conversation rendered as one transcript.
    // Texts alone are weak evidence on a tiny random model, so the final logits row is compared too:
    // it is bit-identical only when the KV the turn decoded against holds the same conversation.
    std::string first, second, rebuilt;
    std::vector<float> second_logits, rebuilt_logits, edited_logits;
    // A continued request whose transcript rewrites the first answer: its prefix trim must remove the
    // divergent tail of the KV.
    meitte::GenerateRequest edited = turn("", false);
    edited.messages = {{"user", "hello there"}, {"assistant", "something else entirely"}, {"user", "and then"}};
    {
        auto s = open_session(model, 0.0f);
        if (!s) return 1;
        auto r1 = s->generate(turn("hello there", true));
        auto r2 = s->generate(turn("and then", false));
        check(r1.ok && r2.ok, "reference conversation");
        first = r1.generated_text;
        second = r2.generated_text;
        second_logits = s->copy_logits();

        meitte::GenerateRequest full = turn("", true);
        full.messages = {{"user", "hello there"}, {"assistant", first}, {"user", "and then"}};
        auto r3 = s->generate(full);
        check(r3.ok, "reference transcript");
        rebuilt = r3.generated_text;
        rebuilt_logits = s->copy_logits();

        meitte::GenerateRequest fresh = edited;
        fresh.clear_kv = true;
        check(s->generate(fresh).ok, "reference edited transcript");
        edited_logits = s->copy_logits();
    }

    // Cancel mid-turn, then continue: the retried turn sees exactly the pre-turn state.
    {
        auto s = open_session(model, 0.0f);
        check(s->generate(turn("hello there", true)).ok, "first turn");
        check(cancelled_turn(*s, turn("and then", false)), "second turn cancels");
        auto r = s->generate(turn("and then", false));
        check(r.ok && r.generated_text == second && s->copy_logits() == second_logits,
              "retry after cancel matches the uninterrupted turn");
        check(s->generate(turn("more", false)).ok, "conversation continues after the retry");
    }

    // The rollback's partial KV removal is refused (recurrent/hybrid memory without a snapshot).
    // The KV must then be cleared together with its records, and the next turn must rebuild the
    // whole conversation instead of decoding after a prefix the context no longer holds.
    {
        auto s = open_session(model, 0.0f);
        check(s->generate(turn("hello there", true)).ok, "first turn (refused rollback)");
        auto req = turn("and then", false);
        auto r = s->generate(req, [&](const meitte::TokenMetrics &) {
            // Arm the refusal for the rollback only: the turn's own prefix trim already ran.
            meitte::testing::fail_next_kv_removals(1);
            s->cancel();
        });
        check(meitte::testing::pending_kv_removal_failures() == 0, "rollback removal was refused");
        meitte::testing::fail_next_kv_removals(0);
        check(r.ok && r.cancelled, "cancelled turn with a refused rollback");
        r = s->generate(turn("and then", false));
        check(r.ok && r.generated_text == rebuilt && s->copy_logits() == rebuilt_logits,
              "turn after a refused rollback rebuilds the conversation");
        check(s->generate(turn("more", false)).ok, "conversation continues after the rebuild");
    }

    // The prefix trim at the start of a continued turn is refused: the turn re-prefills in full.
    {
        auto s = open_session(model, 0.0f);
        check(s->generate(turn("hello there", true)).ok, "first turn (refused prefix trim)");
        meitte::testing::fail_next_kv_removals(1);
        auto r = s->generate(edited);
        check(meitte::testing::pending_kv_removal_failures() == 0, "prefix trim was refused");
        meitte::testing::fail_next_kv_removals(0);
        check(r.ok && s->copy_logits() == edited_logits, "refused prefix trim falls back to a full rebuild");
        check(s->generate(turn("more", false)).ok, "conversation continues after the full rebuild");
    }

    // N-gram speculation: accepted groups, the trim back to what was emitted, and rollback must keep
    // the conversation exact. A repetitive prompt makes the matcher draft.
    {
        const char * rep = "one two three one two three one two three one two";
        std::string t1, t2;
        std::vector<float> l2;
        long long drafted = 0;
        {
            auto s = open_session(model, 0.0f, true);
            auto r1 = s->generate(turn(rep, true, 10));
            auto r2 = s->generate(turn(rep, false, 10));
            check(r1.ok && r2.ok, "n-gram reference conversation");
            drafted = r1.summary.mtp_drafted + r2.summary.mtp_drafted;
            t1 = r1.generated_text;
            t2 = r2.generated_text;
            l2 = s->copy_logits();
        }
        std::printf("n-gram drafted %lld token(s) in the reference\n", drafted);
        auto s = open_session(model, 0.0f, true);
        auto r = s->generate(turn(rep, true, 10));
        check(r.ok && r.generated_text == t1, "n-gram first turn is repeatable");
        check(cancelled_turn(*s, turn(rep, false, 10)), "n-gram turn cancels");
        r = s->generate(turn(rep, false, 10));
        check(r.ok && r.generated_text == t2 && s->copy_logits() == l2, "n-gram retry matches the uninterrupted turn");
        r = s->generate(turn(rep, true, 10));
        check(r.ok && r.generated_text == t1, "n-gram session resets on a new chat");
    }

    // Turn outcome: the engine says why a turn ended and whether the session survives a failure.
    {
        auto s = open_session(model, 0.0f);
        auto r = s->generate(turn("hello there", true));
        const bool length = r.summary.n_generated == 6;
        check(r.ok && r.finish == (length ? meitte::FinishReason::Length : meitte::FinishReason::Stop),
              "finish reason of a completed turn");
        meitte::GenerateRequest bad = turn("and then", false);
        bad.chat_template_kwargs[""] = "true";
        r = s->generate(bad);
        check(!r.ok && r.rejected && !r.fatal && r.finish == meitte::FinishReason::Error,
              "invalid request is rejected, not fatal");
        check(s->generate(turn("and then", false)).ok, "session serves the next request after a rejection");
    }

    // Perplexity ends the conversation (PplRequest): a continued turn afterwards is a first turn, not
    // a turn decoded after the scored text or after a history the KV no longer holds.
    {
        std::vector<float> fresh_logits;
        std::string fresh;
        {
            auto s = open_session(model, 0.0f);
            auto r = s->generate(turn("and then", true));
            fresh = r.generated_text;
            fresh_logits = s->copy_logits();
        }
        auto s = open_session(model, 0.0f);
        check(s->generate(turn("hello there", true)).ok, "turn before perplexity");
        meitte::PplRequest ppl;
        ppl.text = "the quick brown fox jumps over the lazy dog";
        ppl.skip = 1;
        check(s->perplexity(ppl).ok, "perplexity between turns");
        auto r = s->generate(turn("and then", false));
        check(r.ok && r.generated_text == fresh && s->copy_logits() == fresh_logits,
              "turn after perplexity starts a new conversation");
        ppl.text = ""; // too short: rejected before any state changes
        check(!s->perplexity(ppl).ok, "rejected perplexity request");
        r = s->generate(turn("more", false));
        check(r.ok, "conversation continues after a rejected perplexity request");
    }

    // Sampling: the session sampler's RNG is part of the turn transaction, and a request-local
    // override neither replaces nor advances it.
    {
        std::string a1, a2;
        {
            auto s = open_session(model, 0.8f);
            auto r1 = s->generate(turn("hello there", true, 8));
            auto r2 = s->generate(turn("and then", false, 8));
            check(r1.ok && r2.ok, "sampled reference");
            a1 = r1.generated_text;
            a2 = r2.generated_text;
        }
        auto s = open_session(model, 0.8f);
        meitte::GenerateRequest b = turn("hello there", true, 8);
        b.override_sampling = true;
        b.sampling.temp = 1.5f;
        b.sampling.seed = 99;
        check(s->generate(b).ok, "request with a sampling override");
        auto r = s->generate(turn("hello there", true, 8));
        check(r.ok && r.generated_text == a1, "next request without override uses the session sampler");

        check(cancelled_turn(*s, b), "cancelled override request");
        r = s->generate(turn("hello there", true, 8));
        check(r.ok && r.generated_text == a1, "session sampler unaffected by a cancelled override");

        check(cancelled_turn(*s, turn("and then", false, 8)), "cancelled sampled turn");
        r = s->generate(turn("and then", false, 8));
        check(r.ok && r.generated_text == a2, "cancelled turn restores the sampler RNG");
    }

    return failures ? 1 : 0;
}
