// Teacher-forced scoring over a fixed text. It shares the live context, the router hook and the
// warm expert cache with generate(), so a routing policy is measured exactly as generation runs it.
#include "session_impl.h"
#include "llama_glue.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace meitte {

using namespace detail;

// Teacher-forced perplexity. See PplRequest for why the batch is labelled decode.
PplResult Session::perplexity(const PplRequest & req) {
    auto & im = *impl_;
    PplResult r;
    im.cancel_requested.store(false, std::memory_order_release);
    const auto t0 = Clock::now();
    llama_context * ctx = im.ctx.get();

    std::vector<llama_token> tokens;
    const int n = tokenize(im.vocab, req.text, /*add_special*/ true, /*parse_special*/ false, tokens);
    if (n < 2) {
        r.error = "text too short to score";
        return r;
    }
    if (n > im.cfg.n_ctx) {
        r.error =
            "text of " + std::to_string(n) + " tokens exceeds the session n_ctx (" + std::to_string(im.cfg.n_ctx) + ")";
        return r;
    }
    r.n_tokens = n;

    // Scoring needs the context from position 0, so the conversation ends here: every return path
    // below leaves the session as a new chat does (see PplRequest) — no scored text in the KV, no
    // stale history, media or draft positions for a later clear_kv=false turn to build on. An
    // isolated context would keep the conversation, but it would reserve a second KV and compute
    // buffers, and on this engine that memory comes out of the expert cache.
    im.reset_conversation();
    struct ResetOnExit {
        Session::Impl & im;
        ~ResetOnExit() { im.reset_conversation(); }
    } reset_on_exit{im};

    // Warm-up graph, discarded. Both routing policies decide at the terminal node of each layer's
    // weight chain, and which node that is gets LEARNED from the first graph of a run — so on a
    // single-batch pass they would never fire at all, and every cell would score the baseline. One
    // throwaway decode teaches the hook the shape; the scoring pass then runs with the policy live.
    {
        llama_batch warm = llama_batch_init(1, 0, 1);
        batch_fill(warm, tokens.data(), 1, 0, false);
        im.hook->set_batch_phase(req.as_decode ? 1 : 0);
        const int rc = llama_decode(ctx, warm);
        llama_batch_free(warm);
        if (rc != 0) {
            r.error = "warm-up decode failed";
            return r;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    const RoutingCounters routing0 = RoutingCounters::of(*im.hook);

    // Logits at EVERY position, so one pass scores every token — llama_batch_get_one would ask for
    // the last position only.
    llama_batch b = llama_batch_init(im.cfg.n_batch, /*embd*/ 0, /*n_seq_max*/ 1);
    struct BatchGuard {
        llama_batch & b;
        ~BatchGuard() { llama_batch_free(b); }
    } guard{b};

    double nll = 0.0;
    int scored = 0, top1 = 0;
    int last_row = 0; // logits row of the text's final position, for the choices
    // Score the logits at row `row` of the last decode against the token at `pos + 1`.
    auto score = [&](int row, int pos) -> bool {
        const float * lg = llama_get_logits_ith(ctx, row);
        if (!lg) {
            r.error = "no logits at position " + std::to_string(pos);
            return false;
        }
        // log softmax at the token that actually follows.
        nll -= LogSoftmax(lg, im.n_vocab).logp(lg, tokens[pos + 1]);
        ++scored;
        if (argmax(lg, im.n_vocab) == tokens[pos + 1]) ++top1;
        return true;
    };

    if (req.step) {
        // The decode regime, token by token (see PplRequest::step). The unscored prefix goes in as
        // one prefill batch — labelled as such, so a decode-only policy stays out of it exactly as
        // it does in generation — and every scored position is its own one-token decode.
        const int prefix = std::max(1, std::min(req.skip, n - 1));
        batch_fill(b, tokens.data(), prefix, /*pos0*/ 0, /*all_logits*/ false);
        im.hook->set_batch_phase(0);
        if (llama_decode(ctx, b) != 0) {
            r.error = "prefill decode failed";
            return r;
        }
        // With choices the last token is fed too, so the final distribution exists.
        const int last = req.choices.empty() ? n - 1 : n;
        for (int pos = prefix; pos < last; ++pos) {
            batch_fill(b, tokens.data() + pos, 1, pos, /*all_logits*/ true);
            im.hook->set_batch_phase(1);
            if (llama_decode(ctx, b) != 0) {
                r.error = "decode failed at position " + std::to_string(pos);
                return r;
            }
            last_row = 0;
            if (pos < req.skip || pos + 1 >= n) continue;
            if (!score(0, pos)) return r;
        }
    } else {
        for (int i = 0; i < n; i += im.cfg.n_batch) {
            const int chunk = std::min(im.cfg.n_batch, n - i);
            batch_fill(b, tokens.data() + i, chunk, /*pos0*/ i, /*all_logits*/ true);
            im.hook->set_batch_phase(req.as_decode ? 1 : 0);
            if (llama_decode(ctx, b) != 0) {
                r.error = "decode failed at position " + std::to_string(i);
                return r;
            }
            // Position p predicts token p+1, so the last token of the whole text is never scored
            // and the last row of a chunk predicts the first token of the next one.
            for (int j = 0; j < chunk; ++j) {
                const int pos = i + j;
                if (pos + 1 >= n || pos < req.skip) continue;
                if (!score(j, pos)) return r;
            }
            last_row = chunk - 1;
        }
    }
    if (!req.choices.empty()) {
        const float * lg = llama_get_logits_ith(ctx, last_row);
        if (!lg) {
            r.error = "no logits after the text";
            return r;
        }
        const LogSoftmax norm(lg, im.n_vocab);
        const double lse = (double) norm.max + norm.log_sum;
        for (const std::string & c : req.choices) {
            std::vector<llama_token> ct;
            if (tokenize(im.vocab, c, /*add_special*/ false, /*parse_special*/ false, ct) < 1) {
                r.error = "choice '" + c + "' does not tokenize";
                return r;
            }
            r.choice_logp.push_back((double) lg[ct[0]] - lse);
        }
    }
    if (scored == 0 && req.choices.empty()) {
        r.error = "nothing scored — text shorter than skip + 1";
        return r;
    }
    r.nll = scored > 0 ? nll / scored : 0.0;
    r.ppl = std::exp(r.nll);
    r.n_scored = scored;
    r.n_top1 = top1;
    const RoutingCounters routing = RoutingCounters::of(*im.hook) - routing0;
    r.experts_routed = routing.routed;
    r.experts_dropped = routing.dropped;
    r.experts_reranked = routing.reranked;
    r.experts_substituted = routing.substituted;
    r.seconds = secs(t0, Clock::now());
    r.ok = true;
    return r;
}

} // namespace meitte
