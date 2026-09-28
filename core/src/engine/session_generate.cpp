// Session::generate(): render the prompt, reuse the KV prefix, prefill (text or media), then run the
// decode loop — plain, or draft/verify/accept under self-speculation — and report per-token metrics
// and the turn summary. Context growth and compaction are opt-in (see session_context.cpp).
#include "session_impl.h"
#include "chat_parse.h"
#include "chat_render.h"
#include "llama_glue.h"
#include "thinking_control.h"
#include "bmoe/ngram_draft.h"
#include "../io/platform_io.h"

#include "common.h"
#include "sampling.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace meitte {

using namespace detail;

RunResult Session::generate(const GenerateRequest & req,
                            const std::function<void(const TokenMetrics &)> & on_token,
                            IMetricsSink * sink) {
    Impl & im = *impl_;
    const MoeStreamConfig & moe = im.cfg.moe;
    llama_context * ctx = im.ctx.get();

    // Before anything is written about what this run did, say what it was.
    if (sink && !im.info_sent) {
        sink->on_run_info(im.info);
        im.info_sent = true;
    }

    RunResult res;
    // Set once the turn starts to change the KV. A failure before it is a rejected request.
    bool turn_started = false;
    auto fail = [&](std::string msg) {
        res.ok = false;
        res.error = std::move(msg);
        res.finish = res.context_exhausted ? FinishReason::ContextFull : FinishReason::Error;
        res.rejected = !turn_started;
        res.fatal = im.hook->fatal() || im.source.fatal() || im.dense_stream.fatal();
        return res;
    };
    // Let interrupted dense reads stop before the next generation can reuse their addresses.
    if (im.cfg.dense_stream.enabled && im.cancel_requested.load(std::memory_order_acquire) &&
        !im.dense_stream.reset_after_cancel())
        return fail("dense stream could not recover after cancellation");
    im.cancel_requested.store(false, std::memory_order_release);
    if (req.n_predict <= 0 || req.n_predict > std::numeric_limits<int>::max() - 8)
        return fail("n_predict must be positive and leave room for context accounting");

    const bool has_media = !req.media.empty();
    if (has_media && !im.mtmd.enabled()) return fail("media input requires a loaded --mmproj");
    if (has_media && !req.clear_kv)
        return fail("adding media to a preserved KV is not supported; start the media request with clear_kv=true");
    if ((has_media || (!req.clear_kv && im.kv_has_media)) && im.cfg.spec.is_mtp())
        return fail("MTP with media awaits upstream support for embedding batches; use n-gram or no speculation");

    // clear_kv = "new chat": drop the KV and the engine-held conversation. Otherwise this turn
    // continues the conversation, reusing the KV prefix already decoded from earlier turns.
    // A continued turn (clear_kv=false) keeps the sampler stream going, matching the KV it decodes
    // against; a new chat resets it (see reset_conversation).
    if (req.clear_kv) im.reset_conversation();

    // Sampling is fixed at open(), but an HTTP API needs request-scoped temperature and nucleus
    // settings. An override builds a sampler that lives for this request only: the session's own
    // sampler is neither used nor advanced, so the next request without an override samples exactly
    // as if this one had never overridden anything.
    using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    SamplerPtr request_sampler(req.override_sampling ? make_sampler_chain(req.sampling) : nullptr, llama_sampler_free);
    llama_sampler * const smpl = req.override_sampling ? request_sampler.get() : im.smpl;
    // The session sampler's RNG is part of the turn transaction: a rolled-back turn restores it, so
    // the next turn samples as if the failed one had never run.
    SamplerPtr sampler_snapshot(!req.override_sampling && im.smpl ? llama_sampler_clone(im.smpl) : nullptr,
                                llama_sampler_free);

    // Format the prompt. With chat on, render the model's OWN chat template (real Jinja) over the
    // WHOLE conversation so far, and set up reasoning parsing so a thinking model's internal
    // reasoning is stripped from the shown answer. req.think drives enable_thinking, per prompt.
    std::string prompt = req.prompt;
    std::string media_prefix;
    if (has_media) {
        for (size_t i = 0; i < req.media.size(); ++i)
            media_prefix += im.mtmd.marker();
    }
    bool chat_on = im.chat_on && req.chatml;
    bool history_pushed = false;   // did we append this turn's user message to chat_history?
    bool history_replaced = false; // did the caller provide the complete transcript?
    std::vector<common_chat_msg> prior_history;
    const bool prior_kv_has_media = im.kv_has_media;
    const bool context_policy_on = im.cfg.context.grow != ContextMode::Off ||
                                   im.cfg.context.summarize != ContextMode::Off ||
                                   im.cfg.context.trim != ContextMode::Off;
    const auto policy_backup = context_policy_on ? im.chat_history : std::vector<common_chat_msg>{};
    auto rollback_history = [&] {
        if (context_policy_on) {
            im.chat_history = policy_backup;
            history_pushed = false;
            history_replaced = false;
            return;
        }
        if (history_pushed) {
            im.chat_history.pop_back();
            history_pushed = false;
        }
        if (history_replaced) {
            im.chat_history = std::move(prior_history);
            history_replaced = false;
        }
    };
    bool prefilled_answer = false; // closed the reasoning span in the prompt, so skip reasoning parse
    common_chat_parser_params parse_params;
    common_chat_params chat_params;
    std::function<bool(int)> make_context_room;
    if (chat_on) {
        if (std::string invalid = check_template_request(req); !invalid.empty()) return fail(std::move(invalid));
        try {
            if (!req.messages.empty()) {
                // Render structured content BEFORE mutating persistent history.
                std::vector<common_chat_msg> rendered;
                std::string render_error;
                if (!render_request_messages(req, im.mtmd.marker(), im.mtmd.enabled(), media_prefix, rendered,
                                             render_error))
                    return fail(std::move(render_error));
                prior_history = im.chat_history;
                im.chat_history = std::move(rendered);
                history_replaced = true;
            } else {
                common_chat_msg user_msg;
                user_msg.role = "user";
                user_msg.content = has_media ? media_prefix + req.prompt : req.prompt;
                im.chat_history.push_back(user_msg);
                history_pushed = true;
            }

            common_chat_templates_inputs inputs = make_template_inputs(req, im.chat_history);

            // Many templates never read enable_thinking (LFM2.5 among them): the flag reaches the
            // jinja context, is discarded, and the model reasons anyway — the setting silently does
            // nothing. For those, close the reasoning span in the prompt instead, so the model
            // resumes at the first token of its answer with the reasoning already behind it.
            //
            // The span is rendered by llama.cpp's own handler for this template, so no marker for
            // any family — harmony's primed final channel included — is spelled out here. Which
            // models need this was measured at open(), not assumed. See thinking_control.h.
            if (!req.think && im.think_ctl == ThinkControl::Prefill) {
                detail::add_no_think_prefill(inputs);
                prefilled_answer = true;
            }

            chat_params = common_chat_templates_apply(im.chat_tmpls.get(), inputs);
            prompt = chat_params.prompt;
            // Work in complete user turns. Tool exchanges stay attached to their initiating turn.
            const bool automatic = im.cfg.context.grow == ContextMode::Auto ||
                                   im.cfg.context.summarize == ContextMode::Auto ||
                                   im.cfg.context.trim == ContextMode::Auto;
            const int reserve = automatic ? req.n_predict + std::max(0, req.reasoning_budget_tokens) + 8 : 9;
            make_context_room = [&, inputs](int reserve) mutable {
                bool changed_any = false;
                for (bool tried_summary = false; context_policy_on;) {
                    const size_t need = common_tokenize(im.vocab, prompt, true, true).size() + reserve;
                    if (need <= (size_t) im.cfg.n_ctx) break;
                    if (im.cfg.context.grow != ContextMode::Off && im.cfg.n_ctx < im.cfg.context.max_ctx) {
                        const int next = im.grown_context_size((int64_t) need);
                        std::string detail;
                        if (im.resize_context(next, detail)) {
                            ctx = im.ctx.get();
                            res.context_events.push_back("context grew to " + std::to_string(next));
                            changed_any = true;
                            continue;
                        }
                    }
                    size_t first = 0, end = 0;
                    bool protected_content = false;
                    if (!oldest_complete_turn(im.chat_history, im.mtmd.marker(), first, end, protected_content))
                        break; // The newest turn is protected.
                    bool changed = false;
                    if (!tried_summary && !protected_content && im.cfg.context.summarize != ContextMode::Off) {
                        tried_summary = true;
                        std::string summary, detail, original;
                        for (size_t i = first; i < end; ++i)
                            original += im.chat_history[i].content;
                        if (im.summarize(im.chat_history, first, end, summary, detail) &&
                            common_tokenize(im.vocab, summary, false, true).size() + 16 <
                                common_tokenize(im.vocab, original, false, true).size()) {
                            im.chat_history.erase(im.chat_history.begin() + first, im.chat_history.begin() + end);
                            common_chat_msg note;
                            note.role = "user";
                            note.content = "Earlier conversation summary (lossy):\n" + summary;
                            im.chat_history.insert(im.chat_history.begin() + first, std::move(note));
                            res.context_events.push_back("older conversation summarized (lossy)");
                            changed = true;
                        }
                    }
                    if (!changed && im.cfg.context.trim != ContextMode::Off && !protected_content) {
                        im.chat_history.erase(im.chat_history.begin() + first, im.chat_history.begin() + end);
                        res.context_events.push_back("oldest complete turn removed");
                        changed = true;
                    }
                    if (!changed) break;
                    changed_any = true;
                    inputs.messages = im.chat_history;
                    chat_params = common_chat_templates_apply(im.chat_tmpls.get(), inputs);
                    prompt = chat_params.prompt;
                }
                parse_params = detail::build_parse_params(chat_params);
                return changed_any;
            };
            make_context_room(reserve);
            parse_params = detail::build_parse_params(chat_params);
        } catch (const std::exception & e) {
            rollback_history();
            if (!im.cfg.chat_template.empty()) return fail(std::string("chat template apply failed: ") + e.what());
            std::fprintf(stderr, "bmoe: chat template apply failed (%s); using raw prompt\n", e.what());
            chat_on = false;
        }
    }

    if (has_media && !chat_on) prompt = media_prefix + req.prompt;

    const bool media_kv_continuation = !req.clear_kv && im.kv_has_media;
    // A context resize drops the physical KV. Retained media lets this turn rebuild it.
    const bool media_replay = media_kv_continuation && im.kv_n_past == 0;
    if (media_kv_continuation && !chat_on) {
        rollback_history();
        return fail("multimodal KV continuation requires a chat template");
    }

    ReasoningBudget reasoning;
    const bool budget_on = req.reasoning_budget_tokens >= 0;
    if (budget_on) {
        if (!chat_on) {
            rollback_history();
            return fail("reasoning_budget_tokens requires a chat template");
        }
        if (chat_params.thinking_start_tag.empty() || chat_params.thinking_end_tags.empty()) {
            rollback_history();
            return fail("model chat template does not expose reasoning delimiters");
        }

        const SamplingConfig & sampling = req.override_sampling ? req.sampling : im.cfg.sampling;
        if (!reasoning.init(im.model.get(), im.vocab, sampling, req.reasoning_budget_tokens, chat_params, prompt)) {
            rollback_history();
            return fail("could not build the reasoning budget sampler");
        }
    }
    common_sampler * const reasoning_sampler = reasoning.sampler(); // null without a budget
    // Output positions the turn may need: the answer allowance plus, under a budget, the reasoning
    // span that does not consume it.
    const int output_reserve = req.n_predict + (budget_on ? reasoning.allowance() : 0);

    std::vector<llama_token> tokens;
    int n_prompt = 0;
    llama_pos prompt_n_past = 0;
    if (has_media || media_kv_continuation) {
        if (!tokenize_media_logical(im.vocab, prompt, im.mtmd.marker(), tokens) || tokens.empty()) {
            rollback_history();
            return fail("multimodal prompt tokenization failed");
        }
        n_prompt = (int) tokens.size();
    } else {
        n_prompt = tokenize(im.vocab, prompt, /*add_special*/ true, /*parse_special*/ true, tokens);
        if (n_prompt < 1) {
            rollback_history();
            return fail("empty prompt after tokenization");
        }
    }

    // The text to surface: with chat on, parse the raw output so a reasoning model's internal
    // thinking is separated from the answer. The answer is shown inline; the reasoning is handed to
    // the UI as a distinct thinking block rather than dropped, so a Thinking-on run does not sit on a
    // blank screen while the model reasons. Generation always uses the raw tokens.
    struct ShownView {
        std::string content;   // the answer, reasoning stripped
        std::string reasoning; // the thinking span, empty unless the parser split one out
    };
    auto shown_view = [&](const std::string & raw, bool partial) -> ShownView {
        if (!chat_on) return {raw, ""};
        // A prefilled turn resumes mid-answer: the reasoning span and the turn header the parser
        // anchors on are already in the prompt, not in the stream, so there is nothing to strip —
        // the raw stream already IS the answer. (If a model reasons anyway despite the closed span,
        // that reasoning surfaces verbatim rather than being cut out here. Hiding it would only
        // disguise a model this mechanism does not work on; the honest report is ThinkControl::None.)
        if (prefilled_answer) return {raw, ""};
        try {
            common_chat_msg msg = common_chat_parse(raw, partial, parse_params);
            return {msg.content, msg.reasoning_content};
        } catch (const std::exception & e) {
            detail::warn_parse_failed_once(e.what());
            return {raw, ""};
        }
    };

    // Reuse the KV prefix already decoded from earlier turns (chat mode only): find how many
    // leading tokens still match the cache, drop the divergent tail, and prefill only the suffix.
    // Keeping at least one token to decode means a turn is never a no-op. clear_kv leaves
    // kv_tokens empty, so n_common = 0 and this reduces to a full prefill — the one-shot path the
    // byte-identity gates exercise stays unchanged.
    size_t n_common = 0;
    bool media_append_only = false;
    llama_pos media_reuse_n_past = im.kv_n_past;
    if (chat_on && !has_media && !im.kv_tokens.empty()) {
        if (media_kv_continuation) {
            while (n_common < im.kv_tokens.size() && n_common < tokens.size() &&
                   im.kv_tokens[n_common] == tokens[n_common])
                ++n_common;
            if (n_common != im.kv_tokens.size()) {
                if (im.kv_last_generation_start >= im.kv_tokens.size()) {
                    rollback_history();
                    return fail("preserved multimodal KV has no generated assistant span; start a new chat with "
                                "clear_kv=true");
                }
                const auto reply_begin = im.kv_tokens.begin() + im.kv_last_generation_start;
                const auto reply =
                    std::search(tokens.begin() + n_common, tokens.end(), reply_begin, im.kv_tokens.end());
                if (reply == tokens.end()) {
                    rollback_history();
                    return fail(
                        "preserved multimodal KV cannot align the prior assistant response; start a new chat with "
                        "clear_kv=true");
                }
                tokens.assign(reply + (im.kv_tokens.end() - reply_begin), tokens.end());
                n_prompt = (int) tokens.size();
                n_common = 0;
                media_append_only = true;
            }
        } else {
            const size_t max_common = tokens.size() > 0 ? tokens.size() - 1 : 0;
            while (n_common < im.kv_tokens.size() && n_common < max_common &&
                   im.kv_tokens[n_common] == tokens[n_common])
                ++n_common;
            // Drop the divergent tail. Recurrent and hybrid memory can refuse a partial removal; then
            // truncate_kv clears the KV and its records, and the turn re-prefills in full rather than
            // continuing from an inconsistent cache.
            //
            // The draft context mirrors the target's positions, so it is rewound to the SAME point —
            // including when n_common did not move, since the previous turn left it holding
            // everything it generated. Miss this and the first prefill batch of the turn starts at a
            // position the draft context already has, which llama.cpp rejects outright: the second
            // message of a conversation fails while the first always works.
            if ((n_common < im.kv_tokens.size() || im.ctx_dft) &&
                !im.truncate_kv((llama_pos) n_common, n_common, /*in_last_decode*/ false))
                n_common = 0;
        }
    }

    const size_t n_new_prompt_tokens = (size_t) n_prompt - n_common;
    const bool reserve_output = !context_policy_on || im.cfg.context.grow == ContextMode::Auto ||
                                im.cfg.context.summarize == ContextMode::Auto ||
                                im.cfg.context.trim == ContextMode::Auto;
    const long long context_need = (media_kv_continuation ? (long long) media_reuse_n_past : (long long) n_common) +
                                   (long long) n_new_prompt_tokens + (reserve_output ? output_reserve : 1) + 8;
    if (!chat_on && context_need > im.cfg.n_ctx && im.cfg.context.grow != ContextMode::Off &&
        context_need <= im.cfg.context.max_ctx) {
        std::string detail;
        const int next = im.grown_context_size(context_need);
        if (im.resize_context(next, detail)) {
            ctx = im.ctx.get();
            res.context_events.push_back("context grew to " + std::to_string(next));
        }
    }
    if (context_need > im.cfg.n_ctx) {
        rollback_history();
        res.context_exhausted = true;
        return fail("prompt + n_predict exceeds the session n_ctx (" + std::to_string(im.cfg.n_ctx) +
                    "); open the session with a larger n_ctx");
    }

    // Roll this turn back to the state before it started: drop the KV added this turn, forget the
    // tokens we fed, un-append the user message and restore the session sampler. Every failure after
    // this point goes through here, so a cancelled or failed turn leaves prior turns usable.
    const size_t rollback_tokens = media_append_only ? im.kv_tokens.size() : n_common;
    const llama_pos rollback_n_past = media_kv_continuation ? media_reuse_n_past : (llama_pos) n_common;
    const size_t rollback_generation_start = im.kv_last_generation_start;
    auto restore_sampler = [&]() {
        if (!sampler_snapshot) return;
        llama_sampler_free(im.smpl);
        im.smpl = sampler_snapshot.release();
    };
    auto rollback_turn = [&]() {
        if (chat_on) {
            // The draft context follows the target to the same position (truncate_kv): a turn that
            // left the two at different positions would fail the NEXT turn, not this one. When the
            // prefix cannot be kept, the records are cleared with it and the next turn rebuilds.
            if (im.truncate_kv(rollback_n_past, rollback_tokens, /*in_last_decode*/ false)) {
                im.kv_last_generation_start = rollback_generation_start;
                if (media_kv_continuation) im.kv_n_past = rollback_n_past;
            }
            rollback_history();
        } else {
            llama_memory_clear(llama_get_memory(ctx), true);
            if (im.ctx_dft) llama_memory_clear(llama_get_memory(im.ctx_dft.get()), true);
        }
        if (has_media) {
            im.kv_has_media = false;
            im.retained_media.clear();
            im.kv_n_past = 0;
            im.kv_last_generation_start = 0;
        }
        restore_sampler();
    };
    auto abort_turn = [&](std::string msg) {
        rollback_turn();
        return fail(std::move(msg));
    };

    // ── prefill (chunked by n_batch; positions auto-continue from the reused prefix) ──
    // Prefill attribution (#173): the cumulative counters are pinned before the prompt chunks so
    // their deltas across prefill carry the same wall-additive terms the decode phase reports.
    // The session layer already holds everything needed; the streamer is untouched.
    turn_started = true;
    PrefillTally prefill_tally;
    prefill_tally.begin(moe.enabled, im.source);
    const uint64_t prefill_faults0 = pio::major_faults();
    const DenseStreamCounters dense_prefill0 = DenseStreamCounters::of(im.dense_stream);
    double prefill_peak_rss = 0.0;
    const auto t_prefill0 = Clock::now();
    // Two predicates, deliberately distinct. spec_on is "the verify loop runs" — a wide batch, an
    // accept pass, a rollback — and both sources need all of it. mtp_on is "the draft comes from the
    // head", which is the only thing that needs the second context and llama.cpp's `common`
    // speculative driver. Conflating them is what would make the n-gram source pay for a draft
    // context it never uses.
    // The budget sampler has to see every sampled token, while speculative verification accepts a
    // batch at once. Keep the draft context in sync, but decode budgeted requests one token at a time.
    const bool spec_on = im.cfg.spec.enabled() && !reasoning_sampler;
    const bool mtp_on = im.mtp != nullptr;
    const bool media_prefill = has_media || media_replay;
    std::vector<llama_token> media_text_tail;
    double media_prepare_seconds = 0.0;
    double media_projector_seconds = 0.0;
    if (media_prefill) {
        // mtmd performs text/media llama_decode() calls on THIS text context. Its projector graph
        // has no RouterHook, but every embedding decode through ctx retains DriftWood's MoE hook.
        im.hook->set_batch_phase(/*prefill*/ 0);
        llama_pos max_prompt_pos = (llama_pos) im.cfg.n_ctx - (reserve_output ? output_reserve : 1) - 8;
        MtmdRuntime::Prepared prepared;
        std::string media_error;
        const auto & media = has_media ? req.media : im.retained_media;
        const auto media_prepare_start = Clock::now();
        if (!im.mtmd.prepare(prompt, media, prepared, media_error, [&] { return im.cancel_requested.load(); })) {
            rollback_turn();
            return fail(media_error);
        }
        media_prepare_seconds = secs(media_prepare_start, Clock::now());
        const int64_t need =
            std::max<int64_t>(prepared.n_pos, prepared.n_tokens) + (reserve_output ? output_reserve : 1) + 8;
        if (need > im.cfg.n_ctx && im.cfg.context.grow != ContextMode::Off && need <= im.cfg.context.max_ctx) {
            const int next = im.grown_context_size(need);
            if (im.resize_context(next, media_error)) {
                ctx = im.ctx.get();
                max_prompt_pos = im.cfg.n_ctx - (reserve_output ? output_reserve : 1) - 8;
                res.context_events.push_back("context grew to " + std::to_string(next));
            }
        }
        if (prepared.n_pos > max_prompt_pos || prepared.n_tokens > (size_t) std::max(0, max_prompt_pos)) {
            rollback_turn();
            res.context_exhausted = true;
            return fail("multimodal prompt exceeds context capacity; reopen with a larger context");
        }
        MtmdRuntime::DecodeObserver observer;
        observer.before = [&](int pos, int n, MediaKind media_kind) { im.trace_begin(pos, n, 0, media_kind); };
        observer.after = [&](const llama_batch & batch) { im.trace_flush(&batch); };
        MtmdPrefillResult mm =
            im.mtmd.evaluate(ctx, prepared, im.cfg.n_batch, observer, [&] { return im.cancel_requested.load(); });
        if (!mm.ok) {
            const bool cancelled = im.cancel_requested.load(std::memory_order_relaxed);
            rollback_turn();
            if (cancelled) {
                res.ok = true;
                res.cancelled = true;
                res.finish = FinishReason::Cancelled;
                return res;
            }
            if (moe.overlap && im.source.fatal()) return fail("expert stream I/O failed during multimodal prefill");
            return fail(mm.error);
        }
        media_projector_seconds = mm.projector_seconds;
        n_prompt = (int) mm.n_tokens;
        media_text_tail = std::move(mm.text_tail);
        prompt_n_past = mm.n_past;
        im.kv_tokens = tokens;
        if (has_media) im.retained_media = req.media;
        im.kv_has_media = true;
        im.kv_n_past = mm.n_past;
    } else {
        if (media_kv_continuation &&
            std::find(tokens.begin() + n_common, tokens.end(), k_media_kv_sentinel) != tokens.end()) {
            rollback_history();
            return fail("multimodal KV continuation cannot change the original media transcript; start a new chat with "
                        "clear_kv=true");
        }
        for (int i = (int) n_common; i < n_prompt; i += im.cfg.n_batch) {
            const int chunk = std::min(im.cfg.n_batch, n_prompt - i);
            llama_batch pf;
            if (mtp_on) {
                batch_fill(im.mtp_batch, tokens.data() + i, chunk, /*pos0*/ i, /*all_logits*/ false);
                pf = im.mtp_batch;
            } else {
                pf = llama_batch_get_one(tokens.data() + i, chunk);
            }
            im.trace_begin(i, chunk, /*phase*/ 0);
            if (llama_decode(ctx, pf) != 0) {
                if (im.cancel_requested.load(std::memory_order_relaxed)) {
                    rollback_turn();
                    res.ok = true;
                    res.cancelled = true;
                    res.finish = FinishReason::Cancelled;
                    return res;
                }
                if (moe.overlap && im.source.fatal())
                    return abort_turn("expert stream I/O failed during overlap prefill");
                return abort_turn("prefill decode failed");
            }
            if (im.cfg.dense_stream.enabled) {
                pio::ProcessMemory pm;
                if (pio::process_memory(&pm))
                    prefill_peak_rss = std::max(prefill_peak_rss, pm.rss_bytes / (1024.0 * 1024.0));
            }
            im.trace_flush();
            if (mtp_on && !common_speculative_process(im.mtp.get(), pf))
                return abort_turn("MTP draft context failed to process the prefill batch");
        }
        if (chat_on)
            for (int i = (int) n_common; i < n_prompt; ++i)
                im.kv_tokens.push_back(tokens[i]);
        if (mtp_on) common_speculative_begin(im.mtp.get(), /*seq_id*/ 0, tokens);
        prompt_n_past = media_kv_continuation ? media_reuse_n_past + (llama_pos) (n_prompt - n_common) : n_prompt;
    }
    const double prefill_seconds = secs(t_prefill0, Clock::now());
    // Prefill attribution, closed here (the begin() snapshot sits above the chunk loop): deltas
    // of the cumulative counters across the prompt, the quantities the decode phase reports per
    // token. Read with the same rules: io is summed lane busy time under overlap, stall is the
    // union of stalled intervals, cpu is whole-process (upper bound on compute-thread time).
    prefill_tally.end(moe.enabled, im.source);
    const uint64_t prefill_faults = pio::major_faults() - prefill_faults0;
    if (im.cfg.dense_stream.enabled) {
        const DenseStreamCounters now = DenseStreamCounters::of(im.dense_stream);
        prefill_tally.read_mib = (now.bytes - dense_prefill0.bytes) / (1024.0 * 1024.0);
        prefill_tally.io_seconds = (now.io_ns - dense_prefill0.io_ns) / 1e9;
        prefill_tally.stall_seconds = (now.wait_ns - dense_prefill0.wait_ns) / 1e9;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (chat_on) im.kv_last_generation_start = im.kv_tokens.size();

    // ── greedy generation ──
    res.ok = true;
    std::string gen;
    std::vector<llama_token> emitted_tokens;
    int n_gen = 0; // every generated token
    // Tokens charged to the n_predict allowance. Without a reasoning budget that is every token; with
    // one, reasoning tokens up to the budget's allowance are not charged (see ReasoningBudget).
    int n_charged = 0;
    int n_reasoning = 0;
    double gen_seconds = 0.0;

    // Baseline seeded from prefill's closing sample: the summary reports the generation phase
    // only, from real per-token deltas (prefill routes near the whole bank, so folding it into a
    // per-token average would badly inflate the flash-I/O figure). Nothing runs between the two
    // phases, so end()'s snapshot IS the decode baseline — one reading of the counters at the
    // boundary instead of two that could drift apart. In a warm session these counters carry the
    // prior prompts' totals; the deltas make each prompt self-relative.
    GenTally tally;
    tally.overlap = moe.overlap;
    DenseStreamCounters dense_prev = DenseStreamCounters::of(im.dense_stream);
    const DenseStreamCounters dense_decode0 = dense_prev;
    double decode_peak_rss = 0.0;
    if (moe.enabled) tally.seed(prefill_tally.post);
    const IExpertSource::Stats st_spec0 = moe.enabled ? im.source.stats() : IExpertSource::Stats{};
    long long prev_spec_bytes = (long long) st_spec0.spec_read_bytes;
    long long prev_spec_experts = st_spec0.spec_experts;
    long long prev_spec_useful = st_spec0.spec_useful;
    // Taken after prefill, so the drop counters describe generation — the phase the policy is armed
    // for and the one the tok/s number is about.
    const RoutingCounters routing0 = RoutingCounters::of(*im.hook);
    // Per-token cursors for the hook's own eval-thread meters (route-ahead issue + watchdog): the
    // hook accumulates for the session, the rows want this token's share.
    long long prev_ra_issue_ns = im.hook->route_ahead_issue_ns();
    long long prev_ra_wd_ns = im.hook->route_ahead_wd_ns();

    // The decode bracket below measures llama_decode and nothing else, which is what makes
    // compute_ms a clean residual — but it also means everything BETWEEN two decodes (sampling,
    // detokenization, rendering, the sinks) is charged to nobody and disappears from tok/s. Mark
    // where the last decode ended so each token can report the gap it actually waited through.
    auto loop_mark = Clock::now();
    double loop_overhead_s = 0.0;

    // Absolute position the next decoded token occupies. Prefill left the context filled up to
    // n_prompt-1; without speculation this simply tracks n_prompt + n_gen, but a verify decode
    // advances it by a whole accepted group, so it is carried explicitly.
    llama_pos n_past = prompt_n_past;
    // The token sequence the draft source conditions on: the prompt plus everything confirmed since.
    // Only built when speculating — the plain path has no use for it. The head reads it as the
    // sequence to seed from; the n-gram source searches it, and it IS the whole corpus.
    std::vector<llama_token> mtp_ctx;
    if (spec_on) {
        mtp_ctx = media_prefill ? media_text_tail : tokens;
        // A logical media marker is not a vocabulary token. Restrict lookup to the text after it.
        auto boundary = std::find(mtp_ctx.rbegin(), mtp_ctx.rend(), k_media_kv_sentinel);
        if (boundary != mtp_ctx.rend()) mtp_ctx.erase(mtp_ctx.begin(), boundary.base());
    }
    std::vector<llama_token> verify_toks; // [confirmed token, drafts...] for the verify batch
    std::vector<llama_token> confirmed;   // what one decode confirmed, in order
    const SpecCounters spec0 = im.spec_totals;

    // The token the last decode settled on, not yet in the KV. Greedy stays argmax (byte-identical
    // to the resident reference the gates check); with a sampling chain, draw from the context's
    // last-position logits, which llama_sampler_sample reads at index -1 — the same logits argmax
    // would have read.
    llama_token tok = reasoning_sampler ? common_sampler_sample(reasoning_sampler, ctx, -1)
                      : smpl            ? llama_sampler_sample(smpl, ctx, -1)
                                        : argmax(logits, im.n_vocab);

    bool eog = false;
    while (n_charged < req.n_predict) {
        if (llama_vocab_is_eog(im.vocab, tok)) {
            eog = true;
            break;
        }
        if (n_past + 1 >= im.cfg.n_ctx) {
            bool rebuilt = false;
            bool changed = false;
            bool compacted = false;
            std::string detail;
            if (im.cfg.context.grow != ContextMode::Off && im.cfg.n_ctx < im.cfg.context.max_ctx) {
                const int next = (int) std::min<int64_t>(im.cfg.context.max_ctx, 2ll * im.cfg.n_ctx);
                changed = im.resize_context(next, detail);
            }
            if (!changed && make_context_room && !has_media && !media_kv_continuation) {
                compacted = make_context_room((int) emitted_tokens.size() + 9);
                if (compacted) changed = im.resize_context(im.cfg.n_ctx, detail);
            }
            if (changed) {
                const int next = im.cfg.n_ctx;
                ctx = im.ctx.get();
                llama_pos position = 0;
                std::vector<llama_token> replay;
                if (compacted) {
                    tokens = common_tokenize(im.vocab, prompt, true, true);
                    n_prompt = (int) tokens.size();
                    n_common = 0;
                    prompt_n_past = n_prompt;
                }
                if (has_media || media_kv_continuation) {
                    auto mm = im.mtmd.prefill(ctx, prompt, im.retained_media, im.cfg.n_batch, next - 1,
                                              [&] { return im.cancel_requested.load(); });
                    if (!mm.ok) {
                        rollback_turn();
                        return fail("context replay failed: " + mm.error);
                    }
                    position = mm.n_past;
                } else
                    replay = tokens;
                replay.insert(replay.end(), emitted_tokens.begin(), emitted_tokens.end());
                auto batch = llama_batch_init(im.cfg.n_batch, 0, 1);
                bool ok = true;
                for (size_t i = 0; i < replay.size() && ok; i += im.cfg.n_batch) {
                    const int n = (int) std::min<size_t>(im.cfg.n_batch, replay.size() - i);
                    batch_fill(batch, replay.data() + i, n, position, false);
                    im.trace_begin(position, n, 0);
                    ok = llama_decode(ctx, batch) == 0;
                    im.trace_flush();
                    if (ok && mtp_on) ok = common_speculative_process(im.mtp.get(), batch);
                    position += n;
                }
                llama_batch_free(batch);
                if (!ok) {
                    rollback_turn();
                    return fail("context replay failed; clear the conversation before retrying");
                }
                im.kv_tokens = tokens;
                im.kv_tokens.insert(im.kv_tokens.end(), emitted_tokens.begin(), emitted_tokens.end());
                if (spec_on && compacted) mtp_ctx = im.kv_tokens;
                if (mtp_on) common_speculative_begin(im.mtp.get(), 0, im.kv_tokens);
                n_past = position;
                im.kv_n_past = position;
                res.context_events.push_back("context replayed at capacity " + std::to_string(next));
                rebuilt = true;
            }
            if (!rebuilt) {
                res.context_exhausted = true;
                res.ok = false;
                res.error = "context capacity reached; reopen with a larger context or reset the conversation";
                break;
            }
        }

        // ── draft ──
        // The source proposes a continuation of `tok`, capped at the caller's remaining budget: a
        // draft accepted past n_predict would be verified, charged for, and then discarded. The cap
        // goes in BEFORE drafting, so no source is ever asked for tokens with nowhere to go.
        int n_draft = 0;
        double draft_s = 0.0; // this group's drafting + catch-up (see below)
        const int room = std::min(req.n_predict - n_charged - 1, im.cfg.n_ctx - (int) n_past - 2);
        if (spec_on && room > 0) {
            const auto d0 = Clock::now();
            const uint64_t db0 = moe.enabled ? im.source.stats().read_bytes : 0;
            im.draft_buf.clear();
            if (mtp_on) {
                common_speculative_draft_params & dp = common_speculative_get_draft_params(im.mtp.get(), /*seq*/ 0);
                dp.drafting = true;
                dp.n_max = std::min(im.cfg.spec.draft_max, room);
                dp.pos0 = n_past;
                dp.id_last = tok;
                dp.prompt = &mtp_ctx;
                dp.result = &im.draft_buf;
                common_speculative_draft(im.mtp.get());
                if ((int) im.draft_buf.size() > room) im.draft_buf.resize((size_t) room);
                n_draft = (int) im.draft_buf.size();

                // Drafting WROTE into the draft context's KV at the very positions the catch-up
                // below is about to occupy. Rewind to where the draft started, or the second decode
                // collides with the first (llama.cpp requires a batch to begin strictly after the
                // last stored position). The catch-up is what replaces those rows with ones
                // conditioned on the target's own hidden states instead of the head's guesses.
                if (!kv_remove_tail(im.ctx_dft.get(), n_past))
                    return abort_turn("the MTP draft context does not support rewinding its KV cache");
            } else {
                // The n-gram source reads the text and nothing else: no draft context, no decode, no
                // expert read, so the bytes bracketed around this arm are zero by construction. When
                // it finds no confident match it returns 0 and the step below is an ordinary
                // single-token decode — the property the head does not have, and the reason the
                // floor of this source is the baseline rather than a loss.
                n_draft = ngram_draft(mtp_ctx, tok, std::min(im.cfg.spec.draft_max, room), im.cfg.spec.ngram_min_match,
                                      im.cfg.spec.ngram_max_match, im.draft_buf);
            }
            im.spec_totals.drafted += n_draft;
            if (n_draft > 0) ++im.spec_totals.drafted_steps;
            draft_s += secs(d0, Clock::now());
            if (moe.enabled) im.spec_totals.draft_read_bytes += im.source.stats().read_bytes - db0;
        }

        // ── verify batch: the confirmed token, then every draft, all asking for logits ──
        //
        // With nothing drafted there is nothing to verify, so the step takes the plain path — one
        // token, one logits row, no rollback. That is not an optimisation of the speculative loop,
        // it IS the loop's floor: a step that drafts nothing must cost exactly what it would have
        // cost with speculation off, or a source that abstains would still be paying for the
        // scaffolding. It applies to the head too, whenever p_min stops it drafting.
        const bool wide = spec_on && n_draft > 0;
        llama_batch step;
        if (spec_on) {
            verify_toks.clear();
            verify_toks.push_back(tok);
            // n_draft, not draft_buf.size(): on the last token of a run there is no room to draft
            // and the buffer still holds the previous step's proposal.
            verify_toks.insert(verify_toks.end(), im.draft_buf.begin(), im.draft_buf.begin() + n_draft);
        }
        if (wide) {
            batch_fill(im.mtp_batch, verify_toks.data(), (int) verify_toks.size(), n_past, /*all_logits*/ true);
            step = im.mtp_batch;
        } else {
            step = llama_batch_get_one(&tok, 1);
        }

        // Bracket ONLY the decode: major faults and CPU-time deltas here decompose this token's
        // compute residual into flash-fault stalls vs. genuine (or throttled) computation.
        const uint64_t f0 = pio::major_faults();
        const double c0 = pio::process_cpu_seconds();
        auto s0 = Clock::now();
        const double overhead = secs(loop_mark, s0); // everything since the previous decode returned
        loop_overhead_s += overhead;
        im.trace_begin(n_past, /*n_tokens*/ 1 + n_draft, /*phase*/ 1);
        int dec = llama_decode(ctx, step);
        auto s1 = Clock::now();
        loop_mark = s1; // the next token's overhead is measured from here
        const uint64_t f1 = pio::major_faults();
        const double c1 = pio::process_cpu_seconds();
        if (dec != 0) {
            if (im.cancel_requested.load(std::memory_order_relaxed)) {
                res.cancelled = true;
                break;
            }
            if (moe.overlap && im.source.fatal()) return abort_turn("expert stream I/O failed during overlap decode");
            return abort_turn("decode failed during generation");
        }
        if (!spec_on && mtp_on && !common_speculative_process(im.mtp.get(), step))
            return abort_turn("MTP draft context failed to process the budgeted decode");
        im.trace_flush(); // outside the s0..s1 bracket: the trace's own writes must not bill wall_ms
        ++im.spec_totals.decodes;

        // ── accept ──
        // `tok` was settled before the decode, so it is confirmed unconditionally. Each draft is
        // confirmed only while it matches what the target itself would have produced at that
        // position — no approximation enters here. The batch's arithmetic is still not bit-identical
        // to a single-token pass, so a near-tie can land differently; see docs/mtp.md.
        confirmed.clear();
        confirmed.push_back(tok);
        int n_acc = 0;
        bool eog_hit = false;
        for (int i = 0; i < n_draft; ++i) {
            const llama_token want = argmax(llama_get_logits_ith(ctx, i), im.n_vocab);
            if (want != im.draft_buf[i]) break;
            ++n_acc; // accepted: it is in the KV whether or not the caller gets to see it
            if (llama_vocab_is_eog(im.vocab, want)) {
                eog_hit = true; // end-of-generation is never emitted, here as on the plain path
                break;
            }
            confirmed.push_back(want);
        }
        if (spec_on) {
            im.spec_totals.accepted += n_acc;

            // Catch the draft context up on the ACCEPTED PREFIX ONLY.
            //
            // The catch-up re-runs the MTP block over the batch so the draft context holds rows
            // conditioned on the target's own hidden states, and accept() below seeds the next draft
            // from the row at index n_acc. Handing it the whole verify batch — the obvious ordering,
            // and the one upstream's own loop uses — computes the rejected tail as well, and that
            // tail is deleted a few statements later: those drafts were wrong and nothing ever reads
            // their rows. Acceptance depends only on the target's logits, which are already in hand
            // by this point, so the tail simply never has to be submitted.
            //
            // Identical state, strictly less work. accept() seeds from row min(n_acc, n_rows-1),
            // which is row n_acc under either batch, and the KV this leaves behind is exactly the
            // range the rollback used to carve out. What it saves is (n_draft - n_acc) positions
            // through the MTP block — and that block routes experts of its own, so on a streamed
            // device the saving is flash reads, not just arithmetic.
            //
            // The n-gram source has no state to catch up: its next draft is read off the token
            // history the emit block appends to, which is already correct by the time it is read.
            if (mtp_on) {
                const auto p0 = Clock::now();
                const uint64_t pb0 = moe.enabled ? im.source.stats().read_bytes : 0;
                batch_fill(im.mtp_batch, verify_toks.data(), 1 + n_acc, n_past, /*all_logits*/ false);
                if (!common_speculative_process(im.mtp.get(), im.mtp_batch))
                    return abort_turn("MTP draft context failed to process the verify batch");
                draft_s += secs(p0, Clock::now());
                if (moe.enabled) im.spec_totals.draft_read_bytes += im.source.stats().read_bytes - pb0;
            }
            im.spec_totals.draft_seconds += draft_s;

            // Drop the rejected tail from the target; the bounded-rollback snapshots asked for at
            // context creation are what make this a restore rather than a replay. The draft context
            // needs no rollback of its own — it was never given the tail.
            if (n_acc < n_draft) {
                const llama_pos keep = n_past + 1 + n_acc;
                if (!kv_remove_tail(ctx, keep))
                    return abort_turn("failed to roll back the rejected draft tokens from the KV cache");
            }
            if (mtp_on) common_speculative_accept(im.mtp.get(), /*seq*/ 0, (uint16_t) n_acc);
        }

        // ── emit ──
        // One metrics row per confirmed token, but ONE decode produced them all: its cost goes to
        // the first row and the rest carry zeros (see TokenMetrics::mtp_batch). Splitting the wall
        // evenly would read as several equally-cheap tokens, which is not what happened.
        const double wall = secs(s0, s1);
        gen_seconds += wall;
        const IExpertSource::Stats st = moe.enabled ? im.source.stats() : IExpertSource::Stats{};
        // Route-ahead's eval-thread meters accumulate per DECODE, and one decode can confirm a whole
        // group, so they are read once here and charged to the group's first row like every other
        // group cost. Reading them inside the loop would advance the cursors once per token and
        // credit the second and later rows with a delta of zero for the wrong reason.
        double ra_issue_ms = 0.0, ra_wd_ms = 0.0;
        {
            const long long ri = im.hook->route_ahead_issue_ns(), rw = im.hook->route_ahead_wd_ns();
            ra_issue_ms = (ri - prev_ra_issue_ns) / 1e6;
            ra_wd_ms = (rw - prev_ra_wd_ns) / 1e6;
            prev_ra_issue_ns = ri;
            prev_ra_wd_ns = rw;
        }
        for (size_t e = 0; e < confirmed.size() && n_charged < req.n_predict; ++e) {
            const llama_token out = confirmed[e];
            emitted_tokens.push_back(out);
            std::string delta = token_piece(im.vocab, out);
            gen += delta;
            const bool reasoning_token = budget_on && reasoning.accept(out);
            if (reasoning_token && n_reasoning < reasoning.allowance())
                ++n_reasoning;
            else
                ++n_charged;
            if (chat_on) im.kv_tokens.push_back(out);
            if (spec_on) mtp_ctx.push_back(out);
            ++n_gen;

            TokenMetrics m;
            m.step = n_gen;
            m.steps = output_reserve;
            m.mtp_batch = (int) confirmed.size();
            m.loop_overhead_ms = e == 0 ? overhead * 1000.0 : 0.0;
            // Charged to the group's first row like every other group cost. This is a SLICE of
            // loop_overhead_ms, not an addition to it: both measure time outside the decode.
            m.mtp_draft_ms = e == 0 ? draft_s * 1000.0 : 0.0;
            m.ra_issue_ms = e == 0 ? ra_issue_ms : 0.0;
            m.ra_wd_ms = e == 0 ? ra_wd_ms : 0.0;
            m.piece = delta;
            // Only when someone will read it: the parser cannot resume, so this re-parses everything
            // generated so far on every token, and off the chat path it is a full copy of the same.
            if (req.render_text) {
                ShownView sv = shown_view(gen, /*partial*/ true);
                m.text = std::move(sv.content);
                m.reasoning = std::move(sv.reasoning);
            }
            if (e == 0)
                tally.record(m, wall, f1 - f0, c1 - c0, im.turn, moe.enabled ? &st : nullptr);
            else
                tally.record(m, 0.0, 0, 0.0, im.turn, moe.enabled ? &st : nullptr);
            if (im.cfg.dense_stream.enabled && e == 0) {
                const DenseStreamCounters now = DenseStreamCounters::of(im.dense_stream);
                m.read_bytes = now.bytes - dense_prev.bytes;
                m.io_ms = (now.io_ns - dense_prev.io_ns) / 1e6;
                m.stall_ms = (now.wait_ns - dense_prev.wait_ns) / 1e6;
                m.compute_ms = std::max(0.0, m.wall_ms - m.stall_ms);
                im.dense_fixed.sample_residency(pio::vm_page());
                m.dense_resident_frac = im.dense_fixed.resident_frac();
                m.dense_window_resident_frac = im.dense_stream.sample_residency();
                dense_prev = now;
                decode_peak_rss = std::max(decode_peak_rss, m.rss_mib);
            }
            if (on_token) on_token(m);
            if (sink) sink->on_token(m);
        }

        if (eog_hit) {
            eog = true;
            break;
        }
        // The accepted group is now KV-resident, and the logits at the first unverified position
        // hold the target's own continuation — the next token, arrived at for free. Off the
        // speculative path the row index stays -1, exactly as before: one token, one row.
        n_past += 1 + n_acc;
        const int32_t row = wide ? n_acc : -1;
        logits = llama_get_logits_ith(ctx, row);
        tok = reasoning_sampler ? common_sampler_sample(reasoning_sampler, ctx, row)
              : smpl            ? llama_sampler_sample(smpl, ctx, row)
                                : argmax(logits, im.n_vocab);
    }

    // Speculation can leave the KV ahead of what the caller received: an accepted end-of-generation
    // token is decoded but never emitted, and a group can be cut short by n_predict. The KV and
    // kv_tokens must agree exactly or the next turn's prefix reuse decodes from a state that never
    // produced this answer, so trim back to what was actually emitted. The draft context mirrors the
    // target's positions (process() decodes the same batches into it), so truncate_kv trims it to the
    // same point — not cleared, or a continued chat turn would feed it only the new suffix and draft
    // from a state that never saw the conversation. If the trim is refused, nothing survives that we
    // can still describe: truncate_kv clears the KV with its records and the next turn rebuilds.
    llama_pos kv_end = n_past; // the physical end of this turn in the KV
    if (spec_on) {
        const llama_pos emitted_end = prompt_n_past + n_gen;
        kv_end = im.truncate_kv(emitted_end, im.kv_tokens.size(), /*in_last_decode*/ true) ? emitted_end : 0;
    }

    res.finish = res.context_exhausted ? FinishReason::ContextFull
                 : res.cancelled       ? FinishReason::Cancelled
                 : eog                 ? FinishReason::Stop
                                       : FinishReason::Length;

    // ── summary ──
    RunSummary & s = res.summary;
    s.arch = im.arch;
    s.n_generated = n_gen;
    s.n_reasoning = n_reasoning;
    s.gen_seconds = gen_seconds;
    s.s_per_token = n_gen ? gen_seconds / n_gen : 0.0;
    s.tokens_per_second = gen_seconds > 0 ? n_gen / gen_seconds : 0.0;
    // Close the accounting: the tail after the last decode belongs to no row, so add it here.
    loop_overhead_s += secs(loop_mark, Clock::now());
    s.loop_overhead_s_per_token = n_gen ? loop_overhead_s / n_gen : 0.0;
    s.n_prompt = media_prefill ? n_prompt : n_prompt - (int) n_common;
    s.n_past = chat_on && im.kv_has_media ? (int) kv_end
               : has_media                ? (int) (prompt_n_past + n_gen)
               : chat_on                  ? (int) im.kv_tokens.size()
                                          : n_prompt + n_gen;
    s.load_seconds = im.load_seconds;
    s.prefill_seconds = prefill_seconds;
    s.media_prepare_seconds = media_prepare_seconds;
    s.media_projector_seconds = media_projector_seconds;
    s.prefill_cpu_seconds = prefill_tally.cpu_seconds;
    s.prefill_read_mib = prefill_tally.read_mib;
    s.prefill_io_seconds = prefill_tally.io_seconds;
    s.prefill_stall_seconds = prefill_tally.stall_seconds;
    s.prefill_mgmt_seconds = prefill_tally.mgmt_seconds;
    s.prefill_majflt = prefill_faults;
    s.prefill_peak_rss_mib = prefill_peak_rss;
    s.decode_peak_rss_mib = decode_peak_rss;
    if (im.cfg.dense_stream.enabled) {
        const DenseStreamCounters now = DenseStreamCounters::of(im.dense_stream);
        s.dense_read_mib = (now.bytes - dense_decode0.bytes) / (1024.0 * 1024.0);
        s.dense_io_seconds = (now.io_ns - dense_decode0.io_ns) / 1e9;
        s.dense_wait_seconds = (now.wait_ns - dense_decode0.wait_ns) / 1e9;
    }
    s.majflt_per_token = n_gen ? (double) tally.majflt / n_gen : 0.0;
    s.cpu_s_per_token = n_gen ? tally.cpu_seconds / n_gen : 0.0;
    if (moe.enabled) {
        IExpertSource::Stats st = im.source.stats();
        s.moe_read_mib = tally.read_bytes / (1024.0 * 1024.0);
        s.moe_io_seconds = tally.io_seconds;
        s.moe_io_s_per_token = n_gen ? tally.io_seconds / n_gen : 0.0;
        s.moe_mgmt_s_per_token = n_gen ? tally.mgmt_seconds / n_gen : 0.0;
        s.moe_stall_s_per_token = n_gen ? tally.stall_seconds / n_gen : 0.0;
        s.moe_compute_s_per_token =
            s.s_per_token - (moe.overlap ? s.moe_stall_s_per_token : s.moe_io_s_per_token) - s.moe_mgmt_s_per_token;
        if (s.moe_compute_s_per_token < 0) s.moe_compute_s_per_token = 0;
        s.cache_hit_pct = st.cache_lookups > 0 ? 100.0 * st.cache_hits / st.cache_lookups : -1.0;
        s.cache_resident_mib = st.cache_resident_bytes / (1024.0 * 1024.0);
        s.cache_budget_mib = st.cache_budget_bytes / (1024.0 * 1024.0);
        s.cache_resizes = st.cache_resizes;
        s.cache_evictions = st.evictions;
        s.cache_rereads = st.rereads;
        const RowSourceStats rs = im.source.row_stats();
        s.row_table_mib = rs.table_bytes / (1024.0 * 1024.0);
        s.row_resident_mib = rs.resident_bytes / (1024.0 * 1024.0);
        s.row_read_mib = rs.bytes_read / (1024.0 * 1024.0);
        s.row_rows = (long long) rs.rows;
        s.row_slab_reads = (long long) rs.slab_reads;
        s.row_evictions = (long long) rs.evictions;
        s.row_io_errors = (long long) rs.io_errors;
        s.moe_drain_s_per_token = n_gen ? tally.drain_seconds / n_gen : 0.0;
        s.moe_adopt_s_per_token = n_gen ? tally.adopt_seconds / n_gen : 0.0;
        s.token_demand_mib = st.token_demand_bytes / (1024.0 * 1024.0);
        s.layer_demand_mib = st.layer_demand_bytes / (1024.0 * 1024.0);
        s.moe_spec_read_mib = ((long long) st.spec_read_bytes - prev_spec_bytes) / (1024.0 * 1024.0);
        s.moe_spec_experts = st.spec_experts - prev_spec_experts;
        s.moe_spec_useful = st.spec_useful - prev_spec_useful;
    }
    const RoutingCounters routing = RoutingCounters::of(*im.hook) - routing0;
    s.experts_routed = routing.routed;
    s.experts_dropped = routing.dropped;
    s.experts_reranked = routing.reranked;
    s.experts_substituted = routing.substituted;
    // Per-turn deltas, like every other generation figure here: a warm session's counters are
    // cumulative, and an acceptance rate averaged over earlier prompts would describe none of them.
    const SpecCounters & spec = im.spec_totals;
    s.mtp_drafted = spec.drafted - spec0.drafted;
    s.mtp_accepted = spec.accepted - spec0.accepted;
    s.mtp_decodes = spec.decodes - spec0.decodes;
    s.drafted_steps = spec.drafted_steps - spec0.drafted_steps;
    s.mtp_draft_s_per_token = n_gen > 0 ? (spec.draft_seconds - spec0.draft_seconds) / n_gen : 0.0;
    s.mtp_draft_read_mib = (double) (spec.draft_read_bytes - spec0.draft_read_bytes) / (1024.0 * 1024.0);
    if (moe.predict_log) {
        // Session totals, not a per-generation delta: these are an accuracy estimate, and every
        // turn's routings are equally valid samples of it. See RunSummary.
        s.predict_stale = im.hook->predict_stale();
        s.predict_stale2 = im.hook->predict_stale2();
        s.predict_prev = im.hook->predict_prev();
        s.predict_self = im.hook->predict_self();
        s.predict_stale_by_layer = im.hook->predict_stale_by_layer();
        s.predict_prev_by_layer = im.hook->predict_prev_by_layer();
        s.predict_self_by_layer = im.hook->predict_self_by_layer();
        s.predict_unscored = im.hook->predict_unscored();
    }
    if (moe.route_ahead > 0) {
        // Session totals, like the probe's: every overridden routing is an equally valid sample of
        // the perturbation the flag buys, whichever turn produced it.
        s.route_ahead_overridden = im.hook->route_ahead_overridden();
        s.route_ahead_passthrough = im.hook->route_ahead_passthrough();
        s.route_ahead_slots = im.hook->route_ahead_slots();
        s.route_ahead_hits = im.hook->route_ahead_hits();
        s.route_ahead_gemv_ns = im.hook->route_ahead_gemv_ns();
        s.route_ahead_gemv_jobs = im.hook->route_ahead_gemv_jobs();
        s.route_ahead_issue_ns = im.hook->route_ahead_issue_ns();
        s.route_ahead_wd_ns = im.hook->route_ahead_wd_ns();
    }
    if (sink) sink->on_summary(s);

    // One non-partial parse of the finished generation, shared by the returned text and the history
    // commit below. They asked the same question of the same string and each paid a full parse for
    // it; a reasoning model's answer is the whole turn's output, so that was not a small double.
    common_chat_msg final_msg;
    bool final_parsed = false;
    if (chat_on && !prefilled_answer) {
        try {
            final_msg = common_chat_parse(gen, /*is_partial*/ false, parse_params);
            final_parsed = true;
        } catch (const std::exception & e) {
            detail::warn_parse_failed_once(e.what());
        }
    }
    if (final_parsed) {
        res.generated_text = final_msg.content;
        res.reasoning_text = final_msg.reasoning_content;
        res.tool_calls = to_tool_calls(final_msg.tool_calls);
    } else {
        // Not chat, a prefilled turn (the stream IS the answer), or a parse that threw: the raw
        // generation stands on its own, exactly as shown_view would have reported it.
        res.generated_text = gen;
    }

    if (res.context_exhausted) {
        // A failed turn must not leave a compacted transcript or a partly rebuilt KV behind.
        llama_memory_clear(llama_get_memory(ctx), true);
        if (im.ctx_dft) llama_memory_clear(llama_get_memory(im.ctx_dft.get()), true);
        im.kv_tokens.clear();
        im.kv_n_past = 0;
        im.kv_last_generation_start = 0;
        im.kv_has_media = prior_kv_has_media;
        if (!prior_kv_has_media) im.retained_media.clear();
        rollback_history();
        restore_sampler();
    } else if (res.cancelled) {
        // Undo the whole turn (KV, fed tokens, and the pushed user message) so the conversation
        // is left exactly as it was before this prompt and stays continuable.
        rollback_turn();
    } else if (chat_on) {
        if (im.kv_has_media) im.kv_n_past = kv_end;
        // Commit the assistant turn to the running conversation. Parsing separates a thinking
        // model's reasoning from the answer; the next turn re-renders history from these messages.
        // Reuses the parse above. A prefilled turn has no turn header in the stream to parse — the
        // generation is the answer verbatim — and is committed without the prefill, so history holds
        // a normal assistant message and the next turn re-renders cleanly whatever this turn's think
        // setting was. A parse that threw falls back the same way.
        common_chat_msg assistant;
        if (final_parsed)
            assistant = std::move(final_msg);
        else
            assistant.content = gen;
        assistant.role = "assistant";
        im.chat_history.push_back(assistant);
    }
    if (res.ok && !res.cancelled) ++im.turn;
    return res;
}

} // namespace meitte
