#pragma once

// Per-phase measurement for Session::generate(). The expert source and the router hook keep
// cumulative counters across a warm session, so every per-token and per-phase figure
// is a delta between two samples. These types take the samples and close the deltas in one place.

#include "bmoe/expert_source.h"
#include "bmoe/metrics.h"
#include "../io/platform_io.h"
#include "../moe/router_hook.h"

#include <cstdint>

namespace meitte::detail {

// The routing-policy counters of the router hook, sampled at one instant.
struct RoutingCounters {
    long long routed = 0;
    long long dropped = 0;
    long long reranked = 0;
    long long substituted = 0;

    static RoutingCounters of(const RouterHook & hook) {
        return {hook.experts_routed(), hook.experts_dropped(), hook.experts_reranked(), hook.experts_substituted()};
    }
    RoutingCounters operator-(const RoutingCounters & o) const {
        return {routed - o.routed, dropped - o.dropped, reranked - o.reranked, substituted - o.substituted};
    }
};

// Session totals of the self-speculative loop. generate() reports each turn as a delta against the
// totals at the start of that turn.
struct SpecCounters {
    long long drafted = 0;
    long long accepted = 0;
    long long decodes = 0;
    // Steps that drafted anything at all. Against `decodes` this is the n-gram source's match rate —
    // the fraction of the run where it had evidence and widened the verify batch. For the head it is
    // all of them, since it drafts unconditionally unless p_min stops it.
    long long drafted_steps = 0;
    // Time spent drafting and catching the draft context up, i.e. everything speculation adds
    // OUTSIDE the target decode. Without it, this time would land in loop_overhead_ms together with
    // sampling and rendering, where it could only be inferred against an unspeculated run.
    double draft_seconds = 0.0;
    // Flash bytes those passes streamed. The MTP block is a MoE layer of its own, so drafting has an
    // I/O cost as well as a compute one — and it is invisible to the route trace, whose framing
    // brackets the target decode only. Without this the growth in bytes/token under speculation
    // cannot be split between the widened verify union on the trunk and the head's own routing.
    uint64_t draft_read_bytes = 0;
};

// The generation phase's running measurement state, and the one place a generated token's cost is
// written down. The cursors hold the source's absolute totals as of the previous token — its stats
// are cumulative across a warm session, so every per-token flash figure is a delta against these —
// and the totals are what the summary averages over n_gen. Grouped into one object so the per-token
// metrics block can be a method instead of ten more locals threaded through generate().
struct GenTally {
    // Fixed for the run; kept here so record() needs only the token's own measurements.
    bool overlap = false;

    long long prev_bytes = 0;
    double prev_io_s = 0.0;
    double prev_mgmt_s = 0.0;
    double prev_stall_s = 0.0;

    uint64_t read_bytes = 0;
    double io_seconds = 0.0;
    double mgmt_seconds = 0.0;
    double stall_seconds = 0.0;
    double drain_seconds = 0.0;
    double adopt_seconds = 0.0;
    double prev_drain_s = 0.0;
    double prev_adopt_s = 0.0;
    uint64_t majflt = 0;
    double cpu_seconds = 0.0;

    // Start the cursors at `st`, the sample that closed the prefill phase (PrefillTally::post).
    void seed(const IExpertSource::Stats & st) {
        prev_bytes = (long long) st.read_bytes;
        prev_io_s = st.read_seconds;
        prev_mgmt_s = st.mgmt_seconds;
        prev_stall_s = st.stall_seconds;
        prev_drain_s = st.drain_wait_seconds;
        prev_adopt_s = st.adopt_wait_seconds;
    }

    // Fill in everything a generated token is measured by — its wall/fault/CPU decomposition, the
    // memory picture, and (when streaming) the flash figures taken as deltas against the cursors
    // above — then advance those cursors and the run totals. The token's TEXT stays with the
    // caller: what a token says depends on chat state, what it cost does not.
    //
    // `wall` is the decode's wall time in seconds and `faults`/`cpu_s` the deltas measured around
    // that same decode; `st` is the expert source's stats, or null when streaming is off.
    void
    record(TokenMetrics & m, double wall, uint64_t faults, double cpu_s, int turn, const IExpertSource::Stats * st) {
        m.wall_ms = wall * 1000.0;
        // Fault/CPU decomposition is independent of streaming — dense-weight faults show up in the
        // mmap baseline too — so record it for every token before the moe/no-moe split below.
        m.majflt = faults;
        m.cpu_ms = cpu_s * 1000.0;
        m.majflt_mib = (double) m.majflt * (double) pio::fault_bytes() / (1024.0 * 1024.0);
        m.turn = turn;
        majflt += m.majflt;
        cpu_seconds += cpu_s;

        // Read the memory picture AFTER the decode, outside the caller's timing bracket: two /proc
        // reads are cheap but they are not this token's work, and billing them to wall_ms would
        // corrupt the very number the reader is here to trust.
        pio::ProcessMemory pm;
        if (pio::process_memory(&pm)) {
            m.rss_mib = pm.rss_bytes / (1024.0 * 1024.0);
            m.rss_anon_mib = pm.rss_anon_bytes / (1024.0 * 1024.0);
            m.rss_file_mib = pm.rss_file_bytes / (1024.0 * 1024.0);
            m.swap_mib = pm.swap_bytes / (1024.0 * 1024.0);
        }
        pio::DeviceMemory dm;
        if (pio::device_memory(&dm)) {
            m.mem_available_mib = dm.available_bytes / (1024.0 * 1024.0);
            m.mem_free_mib = dm.free_bytes / (1024.0 * 1024.0);
            m.swap_free_mib = dm.swap_free_bytes / (1024.0 * 1024.0);
        }
        if (!st) {
            m.compute_ms = m.wall_ms;
            m.cache_hit_pct = -1.0;
            return;
        }

        m.dense_resident_frac = st->dense_resident_frac;
        m.cache_budget_mib = st->cache_budget_bytes / (1024.0 * 1024.0);
        m.read_bytes = (uint64_t) ((long long) st->read_bytes - prev_bytes);
        m.io_ms = (st->read_seconds - prev_io_s) * 1000.0;
        m.mgmt_ms = (st->mgmt_seconds - prev_mgmt_s) * 1000.0;
        if (overlap) {
            // stall is already wall-additive (the union of stalled intervals), so no thread-count
            // normalization: dividing summed thread time by n_threads was a mean that understated
            // the stall whenever a minority of threads did the waiting, and the difference quietly
            // became "compute".
            m.stall_ms = (st->stall_seconds - prev_stall_s) * 1000.0;
            m.compute_ms = m.wall_ms - m.stall_ms - m.mgmt_ms;
        } else {
            m.compute_ms = m.wall_ms - m.io_ms - m.mgmt_ms;
        }
        if (m.compute_ms < 0) m.compute_ms = 0;
        m.cache_hit_pct = st->cache_lookups > 0 ? 100.0 * st->cache_hits / st->cache_lookups : -1.0;
        m.drain_ms = (st->drain_wait_seconds - prev_drain_s) * 1000.0;
        m.adopt_ms = (st->adopt_wait_seconds - prev_adopt_s) * 1000.0;

        prev_bytes = (long long) st->read_bytes;
        prev_io_s = st->read_seconds;
        prev_mgmt_s = st->mgmt_seconds;
        prev_stall_s = st->stall_seconds;
        prev_drain_s = st->drain_wait_seconds;
        prev_adopt_s = st->adopt_wait_seconds;
        read_bytes += m.read_bytes;
        io_seconds += m.io_ms / 1000.0;
        mgmt_seconds += m.mgmt_ms / 1000.0;
        stall_seconds += m.stall_ms / 1000.0;
        drain_seconds += m.drain_ms / 1000.0;
        adopt_seconds += m.adopt_ms / 1000.0;
    }
};

// The prompt phase's measurement, the prefill counterpart of GenTally above: the source's
// counters are cumulative across a warm session, so begin() pins them (with the process CPU
// clock) just above the prompt chunk loop and end() closes the deltas just after it — the same
// wall-additive terms the decode fields report, read with the same rules. end()'s sample is the
// phase boundary itself: the decode baseline below seeds its cursors from it, so prefill's end
// and decode's start are one reading of the counters, not two that could drift apart.
struct PrefillTally {
    IExpertSource::Stats pre;
    double cpu0 = 0.0;

    // Deltas across this turn's prefill chunks — valid after end().
    double cpu_seconds = 0.0;
    double read_mib = 0.0;
    double io_seconds = 0.0;
    double stall_seconds = 0.0;
    double mgmt_seconds = 0.0;

    // The stats sample end() closed on, for the decode baseline to start from.
    IExpertSource::Stats post;

    void begin(bool moe_on, const IExpertSource & src) {
        pre = moe_on ? src.stats() : IExpertSource::Stats{};
        cpu0 = pio::process_cpu_seconds();
    }
    void end(bool moe_on, const IExpertSource & src) {
        post = moe_on ? src.stats() : IExpertSource::Stats{};
        cpu_seconds = pio::process_cpu_seconds() - cpu0;
        read_mib = (double) ((long long) post.read_bytes - (long long) pre.read_bytes) / (1024.0 * 1024.0);
        io_seconds = post.read_seconds - pre.read_seconds;
        stall_seconds = post.stall_seconds - pre.stall_seconds;
        mgmt_seconds = post.mgmt_seconds - pre.mgmt_seconds;
    }
};

} // namespace meitte::detail
