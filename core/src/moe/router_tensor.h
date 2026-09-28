// Readers for the routing nodes the router hook edits in place: the per-slot routing weights and the
// selected-expert ids. Shared by the capture/dispatch path and the lossy routing policies.
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace meitte::routing {

// Where token j's slot k lives inside a weight node. Single source of truth for the layout: the
// gather below and the drop policy's writer must agree, or the policy would zero another token's
// slot. Returns a mutable pointer; the gather takes a const tensor and only reads through it.
//
// The node is [1, nu, nt] as built (token j at nb[2], slot k at nb[1]) — except the norm variant,
// whose callback fires on the pre-reshape 2-D [nu, nt] (token j at nb[1], slot k at nb[0]). As with
// the topk ids these are views, so only the strides say where a token's row really starts.
//
// The routing's own width and batch size are what tell the two shapes apart. `ne[0] == 1` alone
// does NOT: a top-1 routing makes the 2-D node [1, nt], which passes that test and would be read
// with the token stride of a 3-D node — sending every token after the first to the wrong row, and
// the drop policy's writes to the wrong slot. Match the full extents instead. Both fit only when
// nu and nt are 1, where the two layouts name the same single element.
inline float * weight_at(const ggml_tensor * t, int nu, int nt, int j, int k) {
    const bool fits_3d = t->ne[0] == 1 && t->ne[1] == nu && t->ne[2] == nt;
    const bool fits_2d = t->ne[0] == nu && t->ne[1] == nt && t->ne[2] == 1;
    // Neither fits only if the graph produced a shape build_moe_ffn does not emit; keep the old
    // test as the fallback rather than inventing an offset for a node we do not recognise.
    const bool three_d = fits_3d || (!fits_2d && t->ne[0] == 1);
    const size_t tok_nb = three_d ? t->nb[2] : t->nb[1];
    const size_t slot_nb = three_d ? t->nb[1] : t->nb[0];
    return (float *) ((char *) t->data + (size_t) j * tok_nb + (size_t) k * slot_nb);
}

inline void gather_weights(const ggml_tensor * t, int nu, int nt, std::vector<float> & out) {
    out.assign((size_t) nu * nt, 0.0f);
    for (int j = 0; j < nt; ++j)
        for (int k = 0; k < nu; ++k)
            out[(size_t) j * nu + k] = *weight_at(t, nu, nt, j, k);
}

// Same, for the selected-expert ids. The node is a VIEW of the full argsort, so the row stride is
// nb[1] (n_expert * 4), not n_expert_used * 4 — see the gather at the topk node.
inline int32_t * id_at(ggml_tensor * t, int j, int k) {
    return (int32_t *) ((char *) t->data + (size_t) j * t->nb[1] + (size_t) k * t->nb[0]);
}

} // namespace meitte::routing
