// MoE architecture recipes.
//
// A recipe is the ONLY architecture-specific knowledge in the engine: the gguf tensor
// name suffixes of a layer's expert weight tensors. Everything else — routing, the
// number of experts, the per-expert byte stride — is discovered at runtime from the
// tensors themselves. Supporting a new MoE family is therefore one table row plus a
// byte-identity gate run, never a code change in the streaming path.
//
// This header is pure policy: it has no llama.cpp dependency and compiles standalone.
#pragma once

#include <functional>
#include <string>

namespace meitte {

// The expert weight tensors of a MoE layer, named `blk.<il>.<suffix>.weight` in the gguf.
// A recipe lists the per-layer expert tensors as a suffix table. The common split layout names
// three ({gate, up, down} projections); layouts that fuse gate+up or omit a routed gate tensor
// name two ({gate_up, down} or {up, down}). Unused slots are nullptr. Each named tensor is 3-D
// and its dim-2 indexes the expert (ne[2] == n_expert). The engine streams whatever the recipe
// names and never needs to know which projection a given tensor carries — a fused gate_up is
// just an expert tensor with a larger per-expert stride, discovered at runtime like any other.
struct MoeRecipe {
    static constexpr int max_exps = 3;
    const char * arch;                  // gguf general.architecture, e.g. "qwen3moe"
    const char * exps_suffix[max_exps]; // e.g. {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}
    // The architecture ships either the split layout above or a fused gate_up layout under the same
    // architecture name, and llama.cpp loads both. resolve_moe_recipe() reads the file to pick one.
    bool fused_gate_up_alt = false;
};

// Look up a recipe by gguf architecture string. Returns nullptr if the architecture is
// not in the registry (the engine then refuses to stream rather than guess).
const MoeRecipe * find_moe_recipe(const char * arch);

// Resolve the expert layout that a file actually uses. `has_tensor` tells whether the gguf names a
// tensor; `n_layer` is the layer span to examine. `out` receives `recipe`, or its fused variant
// ({"ffn_gate_up_exps", "ffn_down_exps"}) when the recipe allows it and the file names only fused
// tensors. Returns false when the file mixes the two layouts, which the streamer cannot bind.
bool resolve_moe_recipe(const MoeRecipe & recipe,
                        int n_layer,
                        const std::function<bool(const std::string &)> & has_tensor,
                        MoeRecipe & out);

// Number of registered recipes and indexed access, for `--list-archs` and tests.
int n_moe_recipes();
const MoeRecipe * moe_recipe_at(int i);

} // namespace meitte
