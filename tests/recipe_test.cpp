#include "bmoe/recipe.h"

#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

static void check(bool value) {
    if (!value) std::abort();
}

int main() {
    const char * split[] = {"qwen3moe", "qwen2moe",  "qwen35moe",   "glm-dsa",  "glm5next",   "gpt-oss",
                            "lfm2moe",  "deepseek4", "bailingmoe3", "qwen4exp", "cohere2moe", "hunyuan-moe",
                            "inkling",  "hy_v3",     "hy_v4",      "minimax-m3", "mimo2"};
    for (const char * arch : split) {
        const meitte::MoeRecipe * recipe = meitte::find_moe_recipe(arch);
        check(recipe && std::strcmp(recipe->exps_suffix[0], "ffn_gate_exps") == 0);
        check(std::strcmp(recipe->exps_suffix[1], "ffn_up_exps") == 0);
        check(std::strcmp(recipe->exps_suffix[2], "ffn_down_exps") == 0);
    }
    const meitte::MoeRecipe * fused = meitte::find_moe_recipe("gemma4");
    const meitte::MoeRecipe * up_down = meitte::find_moe_recipe("nemotron_h_moe");
    check(fused && std::strcmp(fused->exps_suffix[0], "ffn_gate_up_exps") == 0);
    check(up_down && std::strcmp(up_down->exps_suffix[0], "ffn_up_exps") == 0);
    check(meitte::n_moe_recipes() == 19);

    // A recipe with a fused alternative follows the file; every other recipe ignores it.
    const meitte::MoeRecipe * alt = meitte::find_moe_recipe("cohere2moe");
    check(alt && alt->fused_gate_up_alt && !meitte::find_moe_recipe("qwen3moe")->fused_gate_up_alt);
    auto names = [](std::initializer_list<const char *> list) {
        return [list](const std::string & name) {
            for (const char * n : list)
                if (name == n) return true;
            return false;
        };
    };
    meitte::MoeRecipe out{};
    check(meitte::resolve_moe_recipe(*alt, 2, names({"blk.1.ffn_gate_up_exps.weight"}), out));
    check(std::strcmp(out.exps_suffix[0], "ffn_gate_up_exps") == 0 && out.exps_suffix[2] == nullptr);
    check(meitte::resolve_moe_recipe(*alt, 2, names({"blk.0.ffn_gate_exps.weight"}), out));
    check(std::strcmp(out.exps_suffix[0], "ffn_gate_exps") == 0);
    check(!meitte::resolve_moe_recipe(*alt, 2, names({"blk.0.ffn_gate_exps.weight", "blk.1.ffn_gate_up_exps.weight"}),
                                      out));
    check(meitte::resolve_moe_recipe(*meitte::find_moe_recipe("qwen3moe"), 2, names({"blk.0.ffn_gate_up_exps.weight"}),
                                     out));
    check(std::strcmp(out.exps_suffix[0], "ffn_gate_exps") == 0);
}
