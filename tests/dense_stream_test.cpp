#include "dense_stream.h"
#include "ggml.h"

#include <array>
#include <atomic>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace meitte;

static DenseTensorRef ref(ggml_tensor & tensor, uint64_t off, uint64_t size, int layer) {
    DenseTensorRef d;
    d.tensor = &tensor;
    d.file_off = off;
    d.size = size;
    d.layer = layer;
    return d;
}

static void exercise(int lanes) {
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("bmoe-dense-stream-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin"))
            .string();
    std::vector<unsigned char> bytes(12288);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = (unsigned char) (i * 17 + 3);
    {
        std::ofstream out(path, std::ios::binary);
        out.write((const char *) bytes.data(), bytes.size());
    }
    ggml_tensor q4{};
    ggml_tensor f16{};
    ggml_tensor alias{};
    q4.type = GGML_TYPE_Q4_0;
    f16.type = GGML_TYPE_F16;
    alias.type = GGML_TYPE_Q4_0;
    std::atomic<bool> cancel{false};
    {
        DenseStream stream;
        auto a = ref(q4, 0, 4096, 0);
        a.aliases.push_back(&alias);
        auto b = ref(f16, 4096, 8192, 1);
        assert(stream.init({a, b}, {path}, 4096, 12288, lanes, false, false, &cancel));
        void * stable = q4.data;
        assert(stable && stable == alias.data && f16.data);
        assert(stream.enter_layer(0));
        assert(stream.weight_ready(&q4));
        assert(std::memcmp(q4.data, bytes.data(), 4096) == 0);
        assert(stream.weight_ready(&f16)); // the next layer was prefetched
        assert(std::memcmp(f16.data, bytes.data() + 4096, 8192) == 0);
        assert(stream.enter_layer(1)); // last consumer of layer 0 has finished
        assert(q4.data == stable);
        assert(!stream.weight_ready(&q4)); // the backing address is stable, its pages are gone
        assert(stream.enter_layer(0));
        assert(stream.weight_ready(&alias));
        assert(std::memcmp(alias.data, bytes.data(), 4096) == 0);
        cancel.store(true);
        stream.notify_cancel();
        assert(!stream.enter_layer(1));
        assert(!stream.fatal());
        assert(stream.reset_after_cancel());
        cancel.store(false);
        assert(q4.data == stable);
        assert(stream.enter_layer(0));
        assert(stream.weight_ready(&alias));
    }
    cancel.store(false);
    q4.data = nullptr;
    alias.data = nullptr;
    {
        DenseStream stream;
        auto a = ref(q4, 0, 4096, 0);
        assert(stream.init({a}, {path}, 4096, 4096, lanes, false, false, &cancel));
        std::ofstream truncate(path, std::ios::binary | std::ios::trunc);
        truncate.close();
        assert(!stream.enter_layer(0));
        assert(stream.fatal());
    }
    std::remove(path.c_str());
}

int main() {
    exercise(1);
    exercise(2);
}
