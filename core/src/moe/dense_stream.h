// Bounded, stable-address storage for whole dense matrices used by CPU MUL_MAT.
#pragma once

#include "dense_weights.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

namespace meitte {

class DenseStream {
public:
    DenseStream() = default;
    ~DenseStream();
    DenseStream(const DenseStream &) = delete;
    DenseStream & operator=(const DenseStream &) = delete;

    bool init(std::vector<DenseTensorRef> refs,
              const std::vector<std::string> & paths,
              size_t align,
              uint64_t window_bytes,
              int lanes,
              bool overlap,
              bool two_wave,
              const std::atomic<bool> * cancel);
    // Called after the first graph node of a layer has finished. The previous layer is no longer
    // read by compute, so its pages can be reclaimed before this layer is filled.
    bool enter_layer(int layer);
    // The fork's CPU hook calls this immediately before MUL_MAT reads src0.
    bool weight_ready(const ggml_tensor * src0);
    bool fatal() const { return fatal_.load(std::memory_order_acquire); }
    uint64_t read_bytes() const;
    uint64_t io_ns() const;
    uint64_t wait_ns() const { return wait_ns_.load(std::memory_order_relaxed); }
    double sample_residency() const;
    bool direct() const;
    // Drain cancelled reads before the next turn and reuse the same tensor addresses.
    bool reset_after_cancel();
    void shutdown();
    void notify_cancel();

private:
    struct Matrix {
        DenseTensorRef ref;
        char * base = nullptr;
        size_t span = 0;
        std::atomic<int> state{0}; // 0 empty, 1 queued/reading, 2 ready, -1 failed, -2 cancelled
    };
    void worker(int lane);
    void schedule(int layer);
    bool wait(Matrix & m);
    void finish(Matrix & m, int state);
    void fail(Matrix & m);

    std::vector<std::unique_ptr<Matrix>> matrices_;
    std::vector<std::vector<Matrix *>> layers_;
    std::unordered_map<const ggml_tensor *, Matrix *> by_tensor_;
    std::vector<std::unique_ptr<FileReader>> readers_;
    std::vector<std::thread> workers_;
    std::deque<Matrix *> jobs_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    const std::atomic<bool> * cancel_ = nullptr;
    std::atomic<bool> fatal_{false};
    std::atomic<uint64_t> wait_ns_{0};
    size_t page_ = 4096;
    bool overlap_ = false;
    bool two_wave_ = false;
    bool stopping_ = false;
};

} // namespace meitte
