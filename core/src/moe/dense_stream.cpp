#include "dense_stream.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace meitte {

namespace {
size_t round_page(uint64_t n, size_t page) {
    return (size_t) ((n + page - 1) / page * page);
}

void drop_mmap_copy(const void * data, uint64_t size, size_t page) {
    if (!data || !size) return;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
    const uintptr_t end = begin + (uintptr_t) size;
    if (end < begin) return;
    const uintptr_t first = (begin + page - 1) & ~(uintptr_t) (page - 1);
    const uintptr_t last = end & ~(uintptr_t) (page - 1);
    if (last > first) pio::vm_drop_file_pages((void *) first, (size_t) (last - first));
}
} // namespace

DenseStream::~DenseStream() {
    shutdown();
}

bool DenseStream::init(std::vector<DenseTensorRef> refs,
                       const std::vector<std::string> & paths,
                       size_t align,
                       uint64_t window_bytes,
                       int lanes,
                       bool overlap,
                       bool two_wave,
                       const std::atomic<bool> * cancel) {
    if (refs.empty() || paths.empty() || lanes < 1 || lanes > 2) return false;
    cancel_ = cancel;
    overlap_ = overlap;
    two_wave_ = two_wave;
    page_ = pio::vm_page();
    int max_layer = -1;
    for (const DenseTensorRef & d : refs)
        max_layer = std::max(max_layer, d.layer);
    if (max_layer < 0) return false;
    layers_.resize((size_t) max_layer + 1);
    std::vector<std::pair<const void *, uint64_t>> mmap_copies;

    for (const std::string & path : paths) {
        readers_.push_back(std::make_unique<FileReader>());
        if (!readers_.back()->open(path, lanes, true, align, (8ull << 20) + 2 * align)) return false;
    }
    for (DenseTensorRef & d : refs) {
        if (!d.tensor || d.size == 0 || d.layer < 0 || d.file_idx < 0 || (size_t) d.file_idx >= readers_.size())
            return false;
        if (d.file_off > readers_[(size_t) d.file_idx]->file_size() ||
            d.size > readers_[(size_t) d.file_idx]->file_size() - d.file_off)
            return false;
        auto m = std::make_unique<Matrix>();
        m->ref = std::move(d);
        m->span = round_page(m->ref.size, page_);
        m->base = static_cast<char *>(pio::vm_reserve(m->span));
        if (!m->base) return false;
        // Every alias keeps this address for the model lifetime. Only physical pages move.
        if (m->ref.tensor->data) mmap_copies.emplace_back(m->ref.tensor->data, m->ref.size);
        m->ref.tensor->data = m->base;
        by_tensor_[m->ref.tensor] = m.get();
        for (ggml_tensor * alias : m->ref.aliases) {
            if (alias->data) mmap_copies.emplace_back(alias->data, m->ref.size);
            alias->data = m->base;
            by_tensor_[alias] = m.get();
        }
        layers_[(size_t) m->ref.layer].push_back(m.get());
        matrices_.push_back(std::move(m));
    }
    uint64_t largest = 0;
    for (size_t l = 0; l < layers_.size(); ++l) {
        uint64_t bytes = 0;
        for (Matrix * m : layers_[l])
            bytes += m->span;
        if (l + 1 < layers_.size())
            for (Matrix * m : layers_[l + 1])
                bytes += m->span;
        largest = std::max(largest, bytes);
    }
    if (largest > window_bytes) {
        std::fprintf(stderr, "bmoe: dense window needs at least %llu MiB for two adjacent layers\n",
                     (unsigned long long) ((largest + (1 << 20) - 1) >> 20));
        return false;
    }
    // Rebound tensors no longer use their model mapping. Drop clean file pages so they do not
    // compete with the fixed anonymous set and the active matrix window.
    for (const auto & mapping : mmap_copies)
        drop_mmap_copy(mapping.first, mapping.second, page_);
    for (int i = 0; i < lanes; ++i)
        workers_.emplace_back(&DenseStream::worker, this, i);
    return true;
}

void DenseStream::fail(Matrix & m) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fatal_.store(true, std::memory_order_release);
        m.state.store(-1, std::memory_order_release);
    }
    cv_.notify_all();
}

void DenseStream::finish(Matrix & m, int state) {
    // Change the predicate under the wait mutex, or a completion can lose its wakeup.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        m.state.store(state, std::memory_order_release);
    }
    cv_.notify_all();
}

void DenseStream::notify_cancel() {
    // Hold the wait mutex so cancellation cannot signal before a waiter sleeps.
    std::lock_guard<std::mutex> lock(mutex_);
    cv_.notify_all();
}

void DenseStream::worker(int lane) {
    for (;;) {
        Matrix * m = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
            if (stopping_) return;
            m = jobs_.front();
            jobs_.pop_front();
        }
        if (fatal()) {
            fail(*m);
            continue;
        }
        if (cancel_ && cancel_->load(std::memory_order_acquire)) {
            finish(*m, -2);
            continue;
        }
        FileReader & reader = *readers_[(size_t) m->ref.file_idx];
        bool ok = true;
        for (uint64_t done = 0; done < m->ref.size;) {
            if (cancel_ && cancel_->load(std::memory_order_acquire)) {
                ok = false;
                break;
            }
            const uint64_t n = std::min<uint64_t>(8ull << 20, m->ref.size - done);
            if (reader.read(lane, m->base + done, m->ref.file_off + done, n) < 0) {
                ok = false;
                break;
            }
            done += n;
        }
        if (!ok) {
            if (!(cancel_ && cancel_->load(std::memory_order_acquire)))
                std::fprintf(stderr, "bmoe: dense matrix read failed: %s (shard %d, offset %llu)\n",
                             m->ref.tensor->name, m->ref.file_idx, (unsigned long long) m->ref.file_off);
            if (cancel_ && cancel_->load(std::memory_order_acquire)) {
                finish(*m, -2);
            } else
                fail(*m);
        } else {
            finish(*m, 2);
        }
    }
}

void DenseStream::schedule(int layer) {
    if (layer < 0 || (size_t) layer >= layers_.size()) return;
    auto & group = layers_[(size_t) layer];
    auto publish = [&](Matrix * m) {
        if (m->state.load(std::memory_order_acquire) != 0) return;
        if (!pio::vm_commit(m->base, m->span)) {
            std::fprintf(stderr, "bmoe: dense matrix commit failed: %s\n", m->ref.tensor->name);
            fail(*m);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            m->state.store(1, std::memory_order_release);
            jobs_.push_back(m);
        }
        cv_.notify_all();
    };
    if (two_wave_ && !group.empty()) {
        publish(group[0]); // the first projection can read while later pages are committed
        for (size_t i = 1; i < group.size(); ++i)
            publish(group[i]);
    } else {
        std::vector<Matrix *> staged;
        for (Matrix * m : group) {
            if (m->state.load(std::memory_order_acquire) != 0) continue;
            if (!pio::vm_commit(m->base, m->span)) {
                std::fprintf(stderr, "bmoe: dense matrix commit failed: %s\n", m->ref.tensor->name);
                fail(*m);
                return;
            }
            staged.push_back(m);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (Matrix * m : staged) {
                m->state.store(1, std::memory_order_release);
                jobs_.push_back(m);
            }
        }
        cv_.notify_all();
    }
}

bool DenseStream::wait(Matrix & m) {
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] {
        return m.state.load(std::memory_order_acquire) != 1 || fatal() ||
               (cancel_ && cancel_->load(std::memory_order_acquire));
    });
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    wait_ns_.fetch_add((uint64_t) ns.count(), std::memory_order_relaxed);
    return m.state.load(std::memory_order_acquire) == 2 && !fatal() &&
           !(cancel_ && cancel_->load(std::memory_order_acquire));
}

bool DenseStream::enter_layer(int layer) {
    if (fatal() || (cancel_ && cancel_->load(std::memory_order_acquire))) return false;
    // The scheduler calls this only after the boundary node has completed. Every earlier layer's
    // last consumer has therefore finished. Keep the current and next layer for lookahead.
    for (size_t l = 0; l < layers_.size(); ++l) {
        if ((int) l == layer || (int) l == layer + 1) continue;
        for (Matrix * m : layers_[l]) {
            if (m->state.load(std::memory_order_acquire) == 2) {
                pio::vm_evict(m->base, m->span);
                m->state.store(0, std::memory_order_release);
            }
        }
    }
    schedule(layer);
    if (overlap_) schedule(layer + 1);
    if (!overlap_ && layer >= 0 && (size_t) layer < layers_.size()) {
        for (Matrix * m : layers_[(size_t) layer])
            if (!wait(*m)) return false;
        schedule(layer + 1); // one-layer lookahead starts while this layer computes
    }
    return !fatal();
}

bool DenseStream::weight_ready(const ggml_tensor * src0) {
    const ggml_tensor * base = src0;
    while (base && by_tensor_.find(base) == by_tensor_.end() && base->src[0])
        base = base->src[0];
    auto it = by_tensor_.find(base);
    return it == by_tensor_.end() || wait(*it->second);
}

uint64_t DenseStream::read_bytes() const {
    uint64_t sum = 0;
    for (const auto & r : readers_)
        sum += (uint64_t) r->read_bytes();
    return sum;
}

uint64_t DenseStream::io_ns() const {
    uint64_t sum = 0;
    for (const auto & r : readers_)
        sum += (uint64_t) r->syscall_ns();
    return sum;
}

double DenseStream::sample_residency() const {
    size_t sampled = 0, resident = 0;
    for (const auto & m : matrices_)
        if (m->state.load(std::memory_order_acquire) == 2)
            pio::vm_resident_sample(m->base, m->span, &sampled, &resident);
    return sampled ? (double) resident / (double) sampled : -1.0;
}

bool DenseStream::direct() const {
    for (const auto & r : readers_)
        if (!r->direct()) return false;
    return !readers_.empty();
}

bool DenseStream::reset_after_cancel() {
    const int lanes = (int) workers_.size();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        jobs_.clear();
    }
    cv_.notify_all();
    for (auto & t : workers_)
        if (t.joinable()) t.join();
    workers_.clear();
    if (fatal()) return false;
    for (auto & m : matrices_) {
        if (m->state.load(std::memory_order_acquire) != 0) pio::vm_evict(m->base, m->span);
        m->state.store(0, std::memory_order_release);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
    }
    for (int i = 0; i < lanes; ++i)
        workers_.emplace_back(&DenseStream::worker, this, i);
    return true;
}

void DenseStream::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto & t : workers_)
        if (t.joinable()) t.join();
    workers_.clear();
    for (auto & m : matrices_)
        if (m->base) pio::vm_release(m->base, m->span);
    matrices_.clear();
    layers_.clear();
    by_tensor_.clear();
    readers_.clear();
}

} // namespace meitte
