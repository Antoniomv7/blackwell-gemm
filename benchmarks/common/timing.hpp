#pragma once

#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace bw {

class CudaEventTimer {
public:
    CudaEventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    float time_ms(const std::function<void()>& fn) {
        CUDA_CHECK(cudaEventRecord(start_));
        fn();
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

template <typename LaunchFn>
float benchmark_kernel_ms(LaunchFn&& launch_fn, int warmup, int reps) {
    CudaEventTimer timer;

    for (int i = 0; i < warmup; ++i) {
        launch_fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(reps));

    for (int i = 0; i < reps; ++i) {
        const float ms = timer.time_ms([&]() {
            launch_fn();
        });
        samples.push_back(ms);
    }

    float best = samples.front();
    for (float x : samples) {
        if (x < best) best = x;
    }
    return best;
}

template <typename LaunchFn>
std::vector<float> benchmark_kernel_samples_ms(LaunchFn&& launch_fn,
                                               int warmup,
                                               int reps) {
    CudaEventTimer timer;

    for (int i = 0; i < warmup; ++i) {
        launch_fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(reps));

    for (int i = 0; i < reps; ++i) {
        const float ms = timer.time_ms([&]() {
            launch_fn();
        });
        samples.push_back(ms);
    }
    return samples;
}

inline float mean_ms(const std::vector<float>& xs) {
    if (xs.empty()) return 0.0f;
    float s = 0.0f;
    for (float x : xs) s += x;
    return s / static_cast<float>(xs.size());
}

inline float min_ms(const std::vector<float>& xs) {
    if (xs.empty()) return 0.0f;
    float best = xs.front();
    for (float x : xs) {
        if (x < best) best = x;
    }
    return best;
}

}  // namespace bw