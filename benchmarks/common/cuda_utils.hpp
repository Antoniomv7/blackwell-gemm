#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bw {

inline void cuda_check_impl(cudaError_t err,
                            const char* expr,
                            const char* file,
                            int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[CUDA ERROR] " << file << ":" << line
            << " | expr: " << expr
            << " | code: " << static_cast<int>(err)
            << " | msg: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void cuda_check_last_kernel(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[KERNEL LAUNCH ERROR] " << file << ":" << line
            << " | code: " << static_cast<int>(err)
            << " | msg: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

inline int device_sm_count(int device = 0) {
    cudaDeviceProp prop{};
    cuda_check_impl(cudaGetDeviceProperties(&prop, device),
                    "cudaGetDeviceProperties",
                    __FILE__,
                    __LINE__);
    return prop.multiProcessorCount;
}

inline int default_grid_1d(int n,
                           int block_size,
                           int device = 0,
                           int grid_multiplier = 20) {
    const int sms = device_sm_count(device);
    const int max_grid = sms * grid_multiplier;
    const int needed = ceil_div(n, block_size);
    return (needed < max_grid) ? needed : max_grid;
}

inline std::string bytes_to_string(std::size_t bytes) {
    constexpr double KB = 1024.0;
    constexpr double MB = 1024.0 * 1024.0;
    constexpr double GB = 1024.0 * 1024.0 * 1024.0;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);

    if (bytes >= static_cast<std::size_t>(GB)) {
        oss << (bytes / GB) << " GiB";
    } else if (bytes >= static_cast<std::size_t>(MB)) {
        oss << (bytes / MB) << " MiB";
    } else if (bytes >= static_cast<std::size_t>(KB)) {
        oss << (bytes / KB) << " KiB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

}  // namespace bw

#define CUDA_CHECK(expr) ::bw::cuda_check_impl((expr), #expr, __FILE__, __LINE__)
#define CUDA_CHECK_LAST_KERNEL() ::bw::cuda_check_last_kernel(__FILE__, __LINE__)