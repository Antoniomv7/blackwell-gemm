#pragma once

#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

namespace bw {

struct DeviceInfo {
    int device = 0;
    std::string name;
    int major = 0;
    int minor = 0;
    int sm_count = 0;
    int warp_size = 0;
    int clock_khz = 0;
    int mem_clock_khz = 0;
    int mem_bus_width_bits = 0;
    std::size_t total_global_mem = 0;
    std::size_t l2_bytes = 0;
    int regs_per_sm = 0;
    std::size_t shared_per_sm = 0;
    std::size_t shared_per_block = 0;
    int regs_per_block = 0;
    int max_threads_per_block = 0;
};

inline DeviceInfo query_device_info(int device = 0) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));

    DeviceInfo info;
    info.device = device;
    info.name = p.name;
    info.major = p.major;
    info.minor = p.minor;
    info.sm_count = p.multiProcessorCount;
    info.warp_size = p.warpSize;
    int sm_clock_khz = 0;
    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, device);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);

    info.clock_khz = sm_clock_khz;
    info.mem_clock_khz = mem_clock_khz;
    info.mem_bus_width_bits = p.memoryBusWidth;
    info.total_global_mem = static_cast<std::size_t>(p.totalGlobalMem);
    info.l2_bytes = static_cast<std::size_t>(p.l2CacheSize);
    info.regs_per_sm = p.regsPerMultiprocessor;
    info.shared_per_sm = static_cast<std::size_t>(p.sharedMemPerMultiprocessor);
    info.shared_per_block = static_cast<std::size_t>(p.sharedMemPerBlock);
    info.regs_per_block = p.regsPerBlock;
    info.max_threads_per_block = p.maxThreadsPerBlock;
    return info;
}

inline void print_device_info_human(const DeviceInfo& d, std::ostream& os = std::cout) {
    os << "device=" << d.device << "\n";
    os << "name=" << d.name << "\n";
    os << "compute_capability=" << d.major << "." << d.minor << "\n";
    os << "sm_count=" << d.sm_count << "\n";
    os << "warp_size=" << d.warp_size << "\n";
    os << "clock_khz=" << d.clock_khz << "\n";
    os << "memory_clock_khz=" << d.mem_clock_khz << "\n";
    os << "memory_bus_width_bits=" << d.mem_bus_width_bits << "\n";
    os << "total_global_mem_bytes=" << d.total_global_mem << "\n";
    os << "l2_bytes=" << d.l2_bytes << "\n";
    os << "regs_per_sm=" << d.regs_per_sm << "\n";
    os << "shared_per_sm_bytes=" << d.shared_per_sm << "\n";
    os << "shared_per_block_bytes=" << d.shared_per_block << "\n";
    os << "regs_per_block=" << d.regs_per_block << "\n";
    os << "max_threads_per_block=" << d.max_threads_per_block << "\n";
}

inline std::vector<std::string> device_info_csv_header() {
    return {
        "device",
        "name",
        "major",
        "minor",
        "sm_count",
        "warp_size",
        "clock_khz",
        "mem_clock_khz",
        "mem_bus_width_bits",
        "total_global_mem",
        "l2_bytes",
        "regs_per_sm",
        "shared_per_sm",
        "shared_per_block",
        "regs_per_block",
        "max_threads_per_block"
    };
}

inline std::vector<std::string> device_info_csv_row(const DeviceInfo& d) {
    return {
        std::to_string(d.device),
        d.name,
        std::to_string(d.major),
        std::to_string(d.minor),
        std::to_string(d.sm_count),
        std::to_string(d.warp_size),
        std::to_string(d.clock_khz),
        std::to_string(d.mem_clock_khz),
        std::to_string(d.mem_bus_width_bits),
        std::to_string(d.total_global_mem),
        std::to_string(d.l2_bytes),
        std::to_string(d.regs_per_sm),
        std::to_string(d.shared_per_sm),
        std::to_string(d.shared_per_block),
        std::to_string(d.regs_per_block),
        std::to_string(d.max_threads_per_block)
    };
}

}  // namespace bw