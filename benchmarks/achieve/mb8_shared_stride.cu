#include "benchmarks/common/cli.hpp"
#include "benchmarks/common/csv_writer.hpp"
#include "benchmarks/common/cuda_utils.hpp"
#include "benchmarks/common/timing.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kSharedElems = 8192;   // 32 KiB of shared memory for FP32
}

__global__ void shared_stride_kernel(float* out, int stride) {
    __shared__ float smem[kSharedElems];

    const int tid = threadIdx.x;
    float acc = 0.f;

    // Initialize shared memory deterministically.
    for (int i = tid; i < kSharedElems; i += blockDim.x) {
        smem[i] = 1.0f + static_cast<float>(i) * 0.001f;
    }
    __syncthreads();

    // Repeated shared-memory accesses with a stride-controlled pattern.
    #pragma unroll
    for (int it = 0; it < 4096; ++it) {
        const int j = (tid * stride + it) & (kSharedElems - 1);
        acc += smem[j];
    }

    if (tid == 0) {
        out[blockIdx.x] = acc;
    }
}

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        const int device = args.get_int("device", 0);
        const int stride = args.get_int("stride", 1);
        const int block = args.get_int("block", 256);
        const int grid = args.get_int("grid", 120);
        const int warmup = args.get_int("warmup", 5);
        const int reps = args.get_int("reps", 20);
        const std::string csv_path = args.get_string("csv", "");
        const bool quiet = args.get_flag("quiet");

        if (stride <= 0) {
            throw std::runtime_error("stride must be > 0");
        }
        if (block <= 0 || block > 1024) {
            throw std::runtime_error("block must be in (0, 1024]");
        }
        if (grid <= 0) {
            throw std::runtime_error("grid must be > 0");
        }

        CUDA_CHECK(cudaSetDevice(device));

        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(grid) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, static_cast<std::size_t>(grid) * sizeof(float)));

        auto launch = [&]() {
            shared_stride_kernel<<<grid, block>>>(d_out, stride);
            CUDA_CHECK_LAST_KERNEL();
        };

        const std::vector<float> samples_ms =
            bw::benchmark_kernel_samples_ms(launch, warmup, reps);
        const float best_ms = bw::min_ms(samples_ms);
        const float mean_ms = bw::mean_ms(samples_ms);

        if (!quiet) {
            std::cout << "# MB8 shared-memory stride probe\n";
            std::cout << "device=" << device << "\n";
            std::cout << "stride=" << stride << "\n";
            std::cout << "block=" << block << "\n";
            std::cout << "grid=" << grid << "\n";
            std::cout << "shared_bytes=" << (kSharedElems * sizeof(float)) << "\n";
            std::cout << "warmup=" << warmup << "\n";
            std::cout << "reps=" << reps << "\n";
            std::cout << "best_ms=" << best_ms << "\n";
            std::cout << "mean_ms=" << mean_ms << "\n";
        }

        if (!csv_path.empty()) {
            bw::CsvWriter writer(csv_path, false);
            writer.write_header({
                "benchmark",
                "device",
                "stride",
                "block",
                "grid",
                "shared_bytes",
                "warmup",
                "reps",
                "best_ms",
                "mean_ms"
            });
            writer.write_row({
                "mb8_shared_stride",
                bw::to_csv_string(device),
                bw::to_csv_string(stride),
                bw::to_csv_string(block),
                bw::to_csv_string(grid),
                bw::to_csv_string(kSharedElems * sizeof(float)),
                bw::to_csv_string(warmup),
                bw::to_csv_string(reps),
                bw::to_csv_string(best_ms),
                bw::to_csv_string(mean_ms)
            });
            writer.flush();

            if (!quiet) {
                std::cout << "[mb8_shared_stride] Wrote CSV: " << csv_path << "\n";
            }
        }

        CUDA_CHECK(cudaFree(d_out));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb8_shared_stride] ERROR: " << e.what() << "\n";
        return 1;
    }
}