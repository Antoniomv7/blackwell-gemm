#include "benchmarks/common/cli.hpp"
#include "benchmarks/common/csv_writer.hpp"
#include "benchmarks/common/cuda_utils.hpp"
#include "benchmarks/common/timing.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void fp32_dep_kernel(float* out) {
    float x = 1.f, y = 2.f, z = 0.5f;

    #pragma unroll
    for (int i = 0; i < 4096; ++i) {
        x = fmaf(x, y, z);
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = x;
    }
}

__global__ void fp32_ilp_kernel(float* out) {
    float a = 1.f, b = 2.f, c = 3.f, d = 4.f, z = 0.5f;

    #pragma unroll
    for (int i = 0; i < 4096; ++i) {
        a = fmaf(a, b, z);
        c = fmaf(c, d, z);
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = a + c;
    }
}

static float run_dep(int grid, int block, int warmup, int reps, float* d_out) {
    auto launch = [&]() {
        fp32_dep_kernel<<<grid, block>>>(d_out);
        CUDA_CHECK_LAST_KERNEL();
    };
    return bw::benchmark_kernel_ms(launch, warmup, reps);
}

static float run_ilp(int grid, int block, int warmup, int reps, float* d_out) {
    auto launch = [&]() {
        fp32_ilp_kernel<<<grid, block>>>(d_out);
        CUDA_CHECK_LAST_KERNEL();
    };
    return bw::benchmark_kernel_ms(launch, warmup, reps);
}

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        const int device = args.get_int("device", 0);
        const int block = args.get_int("block", 256);
        const int grid = args.get_int("grid", 120);
        const int warmup = args.get_int("warmup", 5);
        const int reps = args.get_int("reps", 20);
        const std::string csv_path = args.get_string("csv", "");
        const bool quiet = args.get_flag("quiet");

        if (block <= 0 || grid <= 0) {
            throw std::runtime_error("block and grid must be > 0");
        }

        CUDA_CHECK(cudaSetDevice(device));

        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(grid) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, static_cast<std::size_t>(grid) * sizeof(float)));

        const float dep_ms = run_dep(grid, block, warmup, reps, d_out);
        const float ilp_ms = run_ilp(grid, block, warmup, reps, d_out);

        CUDA_CHECK(cudaFree(d_out));

        if (!quiet) {
            std::cout << "# MB3.2-B FP32 dependency-vs-throughput probe\n";
            std::cout << "device=" << device << "\n";
            std::cout << "block=" << block << "\n";
            std::cout << "grid=" << grid << "\n";
            std::cout << "warmup=" << warmup << "\n";
            std::cout << "reps=" << reps << "\n";
            std::cout << "dep_best_ms=" << dep_ms << "\n";
            std::cout << "ilp_best_ms=" << ilp_ms << "\n";
        }

        if (!csv_path.empty()) {
            bw::CsvWriter writer(csv_path, false);
            writer.write_header({
                "benchmark",
                "device",
                "block",
                "grid",
                "warmup",
                "reps",
                "dep_best_ms",
                "ilp_best_ms"
            });
            writer.write_row({
                "mb32_fp32_probe",
                bw::to_csv_string(device),
                bw::to_csv_string(block),
                bw::to_csv_string(grid),
                bw::to_csv_string(warmup),
                bw::to_csv_string(reps),
                bw::to_csv_string(dep_ms),
                bw::to_csv_string(ilp_ms)
            });
            writer.flush();

            if (!quiet) {
                std::cout << "[mb32_fp32_probe] Wrote CSV: " << csv_path << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb32_fp32_probe] ERROR: " << e.what() << "\n";
        return 1;
    }
}