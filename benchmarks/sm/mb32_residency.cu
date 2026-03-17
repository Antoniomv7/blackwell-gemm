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

template <int REG_FOOTPRINT>
__global__ void occ_sweep_kernel(float* out) {
    extern __shared__ float smem[];

    float r[REG_FOOTPRINT];
    #pragma unroll
    for (int i = 0; i < REG_FOOTPRINT; ++i) {
        r[i] = static_cast<float>(threadIdx.x + i);
    }

    float acc = 0.f;
    #pragma unroll
    for (int it = 0; it < 2048; ++it) {
        acc = fmaf(acc, 1.0001f, r[it % REG_FOOTPRINT]);
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = acc;
    }

    // Keep dynamic shared memory as a launch resource even if not explicitly used.
    if (false && smem != nullptr) {
        out[0] += smem[0];
    }
}

template <int REG_FOOTPRINT>
static float run_case(int grid,
                      int block,
                      std::size_t shared_bytes,
                      int warmup,
                      int reps,
                      float* d_out) {
    auto launch = [&]() {
        occ_sweep_kernel<REG_FOOTPRINT><<<grid, block, shared_bytes>>>(d_out);
        CUDA_CHECK_LAST_KERNEL();
    };

    return bw::benchmark_kernel_ms(launch, warmup, reps);
}

template <int REG_FOOTPRINT>
static int query_blocks_per_sm(int block, std::size_t shared_bytes) {
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        occ_sweep_kernel<REG_FOOTPRINT>,
        block,
        shared_bytes));
    return blocks_per_sm;
}

struct ResultRow {
    int device = 0;
    int reg_footprint = 0;
    int block = 0;
    int grid = 0;
    std::size_t shared_bytes = 0;
    int warmup = 0;
    int reps = 0;
    int blocks_per_sm = 0;
    float best_ms = 0.f;
};

template <int REG_FOOTPRINT>
static ResultRow execute_case(int device,
                              int block,
                              int grid,
                              std::size_t shared_bytes,
                              int warmup,
                              int reps) {
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(grid) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, static_cast<std::size_t>(grid) * sizeof(float)));

    const float best_ms =
        run_case<REG_FOOTPRINT>(grid, block, shared_bytes, warmup, reps, d_out);
    const int blocks_per_sm =
        query_blocks_per_sm<REG_FOOTPRINT>(block, shared_bytes);

    CUDA_CHECK(cudaFree(d_out));

    ResultRow row;
    row.device = device;
    row.reg_footprint = REG_FOOTPRINT;
    row.block = block;
    row.grid = grid;
    row.shared_bytes = shared_bytes;
    row.warmup = warmup;
    row.reps = reps;
    row.blocks_per_sm = blocks_per_sm;
    row.best_ms = best_ms;
    return row;
}

template <typename Fn>
static ResultRow dispatch_reg_footprint(int reg_footprint, Fn&& fn) {
    switch (reg_footprint) {
        case 8:   return fn.template operator()<8>();
        case 16:  return fn.template operator()<16>();
        case 24:  return fn.template operator()<24>();
        case 32:  return fn.template operator()<32>();
        case 48:  return fn.template operator()<48>();
        case 64:  return fn.template operator()<64>();
        case 96:  return fn.template operator()<96>();
        case 128: return fn.template operator()<128>();
        default:
            throw std::runtime_error(
                "Unsupported --reg-footprint. Supported values: "
                "8,16,24,32,48,64,96,128");
    }
}

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        const int device = args.get_int("device", 0);
        const int reg_footprint = args.get_int("reg-footprint", 32);
        const int block = args.get_int("block", 256);
        const int grid = args.get_int("grid", 120);
        const std::size_t shared_bytes =
            static_cast<std::size_t>(args.get_ll("shared-bytes", 0));
        const int warmup = args.get_int("warmup", 5);
        const int reps = args.get_int("reps", 20);
        const std::string csv_path = args.get_string("csv", "");
        const bool quiet = args.get_flag("quiet");

        if (block <= 0 || grid <= 0) {
            throw std::runtime_error("block and grid must be > 0");
        }

        CUDA_CHECK(cudaSetDevice(device));

        struct Runner {
            int device;
            int block;
            int grid;
            std::size_t shared_bytes;
            int warmup;
            int reps;

            template <int RF>
            ResultRow operator()() const {
                return execute_case<RF>(device, block, grid, shared_bytes, warmup, reps);
            }
        };

        const ResultRow row = dispatch_reg_footprint(
            reg_footprint,
            Runner{device, block, grid, shared_bytes, warmup, reps});

        if (!quiet) {
            std::cout << "# MB3.2-A residency sweep\n";
            std::cout << "device=" << row.device << "\n";
            std::cout << "reg_footprint=" << row.reg_footprint << "\n";
            std::cout << "block=" << row.block << "\n";
            std::cout << "grid=" << row.grid << "\n";
            std::cout << "shared_bytes=" << row.shared_bytes << "\n";
            std::cout << "warmup=" << row.warmup << "\n";
            std::cout << "reps=" << row.reps << "\n";
            std::cout << "blocks_per_sm=" << row.blocks_per_sm << "\n";
            std::cout << "best_ms=" << row.best_ms << "\n";
        }

        if (!csv_path.empty()) {
            bw::CsvWriter writer(csv_path, false);
            writer.write_header({
                "benchmark",
                "device",
                "reg_footprint",
                "block",
                "grid",
                "shared_bytes",
                "warmup",
                "reps",
                "blocks_per_sm",
                "best_ms"
            });
            writer.write_row({
                "mb32_residency",
                bw::to_csv_string(row.device),
                bw::to_csv_string(row.reg_footprint),
                bw::to_csv_string(row.block),
                bw::to_csv_string(row.grid),
                bw::to_csv_string(row.shared_bytes),
                bw::to_csv_string(row.warmup),
                bw::to_csv_string(row.reps),
                bw::to_csv_string(row.blocks_per_sm),
                bw::to_csv_string(row.best_ms)
            });
            writer.flush();

            if (!quiet) {
                std::cout << "[mb32_residency] Wrote CSV: " << csv_path << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb32_residency] ERROR: " << e.what() << "\n";
        return 1;
    }
}