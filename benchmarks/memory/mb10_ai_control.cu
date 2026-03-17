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

__global__ void ai_control_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n,
                                  int fmas_per_load) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float v = in[i];
    float acc = 0.f;

    for (int k = 0; k < fmas_per_load; ++k) {
        acc = fmaf(acc, 1.0001f, v);
    }

    out[i] = acc;
}

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        const int device = args.get_int("device", 0);
        const int n = args.get_int("n", 1 << 24);
        const int block = args.get_int("block", 256);
        const int grid = args.get_int("grid", 0);  // auto if 0
        const int fmas_per_load = args.get_int("fmas-per-load", 64);
        const int warmup = args.get_int("warmup", 5);
        const int reps = args.get_int("reps", 20);
        const std::string csv_path = args.get_string("csv", "");
        const bool quiet = args.get_flag("quiet");

        if (n <= 0) {
            throw std::runtime_error("n must be > 0");
        }
        if (block <= 0) {
            throw std::runtime_error("block must be > 0");
        }
        if (fmas_per_load <= 0) {
            throw std::runtime_error("fmas-per-load must be > 0");
        }

        CUDA_CHECK(cudaSetDevice(device));

        const int launch_grid =
            (grid > 0) ? grid : bw::default_grid_1d(n, block, device, 20);

        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

        float* d_in = nullptr;
        float* d_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));

        std::vector<float> h_in(n, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));

        auto launch = [&]() {
            ai_control_kernel<<<launch_grid, block>>>(d_in, d_out, n, fmas_per_load);
            CUDA_CHECK_LAST_KERNEL();
        };

        const std::vector<float> samples_ms =
            bw::benchmark_kernel_samples_ms(launch, warmup, reps);
        const float best_ms = bw::min_ms(samples_ms);
        const float mean_ms = bw::mean_ms(samples_ms);

        // Approximate analytical AI:
        // - 1 load + 1 store = 8 bytes/thread
        // - each FMA ~ 2 FLOPs
        const double approx_flops_per_thread = 2.0 * static_cast<double>(fmas_per_load);
        const double approx_bytes_per_thread = 2.0 * sizeof(float);
        const double approx_ai = approx_flops_per_thread / approx_bytes_per_thread;

        if (!quiet) {
            std::cout << "# MB10 AI-controlled kernel\n";
            std::cout << "device=" << device << "\n";
            std::cout << "n=" << n << "\n";
            std::cout << "block=" << block << "\n";
            std::cout << "grid=" << launch_grid << "\n";
            std::cout << "fmas_per_load=" << fmas_per_load << "\n";
            std::cout << "approx_ai_flops_per_byte=" << approx_ai << "\n";
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
                "n",
                "block",
                "grid",
                "fmas_per_load",
                "approx_ai_flops_per_byte",
                "warmup",
                "reps",
                "best_ms",
                "mean_ms"
            });
            writer.write_row({
                "mb10_ai_control",
                bw::to_csv_string(device),
                bw::to_csv_string(n),
                bw::to_csv_string(block),
                bw::to_csv_string(launch_grid),
                bw::to_csv_string(fmas_per_load),
                bw::to_csv_string(approx_ai),
                bw::to_csv_string(warmup),
                bw::to_csv_string(reps),
                bw::to_csv_string(best_ms),
                bw::to_csv_string(mean_ms)
            });
            writer.flush();

            if (!quiet) {
                std::cout << "[mb10_ai_control] Wrote CSV: " << csv_path << "\n";
            }
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb10_ai_control] ERROR: " << e.what() << "\n";
        return 1;
    }
}