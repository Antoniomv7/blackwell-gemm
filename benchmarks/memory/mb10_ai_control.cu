#include "benchmarks/common/cli.hpp"
#include "benchmarks/common/csv_writer.hpp"
#include "benchmarks/common/cuda_utils.hpp"
#include "benchmarks/common/timing.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

__global__ void init_array_kernel(float* x, float value, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        x[i] = value;
    }
}

__global__ void ai_control_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n,
                                  int fmas_per_load) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        const float v = in[i];
        float acc = v;

        #pragma unroll 1
        for (int k = 0; k < fmas_per_load; ++k) {
            acc = fmaf(acc, 1.0001f, v);
        }

        out[i] = acc;
    }
}

float median_ms(std::vector<float> xs) {
    if (xs.empty()) return 0.0f;
    std::sort(xs.begin(), xs.end());

    const std::size_t n = xs.size();
    if (n % 2 == 1) {
        return xs[n / 2];
    }
    return 0.5f * (xs[n / 2 - 1] + xs[n / 2]);
}

}  // namespace

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
        if (warmup < 0) {
            throw std::runtime_error("warmup must be >= 0");
        }
        if (reps <= 0) {
            throw std::runtime_error("reps must be > 0");
        }
        if (fmas_per_load <= 0) {
            throw std::runtime_error("fmas-per-load must be > 0");
        }

        CUDA_CHECK(cudaSetDevice(device));

        const int launch_grid =
            (grid > 0) ? grid : bw::default_grid_1d(n, block, device, 20);

        const std::size_t bytes_per_array =
            static_cast<std::size_t>(n) * sizeof(float);

        float* d_in = nullptr;
        float* d_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_in, bytes_per_array));
        CUDA_CHECK(cudaMalloc(&d_out, bytes_per_array));

        init_array_kernel<<<launch_grid, block>>>(d_in, 1.0f, n);
        CUDA_CHECK_LAST_KERNEL();
        init_array_kernel<<<launch_grid, block>>>(d_out, 0.0f, n);
        CUDA_CHECK_LAST_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        auto launch = [&]() {
            ai_control_kernel<<<launch_grid, block>>>(d_in, d_out, n, fmas_per_load);
            CUDA_CHECK_LAST_KERNEL();
        };

        const std::vector<float> samples_ms =
            bw::benchmark_kernel_samples_ms(launch, warmup, reps);

        const float best_ms = bw::min_ms(samples_ms);
        const float mean_ms = bw::mean_ms(samples_ms);
        const float median = median_ms(samples_ms);

        // Analytical model for this kernel:
        //   per processed element:
        //     - 1 load from input
        //     - 1 store to output
        //     - fmas_per_load FMAs => 2 * fmas_per_load FLOPs
        //
        // IMPORTANT:
        //   - bytes_effective is a MODEL quantity, not a hardware counter.
        //   - ai_nominal is a MODEL arithmetic intensity, not a measured DRAM intensity.
        const double flops_total =
            static_cast<double>(n) * 2.0 * static_cast<double>(fmas_per_load);
        const double bytes_effective =
            static_cast<double>(n) * 2.0 * sizeof(float);
        const double ai_nominal = flops_total / bytes_effective;

        const double best_s = static_cast<double>(best_ms) * 1e-3;
        const double mean_s = static_cast<double>(mean_ms) * 1e-3;
        const double median_s = static_cast<double>(median) * 1e-3;

        const double best_gflops = flops_total / best_s / 1e9;
        const double mean_gflops = flops_total / mean_s / 1e9;
        const double median_gflops = flops_total / median_s / 1e9;

        const double bw_best_gbs = bytes_effective / best_s / 1e9;
        const double bw_mean_gbs = bytes_effective / mean_s / 1e9;
        const double bw_median_gbs = bytes_effective / median_s / 1e9;

        if (!quiet) {
            std::cout << "# MB10 AI-controlled kernel\n";
            std::cout << "# NOTE: ai_nominal and bytes_effective are analytical model quantities.\n";
            std::cout << "# NOTE: bw_*_gbs is derived from the analytical byte model, not from HW counters.\n";
            std::cout << "device=" << device << "\n";
            std::cout << "n=" << n << "\n";
            std::cout << "bytes_per_array=" << bytes_per_array << "\n";
            std::cout << "bytes_effective=" << bytes_effective << "\n";
            std::cout << "block=" << block << "\n";
            std::cout << "grid=" << launch_grid << "\n";
            std::cout << "fmas_per_load=" << fmas_per_load << "\n";
            std::cout << "ai_nominal=" << ai_nominal << "\n";
            std::cout << "warmup=" << warmup << "\n";
            std::cout << "reps=" << reps << "\n";
            std::cout << "best_ms=" << best_ms << "\n";
            std::cout << "mean_ms=" << mean_ms << "\n";
            std::cout << "median_ms=" << median << "\n";
            std::cout << "best_gflops=" << best_gflops << "\n";
            std::cout << "mean_gflops=" << mean_gflops << "\n";
            std::cout << "median_gflops=" << median_gflops << "\n";
            std::cout << "bw_best_gbs=" << bw_best_gbs << "\n";
            std::cout << "bw_mean_gbs=" << bw_mean_gbs << "\n";
            std::cout << "bw_median_gbs=" << bw_median_gbs << "\n";
        }

        if (!csv_path.empty()) {
            bw::CsvWriter writer(csv_path, false);
            writer.write_header({
                "benchmark",
                "device",
                "n",
                "bytes_per_array",
                "bytes_effective",
                "block",
                "grid",
                "fmas_per_load",
                "ai_nominal",
                "warmup",
                "reps",
                "best_ms",
                "mean_ms",
                "median_ms",
                "best_gflops",
                "mean_gflops",
                "median_gflops",
                "bw_best_gbs",
                "bw_mean_gbs",
                "bw_median_gbs"
            });
            writer.write_row({
                "mb10_ai_control",
                bw::to_csv_string(device),
                bw::to_csv_string(n),
                bw::to_csv_string(bytes_per_array),
                bw::to_csv_string(bytes_effective),
                bw::to_csv_string(block),
                bw::to_csv_string(launch_grid),
                bw::to_csv_string(fmas_per_load),
                bw::to_csv_string(ai_nominal),
                bw::to_csv_string(warmup),
                bw::to_csv_string(reps),
                bw::to_csv_string(best_ms),
                bw::to_csv_string(mean_ms),
                bw::to_csv_string(median),
                bw::to_csv_string(best_gflops),
                bw::to_csv_string(mean_gflops),
                bw::to_csv_string(median_gflops),
                bw::to_csv_string(bw_best_gbs),
                bw::to_csv_string(bw_mean_gbs),
                bw::to_csv_string(bw_median_gbs)
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