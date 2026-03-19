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

__global__ void stream_triad_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c,
                                    int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + 3.0f * b[i];
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
        const int n = args.get_int("n", 1 << 26);         // 67,108,864 elems
        const int block = args.get_int("block", 256);
        const int grid = args.get_int("grid", 0);         // auto if 0
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

        CUDA_CHECK(cudaSetDevice(device));

        const int launch_grid =
            (grid > 0) ? grid : bw::default_grid_1d(n, block, device, 20);

        const std::size_t bytes_per_array =
            static_cast<std::size_t>(n) * sizeof(float);

        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;

        CUDA_CHECK(cudaMalloc(&d_a, bytes_per_array));
        CUDA_CHECK(cudaMalloc(&d_b, bytes_per_array));
        CUDA_CHECK(cudaMalloc(&d_c, bytes_per_array));

        init_array_kernel<<<launch_grid, block>>>(d_a, 1.0f, n);
        CUDA_CHECK_LAST_KERNEL();
        init_array_kernel<<<launch_grid, block>>>(d_b, 2.0f, n);
        CUDA_CHECK_LAST_KERNEL();
        init_array_kernel<<<launch_grid, block>>>(d_c, 0.0f, n);
        CUDA_CHECK_LAST_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        auto launch = [&]() {
            stream_triad_kernel<<<launch_grid, block>>>(d_a, d_b, d_c, n);
            CUDA_CHECK_LAST_KERNEL();
        };

        const std::vector<float> samples_ms =
            bw::benchmark_kernel_samples_ms(launch, warmup, reps);

        const float best_ms = bw::min_ms(samples_ms);
        const float mean_ms = bw::mean_ms(samples_ms);
        const float median = median_ms(samples_ms);

        // STREAM triad effective traffic model:
        //   load A + load B + store C = 3 * sizeof(float) bytes per element
        const double effective_bytes =
            static_cast<double>(n) * 3.0 * sizeof(float);

        const double best_s = static_cast<double>(best_ms) * 1e-3;
        const double mean_s = static_cast<double>(mean_ms) * 1e-3;
        const double median_s = static_cast<double>(median) * 1e-3;

        const double bw_best_gbs = effective_bytes / best_s / 1e9;
        const double bw_mean_gbs = effective_bytes / mean_s / 1e9;
        const double bw_median_gbs = effective_bytes / median_s / 1e9;

        if (!quiet) {
            std::cout << "# MB3.4-A STREAM triad\n";
            std::cout << "device=" << device << "\n";
            std::cout << "n=" << n << "\n";
            std::cout << "bytes_per_array=" << bytes_per_array << "\n";
            std::cout << "effective_bytes=" << effective_bytes << "\n";
            std::cout << "block=" << block << "\n";
            std::cout << "grid=" << launch_grid << "\n";
            std::cout << "warmup=" << warmup << "\n";
            std::cout << "reps=" << reps << "\n";
            std::cout << "best_ms=" << best_ms << "\n";
            std::cout << "mean_ms=" << mean_ms << "\n";
            std::cout << "median_ms=" << median << "\n";
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
                "effective_bytes",
                "block",
                "grid",
                "warmup",
                "reps",
                "best_ms",
                "mean_ms",
                "median_ms",
                "bw_best_gbs",
                "bw_mean_gbs",
                "bw_median_gbs"
            });
            writer.write_row({
                "mb34_stream_triad",
                bw::to_csv_string(device),
                bw::to_csv_string(n),
                bw::to_csv_string(bytes_per_array),
                bw::to_csv_string(effective_bytes),
                bw::to_csv_string(block),
                bw::to_csv_string(launch_grid),
                bw::to_csv_string(warmup),
                bw::to_csv_string(reps),
                bw::to_csv_string(best_ms),
                bw::to_csv_string(mean_ms),
                bw::to_csv_string(median),
                bw::to_csv_string(bw_best_gbs),
                bw::to_csv_string(bw_mean_gbs),
                bw::to_csv_string(bw_median_gbs)
            });
            writer.flush();

            if (!quiet) {
                std::cout << "[mb34_stream_triad] Wrote CSV: " << csv_path << "\n";
            }
        }

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb34_stream_triad] ERROR: " << e.what() << "\n";
        return 1;
    }
}