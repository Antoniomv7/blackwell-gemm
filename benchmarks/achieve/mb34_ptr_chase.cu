#include "benchmarks/common/cli.hpp"
#include "benchmarks/common/csv_writer.hpp"
#include "benchmarks/common/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

__global__ void ptr_chase_kernel(const int* __restrict__ next,
                                 int* __restrict__ idx_out,
                                 unsigned long long* __restrict__ cycles_out,
                                 int iters) {
    int idx = 0;
    const unsigned long long t0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        idx = next[idx];
    }

    const unsigned long long t1 = clock64();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        idx_out[0] = idx;
        cycles_out[0] = t1 - t0;
    }
}

static std::vector<int> make_random_cycle(int n, std::uint64_t seed) {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    std::mt19937_64 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<int> next(n, 0);
    for (int i = 0; i < n - 1; ++i) {
        next[perm[i]] = perm[i + 1];
    }
    next[perm[n - 1]] = perm[0];
    return next;
}

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        const int device = args.get_int("device", 0);
        const int working_set = args.get_int("working-set", 1 << 20); // elements
        const int iters = args.get_int("iters", 1 << 20);
        const std::uint64_t seed =
            static_cast<std::uint64_t>(args.get_ll("seed", 12345));
        const std::string csv_path = args.get_string("csv", "");
        const bool quiet = args.get_flag("quiet");

        if (working_set <= 1) {
            throw std::runtime_error("working-set must be > 1");
        }
        if (iters <= 0) {
            throw std::runtime_error("iters must be > 0");
        }

        CUDA_CHECK(cudaSetDevice(device));

        std::vector<int> h_next = make_random_cycle(working_set, seed);

        int* d_next = nullptr;
        int* d_idx_out = nullptr;
        unsigned long long* d_cycles_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_next, static_cast<std::size_t>(working_set) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_idx_out, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cycles_out, sizeof(unsigned long long)));

        CUDA_CHECK(cudaMemcpy(d_next,
                              h_next.data(),
                              static_cast<std::size_t>(working_set) * sizeof(int),
                              cudaMemcpyHostToDevice));

        ptr_chase_kernel<<<1, 1>>>(d_next, d_idx_out, d_cycles_out, iters);
        CUDA_CHECK_LAST_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_idx_out = 0;
        unsigned long long h_cycles_out = 0;

        CUDA_CHECK(cudaMemcpy(&h_idx_out, d_idx_out, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_cycles_out,
                              d_cycles_out,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));

        const double cycles_per_iter =
            static_cast<double>(h_cycles_out) / static_cast<double>(iters);
        const std::size_t bytes_working_set =
            static_cast<std::size_t>(working_set) * sizeof(int);

        if (!quiet) {
            std::cout << "# MB3.4-B pointer chasing\n";
            std::cout << "device=" << device << "\n";
            std::cout << "working_set=" << working_set << "\n";
            std::cout << "working_set_bytes=" << bytes_working_set << "\n";
            std::cout << "iters=" << iters << "\n";
            std::cout << "seed=" << seed << "\n";
            std::cout << "idx_out=" << h_idx_out << "\n";
            std::cout << "cycles_total=" << h_cycles_out << "\n";
            std::cout << "cycles_per_iter=" << cycles_per_iter << "\n";
        }

        if (!csv_path.empty()) {
            bw::CsvWriter writer(csv_path, false);
            writer.write_header({
                "benchmark",
                "device",
                "working_set",
                "working_set_bytes",
                "iters",
                "seed",
                "idx_out",
                "cycles_total",
                "cycles_per_iter"
            });
            writer.write_row({
                "mb34_ptr_chase",
                bw::to_csv_string(device),
                bw::to_csv_string(working_set),
                bw::to_csv_string(bytes_working_set),
                bw::to_csv_string(iters),
                bw::to_csv_string(seed),
                bw::to_csv_string(h_idx_out),
                bw::to_csv_string(h_cycles_out),
                bw::to_csv_string(cycles_per_iter)
            });
            writer.flush();

            if (!quiet) {
                std::cout << "[mb34_ptr_chase] Wrote CSV: " << csv_path << "\n";
            }
        }

        CUDA_CHECK(cudaFree(d_next));
        CUDA_CHECK(cudaFree(d_idx_out));
        CUDA_CHECK(cudaFree(d_cycles_out));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb34_ptr_chase] ERROR: " << e.what() << "\n";
        return 1;
    }
}