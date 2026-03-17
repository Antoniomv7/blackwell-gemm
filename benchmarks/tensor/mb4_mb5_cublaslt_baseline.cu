#include "benchmarks/common/cli.hpp"
#include "benchmarks/common/csv_writer.hpp"
#include "benchmarks/common/cuda_utils.hpp"
#include "benchmarks/common/timing.hpp"

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUBLASLT_CHECK(expr)                                                         \
    do {                                                                             \
        cublasStatus_t _status = (expr);                                             \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(std::string("[cuBLASLt ERROR] expr: ") + #expr +\
                                     " | code: " + std::to_string((int)_status));    \
        }                                                                            \
    } while (0)

enum class DType {
    FP16,
    BF16
};

static DType parse_dtype(const std::string& s) {
    if (s == "fp16") return DType::FP16;
    if (s == "bf16") return DType::BF16;
    throw std::runtime_error("Unsupported --dtype. Supported: fp16, bf16");
}

static std::string dtype_to_string(DType t) {
    switch (t) {
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
        default: return "unknown";
    }
}

template <typename T>
static void fill_host_buffer(std::vector<T>& buf, float scale = 1.0f);

template <>
void fill_host_buffer<__half>(std::vector<__half>& buf, float scale) {
    for (std::size_t i = 0; i < buf.size(); ++i) {
        const float x = scale * static_cast<float>((i % 13) - 6) * 0.1f;
        buf[i] = __float2half(x);
    }
}

template <>
void fill_host_buffer<__nv_bfloat16>(std::vector<__nv_bfloat16>& buf, float scale) {
    for (std::size_t i = 0; i < buf.size(); ++i) {
        const float x = scale * static_cast<float>((i % 13) - 6) * 0.1f;
        buf[i] = __float2bfloat16(x);
    }
}

struct GemmConfig {
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 5;
    int reps = 20;
    int device = 0;
    DType dtype = DType::FP16;
};

template <typename T>
struct GemmBuffers {
    T* dA = nullptr;
    T* dB = nullptr;
    float* dC = nullptr;

    std::vector<T> hA;
    std::vector<T> hB;
};

template <typename T>
static GemmBuffers<T> allocate_and_init_buffers(const GemmConfig& cfg) {
    GemmBuffers<T> gb;
    const std::size_t elemsA = static_cast<std::size_t>(cfg.m) * cfg.k;
    const std::size_t elemsB = static_cast<std::size_t>(cfg.k) * cfg.n;
    const std::size_t elemsC = static_cast<std::size_t>(cfg.m) * cfg.n;

    gb.hA.resize(elemsA);
    gb.hB.resize(elemsB);

    fill_host_buffer(gb.hA, 1.0f);
    fill_host_buffer(gb.hB, 0.5f);

    CUDA_CHECK(cudaMalloc(&gb.dA, elemsA * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gb.dB, elemsB * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gb.dC, elemsC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(gb.dA, gb.hA.data(), elemsA * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gb.dB, gb.hB.data(), elemsB * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gb.dC, 0, elemsC * sizeof(float)));

    return gb;
}

template <typename T>
static void free_buffers(GemmBuffers<T>& gb) {
    if (gb.dA) CUDA_CHECK(cudaFree(gb.dA));
    if (gb.dB) CUDA_CHECK(cudaFree(gb.dB));
    if (gb.dC) CUDA_CHECK(cudaFree(gb.dC));
    gb.dA = nullptr;
    gb.dB = nullptr;
    gb.dC = nullptr;
}

template <typename T>
static float run_cublaslt_gemm(const GemmConfig& cfg,
                               GemmBuffers<T>& gb,
                               cudaDataType_t ab_type,
                               cublasComputeType_t compute_type,
                               int warmup,
                               int reps,
                               float* mean_ms_out) {
    cublasLtHandle_t lt;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op_desc;
    cublasLtMatrixLayout_t a_desc, b_desc, c_desc;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F));

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Column-major logical layout:
    // A: m x k, ld = m
    // B: k x n, ld = k
    // C: m x n, ld = m
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, ab_type, cfg.m, cfg.k, cfg.m));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, ab_type, cfg.k, cfg.n, cfg.k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, cfg.m, cfg.n, cfg.m));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto launch = [&]() {
        CUBLASLT_CHECK(cublasLtMatmul(
            lt,
            op_desc,
            &alpha,
            gb.dA, a_desc,
            gb.dB, b_desc,
            &beta,
            gb.dC, c_desc,
            gb.dC, c_desc,
            nullptr,
            nullptr,
            0,
            0));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    const std::vector<float> samples = bw::benchmark_kernel_samples_ms(
        launch, warmup, reps);

    const float best_ms = bw::min_ms(samples);
    const float mean_ms = bw::mean_ms(samples);

    if (mean_ms_out) {
        *mean_ms_out = mean_ms;
    }

    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));

    return best_ms;
}

static double gemm_flops(const GemmConfig& cfg) {
    return 2.0 * static_cast<double>(cfg.m) *
           static_cast<double>(cfg.n) *
           static_cast<double>(cfg.k);
}

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        GemmConfig cfg;
        cfg.device = args.get_int("device", 0);
        cfg.m = args.get_int("m", 4096);
        cfg.n = args.get_int("n", 4096);
        cfg.k = args.get_int("k", 4096);
        cfg.warmup = args.get_int("warmup", 5);
        cfg.reps = args.get_int("reps", 20);
        cfg.dtype = parse_dtype(args.get_string("dtype", "fp16"));

        const std::string csv_path = args.get_string("csv", "");
        const bool quiet = args.get_flag("quiet");

        if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
            throw std::runtime_error("m, n, k must be > 0");
        }
        if (cfg.warmup < 0 || cfg.reps <= 0) {
            throw std::runtime_error("warmup must be >= 0 and reps must be > 0");
        }

        CUDA_CHECK(cudaSetDevice(cfg.device));

        float best_ms = 0.0f;
        float mean_ms = 0.0f;

        if (cfg.dtype == DType::FP16) {
            auto gb = allocate_and_init_buffers<__half>(cfg);
            best_ms = run_cublaslt_gemm(
                cfg, gb, CUDA_R_16F, CUBLAS_COMPUTE_32F, cfg.warmup, cfg.reps, &mean_ms);
            free_buffers(gb);
        } else {
            auto gb = allocate_and_init_buffers<__nv_bfloat16>(cfg);
            best_ms = run_cublaslt_gemm(
                cfg, gb, CUDA_R_16BF, CUBLAS_COMPUTE_32F, cfg.warmup, cfg.reps, &mean_ms);
            free_buffers(gb);
        }

        const double flops = gemm_flops(cfg);
        const double best_tflops = flops / (static_cast<double>(best_ms) * 1e-3) / 1e12;
        const double mean_tflops = flops / (static_cast<double>(mean_ms) * 1e-3) / 1e12;

        if (!quiet) {
            std::cout << "# MB4/MB5 cuBLASLt baseline\n";
            std::cout << "device=" << cfg.device << "\n";
            std::cout << "dtype=" << dtype_to_string(cfg.dtype) << "\n";
            std::cout << "m=" << cfg.m << "\n";
            std::cout << "n=" << cfg.n << "\n";
            std::cout << "k=" << cfg.k << "\n";
            std::cout << "warmup=" << cfg.warmup << "\n";
            std::cout << "reps=" << cfg.reps << "\n";
            std::cout << "best_ms=" << best_ms << "\n";
            std::cout << "mean_ms=" << mean_ms << "\n";
            std::cout << "best_tflops=" << best_tflops << "\n";
            std::cout << "mean_tflops=" << mean_tflops << "\n";
        }

        if (!csv_path.empty()) {
            bw::CsvWriter writer(csv_path, false);
            writer.write_header({
                "benchmark",
                "device",
                "dtype",
                "m",
                "n",
                "k",
                "warmup",
                "reps",
                "best_ms",
                "mean_ms",
                "best_tflops",
                "mean_tflops"
            });
            writer.write_row({
                "mb4_mb5_cublaslt_baseline",
                bw::to_csv_string(cfg.device),
                dtype_to_string(cfg.dtype),
                bw::to_csv_string(cfg.m),
                bw::to_csv_string(cfg.n),
                bw::to_csv_string(cfg.k),
                bw::to_csv_string(cfg.warmup),
                bw::to_csv_string(cfg.reps),
                bw::to_csv_string(best_ms),
                bw::to_csv_string(mean_ms),
                bw::to_csv_string(best_tflops),
                bw::to_csv_string(mean_tflops)
            });
            writer.flush();

            if (!quiet) {
                std::cout << "[mb4_mb5_cublaslt_baseline] Wrote CSV: " << csv_path << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb4_mb5_cublaslt_baseline] ERROR: " << e.what() << "\n";
        return 1;
    }
}