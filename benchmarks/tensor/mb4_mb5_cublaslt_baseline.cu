#include "benchmarks/common/cli.hpp"
#include "benchmarks/common/csv_writer.hpp"
#include "benchmarks/common/cuda_utils.hpp"
#include "benchmarks/common/timing.hpp"

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUBLASLT_CHECK(expr)                                                                   \
    do {                                                                                       \
        cublasStatus_t _status = (expr);                                                       \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                                \
            throw std::runtime_error(std::string("[cuBLASLt ERROR] expr: ") + #expr +         \
                                     " | code: " + std::to_string(static_cast<int>(_status)));\
        }                                                                                      \
    } while (0)

enum class DType {
    FP16,
    BF16
};

enum class Op {
    N,
    T
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

static Op parse_op(const std::string& s) {
    if (s == "N" || s == "n") return Op::N;
    if (s == "T" || s == "t") return Op::T;
    throw std::runtime_error("Unsupported transpose flag. Supported: N, T");
}

static std::string op_to_string(Op op) {
    return (op == Op::N) ? "N" : "T";
}

static std::string layout_string(Op transa, Op transb) {
    return op_to_string(transa) + op_to_string(transb);
}

static cublasOperation_t to_cublas_op(Op op) {
    return (op == Op::N) ? CUBLAS_OP_N : CUBLAS_OP_T;
}

static cudaDataType_t to_ab_cuda_type(DType dtype) {
    return (dtype == DType::FP16) ? CUDA_R_16F : CUDA_R_16BF;
}

static std::size_t ab_bytes_per_element(DType dtype) {
    switch (dtype) {
        case DType::FP16: return sizeof(__half);
        case DType::BF16: return sizeof(__nv_bfloat16);
        default: return 2;
    }
}

template <typename T>
static std::string csv_string(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

static std::int64_t parse_i64(const std::string& s, const std::string& flag_name) {
    try {
        std::size_t pos = 0;
        long long v = std::stoll(s, &pos, 10);
        if (pos != s.size()) {
            throw std::runtime_error("");
        }
        return static_cast<std::int64_t>(v);
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + flag_name + ": " + s);
    }
}

static float min_of(const std::vector<float>& xs) {
    if (xs.empty()) {
        throw std::runtime_error("Internal error: empty timing sample vector");
    }
    return *std::min_element(xs.begin(), xs.end());
}

static float mean_of(const std::vector<float>& xs) {
    if (xs.empty()) {
        throw std::runtime_error("Internal error: empty timing sample vector");
    }
    const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    return static_cast<float>(sum / static_cast<double>(xs.size()));
}

static float median_of(std::vector<float> xs) {
    if (xs.empty()) {
        throw std::runtime_error("Internal error: empty timing sample vector");
    }
    std::sort(xs.begin(), xs.end());
    const std::size_t n = xs.size();
    if (n % 2 == 1) {
        return xs[n / 2];
    }
    return 0.5f * (xs[n / 2 - 1] + xs[n / 2]);
}

static std::int64_t default_lda(int m, int k, Op transa) {
    return (transa == Op::N) ? static_cast<std::int64_t>(m)
                             : static_cast<std::int64_t>(k);
}

static std::int64_t default_ldb(int n, int k, Op transb) {
    return (transb == Op::N) ? static_cast<std::int64_t>(k)
                             : static_cast<std::int64_t>(n);
}

static std::int64_t default_ldc(int m) {
    return static_cast<std::int64_t>(m);
}

static std::int64_t min_valid_lda(int m, int k, Op transa) {
    return (transa == Op::N) ? static_cast<std::int64_t>(m)
                             : static_cast<std::int64_t>(k);
}

static std::int64_t min_valid_ldb(int n, int k, Op transb) {
    return (transb == Op::N) ? static_cast<std::int64_t>(k)
                             : static_cast<std::int64_t>(n);
}

static std::int64_t min_valid_ldc(int m) {
    return static_cast<std::int64_t>(m);
}

static std::int64_t a_rows_physical(int m, int k, Op transa) {
    return (transa == Op::N) ? static_cast<std::int64_t>(m)
                             : static_cast<std::int64_t>(k);
}

static std::int64_t a_cols_physical(int m, int k, Op transa) {
    return (transa == Op::N) ? static_cast<std::int64_t>(k)
                             : static_cast<std::int64_t>(m);
}

static std::int64_t b_rows_physical(int n, int k, Op transb) {
    return (transb == Op::N) ? static_cast<std::int64_t>(k)
                             : static_cast<std::int64_t>(n);
}

static std::int64_t b_cols_physical(int n, int k, Op transb) {
    return (transb == Op::N) ? static_cast<std::int64_t>(n)
                             : static_cast<std::int64_t>(k);
}

template <typename T>
static void fill_host_buffer(std::vector<T>& buf, float scale = 1.0f);

template <>
void fill_host_buffer<__half>(std::vector<__half>& buf, float scale) {
    for (std::size_t i = 0; i < buf.size(); ++i) {
        const float x = scale * static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.1f;
        buf[i] = __float2half(x);
    }
}

template <>
void fill_host_buffer<__nv_bfloat16>(std::vector<__nv_bfloat16>& buf, float scale) {
    for (std::size_t i = 0; i < buf.size(); ++i) {
        const float x = scale * static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.1f;
        buf[i] = __float2bfloat16(x);
    }
}

struct GemmConfig {
    std::string case_id = "";
    std::string family = "gemm";

    int m = 4096;
    int n = 4096;
    int k = 4096;

    int warmup = 10;
    int reps = 30;
    int device = 0;

    DType dtype = DType::FP16;
    Op transa = Op::N;
    Op transb = Op::N;

    std::int64_t lda = 0;
    std::int64_t ldb = 0;
    std::int64_t ldc = 0;

    std::size_t workspace_bytes = 0;
};

template <typename T>
struct GemmBuffers {
    T* dA = nullptr;
    T* dB = nullptr;
    float* dC = nullptr;

    std::vector<T> hA;
    std::vector<T> hB;

    std::size_t elemsA = 0;
    std::size_t elemsB = 0;
    std::size_t elemsC = 0;
};

static std::string default_case_id(const GemmConfig& cfg) {
    std::ostringstream oss;
    oss << "m" << cfg.m
        << "_n" << cfg.n
        << "_k" << cfg.k
        << "_" << dtype_to_string(cfg.dtype)
        << "_" << layout_string(cfg.transa, cfg.transb)
        << "_ws" << cfg.workspace_bytes;
    return oss.str();
}

static void finalize_config(GemmConfig& cfg) {
    if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
        throw std::runtime_error("m, n, k must be > 0");
    }
    if (cfg.warmup < 0 || cfg.reps <= 0) {
        throw std::runtime_error("warmup must be >= 0 and reps must be > 0");
    }

    if (cfg.lda == 0) cfg.lda = default_lda(cfg.m, cfg.k, cfg.transa);
    if (cfg.ldb == 0) cfg.ldb = default_ldb(cfg.n, cfg.k, cfg.transb);
    if (cfg.ldc == 0) cfg.ldc = default_ldc(cfg.m);

    if (cfg.lda < min_valid_lda(cfg.m, cfg.k, cfg.transa)) {
        throw std::runtime_error("lda is too small for the selected transa/m/k");
    }
    if (cfg.ldb < min_valid_ldb(cfg.n, cfg.k, cfg.transb)) {
        throw std::runtime_error("ldb is too small for the selected transb/n/k");
    }
    if (cfg.ldc < min_valid_ldc(cfg.m)) {
        throw std::runtime_error("ldc is too small for matrix C");
    }

    if (cfg.case_id.empty()) {
        cfg.case_id = default_case_id(cfg);
    }
}

template <typename T>
static GemmBuffers<T> allocate_and_init_buffers(const GemmConfig& cfg) {
    GemmBuffers<T> gb;

    const std::int64_t a_cols = a_cols_physical(cfg.m, cfg.k, cfg.transa);
    const std::int64_t b_cols = b_cols_physical(cfg.n, cfg.k, cfg.transb);

    gb.elemsA = static_cast<std::size_t>(cfg.lda) * static_cast<std::size_t>(a_cols);
    gb.elemsB = static_cast<std::size_t>(cfg.ldb) * static_cast<std::size_t>(b_cols);
    gb.elemsC = static_cast<std::size_t>(cfg.ldc) * static_cast<std::size_t>(cfg.n);

    gb.hA.resize(gb.elemsA);
    gb.hB.resize(gb.elemsB);

    fill_host_buffer(gb.hA, 1.0f);
    fill_host_buffer(gb.hB, 0.5f);

    CUDA_CHECK(cudaMalloc(&gb.dA, gb.elemsA * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gb.dB, gb.elemsB * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gb.dC, gb.elemsC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(gb.dA, gb.hA.data(), gb.elemsA * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gb.dB, gb.hB.data(), gb.elemsB * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gb.dC, 0, gb.elemsC * sizeof(float)));

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

struct BenchmarkResult {
    float best_ms = 0.0f;
    float mean_ms = 0.0f;
    float median_ms = 0.0f;

    double flops = 0.0;
    double bytes_model = 0.0;
    double ai_model = 0.0;

    double best_tflops = 0.0;
    double mean_tflops = 0.0;
    double median_tflops = 0.0;

    int algo_id = -1;
    std::size_t heuristic_workspace_bytes = 0;
};

static double gemm_flops(const GemmConfig& cfg) {
    return 2.0 *
           static_cast<double>(cfg.m) *
           static_cast<double>(cfg.n) *
           static_cast<double>(cfg.k);
}

static double gemm_bytes_model(const GemmConfig& cfg) {
    const double ab_bpe = static_cast<double>(ab_bytes_per_element(cfg.dtype));
    const double c_bpe = static_cast<double>(sizeof(float));

    const double bytes_a = ab_bpe * static_cast<double>(cfg.m) * static_cast<double>(cfg.k);
    const double bytes_b = ab_bpe * static_cast<double>(cfg.k) * static_cast<double>(cfg.n);
    const double bytes_c = c_bpe  * static_cast<double>(cfg.m) * static_cast<double>(cfg.n);

    // Modelo simple para roofline:
    // read A + read B + read C + write C.
    return bytes_a + bytes_b + 2.0 * bytes_c;
}

template <typename T>
static BenchmarkResult run_cublaslt_gemm(const GemmConfig& cfg, GemmBuffers<T>& gb) {
    BenchmarkResult out;

    const cudaDataType_t ab_type = to_ab_cuda_type(cfg.dtype);
    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const cudaDataType_t c_type = CUDA_R_32F;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    void* workspace = nullptr;

    try {
        CUBLASLT_CHECK(cublasLtCreate(&lt));
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F));

        const cublasOperation_t transA = to_cublas_op(cfg.transa);
        const cublasOperation_t transB = to_cublas_op(cfg.transb);

        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

        const std::int64_t a_rows = a_rows_physical(cfg.m, cfg.k, cfg.transa);
        const std::int64_t a_cols = a_cols_physical(cfg.m, cfg.k, cfg.transa);
        const std::int64_t b_rows = b_rows_physical(cfg.n, cfg.k, cfg.transb);
        const std::int64_t b_cols = b_cols_physical(cfg.n, cfg.k, cfg.transb);

        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, ab_type, a_rows, a_cols, cfg.lda));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, ab_type, b_rows, b_cols, cfg.ldb));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, c_type, cfg.m, cfg.n, cfg.ldc));

        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        const std::uint64_t workspace_limit = static_cast<std::uint64_t>(cfg.workspace_bytes);
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit,
            sizeof(workspace_limit)));

        constexpr int kRequestedAlgoCount = 8;
        cublasLtMatmulHeuristicResult_t heuristics[kRequestedAlgoCount];
        int returned_algo_count = 0;

        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            lt,
            op_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            pref,
            kRequestedAlgoCount,
            heuristics,
            &returned_algo_count));

        int selected_index = -1;
        for (int i = 0; i < returned_algo_count; ++i) {
            if (heuristics[i].state == CUBLAS_STATUS_SUCCESS) {
                selected_index = i;
                break;
            }
        }

        if (selected_index < 0) {
            std::ostringstream oss;
            oss << "No valid cuBLASLt heuristic result for case_id=" << cfg.case_id
                << " dtype=" << dtype_to_string(cfg.dtype)
                << " layout=" << layout_string(cfg.transa, cfg.transb)
                << " workspace_bytes=" << cfg.workspace_bytes;
            throw std::runtime_error(oss.str());
        }

        cublasLtMatmulHeuristicResult_t selected = heuristics[selected_index];
        out.heuristic_workspace_bytes = selected.workspaceSize;

        if (selected.workspaceSize > cfg.workspace_bytes) {
            std::ostringstream oss;
            oss << "Selected heuristic requires " << selected.workspaceSize
                << " bytes of workspace, exceeding requested limit " << cfg.workspace_bytes;
            throw std::runtime_error(oss.str());
        }

        size_t size_written = 0;
        int algo_id = -1;
        cublasStatus_t algo_attr_status = cublasLtMatmulAlgoConfigGetAttribute(
            &selected.algo,
            CUBLASLT_ALGO_CONFIG_ID,
            &algo_id,
            sizeof(algo_id),
            &size_written);
        if (algo_attr_status == CUBLAS_STATUS_SUCCESS) {
            out.algo_id = algo_id;
        }

        if (selected.workspaceSize > 0) {
            CUDA_CHECK(cudaMalloc(&workspace, selected.workspaceSize));
        }

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
                &selected.algo,
                workspace,
                selected.workspaceSize,
                0));
            CUDA_CHECK(cudaDeviceSynchronize());
        };

        const std::vector<float> samples =
            bw::benchmark_kernel_samples_ms(launch, cfg.warmup, cfg.reps);

        out.best_ms = min_of(samples);
        out.mean_ms = mean_of(samples);
        out.median_ms = median_of(samples);
    } catch (...) {
        if (workspace) cudaFree(workspace);
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        if (lt) cublasLtDestroy(lt);
        throw;
    }

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    if (pref) CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    if (a_desc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    if (b_desc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    if (c_desc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    if (op_desc) CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    if (lt) CUBLASLT_CHECK(cublasLtDestroy(lt));

    out.flops = gemm_flops(cfg);
    out.bytes_model = gemm_bytes_model(cfg);
    out.ai_model = out.flops / out.bytes_model;

    out.best_tflops = out.flops / (static_cast<double>(out.best_ms) * 1e-3) / 1e12;
    out.mean_tflops = out.flops / (static_cast<double>(out.mean_ms) * 1e-3) / 1e12;
    out.median_tflops = out.flops / (static_cast<double>(out.median_ms) * 1e-3) / 1e12;

    return out;
}

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        GemmConfig cfg;
        cfg.case_id = args.get_string("case-id", "");
        cfg.family = args.get_string("family", "gemm");
        cfg.device = args.get_int("device", 0);
        cfg.m = args.get_int("m", 4096);
        cfg.n = args.get_int("n", 4096);
        cfg.k = args.get_int("k", 4096);
        cfg.warmup = args.get_int("warmup", 10);
        cfg.reps = args.get_int("reps", 30);
        cfg.dtype = parse_dtype(args.get_string("dtype", "fp16"));
        cfg.transa = parse_op(args.get_string("transa", "N"));
        cfg.transb = parse_op(args.get_string("transb", "N"));
        cfg.lda = parse_i64(args.get_string("lda", "0"), "--lda");
        cfg.ldb = parse_i64(args.get_string("ldb", "0"), "--ldb");
        cfg.ldc = parse_i64(args.get_string("ldc", "0"), "--ldc");

        const std::int64_t workspace_i64 =
            parse_i64(args.get_string("workspace-bytes", "0"), "--workspace-bytes");
        if (workspace_i64 < 0) {
            throw std::runtime_error("--workspace-bytes must be >= 0");
        }
        cfg.workspace_bytes = static_cast<std::size_t>(workspace_i64);

        const std::string csv_path = args.get_string("csv", "");
        const bool quiet = args.get_flag("quiet");

        finalize_config(cfg);
        CUDA_CHECK(cudaSetDevice(cfg.device));

        BenchmarkResult result;

        if (cfg.dtype == DType::FP16) {
            auto gb = allocate_and_init_buffers<__half>(cfg);
            result = run_cublaslt_gemm(cfg, gb);
            free_buffers(gb);
        } else {
            auto gb = allocate_and_init_buffers<__nv_bfloat16>(cfg);
            result = run_cublaslt_gemm(cfg, gb);
            free_buffers(gb);
        }

        if (!quiet) {
            std::cout << "# MB4/MB5 cuBLASLt GEMM backend\n";
            std::cout << "case_id=" << cfg.case_id << "\n";
            std::cout << "family=" << cfg.family << "\n";
            std::cout << "device=" << cfg.device << "\n";
            std::cout << "dtype=" << dtype_to_string(cfg.dtype) << "\n";
            std::cout << "transa=" << op_to_string(cfg.transa) << "\n";
            std::cout << "transb=" << op_to_string(cfg.transb) << "\n";
            std::cout << "layout=" << layout_string(cfg.transa, cfg.transb) << "\n";
            std::cout << "m=" << cfg.m << "\n";
            std::cout << "n=" << cfg.n << "\n";
            std::cout << "k=" << cfg.k << "\n";
            std::cout << "lda=" << cfg.lda << "\n";
            std::cout << "ldb=" << cfg.ldb << "\n";
            std::cout << "ldc=" << cfg.ldc << "\n";
            std::cout << "workspace_bytes=" << cfg.workspace_bytes << "\n";
            std::cout << "heuristic_workspace_bytes=" << result.heuristic_workspace_bytes << "\n";
            std::cout << "algo_id=" << result.algo_id << "\n";
            std::cout << "warmup=" << cfg.warmup << "\n";
            std::cout << "reps=" << cfg.reps << "\n";
            std::cout << "best_ms=" << result.best_ms << "\n";
            std::cout << "mean_ms=" << result.mean_ms << "\n";
            std::cout << "median_ms=" << result.median_ms << "\n";
            std::cout << "flops=" << result.flops << "\n";
            std::cout << "bytes_model=" << result.bytes_model << "\n";
            std::cout << "ai_model=" << result.ai_model << "\n";
            std::cout << "best_tflops=" << result.best_tflops << "\n";
            std::cout << "mean_tflops=" << result.mean_tflops << "\n";
            std::cout << "median_tflops=" << result.median_tflops << "\n";
        }

        if (!csv_path.empty()) {
            bw::CsvWriter writer(csv_path, false);
            writer.write_header({
                "benchmark",
                "case_id",
                "family",
                "device",
                "dtype",
                "a_type",
                "b_type",
                "c_type",
                "d_type",
                "compute_type",
                "transa",
                "transb",
                "layout",
                "m",
                "n",
                "k",
                "lda",
                "ldb",
                "ldc",
                "workspace_bytes",
                "heuristic_workspace_bytes",
                "algo_id",
                "warmup",
                "reps",
                "best_ms",
                "mean_ms",
                "median_ms",
                "flops",
                "bytes_model",
                "ai_model",
                "best_tflops",
                "mean_tflops",
                "median_tflops"
            });
            writer.write_row({
                "mb4_mb5_cublaslt_baseline",
                cfg.case_id,
                cfg.family,
                csv_string(cfg.device),
                dtype_to_string(cfg.dtype),
                dtype_to_string(cfg.dtype),
                dtype_to_string(cfg.dtype),
                "fp32",
                "fp32",
                "fp32",
                op_to_string(cfg.transa),
                op_to_string(cfg.transb),
                layout_string(cfg.transa, cfg.transb),
                csv_string(cfg.m),
                csv_string(cfg.n),
                csv_string(cfg.k),
                csv_string(cfg.lda),
                csv_string(cfg.ldb),
                csv_string(cfg.ldc),
                csv_string(cfg.workspace_bytes),
                csv_string(result.heuristic_workspace_bytes),
                csv_string(result.algo_id),
                csv_string(cfg.warmup),
                csv_string(cfg.reps),
                csv_string(result.best_ms),
                csv_string(result.mean_ms),
                csv_string(result.median_ms),
                csv_string(result.flops),
                csv_string(result.bytes_model),
                csv_string(result.ai_model),
                csv_string(result.best_tflops),
                csv_string(result.mean_tflops),
                csv_string(result.median_tflops)
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