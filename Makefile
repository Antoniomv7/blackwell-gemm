# ============================================================================
# blackwell-gemm - unified build system
# ============================================================================
#
# Usage examples:
#   make help
#   make list
#   make all
#   make mb31_inventory
#   make mb34_stream_triad
#   make sprint9-pending
#
# Notes:
#   - source scripts/env/env.sh before using if you want project defaults
#   - environment variables may override defaults:
#       CUDA_ARCH=sm_103
#       NVCC=nvcc
#       CXX=g++
# ============================================================================

SHELL := /usr/bin/env bash

NVCC ?= nvcc
CXX  ?= g++

PROJECT_ROOT := $(CURDIR)
BENCH_DIR     := $(PROJECT_ROOT)/benchmarks
COMMON_DIR    := $(BENCH_DIR)/common
DEVICE_DIR    := $(BENCH_DIR)/device
SM_DIR        := $(BENCH_DIR)/sm
MEMORY_DIR    := $(BENCH_DIR)/memory
TENSOR_DIR    := $(BENCH_DIR)/tensor

BIN_DIR       := $(PROJECT_ROOT)/bin
BUILD_DIR     := $(PROJECT_ROOT)/build

CUDA_ARCH ?= sm_103

INCLUDES := -I$(PROJECT_ROOT) -I$(BENCH_DIR) -I$(COMMON_DIR) -I/usr/local/cuda/include

NVCC_STD_FLAGS   := -std=c++17 -O3 -lineinfo
NVCC_ARCH_FLAGS  := -arch=$(CUDA_ARCH)
NVCC_WARN_FLAGS  :=
NVCC_LIBS        := -lcudart

CXX_STD_FLAGS    := -std=c++17 -O3
CXX_WARN_FLAGS   := -Wall -Wextra -Wpedantic
CXX_LIBS         := -L/usr/local/cuda/lib64 -lcudart

NVCCFLAGS := $(NVCC_STD_FLAGS) $(NVCC_ARCH_FLAGS) $(NVCC_WARN_FLAGS) $(INCLUDES)
CXXFLAGS  := $(CXX_STD_FLAGS) $(CXX_WARN_FLAGS) $(INCLUDES)

CPP_TARGETS := \
	mb31_inventory

CUDA_TARGETS := \
	mb34_stream_triad \
	mb34_ptr_chase \
	mb32_residency \
	mb32_fp32_probe \
	mb8_shared_stride \
	mb3_mem_compute_mix \
	mb10_ai_control \
	mb4_mb5_cublaslt_baseline

SPRINT9_PENDING_TARGETS := \
	mb33_tensor_peak_cutlass \
	mb33_layout_sensitivity

mb31_inventory_SRC               := $(DEVICE_DIR)/mb31_inventory.cpp
mb34_stream_triad_SRC            := $(MEMORY_DIR)/mb34_stream_triad.cu
mb34_ptr_chase_SRC               := $(MEMORY_DIR)/mb34_ptr_chase.cu
mb32_residency_SRC               := $(SM_DIR)/mb32_residency.cu
mb32_fp32_probe_SRC              := $(SM_DIR)/mb32_fp32_probe.cu
mb8_shared_stride_SRC            := $(SM_DIR)/mb8_shared_stride.cu
mb3_mem_compute_mix_SRC          := $(MEMORY_DIR)/mb3_mem_compute_mix.cu
mb10_ai_control_SRC              := $(MEMORY_DIR)/mb10_ai_control.cu
mb4_mb5_cublaslt_baseline_SRC    := $(TENSOR_DIR)/mb4_mb5_cublaslt_baseline.cu
mb33_tensor_peak_cutlass_SRC     := $(TENSOR_DIR)/mb33_tensor_peak_cutlass.cu
mb33_layout_sensitivity_SRC      := $(TENSOR_DIR)/mb33_layout_sensitivity.cu

ALL_BUILDABLE_TARGETS := $(CPP_TARGETS) $(CUDA_TARGETS)
EXISTING_BUILDABLE_TARGETS := $(foreach t,$(ALL_BUILDABLE_TARGETS),$(if $(wildcard $($(t)_SRC)),$(t),))
MISSING_BUILDABLE_TARGETS := $(foreach t,$(ALL_BUILDABLE_TARGETS),$(if $(wildcard $($(t)_SRC)),,$(t)))

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
define print_missing_source
	@if [[ ! -f "$($(1)_SRC)" ]]; then \
		echo "[make] ERROR: source file not found for target '$(1)': $($(1)_SRC)"; \
		exit 1; \
	fi
endef

.PHONY: help list sprint9-pending dirs clean distclean all \
        $(ALL_BUILDABLE_TARGETS) $(SPRINT9_PENDING_TARGETS)

help:
	@echo "Targets:"
	@echo "  make help                - show this help"
	@echo "  make list                - list buildable targets and their sources"
	@echo "  make all                 - build all sources that currently exist"
	@echo "  make sprint9-pending     - list targets intentionally deferred to Sprint 9"
	@echo "  make <target>            - build one known target"
	@echo "  make clean               - remove build artifacts"
	@echo "  make distclean           - remove build/ and bin/"
	@echo
	@echo "Variables:"
	@echo "  CUDA_ARCH=sm_103"
	@echo "  NVCC=nvcc"
	@echo "  CXX=g++"

list:
	@echo "[make] Buildable C++ targets:"
	@$(foreach t,$(CPP_TARGETS),echo "  $(t) -> $($(t)_SRC)";)
	@echo "[make] Buildable CUDA targets:"
	@$(foreach t,$(CUDA_TARGETS),echo "  $(t) -> $($(t)_SRC)";)
	@echo "[make] Deferred to Sprint 9:"
	@$(foreach t,$(SPRINT9_PENDING_TARGETS),echo "  $(t) -> $($(t)_SRC) [pending Sprint 9]";)

sprint9-pending:
	@echo "[make] Targets deferred to Sprint 9:"
	@$(foreach t,$(SPRINT9_PENDING_TARGETS),echo "  $(t) -> $($(t)_SRC)";)

dirs:
	@mkdir -p $(BIN_DIR) $(BUILD_DIR)

all: dirs $(EXISTING_BUILDABLE_TARGETS)
	@echo "[make] Built existing targets: $(EXISTING_BUILDABLE_TARGETS)"
	@if [[ -n "$(strip $(MISSING_BUILDABLE_TARGETS))" ]]; then \
		echo "[make] Skipped missing sources: $(strip $(MISSING_BUILDABLE_TARGETS))"; \
	fi

mb31_inventory: dirs
	$(call print_missing_source,mb31_inventory)
	$(CXX) $(CXXFLAGS) $(mb31_inventory_SRC) -o $(BIN_DIR)/$@ $(CXX_LIBS)

mb34_stream_triad: dirs
	$(call print_missing_source,mb34_stream_triad)
	$(NVCC) $(NVCCFLAGS) $(mb34_stream_triad_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS)

mb34_ptr_chase: dirs
	$(call print_missing_source,mb34_ptr_chase)
	$(NVCC) $(NVCCFLAGS) $(mb34_ptr_chase_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS)

mb32_residency: dirs
	$(call print_missing_source,mb32_residency)
	$(NVCC) $(NVCCFLAGS) $(mb32_residency_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS)

mb32_fp32_probe: dirs
	$(call print_missing_source,mb32_fp32_probe)
	$(NVCC) $(NVCCFLAGS) $(mb32_fp32_probe_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS)

mb8_shared_stride: dirs
	$(call print_missing_source,mb8_shared_stride)
	$(NVCC) $(NVCCFLAGS) $(mb8_shared_stride_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS)

mb3_mem_compute_mix: dirs
	$(call print_missing_source,mb3_mem_compute_mix)
	$(NVCC) $(NVCCFLAGS) $(mb3_mem_compute_mix_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS)

mb10_ai_control: dirs
	$(call print_missing_source,mb10_ai_control)
	$(NVCC) $(NVCCFLAGS) $(mb10_ai_control_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS)

mb4_mb5_cublaslt_baseline: dirs
	$(call print_missing_source,mb4_mb5_cublaslt_baseline)
	$(NVCC) $(NVCCFLAGS) $(mb4_mb5_cublaslt_baseline_SRC) -o $(BIN_DIR)/$@ $(NVCC_LIBS) -lcublasLt -lcublas

mb33_tensor_peak_cutlass:
	@echo "[make] Target '$@' is intentionally deferred to Sprint 9."
	@echo "[make] Source placeholder: $(mb33_tensor_peak_cutlass_SRC)"
	@exit 1

mb33_layout_sensitivity:
	@echo "[make] Target '$@' is intentionally deferred to Sprint 9."
	@echo "[make] Source placeholder: $(mb33_layout_sensitivity_SRC)"
	@exit 1

clean:
	@echo "[make] Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)/*

distclean:
	@echo "[make] Removing build/ and bin/ ..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
