# blackwell-gemm

`blackwell-gemm` is a research repository for **reproducible GPU performance characterization on NVIDIA Blackwell systems**. The current focus is not generic CUDA benchmarking, but a single, coherent experimental axis:

- execution environment inventory and control,
- sustained HBM bandwidth measurement,
- synthetic roofline calibration,
- cuBLASLt GEMM sweeps,
- empirical roofline construction,
- and selected profiler-backed case studies.

The repository is organized so that a reader can understand both **what is being measured** and **how the measurements are produced**.

---

## Research question

The main question behind this repository is:

> How close does real GEMM execution on Blackwell get to sustained memory and compute ceilings, and how do shape, layout, datatype, and workspace move GEMM between different performance regimes?

This is therefore a **measurement and analysis repository**, not a kernel-development repo and not a generic architecture reverse-engineering project.

---

## Scope

The mainline workflow covers six components:

1. **Inventory / environment control**  
   Capture device, runtime, and provenance information for each campaign.

2. **HBM ceiling**  
   Measure sustained memory bandwidth with a STREAM-like triad kernel.

3. **Roofline calibration**  
   Generate synthetic points with controlled arithmetic intensity.

4. **GEMM sweep with cuBLASLt**  
   Sweep representative GEMM families across datatype, layout, and workspace.

5. **Empirical roofline analysis**  
   Build a measured roofline from sustained ceilings and place GEMM points on it.

6. **Selected profiler case studies**  
   Profile a small subset of representative GEMMs to explain behavior, not to replace the sweep.

Broader microbenchmarks that are not part of this central story are intentionally kept out of the main workflow.

---

## What this repository is not

This repository is not intended to be:

- a general CUDA tutorial collection,
- a CUTLASS kernel-development sandbox,
- a multi-GPU scaling benchmark suite,
- or a complete public model of Blackwell microarchitecture.

The emphasis is on **experimental coherence**, not breadth.

---

## Repository structure

```text
blackwell-gemm/
├── benchmarks/
│   ├── common/
│   ├── device/
│   │   ├── mb31_inventory.cpp
│   │   ├── mb31_power_trace.sh
│   │   └── mb31_topology.sh
│   ├── memory/
│   │   ├── mb34_stream_triad.cu
│   │   ├── mb10_ai_control.cu
│   │   └── mb3_mem_compute_mix.cu
│   ├── tensor/
│   │   └── mb4_mb5_cublaslt_baseline.cu
│   └── archive/
│
├── scripts/
│   ├── env/
│   ├── lib/
│   │   └── common.sh
│   ├── run/
│   │   ├── run_inventory.sh
│   │   ├── run_hbm_ceiling.sh
│   │   ├── run_roofline_calibration.sh
│   │   ├── run_gemm_case.sh
│   │   ├── run_gemm_sweep.sh
│   │   └── run_roofline_analysis.sh
│   ├── profile/
│   │   └── ncu_gemm_cases.sh
│   └── campaign/
│       ├── campaign_mvp_gemm_roofline.sh
│       └── campaign_ncu_selected.sh
│
├── analysis/
│   ├── build_empirical_roofline.py
│   ├── plot_roofline.py
│   ├── summarize_campaign.py
│   ├── summarize_gemm.py
│   └── plot_gemm_heatmaps.py
│
├── configs/
│   ├── gemm_shapes_mvp.json
│   └── ncu_cases.json
│
├── results/
│   ├── raw/
│   └── processed/
│
└── README.md
```

---

## Experimental workflow

The repository is designed around a campaign-based workflow.

### 1. Inventory

Each campaign starts by recording:

- selected GPU,
- runtime environment,
- provenance metadata,
- and device information.

### 2. HBM ceiling

A STREAM-like triad benchmark is used to estimate a sustained HBM bandwidth ceiling.

### 3. Roofline calibration

Synthetic kernels sweep arithmetic intensity to generate calibration points spanning memory-bound and compute-like behavior.

### 4. GEMM sweep

Representative GEMM families are evaluated with cuBLASLt while varying:

- shape,
- datatype,
- layout,
- and workspace budget.

### 5. Empirical roofline

The analysis stage combines the measured ceilings and GEMM points into a single empirical roofline view.

### 6. Selected profiling cases

A small, curated set of GEMM cases is profiled for explanation and interpretation.

---

## Main entry point

Once the required binaries are built, the main campaign is launched with:

```bash
bash scripts/campaign/campaign_mvp_gemm_roofline.sh <campaign_id>
```

Example:

```bash
bash scripts/campaign/campaign_mvp_gemm_roofline.sh test_full
```

This runs, in order:

1. `run_inventory.sh`
2. `run_hbm_ceiling.sh`
3. `run_roofline_calibration.sh`
4. `run_gemm_sweep.sh`
5. `run_roofline_analysis.sh`

You can also provide a custom GEMM sweep configuration:

```bash
bash scripts/campaign/campaign_mvp_gemm_roofline.sh test_full configs/gemm_shapes_mvp.json
```

---

## Running steps individually

The campaign can also be executed step by step.

### Inventory

```bash
bash scripts/run/run_inventory.sh test_inventory
```

### HBM ceiling

```bash
bash scripts/run/run_hbm_ceiling.sh test_hbm
```

### Roofline calibration

```bash
bash scripts/run/run_roofline_calibration.sh test_calib
```

### GEMM sweep

```bash
bash scripts/run/run_gemm_sweep.sh test_gemm configs/gemm_shapes_mvp.json
```

### Roofline analysis

```bash
bash scripts/run/run_roofline_analysis.sh test_full
```

---

## Input configurations

### GEMM sweep

The default GEMM sweep is defined in:

```text
configs/gemm_shapes_mvp.json
```

It covers representative families such as:

- square,
- tall-skinny,
- wide-short,
- skinny-K,
- fat-K,

and varies:

- datatype (`fp16`, `bf16`),
- layout (`NN`, `NT`),
- workspace budget.

The aim is not exhaustive combinatorics, but a compact and interpretable map of GEMM regimes.

### Selected profiling cases

Profiler-backed case studies are defined in:

```text
configs/ncu_cases.json
```

These are intended to be few, representative, and easy to relate back to the sweep.

---

## Output structure

### Raw results

Raw outputs are written under:

```text
results/raw/<campaign_id>/
├── system/
├── hbm/
├── roofline_calibration/
├── gemm/
└── ncu/
```

Typical files include:

- `device_inventory.csv`
- `hbm_stream_triad.csv`
- `ai_sweep.csv`
- `mem_compute_mix.csv`
- `gemm_sweep.csv`
- per-case CSVs and logs
- optional profiler reports

### Processed results

Processed analysis artifacts are written under:

```text
results/processed/<campaign_id>/
├── merged/
├── roofline/
├── figures/
└── tables/
```

Typical outputs include:

- empirical roofline summary JSON,
- roofline points CSV,
- roofline figure,
- GEMM summary table,
- GEMM heatmaps,
- campaign summary text.

---

## Reproducibility conventions

Each campaign is expected to preserve:

- a unique `campaign_id`,
- separation between raw and processed outputs,
- provenance metadata,
- environment snapshots,
- deterministic input configs,
- and stable CSV contracts.

The goal is that a run can be reproduced, inspected later, and compared against future runs on the same or another system.

---

## Profiling note for shared systems

Selected Nsight Compute runs are supported by the repository, but in shared environments they may be blocked by permissions on GPU performance counters.

When that happens:

- the benchmark itself may still execute,
- the benchmark CSV may still be written,
- but Nsight Compute may fail to generate a `.ncu-rep` report.

In that case, the limitation is environmental rather than methodological.

---

## Expected usage model

This repository is intended to be understandable and useful for:

- researchers studying GPU performance,
- students reproducing a measurement campaign,
- lab members reviewing methodology,
- and HPC practitioners interested in GEMM/roofline behavior on Blackwell systems.

If you want the shortest operational summary, it is this:

```bash
bash scripts/campaign/campaign_mvp_gemm_roofline.sh <campaign_id>
```

That command is the main entry point of the repository.
