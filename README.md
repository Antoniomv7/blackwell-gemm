# blackwell-gemm

Microbenchmarking and analysis scaffold for the TFM workflow on a shared Blackwell server.

## Official workflow: VS Code Dev Container

This repository is intended to be used from a **devcontainer**. The recommended flow is:

1. On the host, choose a free physical GPU.
2. Export the selected **physical** GPU index:

```bash
export BLACKWELL_GPU_INDEX=3
code .
```

3. Reopen the project in the devcontainer.
4. Build and run benchmarks **inside** the container.

### Important indexing rule
If the container exposes a single physical GPU, then **inside the container that GPU normally appears as logical device `0`**.

So the normal pattern is:
- host: choose `BLACKWELL_GPU_INDEX=<physical index>`
- container: run benchmarks with `--device 0`

Example:

```bash
bash scripts/run/build_one.sh mb31_inventory
bash scripts/run/run_mb31.sh --device 0 --trace-seconds 3
```

## Container/toolchain expectations
The devcontainer is aligned with the validated server toolchain for Blackwell:
- CUDA 13.0 base image
- `sm_103` expected in `Makefile` and `scripts/env/env.sh`

Before opening the devcontainer, `BLACKWELL_GPU_INDEX` **must** be exported on the host.

## Core benchmark flow vs optional profiling
The repository has two layers of workflow:

### Core flow
This is the path that should work for normal benchmarking and result generation:
- build targets
- run benchmarks
- collect CSV/JSON outputs
- generate analysis artifacts

This core flow does **not** require `ncu` or `nsys`.

### Optional profiling flow
Some scripts under `scripts/profile/` and `scripts/campaign/` require:
- `ncu`
- `nsys`

Those tools are **optional but recommended**. If they are not available inside the container, the core benchmark flow can still work, but Nsight-based profiling helpers will fail.

## Environment validation
Inside the container:

```bash
source scripts/env/env.sh
bw_env_summary
bash scripts/env/check_requirements.sh
bash scripts/collect_env.sh
```

`collect_env.sh` writes by default to:

```text
results/system/env_<timestamp>.txt
```

## Build commands
List available targets:

```bash
make list
```

Build all existing targets:

```bash
make all
```

Build one target:

```bash
bash scripts/run/build_one.sh mb34_stream_triad
```

Targets intentionally deferred to Sprint 9:

```bash
make sprint9-pending
```

## Power-trace profiling
The power-trace helper verifies the binary and rebuilds it if needed:

```bash
bash scripts/profile/profile_power_trace.sh \
  --target mb34_stream_triad \
  --out-prefix triad \
  -- --device 0 --n 67108864 --block 256 --grid 120
```

To verify only and forbid automatic rebuild:

```bash
bash scripts/profile/profile_power_trace.sh \
  --target mb34_stream_triad \
  --out-prefix triad \
  --no-build \
  -- --device 0 --n 67108864 --block 256 --grid 120
```

## Power-cap campaigns
The power-cap campaigns attempt to modify the device power limit using `nvidia-smi -pl`.
On shared systems this may fail because of permissions, container restrictions, or cluster policy.

The scripts now try to restore the original limit at exit, but you should still treat these campaigns as **environment-dependent** and verify that the node policy allows them before using them in production runs.

## Notes
- `CUDA_ARCH` defaults to `sm_103` in `scripts/env/env.sh` and the `Makefile`.
- `mb33_tensor_peak_cutlass` and `mb33_layout_sensitivity` are intentionally deferred to **Sprint 9**.
- Results are written under `results/`.
- `docker/run_blackwell_container.sh` is a manual/fallback path; the official workflow is the VS Code devcontainer.
