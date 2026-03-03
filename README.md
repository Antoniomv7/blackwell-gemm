# Blackwell GEMM – TFM Research Repository

**Master Thesis Project**  
**Title:** *The Anatomy of an Efficient Blackwell GEMM*  
**Author:** Antonio Moral  
**Focus:** GPU microbenchmarking, architectural characterization, and performance modeling of NVIDIA Blackwell GPUs.

---

## 🎯 Objective

This repository contains the experimental framework used to:

- Characterize NVIDIA Blackwell GPU hardware
- Extract architectural parameters (SM count, memory bandwidth, compute capability)
- Execute controlled CUDA microbenchmarks
- Generate reproducible logs for roofline modeling
- Support compute-bound vs memory-bound analysis

The goal is to establish a rigorous and reproducible methodology for GEMM performance evaluation.

---

## 🏗 Repository Structure
blackwell-gemm/
│  
├── docker/ # Docker environment definitions  
├── env/ # Environment configuration files  
├── scripts/ # Benchmark automation and helper scripts  
├── experiments/ # Experiment configurations and notes  
├── results/ # Raw benchmark outputs and hardware logs  
├── .devcontainer/ # VSCode Dev Container setup  
└── README.md


---

## 🧪 Experimental Workflow

All experiments are executed inside a Docker container to guarantee:

- Environment reproducibility  
- CUDA version consistency  
- Isolation from host dependencies  
- Deterministic benchmarking  

