#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any
import math


@dataclass
class RooflinePoint:
    benchmark: str
    ai_flops_per_byte: float
    perf_gflops: float
    metadata: Dict[str, Any]


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0:
        return default
    return num / den


def flops_per_second_to_gflops(x: float) -> float:
    return x / 1e9


def flops_per_second_to_tflops(x: float) -> float:
    return x / 1e12


def compute_ai(flops: float, bytes_moved: float) -> float:
    return safe_div(flops, bytes_moved, 0.0)


def compute_perf_flops(flops: float, time_seconds: float) -> float:
    return safe_div(flops, time_seconds, 0.0)


def compute_perf_gflops(flops: float, time_seconds: float) -> float:
    return flops_per_second_to_gflops(compute_perf_flops(flops, time_seconds))


def compute_ridge_point_ai(peak_flops_per_s: float, peak_bw_bytes_per_s: float) -> float:
    return safe_div(peak_flops_per_s, peak_bw_bytes_per_s, 0.0)


def memory_roof_perf(ai_flops_per_byte: float, bw_bytes_per_s: float) -> float:
    return ai_flops_per_byte * bw_bytes_per_s


def roofline_perf(ai_flops_per_byte: float,
                  peak_flops_per_s: float,
                  bw_bytes_per_s: float) -> float:
    return min(peak_flops_per_s, memory_roof_perf(ai_flops_per_byte, bw_bytes_per_s))


def make_roofline_curve(ai_values: Iterable[float],
                        peak_flops_per_s: float,
                        bw_bytes_per_s: float) -> List[float]:
    return [roofline_perf(ai, peak_flops_per_s, bw_bytes_per_s) for ai in ai_values]


def logspace(start_exp: float, stop_exp: float, num: int) -> List[float]:
    if num <= 1:
        return [10 ** start_exp]
    step = (stop_exp - start_exp) / (num - 1)
    return [10 ** (start_exp + i * step) for i in range(num)]


def point_from_kernel(benchmark: str,
                      flops: float,
                      bytes_moved: float,
                      time_seconds: float,
                      metadata: Dict[str, Any] | None = None) -> RooflinePoint:
    metadata = metadata or {}
    ai = compute_ai(flops, bytes_moved)
    perf = compute_perf_gflops(flops, time_seconds)
    return RooflinePoint(
        benchmark=benchmark,
        ai_flops_per_byte=ai,
        perf_gflops=perf,
        metadata=metadata
    )