#!/usr/bin/env python3
"""
tests/test_analysis_pipeline.py

Unit and integration tests for the statistical analysis and evidence reproduction pipeline:
  - Median calculation with even and odd number of observations.
  - Nearest-rank P95 calculation matching benchmark convention.
  - Sample standard deviation (Bessel's correction, n - 1).
  - Benjamini-Hochberg (BH) false discovery rate adjustment with monotonicity.
  - Holm step-down family-wise error rate adjustment with cumulative maximum.
  - Known test case where unadjusted Holm products are non-monotonic.
  - Matched rank-biserial correlation and directionality.
  - Cross-validation of all 27 Profile P4 summary statistics against immutable summary_results.csv.
"""

import os
import csv
import math
import pytest
import statistics
from collections import defaultdict

from scripts.analysis.analyze_phase3_sensitivity import (
    benjamini_hochberg,
    holm_step_down,
    wilcoxon_signed_rank,
    chi2_sf,
    normal_sf,
    PROJECT_ROOT
)


def test_median_even_and_odd():
    """Verify median calculation for even and odd count lists."""
    odd_list = [1.0, 3.0, 2.0]
    assert statistics.median(odd_list) == 2.0

    even_list = [1.0, 2.0, 4.0, 3.0]
    assert statistics.median(even_list) == 2.5

    # 100 elements test
    vals_100 = list(range(1, 101))
    assert statistics.median(vals_100) == 50.5


def test_nearest_rank_p95():
    """Verify nearest-rank P95 calculation matching benchmark formula."""
    # For n = 100, ceil(0.95 * 100) - 1 = 95 - 1 = 94 (0-indexed 95th element)
    values = list(range(1, 101))
    sorted_v = sorted(values)
    p95_idx = max(0, int(math.ceil(0.95 * len(values))) - 1)
    assert p95_idx == 94
    assert sorted_v[p95_idx] == 95

    # For n = 20 queries, ceil(0.95 * 20) - 1 = 19 - 1 = 18 (0-indexed 19th element)
    vals_20 = list(range(1, 21))
    sorted_20 = sorted(vals_20)
    p95_20_idx = max(0, int(math.ceil(0.95 * 20)) - 1)
    assert p95_20_idx == 18
    assert sorted_20[p95_20_idx] == 19


def test_sample_stdev_convention():
    """Verify sample standard deviation uses Bessel's correction (n - 1 denominator)."""
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    n = len(values)
    mean = sum(values) / n
    sample_var = sum((x - mean) ** 2 for x in values) / (n - 1)
    sample_sd = math.sqrt(sample_var)
    assert abs(statistics.stdev(values) - sample_sd) < 1e-12

    # Verify that population variance would differ
    pop_var = sum((x - mean) ** 2 for x in values) / n
    pop_sd = math.sqrt(pop_var)
    assert sample_sd > pop_sd


def test_benjamini_hochberg_monotonic():
    """Verify Benjamini-Hochberg adjustment enforces right-to-left cumulative minimum."""
    raw_p = [0.01, 0.04, 0.03, 0.20]
    # m = 4. Sorted:
    # rank 1: 0.01 -> 0.01 * 4 / 1 = 0.04
    # rank 2: 0.03 -> 0.03 * 4 / 2 = 0.06
    # rank 3: 0.04 -> 0.04 * 4 / 3 = 0.05333...
    # rank 4: 0.20 -> 0.20 * 4 / 4 = 0.20
    # Enforcing monotonicity backwards:
    # rank 3 is 0.0533, so rank 2 (0.06) must be clamped down to 0.05333...
    adj_p = benjamini_hochberg(raw_p)
    assert len(adj_p) == len(raw_p)
    # Check monotonicity in sorted order
    sorted_pairs = sorted(zip(raw_p, adj_p), key=lambda t: t[0])
    for i in range(len(sorted_pairs) - 1):
        assert sorted_pairs[i][1] <= sorted_pairs[i + 1][1] + 1e-12


def test_holm_step_down_cumulative_max():
    """Verify Holm step-down adjustment enforces left-to-right cumulative maximum."""
    raw_p = [0.005, 0.015, 0.02, 0.50]
    adj_p = holm_step_down(raw_p)
    assert len(adj_p) == len(raw_p)
    sorted_pairs = sorted(zip(raw_p, adj_p), key=lambda t: t[0])
    for i in range(len(sorted_pairs) - 1):
        assert sorted_pairs[i][1] <= sorted_pairs[i + 1][1] + 1e-12


def test_holm_step_down_non_monotonic_case():
    """
    Specifically test a known case where raw product p_i * (m - i + 1) is non-monotonic,
    confirming that the cumulative maximum prevents adjusted p from decreasing.
    """
    # Specifically mirror the P4 forced-HNSW comparison where rank 2 has raw p = 0.017758
    # and rank 3 has raw p = 0.021678 across m = 6 tests:
    # rank 2: 0.017758 * 5 = 0.08879
    # rank 3: 0.021678 * 4 = 0.086712 (lower than rank 2!)
    raw_p = [0.0042908, 0.017758, 0.021678, 0.064607, 0.61427, 0.80827]
    adj_p = holm_step_down(raw_p)

    # Without cumulative max, rank 3 would be 0.086712.
    # With cumulative max, rank 3 must equal rank 2's adjusted p (0.08879):
    assert abs(adj_p[0] - 0.025745) < 1e-4
    assert abs(adj_p[1] - 0.088790) < 1e-4
    assert abs(adj_p[2] - 0.088790) < 1e-4
    assert abs(adj_p[3] - 0.193821) < 1e-4
    assert abs(adj_p[4] - 1.000000) < 1e-4
    assert abs(adj_p[5] - 1.000000) < 1e-4


def test_matched_rank_biserial_direction():
    """
    Verify matched rank-biserial correlation direction:
    Negative r_rb indicates that x (e.g. FP16) has lower latency than y (e.g. FP32).
    """
    # Synthetic case where x is strictly faster than y (n = 10):
    x_fast = [1.0 + 0.05 * i for i in range(10)]
    y_slow = [2.0 + 0.05 * i for i in range(10)]
    w, p, r_rb = wilcoxon_signed_rank(x_fast, y_slow)
    assert r_rb == -1.0  # Perfect negative rank-biserial correlation
    assert p < 0.01

    # Synthetic case where x is strictly slower than y:
    w2, p2, r_rb2 = wilcoxon_signed_rank(y_slow, x_fast)
    assert r_rb2 == +1.0  # Perfect positive rank-biserial correlation

    # Symmetric case
    x_sym = [1.0, 3.0, 1.0, 3.0]
    y_sym = [3.0, 1.0, 3.0, 1.0]
    w3, p3, r_rb3 = wilcoxon_signed_rank(x_sym, y_sym)
    assert r_rb3 == 0.0


def test_p4_summary_reconciliation():
    """
    Cross-validates all 27 Profile P4 scale x representation x mode groups
    against the immutable Phase 2 summary_results.csv benchmark artifact.
    Fails loudly on any mismatch.
    """
    summary_path = os.path.join(PROJECT_ROOT, "results", "planner_sensitivity", "full_run_20261031_01", "summary_results.csv")
    latency_path = os.path.join(PROJECT_ROOT, "results", "planner_sensitivity", "full_run_20261031_01", "latency_results.csv")

    if not os.path.exists(summary_path) or not os.path.exists(latency_path):
        pytest.skip("Benchmark evidence CSVs not present locally.")

    summary_p4 = {}
    with open(summary_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["profile_id"] == "P4":
                k = (int(r["scale"]), r["representation"], r["mode"])
                summary_p4[k] = {
                    "trials": int(r["trials"]),
                    "mean": float(r["mean_latency_ms"]),
                    "median": float(r["median_latency_ms"]),
                    "p95": float(r["p95_latency_ms"])
                }

    raw_p4 = defaultdict(list)
    with open(latency_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["profile_id"] == "P4":
                k = (int(r["scale"]), r["representation"], r["mode"])
                raw_p4[k].append(float(r["latency_ms"]))

    assert len(summary_p4) == 27, f"Expected 27 P4 summary groups, found {len(summary_p4)}"

    for k, s in summary_p4.items():
        vals = raw_p4[k]
        sorted_v = sorted(vals)
        n = len(vals)
        c_mean = statistics.mean(vals)
        c_p50 = statistics.median(vals)
        p95_idx = max(0, int(math.ceil(0.95 * n)) - 1)
        c_p95 = sorted_v[p95_idx]

        assert n == s["trials"], f"Trials mismatch for {k}: {n} != {s['trials']}"
        assert abs(c_mean - s["mean"]) < 1e-4, f"Mean mismatch for {k}: {c_mean} vs {s['mean']}"
        assert abs(c_p50 - s["median"]) < 1e-4, f"Median mismatch for {k}: {c_p50} vs {s['median']}"
        assert abs(c_p95 - s["p95"]) < 1e-4, f"P95 mismatch for {k}: {c_p95} vs {s['p95']}"
