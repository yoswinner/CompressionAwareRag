#!/usr/bin/env python3
"""
scripts/analysis/analyze_phase3_sensitivity.py

Comprehensive Statistical Analysis & Evidence Synthesis for:
Precision-Aware Query Planning for Vector Retrieval in PostgreSQL

This script independently ingests the benchmark evidence from:
  - results/planner_sensitivity/storage_results.csv
  - results/planner_sensitivity/full_run_20261031_01/latency_results.csv
  - results/planner_sensitivity/full_run_20261031_01/plan_capture.jsonl
  - results/planner_sensitivity/full_run_20261031_01/diagnostic_root_buffer_summary.csv
  - results/planner_sensitivity/phase3_tie_sensitivity_20261031/tie_sensitivity_summary.csv

It calculates:
  1. Storage reductions (heap, TOAST, HNSW index, total)
  2. Planner plan-selection distributions (HNSW Index Scan vs Seq Scan) across profiles
  3. Latency distributions (P50, P95, mean, std) under canonical profile P4
  4. Retrieval quality and boundary-tie sensitivity across epsilons [1e-7, 1e-6, 1e-5]
  5. Corrected root-plan buffer consumption
  6. Non-parametric statistical tests (Friedman test across profiles, paired Wilcoxon signed-rank
     tests with Holm-Bonferroni adjustment, Cliff's delta effect sizes)

Zero database access required. Self-contained and reproducible.
"""

import os
import sys
import csv
import json
import math
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "results", "planner_sensitivity")
FULL_RUN_DIR = os.path.join(BENCHMARK_DIR, "full_run_20261031_01")
TIE_DIR = os.path.join(BENCHMARK_DIR, "phase3_tie_sensitivity_20261031")


def chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) for Chi-Square distribution with integer df."""
    if x <= 0:
        return 1.0
    # Use regularized incomplete gamma Q(s, x) = 1 - P(s, x)
    s = df / 2.0
    z = x / 2.0
    # Series expansion for lower incomplete gamma P(s, z), then 1 - P
    # For small to moderate df and z
    sum_val = 1.0 / s
    term = 1.0 / s
    for n in range(1, 100):
        term *= z / (s + n)
        sum_val += term
        if abs(term) < 1e-12:
            break
    p_lower = sum_val * math.exp(-z + s * math.log(z) - math.lgamma(s))
    sf = 1.0 - p_lower
    return max(0.0, min(1.0, sf))


def normal_sf(z: float) -> float:
    """Survival function (upper tail) for standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Computes two-sided Wilcoxon signed-rank test for paired samples.
    Returns (W_statistic, p_value).
    """
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    abs_diffs = [(abs(d), i, d > 0) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda t: t[0])

    # Assign ranks with average ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    w_pos = sum(ranks[k] for k in range(n) if abs_diffs[k][2])
    w_neg = sum(ranks[k] for k in range(n) if not abs_diffs[k][2])
    w = min(w_pos, w_neg)

    # Normal approximation with continuity correction
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (abs(w - mean_w) - 0.5) / math.sqrt(var_w)
    p_val = 2.0 * normal_sf(z)
    return w, min(1.0, p_val)


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Computes Cliff's delta effect size between two distributions."""
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    return (gt - lt) / (n_x * n_y)


def analyze_storage():
    storage_csv = os.path.join(BENCHMARK_DIR, "storage_results.csv")
    if not os.path.exists(storage_csv):
        print(f"[WARN] Storage CSV not found: {storage_csv}")
        return

    print("=" * 80)
    print("TABLE 1: PHYSICAL STORAGE METRICS AND REDUCTIONS")
    print("=" * 80)
    rows_by_scale = defaultdict(dict)
    with open(storage_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scale = int(r["scale"])
            rep = r["representation"]
            rows_by_scale[scale][rep] = {
                "heap": int(r["heap_bytes"]),
                "toast": int(r["toast_total_bytes"]),
                "hnsw": int(r["hnsw_index_bytes"]),
                "total": int(r["total_inclusive_bytes"]),
            }

    header = f"{'Scale':>6} | {'Representation':<15} | {'Heap (B)':>11} | {'TOAST (B)':>11} | {'HNSW (B)':>11} | {'Total (B)':>11} | {'Total Red. vs FP32 Ext':>22}"
    print(header)
    print("-" * len(header))

    for scale in sorted(rows_by_scale.keys()):
        base_tot = rows_by_scale[scale]["fp32_extended"]["total"]
        for rep in ["fp32_extended", "fp32_plain", "fp16_extended"]:
            d = rows_by_scale[scale][rep]
            red = (1.0 - d["total"] / base_tot) * 100.0
            red_str = f"{red:+.2f}%" if rep != "fp32_extended" else "Baseline (0.0%)"
            print(f"{scale:>6} | {rep:<15} | {d['heap']:>11,} | {d['toast']:>11,} | {d['hnsw']:>11,} | {d['total']:>11,} | {red_str:>22}")
    print()


def analyze_planner_selection():
    plan_jsonl = os.path.join(FULL_RUN_DIR, "plan_capture.jsonl")
    if not os.path.exists(plan_jsonl):
        print(f"[WARN] Plan capture not found: {plan_jsonl}")
        return

    print("=" * 80)
    print("TABLE 2: PLANNER ACCESS-PATH SELECTION (HNSW INDEX SCAN vs SEQ SCAN)")
    print("=" * 80)

    plans = {}
    with open(plan_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("mode") == "planner_selected":
                k = (int(item["scale"]), item["representation"], item["profile_id"], int(item["query_id"]))
                plans[k] = item["scan_node_type"]

    total_plans = len(plans)
    hnsw_count = sum(1 for scan in plans.values() if scan == "Index Scan")
    seq_count = sum(1 for scan in plans.values() if scan == "Seq Scan")

    print(f"Total Planner-Selected Plans: {total_plans}")
    print(f"Overall Choices: Index Scan (HNSW) = {hnsw_count:,} ({hnsw_count/total_plans*100:.1f}%), Seq Scan = {seq_count:,} ({seq_count/total_plans*100:.1f}%)\n")

    # Group by scale and profile
    by_scale_prof = defaultdict(lambda: Counter())
    for (scale, rep, prof, qid), scan in plans.items():
        by_scale_prof[(scale, prof)][scan] += 1

    profiles = [f"P{i}" for i in range(1, 10)]
    print(f"{'Scale':>6} | {'Profile':<7} | {'Index Scan (HNSW)':>18} | {'Seq Scan':>10} | {'Total':>6} | {'Index Scan %':>12}")
    print("-" * 72)
    for scale in [1001, 5000, 10000]:
        for prof in profiles:
            c = by_scale_prof[(scale, prof)]
            idx_s = c["Index Scan"]
            seq_s = c["Seq Scan"]
            tot = idx_s + seq_s
            pct = (idx_s / tot * 100.0) if tot > 0 else 0.0
            print(f"{scale:>6} | {prof:<7} | {idx_s:>18} | {seq_s:>10} | {tot:>6} | {pct:>11.1f}%")
    print()


def analyze_p4_latency():
    lat_csv = os.path.join(FULL_RUN_DIR, "latency_results.csv")
    if not os.path.exists(lat_csv):
        print(f"[WARN] Latency CSV not found: {lat_csv}")
        return

    print("=" * 80)
    print("TABLE 3: CANONICAL PROFILE P4 LATENCY SUMMARY (5 MEASURED PASSES x 20 QUERIES)")
    print("=" * 80)

    # Group by (scale, rep, mode)
    lats = defaultdict(list)
    with open(lat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["profile_id"] == "P4":
                k = (int(r["scale"]), r["representation"], r["mode"])
                lats[k].append(float(r["latency_ms"]))

    header = f"{'Scale':>6} | {'Representation':<15} | {'Mode':<17} | {'N':>4} | {'P50 (ms)':>9} | {'P95 (ms)':>9} | {'Mean (ms)':>9} | {'Std (ms)':>9}"
    print(header)
    print("-" * len(header))

    for k in sorted(lats.keys()):
        scale, rep, mode = k
        vals = sorted(lats[k])
        n = len(vals)
        p50 = vals[int(n * 0.50)]
        p95 = vals[int(n * 0.95)]
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        std = math.sqrt(var)
        print(f"{scale:>6} | {rep:<15} | {mode:<17} | {n:>4} | {p50:>9.3f} | {p95:>9.3f} | {mean:>9.3f} | {std:>9.3f}")
    print()


def analyze_boundary_ties():
    tie_csv = os.path.join(TIE_DIR, "tie_sensitivity_summary.csv")
    if not os.path.exists(tie_csv):
        print(f"[WARN] Tie sensitivity CSV not found: {tie_csv}")
        return

    print("=" * 80)
    print("TABLE 4: BOUNDARY-TIE TOLERANCE SENSITIVITY SUMMARY (ALL MODES & REPS COMBINED)")
    print("=" * 80)

    print(f"{'Scale':>6} | {'Epsilon (tol)':<13} | {'Obs':>5} | {'Tied Obs':>8} | {'Conv Recall@10':>15} | {'Tie-Aware Recall@10':>20} | {'Boost':>8}")
    print("-" * 88)

    with open(tie_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["representation"] == "ALL" and r["mode"] == "ALL":
                scale = r["scale"]
                eps = r["epsilon"]
                n = r["num_observations"]
                tied = r["tie_prevalence_count"]
                c_rec = float(r["mean_conventional_recall"])
                t_rec = float(r["mean_tie_aware_recall"])
                boost = float(r["mean_tie_recall_boost"])
                print(f"{scale:>6} | {eps:<13} | {n:>5} | {tied:>8} | {c_rec:>15.4f} | {t_rec:>20.4f} | {boost:>+8.4f}")
    print()


def analyze_root_buffers():
    buf_csv = os.path.join(FULL_RUN_DIR, "diagnostic_root_buffer_summary.csv")
    if not os.path.exists(buf_csv):
        print(f"[WARN] Root buffer CSV not found: {buf_csv}")
        return

    print("=" * 80)
    print("TABLE 5: CORRECTED ROOT-PLAN SHARED BUFFER CONSUMPTION")
    print("=" * 80)

    header = f"{'Scale':>6} | {'Representation':<15} | {'Mode':<17} | {'Mean Shared Hit':>16} | {'Mean Shared Read':>17}"
    print(header)
    print("-" * len(header))

    with open(buf_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scale = r["scale"]
            rep = r["representation"]
            mode = r["mode"]
            hit = float(r["reconstructed_mean_shared_hit_buffers"])
            read = float(r["reconstructed_mean_shared_read_buffers"])
            print(f"{scale:>6} | {rep:<15} | {mode:<17} | {hit:>16.1f} | {read:>17.2f}")
    print()


def analyze_inferential_statistics():
    lat_csv = os.path.join(FULL_RUN_DIR, "latency_results.csv")
    if not os.path.exists(lat_csv):
        return

    print("=" * 80)
    print("TABLE 6: NON-PARAMETRIC STATISTICAL INFERENCE (N=20 QUERY-LEVEL OBSERVATIONS)")
    print("=" * 80)

    # Aggregate query-level mean latency across the 5 measured passes: (scale, rep, prof, mode, qid) -> float
    q_lat = defaultdict(list)
    with open(lat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (int(r["scale"]), r["representation"], r["profile_id"], r["mode"], int(r["query_id"]))
            q_lat[k].append(float(r["latency_ms"]))

    # Mean per query
    q_mean = {k: sum(v) / len(v) for k, v in q_lat.items()}

    # 1. Friedman Test across 9 cost profiles for planner_selected mode
    print("--- 1. Friedman Test Across 9 Planner Cost Profiles (P1 - P9) ---")
    profiles = [f"P{i}" for i in range(1, 10)]
    k_prof = len(profiles)  # 9
    n_queries = 20

    for scale in [1001, 5000, 10000]:
        for rep in ["fp32_extended", "fp32_plain", "fp16_extended"]:
            # Matrix: rows=queries (20), cols=profiles (9)
            ranks_sum = [0.0] * k_prof
            for qid in range(n_queries):
                row_vals = [(q_mean[(scale, rep, prof, "planner_selected", qid)], idx) for idx, prof in enumerate(profiles)]
                row_vals.sort(key=lambda t: t[0])
                # Assign ranks 1 to 9 (handling ties if any)
                i = 0
                while i < k_prof:
                    j = i
                    while j < k_prof and row_vals[j][0] == row_vals[i][0]:
                        j += 1
                    avg_r = (i + 1 + j) / 2.0
                    for m in range(i, j):
                        orig_col = row_vals[m][1]
                        ranks_sum[orig_col] += avg_r
                    i = j

            # Friedman statistic Q
            # Q = (12 / (N * k * (k + 1))) * sum(R_j^2) - 3 * N * (k + 1)
            sum_r2 = sum(r_j ** 2 for r_j in ranks_sum)
            q_stat = (12.0 / (n_queries * k_prof * (k_prof + 1))) * sum_r2 - 3.0 * n_queries * (k_prof + 1)
            p_val = chi2_sf(q_stat, k_prof - 1)
            print(f"Scale {scale:>5} | {rep:<15} | Friedman Q = {q_stat:>7.3f} (df=8), p-value = {p_val:>8.2e}")

    # 2. Paired Wilcoxon Tests under Canonical Profile P4: FP16 vs FP32 Plain and Extended
    print("\n--- 2. Paired Wilcoxon Signed-Rank Tests (Profile P4, Forced-HNSW Mode) ---")
    comparisons = []
    for scale in [1001, 5000, 10000]:
        fp16_vals = [q_mean[(scale, "fp16_extended", "P4", "forced_hnsw", qid)] for qid in range(20)]
        fp32_ext_vals = [q_mean[(scale, "fp32_extended", "P4", "forced_hnsw", qid)] for qid in range(20)]
        fp32_pln_vals = [q_mean[(scale, "fp32_plain", "P4", "forced_hnsw", qid)] for qid in range(20)]

        # FP16 vs FP32 Ext
        w1, p1 = wilcoxon_signed_rank(fp16_vals, fp32_ext_vals)
        d1 = cliffs_delta(fp16_vals, fp32_ext_vals)
        comparisons.append((scale, "FP16 Ext vs FP32 Ext", w1, p1, d1))

        # FP16 vs FP32 Plain
        w2, p2 = wilcoxon_signed_rank(fp16_vals, fp32_pln_vals)
        d2 = cliffs_delta(fp16_vals, fp32_pln_vals)
        comparisons.append((scale, "FP16 Ext vs FP32 Plain", w2, p2, d2))

    # Apply Holm-Bonferroni correction across the 6 paired comparisons
    comparisons.sort(key=lambda t: t[3])
    m = len(comparisons)
    print(f"{'Scale':>6} | {'Comparison':<24} | {'W Stat':>7} | {'Raw p-value':>12} | {'Holm-Adj p':>11} | {'Cliff delta':>11} | {'Sig (alpha=0.05)':>16}")
    print("-" * 102)
    for rank, (scale, name, w, p_raw, delta) in enumerate(comparisons):
        adj_p = min(1.0, p_raw * (m - rank))
        sig = "Significant" if adj_p < 0.05 else "Not Significant"
        print(f"{scale:>6} | {name:<24} | {w:>7.1f} | {p_raw:>12.4e} | {adj_p:>11.4e} | {delta:>+11.2f} | {sig:>16}")
    print()


def main():
    print("=" * 80)
    print("PRECISION-AWARE QUERY PLANNING FOR VECTOR RETRIEVAL IN POSTGRESQL")
    print("COMPREHENSIVE STATISTICAL ANALYSIS & EMPIRICAL EVIDENCE SYNTHESIS")
    print("=" * 80)
    print()

    analyze_storage()
    analyze_planner_selection()
    analyze_p4_latency()
    analyze_boundary_ties()
    analyze_root_buffers()
    analyze_inferential_statistics()


if __name__ == "__main__":
    main()
