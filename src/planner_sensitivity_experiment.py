#!/usr/bin/env python3
"""
src/planner_sensitivity_experiment.py

Benchmark runner for the 'Precision-Aware Query Planning for Vector Retrieval in PostgreSQL' research experiment.
Implements strictly read-only, transaction-scoped planner sensitivity benchmarking.

Key Architectural Guarantees:
  1. Strict Read-Only Mode: conn.read_only = True enforced immediately upon connection.
  2. Transactional GUC Scoping: Every query executes inside an isolated transaction with SET LOCAL
     and is unconditionally concluded with ROLLBACK in a finally block to prevent GUC leakage.
  3. Strict Separation of Concerns:
     - Phase A: Execution-Free Plan Capture via EXPLAIN (SETTINGS, FORMAT JSON).
       Incurs optimizer planning and JSON serialization overhead, but is strictly excluded from primary SELECT timing.
     - Phase B: Pure Wall-Clock Latency via parameterized SELECT with time.perf_counter().
       [Precomputed embeddings; warmups; measured passes; never preceded by EXPLAIN ANALYZE; exact K=10 LIMIT].
     - Phase C: Execution Diagnostics via EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON)
       [Executed separately after timing passes across all evaluation queries under canonical profile P4].
  4. Exact Boundary Tie Detection & Recall Evaluation:
     - Auxiliary un-timed query executed outside Phase B latency timing to identify cosine distance (<=>) ties
       at rank K vs K+1 using an exact sequential scan with plan verification.
     - Preserves conventional Recall@10 as the primary retrieval accuracy metric.
     - Retains tie indicators and computes separate Tie-Aware Recall@10 where boundary distance ties occur.
  5. Optimized Matrix Accounting:
     - Planner-selected mode: timed across all 9 cost profiles.
     - Forced exact mode: timed once per (scale, rep) under canonical profile (P4).
     - Forced HNSW mode: timed once per (scale, rep) under canonical profile (P4).
     - Full plan curves across all 9 profiles captured via execution-free Phase A EXPLAIN.
     - Total timed executions: 8,100 + 900 + 900 = 9,900 for full matrix (excludes warmups/diagnostics).
     - Diagnostic executions: 9 combinations x 3 modes x 20 queries = 540 at canonical profile P4.
  6. Reproducibility & Seeded Trial Interleaving:
     - Configurable fixed random seed (default 42).
     - Randomized trial sequencing within each measured pass to mitigate systematic ordering effects
       (such as presentation order and sequential cache warming effects) without eliminating underlying
       hardware, OS, or buffer cache effects.
     - Seed and randomized execution sequence recorded in manifest.
  7. Comprehensive Diagnostic & Workload Accounting:
     - Diagnostic runtime accounting tracks expected, attempted, successful, and failed executions.
     - Preserves explicit records for all diagnostic attempts including errors; marks dataset incomplete on any failure.
     - Separately tracks and identifies the primary latency benchmark workload.
  8. Fail-Closed Workload Validation: Compares actual timed and diagnostic executions against declared manifest.
  9. Output Persistence: Emits structured JSONL and CSV artifacts to results/planner_sensitivity/.
"""

import os
import sys
import json
import csv
import time
import math
import random
import ctypes
import platform
import statistics
import subprocess
from datetime import datetime, timezone
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_db_connection_string
from src.representations import EVALUATION_QUERIES
from src.embeddings import get_embedding

SCHEMA_NAME = "planner_sensitivity"

ALLOWED_DATABASES = {
    "compression_aware_rag_exp",
    "compression_aware_rag_scale",
    "compression_aware_rag_test"
}

DISALLOWED_DATABASES = {
    "postgres",
    "compression_aware_rag",  # Production baseline
    "smart_document_chatbot"
}

SCALE_DATABASE_MAP = {
    1001: "compression_aware_rag_exp",
    5000: "compression_aware_rag_scale",
    10000: "compression_aware_rag_scale",
    "test": "compression_aware_rag_test"
}

DEFAULT_RANDOM_PAGE_COSTS = [1.0, 4.0, 8.0]
DEFAULT_CPU_OPERATOR_COSTS = [0.0025, 0.010, 0.050]
FIXED_SEQ_PAGE_COST = 1.0
FIXED_CPU_TUPLE_COST = 0.01
FIXED_EF_SEARCH = 40
FIXED_LIMIT = 10
DEFAULT_CANONICAL_FORCED_PROFILE = "P4"  # P4 is standard PostgreSQL default (seq=1.0, rpc=4.0, cpu_op=0.0025, cpu_tuple=0.01)
DEFAULT_RANDOM_SEED = 42
COSINE_DISTANCE_TOLERANCE = 1e-6
SENSITIVITY_TOLERANCES = [1e-7, 1e-6, 1e-5]
TOLERANCE_POLICY = (
    "1.0e-6 operational numerical-equivalence threshold (with sensitivity checks across 1e-7 and 1e-5). "
    "Serves as an operational convention for identifying boundary distance ties across 768-dimensional vector dot-product "
    "and norm accumulations under hardware floating-point math, and is not a mathematically proven IEEE-754 error bound."
)


def generate_cartesian_cost_grid(
    random_page_costs: list[float] | None = None,
    cpu_operator_costs: list[float] | None = None,
    seq_page_cost: float = FIXED_SEQ_PAGE_COST,
    cpu_tuple_cost: float = FIXED_CPU_TUPLE_COST
) -> dict[str, dict]:
    """
    Generates the 9-profile Cartesian cost grid:
    random_page_cost in {1.0, 4.0, 8.0} x cpu_operator_cost in {0.0025, 0.010, 0.050}.
    Cost values represent arbitrary relative optimizer units normalized to seq_page_cost = 1.0.
    They do not represent physical disk reads (such as 8 KiB page fetches) or measured execution latency.
    """
    rpc_list = random_page_costs if random_page_costs is not None else DEFAULT_RANDOM_PAGE_COSTS
    op_list = cpu_operator_costs if cpu_operator_costs is not None else DEFAULT_CPU_OPERATOR_COSTS

    profiles = {}
    idx = 1
    for rpc in sorted(rpc_list):
        for op in sorted(op_list):
            prof_name = f"P{idx}"
            desc = f"rpc={rpc}_op={op}"
            if prof_name == "P4":
                desc += " (PostgreSQL Standard Default - Canonical Forced Profile)"
            elif prof_name == "P5":
                desc += " (Elevated CPU Operator Cost Sensitivity Setting)"
            profiles[prof_name] = {
                "profile_id": prof_name,
                "random_page_cost": float(rpc),
                "cpu_operator_cost": float(op),
                "seq_page_cost": float(seq_page_cost),
                "cpu_tuple_cost": float(cpu_tuple_cost),
                "description": desc
            }
            idx += 1
    return profiles


PLANNER_COST_PROFILES = generate_cartesian_cost_grid()


def calculate_expected_workload(
    scales: list[int | str],
    representations: list[str],
    profiles: dict[str, dict],
    modes: list[str],
    num_queries: int,
    measured_passes: int,
    warmup_passes: int,
    canonical_profile_id: str = DEFAULT_CANONICAL_FORCED_PROFILE
) -> dict:
    """
    Computes exact expected execution counts for the optimized experimental matrix:
    - Planner-selected mode: measured across all N cost profiles.
    - Forced exact mode: measured once per (scale, rep) under the canonical profile.
    - Forced HNSW mode: measured once per (scale, rep) under the canonical profile.
    - Planning-only plan captures: captured for all modes and all profiles (execution-free).
    - Diagnostic EXPLAIN ANALYZE runs: executed across all queries for all 3 modes under canonical profile.
    """
    num_scales = len(scales)
    num_reps = len(representations)
    num_profiles = len(profiles)
    rep_scale_combinations = num_scales * num_reps

    timed_planner = 0
    if "planner_selected" in modes:
        timed_planner = rep_scale_combinations * num_profiles * num_queries * measured_passes

    timed_exact = 0
    if "exact" in modes:
        timed_exact = rep_scale_combinations * 1 * num_queries * measured_passes

    timed_hnsw = 0
    if "forced_hnsw" in modes:
        timed_hnsw = rep_scale_combinations * 1 * num_queries * measured_passes

    total_timed = timed_planner + timed_exact + timed_hnsw

    warmup_planner = 0
    if "planner_selected" in modes:
        warmup_planner = rep_scale_combinations * num_profiles * num_queries * warmup_passes

    warmup_exact = 0
    if "exact" in modes:
        warmup_exact = rep_scale_combinations * 1 * num_queries * warmup_passes

    warmup_hnsw = 0
    if "forced_hnsw" in modes:
        warmup_hnsw = rep_scale_combinations * 1 * num_queries * warmup_passes

    total_warmup = warmup_planner + warmup_exact + warmup_hnsw

    # Phase A Planning-only plan captures (execution-free) across all modes, profiles, and queries
    plan_captures = rep_scale_combinations * len(modes) * num_profiles * num_queries

    # Phase C Diagnostic runs: executed for all queries across all 3 modes under the canonical profile
    diagnostic_runs = rep_scale_combinations * len(modes) * num_queries

    return {
        "num_scales": num_scales,
        "num_representations": num_reps,
        "num_profiles": num_profiles,
        "num_queries": num_queries,
        "measured_passes": measured_passes,
        "warmup_passes": warmup_passes,
        "rep_scale_combinations": rep_scale_combinations,
        "canonical_forced_profile_id": canonical_profile_id,
        "timed_planner_selected": timed_planner,
        "timed_forced_exact": timed_exact,
        "timed_forced_hnsw": timed_hnsw,
        "total_timed_executions": total_timed,
        "total_warmup_executions": total_warmup,
        "total_plan_captures": plan_captures,
        "total_diagnostic_executions": diagnostic_runs
    }


def assert_sensitivity_database(conn: psycopg.Connection, expected_db: str | None = None) -> str:
    """
    Strict fail-closed safety guard:
    Verifies that the target database is in ALLOWED_DATABASES and rejects production catalogs.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        active_db = cur.fetchone()[0]
        if active_db in DISALLOWED_DATABASES:
            raise RuntimeError(
                f"CRITICAL SAFETY VIOLATION: Database '{active_db}' is protected against benchmark execution!"
            )
        if active_db not in ALLOWED_DATABASES:
            raise RuntimeError(
                f"CRITICAL SAFETY VIOLATION: Database '{active_db}' is not an approved benchmark catalog!"
            )
        if expected_db and active_db != expected_db:
            raise RuntimeError(
                f"DATABASE MISMATCH: Connected to '{active_db}', expected '{expected_db}'."
            )
        return active_db


def get_sensitivity_connection(target_db: str) -> psycopg.Connection:
    """
    Opens a connection strictly enforcing read_only mode and disabling automatic prepared statements.
    Autocommit is enabled on the outer connection so that explicit BEGIN READ ONLY / ROLLBACK
    blocks can be managed per query.

    CRITICAL PREPARED STATEMENT SAFETY:
      prepare_threshold=None disables Psycopg 3's automatic server-side prepared statements.
      This guarantees that every parameterized query is planned freshly under its transaction-local
      planner GUCs (SET LOCAL random_page_cost, cpu_operator_cost, etc.) and prevents reuse of
      stale cached generic plans across different cost profiles or execution modes.
    """
    if target_db not in ALLOWED_DATABASES:
        raise RuntimeError(
            f"REFUSING CONNECTION: Target database '{target_db}' is not in approved list {ALLOWED_DATABASES}."
        )
    conn = psycopg.connect(
        get_db_connection_string(target_db),
        autocommit=True,
        prepare_threshold=None
    )
    conn.read_only = True
    if conn.prepare_threshold is not None:
        raise RuntimeError(
            f"PREPARE THRESHOLD SAFETY VIOLATION: Expected prepare_threshold=None on '{target_db}', "
            f"got {conn.prepare_threshold}!"
        )
    assert_sensitivity_database(conn, target_db)
    register_vector(conn)
    return conn


def parse_json_plan(plan_json_obj: list | dict) -> dict:
    """
    Robust recursive parser for PostgreSQL EXPLAIN JSON outputs.
    Traverses the plan tree to identify:
      - Root node (e.g. Limit, Sort) and its cost/timing estimates.
      - Core scan node (e.g. Seq Scan, Index Scan, Bitmap Heap Scan) and its specific cost.
      - Presence of an explicit Top-N Sort node.
      - Buffer hits, reads, dirtying, and writes (when ANALYZE, BUFFERS is active).
    """
    if isinstance(plan_json_obj, list):
        if not plan_json_obj:
            return {}
        top_wrapper = plan_json_obj[0]
    else:
        top_wrapper = plan_json_obj

    root_node = top_wrapper.get("Plan", {})
    planning_time = top_wrapper.get("Planning Time")
    execution_time = top_wrapper.get("Execution Time")
    settings = top_wrapper.get("Settings", {})

    # Traverse child plans to identify Sort and Scan nodes
    scan_node = None
    sort_node = None

    # For query-level total buffers, extract counters from the root Plan node only.
    # PostgreSQL reports cumulative buffer usage at parent nodes as well as their children;
    # summing across descendants causes double/triple counting across parent Limit/Sort/Scan nodes.
    shared_hit = root_node.get("Shared Hit Blocks", 0)
    shared_read = root_node.get("Shared Read Blocks", 0)
    shared_dirtied = root_node.get("Shared Dirtied Blocks", 0)
    shared_written = root_node.get("Shared Written Blocks", 0)

    def traverse(node):
        nonlocal scan_node, sort_node
        if not isinstance(node, dict):
            return

        node_type = node.get("Node Type", "")
        if "Sort" in node_type:
            sort_node = node
        elif any(s in node_type for s in ["Scan", "Index", "Bitmap"]):
            if scan_node is None:
                scan_node = node

        for child in node.get("Plans", []):
            traverse(child)

    traverse(root_node)

    if scan_node is None:
        scan_node = root_node

    return {
        "root_node_type": root_node.get("Node Type"),
        "root_startup_cost": float(root_node.get("Startup Cost", 0.0)),
        "root_total_cost": float(root_node.get("Total Cost", 0.0)),
        "root_plan_rows": int(root_node.get("Plan Rows", 0)),
        "root_plan_width": int(root_node.get("Plan Width", 0)),
        "scan_node_type": scan_node.get("Node Type"),
        "index_name": scan_node.get("Index Name"),
        "relation_name": scan_node.get("Relation Name"),
        "scan_startup_cost": float(scan_node.get("Startup Cost", 0.0)),
        "scan_total_cost": float(scan_node.get("Total Cost", 0.0)),
        "scan_plan_rows": int(scan_node.get("Plan Rows", 0)),
        "scan_plan_width": int(scan_node.get("Plan Width", 0)),
        "has_sort_node": (sort_node is not None),
        "actual_startup_time_ms": float(root_node.get("Actual Startup Time", 0.0)) if "Actual Startup Time" in root_node else None,
        "actual_total_time_ms": float(root_node.get("Actual Total Time", 0.0)) if "Actual Total Time" in root_node else None,
        "actual_rows": int(root_node.get("Actual Rows", 0)) if "Actual Rows" in root_node else None,
        "shared_hit_buffers": int(shared_hit),
        "shared_read_buffers": int(shared_read),
        "shared_dirtied_buffers": int(shared_dirtied),
        "shared_written_buffers": int(shared_written),
        "planning_time_ms": float(planning_time) if planning_time is not None else None,
        "execution_time_ms": float(execution_time) if execution_time is not None else None,
        "settings": settings,
        "raw_plan": plan_json_obj
    }


def get_host_memory_info() -> dict:
    """Captures total and available physical RAM on Windows/Linux host."""
    mem_info = {"total_gb": None, "available_gb": None}
    if platform.system() == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                mem_info["total_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 2)
                mem_info["available_gb"] = round(stat.ullAvailPhys / (1024 ** 3), 2)
        except Exception:
            pass
    return mem_info


def capture_environment_snapshot(conn: psycopg.Connection) -> dict:
    """
    Captures complete host, PostgreSQL engine, pgvector extension, and planner GUC snapshot.
    Reads pg_settings directly to record effective configurations.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        pg_version = cur.fetchone()[0]

        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        ext_row = cur.fetchone()
        vector_version = ext_row[0] if ext_row else "missing"

        cur.execute("SELECT current_database();")
        db_name = cur.fetchone()[0]

        # Read core planner, parallel, and resource settings directly from pg_settings
        guc_names = [
            "shared_buffers", "work_mem", "maintenance_work_mem",
            "random_page_cost", "seq_page_cost", "cpu_tuple_cost",
            "cpu_index_tuple_cost", "cpu_operator_cost", "effective_cache_size",
            "enable_seqscan", "enable_indexscan", "enable_bitmapscan",
            "max_parallel_workers_per_gather", "max_parallel_workers",
            "max_parallel_maintenance_workers"
        ]
        cur.execute("""
            SELECT name, setting, unit, boot_val, reset_val, source
            FROM pg_settings
            WHERE name = ANY(%s);
        """, (guc_names,))
        guc_rows = cur.fetchall()
        pg_settings_map = {
            r[0]: {
                "setting": r[1],
                "unit": r[2],
                "boot_val": r[3],
                "reset_val": r[4],
                "source": r[5]
            }
            for r in guc_rows
        }

        # Check hnsw.ef_search
        try:
            cur.execute("SHOW hnsw.ef_search;")
            ef_search_active = cur.fetchone()[0]
        except Exception:
            ef_search_active = "unknown"

    # Git metadata
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass

    host_mem = get_host_memory_info()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_name,
        "postgresql_version": pg_version,
        "pgvector_version": vector_version,
        "python_version": platform.python_version(),
        "psycopg_version": psycopg.__version__,
        "prepare_threshold": getattr(conn, "prepare_threshold", None),
        "automatic_prepared_statements_disabled": (getattr(conn, "prepare_threshold", None) is None),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "physical_memory_gb": host_mem["total_gb"],
            "available_memory_gb": host_mem["available_gb"]
        },
        "git_commit": git_commit,
        "active_pg_settings": pg_settings_map,
        "active_hnsw_ef_search": ef_search_active,
        "baseline_random_page_cost": pg_settings_map.get("random_page_cost", {}).get("setting")
    }


def calculate_recall_metrics(
    ref_results: list[dict],
    cand_results: list[dict],
    gt_tie_group: dict | None = None,
    gt_tie_info: dict | None = None,
    top_k: int = FIXED_LIMIT,
    tolerance: float = COSINE_DISTANCE_TOLERANCE
) -> dict:
    """
    Computes both conventional Recall@K and Tie-Aware Recall@K using a disjoint-set formulation.

    Mathematical Formulation:
      Let:
        - B = all corpus documents strictly closer than the FP32 exact top-K boundary distance D:
              B = { c in Corpus | dist_FP32(c, q) < D - tolerance }
        - T = the complete set of documents tied at the FP32 exact boundary distance D:
              T = { c in Corpus | |dist_FP32(c, q) - D| <= tolerance }
        - S = the candidate top-K document IDs (retrieved by candidate method, |S| <= K).
        - r = K - |B| (the exact number of top-K slots available for boundary-tied documents).

      Disjoint-Set Proof:
        By construction, for any c in B, dist(c) < D - tolerance.
        For any c in T, D - tolerance <= dist(c) <= D + tolerance.
        Therefore B intersect T = empty set.
        Consequently, (S intersect B) and (S intersect T) are strictly disjoint subsets of S,
        eliminating any possibility of double-counting.

      Metric Definitions:
        1. Conventional Recall@K (Primary Benchmark Metric):
           conventional_recall = |S intersect Top-K_ref| / K
           Evaluates strict ID agreement against the arbitrarily tie-broken ground truth.

        2. Tie-Aware Recall@K (Auxiliary Discrepancy Metric):
           tie_aware_recall = (|S intersect B| + min(|S intersect T|, r)) / K
           Recognizes equally distant alternatives at the boundary without treating them as errors.

      Guarantees & Properties:
        1. When |T| <= r (no tie crosses into rank K+1), tie_aware_recall == conventional_recall.
        2. tie_aware_recall is mathematically bounded in [0.0, 1.0] since |S intersect B| <= |B| = K - r
           and min(|S intersect T|, r) <= r, so numerator <= (K - r) + r = K.
        3. Missing any strictly closer document (c in B) is always penalized.
        4. Membership of candidate documents in B or T is strictly determined from FP32 reference distances,
           insulating FP16 (halfvec) evaluations from quantization noise.
    """
    if top_k <= 0 or not ref_results:
        return {
            "conventional_recall": 0.0,
            "tie_aware_recall": 0.0,
            "has_boundary_tie": False,
            "boundary_distance": None,
            "size_B": 0,
            "size_T": 0,
            "available_tie_slots": 0,
            "hits_B": 0,
            "hits_T": 0,
            "credited_T": 0,
            "conventional_hits": 0,
            "tie_aware_hits": 0,
            "tolerance": tolerance
        }

    # S: Candidate top-K chunk IDs
    S = {c["chunk_id"] for c in cand_results[:top_k]}

    # Top-K_ref: Conventional reference top-K chunk IDs
    ref_ids = {r["chunk_id"] for r in ref_results[:top_k]}
    conventional_hits = len(S.intersection(ref_ids))
    conventional_recall = conventional_hits / float(top_k)

    if gt_tie_group is None and gt_tie_info is not None:
        gt_tie_group = gt_tie_info

    if not gt_tie_group or "B_ids" not in gt_tie_group or "T_ids" not in gt_tie_group:
        # Fallback if no tie group provided
        sens_recalls = {f"{t:.1e}": conventional_recall for t in SENSITIVITY_TOLERANCES}
        sens_ties = {f"{t:.1e}": False for t in SENSITIVITY_TOLERANCES}
        return {
            "conventional_recall": conventional_recall,
            "tie_aware_recall": conventional_recall,
            "has_boundary_tie": False,
            "boundary_distance": None,
            "size_B": len(ref_ids),
            "size_T": 0,
            "available_tie_slots": 0,
            "hits_B": conventional_hits,
            "hits_T": 0,
            "credited_T": 0,
            "conventional_hits": conventional_hits,
            "tie_aware_hits": conventional_hits,
            "tolerance": tolerance,
            "sensitivity_recalls": sens_recalls,
            "sensitivity_ties": sens_ties,
            "sensitivity_tolerances": [f"{t:.1e}" for t in SENSITIVITY_TOLERANCES]
        }

    B = set(gt_tie_group["B_ids"])
    T = set(gt_tie_group["T_ids"])
    D = gt_tie_group.get("boundary_distance")
    r = gt_tie_group.get("available_tie_slots", max(0, top_k - len(B)))
    has_boundary_tie = bool(gt_tie_group.get("has_boundary_tie", len(T) > r))

    # If no boundary tie exists and B was not populated, tie-aware recall simplifies exactly to conventional Recall@K
    if not has_boundary_tie and not gt_tie_group.get("has_tie", False) and len(B) == 0:
        sens_recalls = {f"{t:.1e}": conventional_recall for t in SENSITIVITY_TOLERANCES}
        sens_ties = {f"{t:.1e}": False for t in SENSITIVITY_TOLERANCES}
        return {
            "conventional_recall": conventional_recall,
            "tie_aware_recall": conventional_recall,
            "has_boundary_tie": False,
            "boundary_distance": D,
            "size_B": len(ref_ids),
            "size_T": len(T),
            "available_tie_slots": r,
            "hits_B": conventional_hits,
            "hits_T": 0,
            "credited_T": 0,
            "delta": gt_tie_group.get("delta") if gt_tie_group else None,
            "conventional_hits": conventional_hits,
            "tie_aware_hits": conventional_hits,
            "tolerance": tolerance,
            "sensitivity_recalls": sens_recalls,
            "sensitivity_ties": sens_ties,
            "sensitivity_tolerances": [f"{t:.1e}" for t in SENSITIVITY_TOLERANCES]
        }

    # Disjoint intersections
    hits_B = len(S.intersection(B))
    hits_T = len(S.intersection(T))
    credited_T = min(hits_T, r)
    tie_aware_hits = hits_B + credited_T
    tie_aware_recall = min(1.0, tie_aware_hits / float(top_k))

    # Tolerance sensitivity analysis across predefined robustness thresholds (e.g. 1e-7, 1e-6, 1e-5)
    sensitivity_recalls = {}
    sensitivity_ties = {}
    sens_data = gt_tie_group.get("sensitivity_results", {})
    if sens_data:
        for tol_k, s_info in sens_data.items():
            b_set = set(s_info.get("B_ids", []))
            t_set = set(s_info.get("T_ids", []))
            r_sl = s_info.get("available_tie_slots", max(0, top_k - len(b_set)))
            h_b = len(S.intersection(b_set))
            h_t = len(S.intersection(t_set))
            c_t = min(h_t, r_sl)
            sensitivity_recalls[tol_k] = min(1.0, (h_b + c_t) / float(top_k))
            sensitivity_ties[tol_k] = bool(s_info.get("has_boundary_tie", len(t_set) > r_sl))
    else:
        for tol_v in SENSITIVITY_TOLERANCES:
            tol_k = f"{tol_v:.1e}"
            sensitivity_recalls[tol_k] = tie_aware_recall
            sensitivity_ties[tol_k] = has_boundary_tie

    return {
        "conventional_recall": conventional_recall,
        "tie_aware_recall": tie_aware_recall,
        "has_boundary_tie": has_boundary_tie,
        "boundary_distance": D,
        "size_B": len(B),
        "size_T": len(T),
        "available_tie_slots": r,
        "hits_B": hits_B,
        "hits_T": hits_T,
        "credited_T": credited_T,
        "delta": gt_tie_group.get("delta") if gt_tie_group else None,
        "conventional_hits": conventional_hits,
        "tie_aware_hits": tie_aware_hits,
        "tolerance": tolerance,
        "sensitivity_recalls": sensitivity_recalls,
        "sensitivity_ties": sensitivity_ties,
        "sensitivity_tolerances": [f"{t:.1e}" for t in SENSITIVITY_TOLERANCES]
    }


def calculate_recall_at_k(ref_results: list[dict], cand_results: list[dict], top_k: int = FIXED_LIMIT) -> float:
    """
    Conventional Recall@K = |Top-K_ref intersect Top-K_cand| / K.
    Preserved as the primary retrieval accuracy metric.
    """
    metrics = calculate_recall_metrics(ref_results, cand_results, gt_tie_group=None, top_k=top_k)
    return metrics["conventional_recall"]


def retrieve_exact_ground_truth(
    conn: psycopg.Connection,
    query_vector: list[float],
    schema_name: str,
    table_name: str,
    cast_type: str,
    profile_params: dict,
    top_k: int = FIXED_LIMIT
) -> list[dict]:
    """
    Retrieves exact FP32 ground-truth candidate list with mandatory execution plan verification.

    Guarantees:
      1. Executes inside an isolated transaction with SET LOCAL.
      2. Enforces exact sequential scan:
           SET LOCAL enable_indexscan = off;
           SET LOCAL enable_bitmapscan = off;
           SET LOCAL enable_seqscan = on;
           SET LOCAL max_parallel_workers_per_gather = 0;
      3. Captures execution plan via EXPLAIN (FORMAT JSON) with prepare=False.
      4. Explicitly asserts scan_node_type == 'Seq Scan'. Fails closed if any index scan or non-sequential scan is used.
      5. Executes the timed-equivalent SELECT query with prepare=False.
      6. Unconditionally concludes with ROLLBACK.
    """
    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY;")
        try:
            cur.execute(sql.SQL("SET LOCAL seq_page_cost = {};").format(sql.Literal(profile_params["seq_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL random_page_cost = {};").format(sql.Literal(profile_params["random_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_operator_cost = {};").format(sql.Literal(profile_params["cpu_operator_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_tuple_cost = {};").format(sql.Literal(profile_params["cpu_tuple_cost"])))
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")
            cur.execute("SET LOCAL enable_indexscan = off;")
            cur.execute("SET LOCAL enable_bitmapscan = off;")
            cur.execute("SET LOCAL enable_seqscan = on;")

            # 1. Mandatory Plan Verification
            explain_sql = sql.SQL("""
                EXPLAIN (FORMAT JSON)
                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                FROM {schema}.{table}
                ORDER BY distance ASC
                LIMIT %s;
            """).format(
                cast=sql.SQL(cast_type),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(explain_sql, (query_vector, top_k), prepare=False)
            plan_json = cur.fetchone()[0]
            parsed_plan = parse_json_plan(plan_json)
            actual_scan = parsed_plan.get("scan_node_type")
            if actual_scan != "Seq Scan":
                raise RuntimeError(
                    f"GROUND TRUTH SCAN VERIFICATION FAILURE on '{schema_name}.{table_name}': "
                    f"Ground-truth query must strictly execute via 'Seq Scan', but planner selected '{actual_scan}'! "
                    f"Failing closed to prevent corrupted ground-truth baseline."
                )

            # 2. Query Execution with prepare=False
            query_sql = sql.SQL("""
                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                FROM {schema}.{table}
                ORDER BY distance ASC
                LIMIT %s;
            """).format(
                cast=sql.SQL(cast_type),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(query_sql, (query_vector, top_k), prepare=False)
            rows = cur.fetchall()
            return [{"chunk_id": r[0], "distance": float(r[1])} for r in rows]
        finally:
            cur.execute("ROLLBACK;")


def retrieve_boundary_tie_group(
    conn: psycopg.Connection,
    query_vector: list[float],
    schema_name: str,
    table_name: str,
    top_k: int = FIXED_LIMIT,
    boundary_distance: float | None = None,
    tolerance: float = COSINE_DISTANCE_TOLERANCE
) -> dict:
    """
    Identifies the complete boundary tie group using an exact sequential scan on the FP32 reference table.

    Procedure:
      1. Runs inside an isolated read-only transaction with unconditional ROLLBACK.
      2. Strictly enforces exact sequential scan:
           SET LOCAL enable_indexscan = off;
           SET LOCAL enable_bitmapscan = off;
           SET LOCAL enable_seqscan = on;
           SET LOCAL max_parallel_workers_per_gather = 0;
      3. Establishes the FP32 exact rank-K boundary distance D (0-indexed OFFSET top_k - 1 LIMIT 1).
      4. Captures and verifies the execution plan of the tie query via EXPLAIN (FORMAT JSON),
         strictly asserting that scan_node_type == "Seq Scan".
      5. Executes the exact sequential scan to retrieve all rows with cosine distance <= D + tolerance.
      6. Constructs mutually disjoint sets:
           - B: chunk IDs strictly closer than the boundary (dist < D - tolerance)
           - T: chunk IDs tied at the boundary (|dist - D| <= tolerance)
      7. Computes r = max(0, top_k - |B|) (available boundary slots).
      8. Detects whether a tie crosses the top-K boundary: has_boundary_tie = (|T| > r).
    """
    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY;")
        try:
            cur.execute("SET LOCAL enable_indexscan = off;")
            cur.execute("SET LOCAL enable_bitmapscan = off;")
            cur.execute("SET LOCAL enable_seqscan = on;")
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")

            # 1. Establish the FP32 exact rank-K boundary distance D
            if boundary_distance is not None:
                D = float(boundary_distance)
            else:
                boundary_sql = sql.SQL("""
                    SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                    FROM {schema}.{table}
                    ORDER BY distance ASC
                    LIMIT 1 OFFSET %s;
                """).format(
                    schema=sql.Identifier(schema_name),
                    table=sql.Identifier(table_name)
                )
                cur.execute(boundary_sql, (query_vector, top_k - 1), prepare=False)
                b_row = cur.fetchone()
                if not b_row:
                    return {
                        "boundary_distance": None,
                        "tolerance": tolerance,
                        "size_B": 0,
                        "size_T": 0,
                        "available_tie_slots": top_k,
                        "has_boundary_tie": False,
                        "has_tie": False,
                        "delta": None,
                        "B_ids": set(),
                        "T_ids": set(),
                        "B_ids_list": [],
                        "T_ids_list": [],
                        "sensitivity_results": {},
                        "scan_node_type": "Seq Scan"
                    }

                # Rigorous validation replacing any mock fallback
                if not isinstance(b_row, (tuple, list)) or len(b_row) < 2:
                    raise ValueError(
                        f"Boundary query on '{schema_name}.{table_name}' returned malformed row structure: "
                        f"expected (chunk_id, numeric distance), got {b_row!r}."
                    )
                chunk_id_val, dist_val = b_row[0], b_row[1]
                if isinstance(chunk_id_val, (dict, list)) or isinstance(dist_val, (dict, list)):
                    raise ValueError(
                        f"Boundary query on '{schema_name}.{table_name}' returned unexpected container types: "
                        f"chunk_id={chunk_id_val!r}, distance={dist_val!r}. Dicts/lists are not valid row data."
                    )
                try:
                    D = float(dist_val)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Boundary query on '{schema_name}.{table_name}' returned non-numeric distance: {dist_val!r}."
                    ) from e

                if not math.isfinite(D):
                    raise ValueError(
                        f"Boundary query on '{schema_name}.{table_name}' returned non-finite distance: {D}."
                    )

            # 2. Plan capture and verification for the tie query
            explain_sql = sql.SQL("""
                EXPLAIN (FORMAT JSON)
                SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                FROM {schema}.{table}
                WHERE (embedding <=> %s::vector(768)) <= %s
                ORDER BY distance ASC;
            """).format(
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(explain_sql, (query_vector, query_vector, D + tolerance), prepare=False)
            raw_plan = cur.fetchone()[0]
            parsed_plan = parse_json_plan(raw_plan if isinstance(raw_plan, list) else json.loads(raw_plan))
            scan_type = parsed_plan.get("scan_node_type")
            if scan_type != "Seq Scan":
                raise RuntimeError(
                    f"EXACT TIE GROUP SCAN ASSERTION FAILED on '{table_name}': Expected 'Seq Scan', "
                    f"got '{scan_type}' (index: '{parsed_plan.get('index_name')}'). "
                    f"The auxiliary boundary tie detection must use an exact sequential scan."
                )

            # 3. Retrieve all corpus rows up to D + max(SENSITIVITY_TOLERANCES)
            max_tol = max(SENSITIVITY_TOLERANCES) if SENSITIVITY_TOLERANCES else tolerance
            group_sql = sql.SQL("""
                SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                FROM {schema}.{table}
                WHERE (embedding <=> %s::vector(768)) <= %s
                ORDER BY distance ASC;
            """).format(
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(group_sql, (query_vector, query_vector, D + max_tol), prepare=False)
            rows = cur.fetchall()

            items_with_distance = []
            for r in rows:
                if isinstance(r, (tuple, list)) and len(r) >= 2:
                    try:
                        d_f = float(r[1])
                        if math.isfinite(d_f):
                            items_with_distance.append((r[0], d_f))
                    except (TypeError, ValueError):
                        pass

            B_ids = {cid for cid, dist in items_with_distance if dist < (D - tolerance)}
            T_ids = {cid for cid, dist in items_with_distance if abs(dist - D) <= tolerance}

            r_slots = max(0, top_k - len(B_ids))
            has_boundary_tie = len(T_ids) > r_slots

            # Robustness sensitivity analysis across tolerances (e.g. 1e-7, 1e-6, 1e-5)
            sensitivity_results = {}
            for tol_val in (SENSITIVITY_TOLERANCES or [tolerance]):
                b_set = {cid for cid, dist in items_with_distance if dist < (D - tol_val)}
                t_set = {cid for cid, dist in items_with_distance if abs(dist - D) <= tol_val}
                r_sl = max(0, top_k - len(b_set))
                has_tie_tol = len(t_set) > r_sl
                tol_key = f"{tol_val:.1e}"
                sensitivity_results[tol_key] = {
                    "tolerance": tol_val,
                    "has_boundary_tie": has_tie_tol,
                    "size_B": len(b_set),
                    "size_T": len(t_set),
                    "available_tie_slots": r_sl,
                    "B_ids": b_set,
                    "T_ids": t_set
                }

            return {
                "boundary_distance": D,
                "tolerance": tolerance,
                "size_B": len(B_ids),
                "size_T": len(T_ids),
                "available_tie_slots": r_slots,
                "has_boundary_tie": has_boundary_tie,
                "has_tie": has_boundary_tie,
                "delta": None,
                "B_ids": B_ids,
                "T_ids": T_ids,
                "B_ids_list": sorted(list(B_ids)),
                "T_ids_list": sorted(list(T_ids)),
                "sensitivity_results": sensitivity_results,
                "scan_node_type": scan_type
            }
        finally:
            cur.execute("ROLLBACK;")


def check_boundary_distance_tie(
    conn: psycopg.Connection,
    query_vector: list[float],
    schema_name: str,
    table_name: str,
    cast_type: str = "vector(768)",
    profile_params: dict | None = None,
    mode: str = "exact",
    top_k: int = FIXED_LIMIT,
    tolerance: float = COSINE_DISTANCE_TOLERANCE
) -> dict:
    """
    Auxiliary un-timed query executed outside latency measurements to detect cosine distance (<=>)
    ties between rank K and rank K+1 using an exact sequential scan with plan verification.
    """
    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY;")
        try:
            # Strictly enforce exact sequential scan path
            cur.execute("SET LOCAL enable_indexscan = off;")
            cur.execute("SET LOCAL enable_bitmapscan = off;")
            cur.execute("SET LOCAL enable_seqscan = on;")
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")

            if profile_params:
                cur.execute(sql.SQL("SET LOCAL seq_page_cost = {};").format(
                    sql.Literal(profile_params.get("seq_page_cost", FIXED_SEQ_PAGE_COST))
                ))
                cur.execute(sql.SQL("SET LOCAL cpu_operator_cost = {};").format(
                    sql.Literal(profile_params.get("cpu_operator_cost", 0.0025))
                ))
                cur.execute(sql.SQL("SET LOCAL cpu_tuple_cost = {};").format(
                    sql.Literal(profile_params.get("cpu_tuple_cost", FIXED_CPU_TUPLE_COST))
                ))

            # Step 1: Capture and verify plan
            explain_sql = sql.SQL("""
                EXPLAIN (FORMAT JSON)
                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                FROM {schema}.{table}
                ORDER BY distance ASC
                LIMIT 2 OFFSET %s;
            """).format(
                cast=sql.SQL(cast_type),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(explain_sql, (query_vector, top_k - 1), prepare=False)
            raw_plan = cur.fetchone()[0]
            parsed_plan = parse_json_plan(raw_plan if isinstance(raw_plan, list) else json.loads(raw_plan))
            scan_type = parsed_plan.get("scan_node_type")
            if scan_type != "Seq Scan":
                raise RuntimeError(
                    f"EXACT TIE CHECK SCAN ASSERTION FAILED on '{table_name}': Expected 'Seq Scan', "
                    f"got '{scan_type}' (index: '{parsed_plan.get('index_name')}'). "
                    f"The auxiliary boundary tie check must use exact sequential scan."
                )

            # Step 2: Execute exact sequential scan query
            tie_sql = sql.SQL("""
                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                FROM {schema}.{table}
                ORDER BY distance ASC
                LIMIT 2 OFFSET %s;
            """).format(
                cast=sql.SQL(cast_type),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(tie_sql, (query_vector, top_k - 1), prepare=False)
            rows = cur.fetchall()

            if len(rows) >= 2:
                r10_id, r10_dist = rows[0][0], float(rows[0][1])
                r11_id, r11_dist = rows[1][0], float(rows[1][1])
                delta = abs(r10_dist - r11_dist)
                has_tie = (delta <= tolerance)
            elif len(rows) == 1:
                r10_id, r10_dist = rows[0][0], float(rows[0][1])
                r11_id, r11_dist, delta = None, None, None
                has_tie = False
            else:
                r10_id, r10_dist = None, None
                r11_id, r11_dist, delta = None, None, None
                has_tie = False

            # Construct complete tie group and strictly better set B
            sensitivity_results = {}
            if r10_dist is not None:
                max_tol = max(SENSITIVITY_TOLERANCES) if SENSITIVITY_TOLERANCES else tolerance
                group_sql = sql.SQL("""
                    SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                    FROM {schema}.{table}
                    WHERE (embedding <=> %s::{cast}) <= %s
                    ORDER BY distance ASC;
                """).format(
                    cast=sql.SQL(cast_type),
                    schema=sql.Identifier(schema_name),
                    table=sql.Identifier(table_name)
                )
                cur.execute(group_sql, (query_vector, query_vector, r10_dist + max_tol), prepare=False)
                group_rows = cur.fetchall()
                items_with_distance = []
                for r in group_rows:
                    if isinstance(r, (tuple, list)) and len(r) >= 2:
                        try:
                            d_f = float(r[1])
                            if math.isfinite(d_f):
                                items_with_distance.append((r[0], d_f))
                        except (TypeError, ValueError):
                            pass

                B_ids = {cid for cid, dist in items_with_distance if dist < (r10_dist - tolerance)}
                T_ids = {cid for cid, dist in items_with_distance if abs(dist - r10_dist) <= tolerance}
                r_slots = max(0, top_k - len(B_ids))

                for tol_val in (SENSITIVITY_TOLERANCES or [tolerance]):
                    b_set = {cid for cid, dist in items_with_distance if dist < (r10_dist - tol_val)}
                    t_set = {cid for cid, dist in items_with_distance if abs(dist - r10_dist) <= tol_val}
                    r_sl = max(0, top_k - len(b_set))
                    has_tie_tol = len(t_set) > r_sl
                    tol_key = f"{tol_val:.1e}"
                    sensitivity_results[tol_key] = {
                        "tolerance": tol_val,
                        "has_boundary_tie": has_tie_tol,
                        "size_B": len(b_set),
                        "size_T": len(t_set),
                        "available_tie_slots": r_sl,
                        "B_ids": b_set,
                        "T_ids": t_set
                    }
            else:
                B_ids = set()
                T_ids = {r10_id} if r10_id is not None else set()
                r_slots = 1

            return {
                "has_tie": bool(has_tie),
                "has_boundary_tie": bool(has_tie),
                "boundary_distance": r10_dist,
                "rank_10_chunk_id": r10_id,
                "rank_10_distance": r10_dist,
                "rank_11_chunk_id": r11_id,
                "rank_11_distance": r11_dist,
                "delta": delta,
                "size_B": len(B_ids),
                "size_T": len(T_ids),
                "available_tie_slots": r_slots,
                "B_ids": B_ids,
                "T_ids": T_ids,
                "sensitivity_results": sensitivity_results,
                "scan_node_type": scan_type
            }
        finally:
            cur.execute("ROLLBACK;")


def capture_query_plan(
    conn: psycopg.Connection,
    query_vector: list[float],
    schema_name: str,
    table_name: str,
    cast_type: str,
    profile_params: dict,
    mode: str = "planner_selected",
    top_k: int = FIXED_LIMIT
) -> dict:
    """
    Phase A: Planning-Only EXPLAIN (Execution-Free Plan Capture).
    Executes EXPLAIN (SETTINGS, FORMAT JSON) strictly without ANALYZE or query execution.
    Incurs server-side planning and JSON serialization overhead, but is strictly excluded from Phase B timing.
    Transaction is unconditionally rolled back.
    """
    plan_dict = {}
    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY;")
        try:
            cur.execute(sql.SQL("SET LOCAL seq_page_cost = {};").format(sql.Literal(profile_params["seq_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL random_page_cost = {};").format(sql.Literal(profile_params["random_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_operator_cost = {};").format(sql.Literal(profile_params["cpu_operator_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_tuple_cost = {};").format(sql.Literal(profile_params["cpu_tuple_cost"])))
            cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {};").format(sql.Literal(FIXED_EF_SEARCH)))
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")

            if mode == "exact":
                cur.execute("SET LOCAL enable_indexscan = off;")
                cur.execute("SET LOCAL enable_bitmapscan = off;")
                cur.execute("SET LOCAL enable_seqscan = on;")
            elif mode == "forced_hnsw":
                cur.execute("SET LOCAL enable_indexscan = on;")
                cur.execute("SET LOCAL enable_bitmapscan = on;")
                cur.execute("SET LOCAL enable_seqscan = off;")
            else:  # planner_selected
                cur.execute("SET LOCAL enable_indexscan = on;")
                cur.execute("SET LOCAL enable_bitmapscan = on;")
                cur.execute("SET LOCAL enable_seqscan = on;")

            explain_sql = sql.SQL("""
                EXPLAIN (SETTINGS, FORMAT JSON)
                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                FROM {schema}.{table}
                ORDER BY distance ASC
                LIMIT %s;
            """).format(
                cast=sql.SQL(cast_type),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(explain_sql, (query_vector, top_k), prepare=False)
            raw_res = cur.fetchone()[0]
            parsed = parse_json_plan(raw_res if isinstance(raw_res, list) else json.loads(raw_res))
            plan_dict = parsed
        finally:
            cur.execute("ROLLBACK;")

    return plan_dict


def measure_query_latency(
    conn: psycopg.Connection,
    query_vector: list[float],
    schema_name: str,
    table_name: str,
    cast_type: str,
    profile_params: dict,
    mode: str = "planner_selected",
    top_k: int = FIXED_LIMIT
) -> tuple[list[dict], float]:
    """
    Phase B: Pure Performance Measurement.
    Executes the ordinary parameterized SELECT inside an isolated transaction with SET LOCAL.
    Measures round-trip execution latency with time.perf_counter().
    Fetches exactly top_k (10) results; never preceded by EXPLAIN ANALYZE. Concludes with ROLLBACK.
    """
    results = []
    elapsed_ms = 0.0

    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY;")
        try:
            cur.execute(sql.SQL("SET LOCAL seq_page_cost = {};").format(sql.Literal(profile_params["seq_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL random_page_cost = {};").format(sql.Literal(profile_params["random_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_operator_cost = {};").format(sql.Literal(profile_params["cpu_operator_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_tuple_cost = {};").format(sql.Literal(profile_params["cpu_tuple_cost"])))
            cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {};").format(sql.Literal(FIXED_EF_SEARCH)))
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")

            if mode == "exact":
                cur.execute("SET LOCAL enable_indexscan = off;")
                cur.execute("SET LOCAL enable_bitmapscan = off;")
                cur.execute("SET LOCAL enable_seqscan = on;")
            elif mode == "forced_hnsw":
                cur.execute("SET LOCAL enable_indexscan = on;")
                cur.execute("SET LOCAL enable_bitmapscan = on;")
                cur.execute("SET LOCAL enable_seqscan = off;")
            else:  # planner_selected
                cur.execute("SET LOCAL enable_indexscan = on;")
                cur.execute("SET LOCAL enable_bitmapscan = on;")
                cur.execute("SET LOCAL enable_seqscan = on;")

            query_sql = sql.SQL("""
                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                FROM {schema}.{table}
                ORDER BY distance ASC
                LIMIT %s;
            """).format(
                cast=sql.SQL(cast_type),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )

            t0 = time.perf_counter()
            cur.execute(query_sql, (query_vector, top_k), prepare=False)  # exactly top_k (10)
            rows = cur.fetchall()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            results = [{"chunk_id": r[0], "distance": float(r[1])} for r in rows]
        finally:
            cur.execute("ROLLBACK;")

    return results, elapsed_ms


def execute_diagnostic_pass(
    conn: psycopg.Connection,
    query_vector: list[float],
    schema_name: str,
    table_name: str,
    cast_type: str,
    profile_params: dict,
    mode: str = "planner_selected",
    top_k: int = FIXED_LIMIT
) -> dict:
    """
    Phase C: Diagnostic Inspection.
    Executes EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON) strictly AFTER the timed SELECT trials.
    Concludes with ROLLBACK.
    """
    diagnostic_dict = {}
    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY;")
        try:
            cur.execute(sql.SQL("SET LOCAL seq_page_cost = {};").format(sql.Literal(profile_params["seq_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL random_page_cost = {};").format(sql.Literal(profile_params["random_page_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_operator_cost = {};").format(sql.Literal(profile_params["cpu_operator_cost"])))
            cur.execute(sql.SQL("SET LOCAL cpu_tuple_cost = {};").format(sql.Literal(profile_params["cpu_tuple_cost"])))
            cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {};").format(sql.Literal(FIXED_EF_SEARCH)))
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")

            if mode == "exact":
                cur.execute("SET LOCAL enable_indexscan = off;")
                cur.execute("SET LOCAL enable_bitmapscan = off;")
                cur.execute("SET LOCAL enable_seqscan = on;")
            elif mode == "forced_hnsw":
                cur.execute("SET LOCAL enable_indexscan = on;")
                cur.execute("SET LOCAL enable_bitmapscan = on;")
                cur.execute("SET LOCAL enable_seqscan = off;")
            else:  # planner_selected
                cur.execute("SET LOCAL enable_indexscan = on;")
                cur.execute("SET LOCAL enable_bitmapscan = on;")
                cur.execute("SET LOCAL enable_seqscan = on;")

            explain_sql = sql.SQL("""
                EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON)
                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                FROM {schema}.{table}
                ORDER BY distance ASC
                LIMIT %s;
            """).format(
                cast=sql.SQL(cast_type),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name)
            )
            cur.execute(explain_sql, (query_vector, top_k), prepare=False)
            raw_res = cur.fetchone()[0]
            parsed = parse_json_plan(raw_res if isinstance(raw_res, list) else json.loads(raw_res))
            diagnostic_dict = parsed
            diagnostic_dict["status"] = "success"
        except Exception as e:
            diagnostic_dict = {"status": "error", "error_message": str(e)}
        finally:
            cur.execute("ROLLBACK;")

    return diagnostic_dict


def compute_distribution_stats(values: list[float]) -> dict:
    """Computes mean, median, IQR, p95, min, max, and stdev for a numeric distribution."""
    if not values:
        return {}
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = statistics.quantiles(sorted_v, n=4)[0] if n >= 4 else min(sorted_v)
    q3 = statistics.quantiles(sorted_v, n=4)[2] if n >= 4 else max(sorted_v)
    p95_idx = max(0, int(math.ceil(0.95 * n)) - 1)
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "iqr": float(q3 - q1),
        "p95": float(sorted_v[p95_idx]),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.stdev(values)) if n > 1 else 0.0
    }


def run_planner_sensitivity_benchmark(
    scales: list[int | str] | None = None,
    representations: list[str] | None = None,
    profiles: dict[str, dict] | None = None,
    canonical_profile_id: str | None = None,
    modes: list[str] | None = None,
    queries: list[str] | None = None,
    precomputed_embeddings: list[list[float]] | None = None,
    warmup_passes: int = 2,
    measured_passes: int = 5,
    top_k: int = FIXED_LIMIT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    output_dir: str = "results/planner_sensitivity",
    dry_run: bool = False
) -> dict:
    """
    Executes the full controlled planner sensitivity experimental matrix.
    Preserves raw plan JSONs, raw latency trials, and raw diagnostic records in append-only format.
    Optimizes execution count:
      - planner_selected timed across all profiles
      - forced paths timed once per (scale, rep) under canonical profile
      - planning-only plan capture executed across all profiles for complete cost-curve modeling
      - diagnostics captured for all queries under canonical profile
      - randomized, seed-deterministic trial execution order within each pass
      - fails closed if actual timed or diagnostic executions differ from declared workload manifest
    """
    time_start_iso = datetime.now(timezone.utc).isoformat()
    target_scales = scales if scales is not None else [1001, 5000, 10000]
    target_reps = representations if representations is not None else ["fp32_extended", "fp32_plain", "fp16_extended"]
    target_profiles = profiles if profiles is not None else PLANNER_COST_PROFILES
    target_modes = modes if modes is not None else ["exact", "forced_hnsw", "planner_selected"]
    query_texts = queries if queries is not None else EVALUATION_QUERIES

    # Determine canonical profile for forced-path timing
    if canonical_profile_id and canonical_profile_id in target_profiles:
        active_canonical_id = canonical_profile_id
    elif DEFAULT_CANONICAL_FORCED_PROFILE in target_profiles:
        active_canonical_id = DEFAULT_CANONICAL_FORCED_PROFILE
    else:
        active_canonical_id = sorted(target_profiles.keys())[0]

    # Programmatic Workload Validation Calculation
    workload_spec = calculate_expected_workload(
        scales=target_scales,
        representations=target_reps,
        profiles=target_profiles,
        modes=target_modes,
        num_queries=len(query_texts),
        measured_passes=measured_passes,
        warmup_passes=warmup_passes,
        canonical_profile_id=active_canonical_id
    )

    manifest_path = os.path.join(output_dir, "experiment_manifest.json")
    env_snapshot_path = os.path.join(output_dir, "environment_snapshot.json")
    plan_jsonl_path = os.path.join(output_dir, "plan_capture.jsonl")
    latency_csv_path = os.path.join(output_dir, "latency_results.csv")
    diagnostic_jsonl_path = os.path.join(output_dir, "diagnostic_results.jsonl")
    summary_csv_path = os.path.join(output_dir, "summary_results.csv")

    expected_output_artifacts = [
        manifest_path, env_snapshot_path, plan_jsonl_path,
        latency_csv_path, diagnostic_jsonl_path, summary_csv_path
    ]

    if not dry_run:
        existing_artifacts = [p for p in expected_output_artifacts if os.path.exists(p)]
        if existing_artifacts:
            raise FileExistsError(
                f"OUTPUT DIRECTORY COLLISION: Target output directory '{output_dir}' already contains "
                f"pre-existing benchmark artifacts:\n"
                f"  {existing_artifacts}\n"
                f"To prevent data contamination and ensure experimental reproducibility, every benchmark execution "
                f"requires a fresh, empty output directory. Existing results are preserved; please specify a new "
                f"--output-dir (e.g. results/planner_sensitivity/run_<timestamp>)."
            )
        os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print("PLANNER SENSITIVITY BENCHMARK EXECUTION WORKLOAD MANIFEST")
    print(f"Scales: {target_scales} | Representations: {target_reps}")
    print(f"Profiles: {list(target_profiles.keys())} | Canonical Forced Profile: {active_canonical_id}")
    print(f"Queries: {len(query_texts)} | Warmups: {warmup_passes} | Passes: {measured_passes} | Top-K: {top_k}")
    print(f"Random Seed: {random_seed} (Deterministic interleaved execution)")
    print(f"Planned Workload Breakdown:")
    print(f"  - Timed Executions (Phase B):")
    print(f"      planner_selected ({workload_spec['num_profiles']} profiles): {workload_spec['timed_planner_selected']:,}")
    print(f"      forced_exact (canonical {active_canonical_id}):       {workload_spec['timed_forced_exact']:,}")
    print(f"      forced_hnsw (canonical {active_canonical_id}):        {workload_spec['timed_forced_hnsw']:,}")
    print(f"    TOTAL TIMED EXECUTIONS:                               {workload_spec['total_timed_executions']:,}")
    print(f"  - Warm-up Executions:                                   {workload_spec['total_warmup_executions']:,}")
    print(f"  - Planning-Only Plan Captures (Phase A):                {workload_spec['total_plan_captures']:,}")
    print(f"  - Diagnostic Runs (Phase C, canonical {active_canonical_id}):      {workload_spec['total_diagnostic_executions']:,}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE READ-ONLY BENCHMARK'}")
    print("================================================================================")

    if dry_run:
        print("[DRY RUN] Validating parameters and connections without executing queries.")
        for sc in target_scales:
            db = SCALE_DATABASE_MAP[sc]
            conn = get_sensitivity_connection(db)
            if conn.prepare_threshold is not None:
                raise RuntimeError(
                    f"PREPARE THRESHOLD SAFETY VIOLATION: Dry-run scale {sc} connection on '{db}' "
                    f"has prepare_threshold={conn.prepare_threshold}; must be None!"
                )
            conn.close()
        print("[DRY RUN] All target databases accessible and verified read-only with prepare_threshold=None.")
        return {
            "status": "dry_run_success",
            "scales": target_scales,
            "profiles": list(target_profiles.keys()),
            "workload_spec": workload_spec
        }

    # 1. Precompute query embeddings outside of latency timer
    print("\n--- Step 1: Precomputing Evaluation Query Embeddings (Excluded from Timer) ---")
    t0_emb = time.perf_counter()
    if precomputed_embeddings is not None:
        embeddings = precomputed_embeddings
        print(f"Using {len(embeddings)} supplied precomputed vector embeddings.")
    else:
        embeddings = [get_embedding(q) for q in query_texts]
        print(f"Precomputed {len(embeddings)} query vectors via Ollama in {time.perf_counter() - t0_emb:.2f}s.")

    # 2. Capture Environment Snapshot on the primary scaling database
    primary_db = SCALE_DATABASE_MAP[target_scales[0]]
    conn_primary = get_sensitivity_connection(primary_db)
    env_snapshot = capture_environment_snapshot(conn_primary)
    conn_primary.close()
    with open(env_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(env_snapshot, f, indent=2)
    print(f"[RECORDED] Host environment snapshot written to {env_snapshot_path}")

    # Initialize CSV writers in fresh write mode
    latency_file = open(latency_csv_path, "w", newline="", encoding="utf-8")
    latency_writer = csv.writer(latency_file)
    latency_writer.writerow([
        "timestamp", "scale", "representation", "profile_id", "mode",
        "pass_id", "query_id", "latency_ms", "recall_at_10", "tie_aware_recall_at_10",
        "has_distance_tie", "boundary_distance", "boundary_delta"
    ])

    plan_jsonl_file = open(plan_jsonl_path, "w", encoding="utf-8")
    diagnostic_jsonl_file = open(diagnostic_jsonl_path, "w", encoding="utf-8")

    all_trial_records = []
    summary_records = []
    actual_execution_order_log = []

    actual_timed_executions = 0
    actual_warmup_executions = 0
    actual_plan_captures = 0
    actual_diagnostic_executions = 0

    latency_accounting = {
        "expected": workload_spec["total_timed_executions"],
        "attempted": 0,
        "successful": 0,
        "failed": 0,
        "dataset_status": "in_progress",
        "errors": []
    }

    diagnostic_accounting = {
        "expected": workload_spec["total_diagnostic_executions"],
        "attempted": 0,
        "successful": 0,
        "failed": 0,
        "dataset_status": "in_progress",
        "errors": []
    }

    # Outer Scale Loop
    for scale in target_scales:
        target_db = SCALE_DATABASE_MAP[scale]
        conn = get_sensitivity_connection(target_db)
        if conn.prepare_threshold is not None:
            raise RuntimeError(
                f"PREPARE THRESHOLD SAFETY VIOLATION: Benchmark connection for scale {scale} on '{target_db}' "
                f"has prepare_threshold={conn.prepare_threshold}; must strictly be None!"
            )
        print(f"\n=======================================================")
        print(f"PROCESSING SCALE: {scale} (Database: '{target_db}')")
        print(f"=======================================================")

        try:
            # 3. Compute scale-specific FP32 Exact Ground Truth and Boundary Ties for Recall@10
            # Ground truth is obtained from fp32_extended in exact mode with Seq Scan plan verification
            fp32_ext_tbl = f"chunks_{scale}_fp32_extended"
            ground_truths = {}
            ground_truth_ties = {}
            canonical_prof_params = target_profiles[active_canonical_id]

            print(f"Computing FP32 Exact Ground Truth for scale {scale} (Canonical Profile: {active_canonical_id}, Top-K: {top_k})...")
            for q_idx, q_vec in enumerate(embeddings):
                gt_rows = retrieve_exact_ground_truth(
                    conn, q_vec, SCHEMA_NAME, fp32_ext_tbl, "vector(768)",
                    canonical_prof_params, top_k=top_k
                )
                ground_truths[q_idx] = gt_rows[:top_k]

                # Retrieve complete boundary tie group using exact Seq Scan
                gt_tie_res = retrieve_boundary_tie_group(
                    conn, q_vec, SCHEMA_NAME, fp32_ext_tbl, top_k=top_k, tolerance=COSINE_DISTANCE_TOLERANCE
                )
                ground_truth_ties[q_idx] = gt_tie_res
                if gt_tie_res.get("has_boundary_tie") or gt_tie_res.get("has_tie"):
                    print(f"  [BOUNDARY TIE DETECTED] Query {q_idx}: Rank 10 dist={gt_tie_res['boundary_distance']:.6f}, |B|={gt_tie_res['size_B']}, |T|={gt_tie_res['size_T']}")

            # 4. Phase A: Execution-Free Plan Captures (All representations, profiles, modes, queries)
            print("\n--- Phase A: Execution-Free Plan Capture ---")
            plan_capture_cache = {}
            for rep in target_reps:
                table_name = f"chunks_{scale}_{rep}"
                cast_type = "halfvec(768)" if "fp16" in rep else "vector(768)"
                expected_index_name = f"{table_name}_hnsw_idx"

                for prof_id, prof_params in target_profiles.items():
                    for mode in target_modes:
                        sample_plan = None
                        for q_idx, q_vec in enumerate(embeddings):
                            plan_info = capture_query_plan(
                                conn, q_vec, SCHEMA_NAME, table_name, cast_type,
                                prof_params, mode=mode, top_k=top_k
                            )
                            actual_plan_captures += 1
                            if q_idx == 0:
                                sample_plan = plan_info

                            plan_record = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "scale": scale,
                                "representation": rep,
                                "profile_id": prof_id,
                                "mode": mode,
                                "query_id": q_idx,
                                "root_node_type": plan_info.get("root_node_type"),
                                "root_startup_cost": plan_info.get("root_startup_cost"),
                                "root_total_cost": plan_info.get("root_total_cost"),
                                "scan_node_type": plan_info.get("scan_node_type"),
                                "index_name": plan_info.get("index_name"),
                                "settings": plan_info.get("settings"),
                                "raw_plan": plan_info.get("raw_plan")
                            }
                            plan_jsonl_file.write(json.dumps(plan_record) + "\n")
                            plan_jsonl_file.flush()

                            # Assertion for forced HNSW mode: verify index name
                            if mode == "forced_hnsw":
                                if plan_info.get("scan_node_type") != "Index Scan" or plan_info.get("index_name") != expected_index_name:
                                    raise RuntimeError(
                                        f"FORCED-INDEX VERIFICATION FAILURE on '{table_name}': Expected Index Scan using "
                                        f"'{expected_index_name}', but planner selected '{plan_info.get('scan_node_type')}' "
                                        f"with index '{plan_info.get('index_name')}'."
                                    )

                        plan_capture_cache[(rep, prof_id, mode)] = sample_plan

            # 5. Phase B: Pure Wall-Clock Latency Measurements
            print("\n--- Phase B: Performance Measurement (Interleaved Seeded Passes) ---")

            # Collect trial performance per (rep, prof_id, mode)
            measured_data = {}
            for rep in target_reps:
                for prof_id in target_profiles:
                    for mode in target_modes:
                        if mode == "planner_selected" or prof_id == active_canonical_id:
                            measured_data[(rep, prof_id, mode)] = {
                                "latencies": [],
                                "recalls": [],
                                "tie_aware_recalls": [],
                                "per_query_recalls": {q_i: [] for q_i in range(len(embeddings))},
                                "per_query_tie_recalls": {q_i: [] for q_i in range(len(embeddings))},
                                "tie_flags": []
                            }

            # Warm-up passes (unrecorded)
            print(f"Running {warmup_passes} warm-up passes...")
            for w_pass in range(warmup_passes):
                for rep in target_reps:
                    table_name = f"chunks_{scale}_{rep}"
                    cast_type = "halfvec(768)" if "fp16" in rep else "vector(768)"
                    for prof_id, prof_params in target_profiles.items():
                        for mode in target_modes:
                            if mode == "planner_selected" or prof_id == active_canonical_id:
                                for q_vec in embeddings:
                                    measure_query_latency(
                                        conn, q_vec, SCHEMA_NAME, table_name, cast_type,
                                        prof_params, mode=mode, top_k=top_k
                                    )
                                    actual_warmup_executions += 1

            # Measured passes (interleaved & randomized by seed to mitigate systematic presentation bias)
            print(f"Running {measured_passes} measured passes (Random Seed: {random_seed})...")
            for pass_id in range(1, measured_passes + 1):
                pass_trials = []
                for rep in target_reps:
                    table_name = f"chunks_{scale}_{rep}"
                    cast_type = "halfvec(768)" if "fp16" in rep else "vector(768)"
                    for prof_id, prof_params in target_profiles.items():
                        for mode in target_modes:
                            if mode == "planner_selected" or prof_id == active_canonical_id:
                                for q_idx, q_vec in enumerate(embeddings):
                                    pass_trials.append({
                                        "scale": scale,
                                        "rep": rep,
                                        "table_name": table_name,
                                        "cast_type": cast_type,
                                        "prof_id": prof_id,
                                        "prof_params": prof_params,
                                        "mode": mode,
                                        "query_id": q_idx,
                                        "query_vec": q_vec,
                                        "pass_id": pass_id
                                    })

                # Deterministic shuffle for this pass
                pass_rng = random.Random(random_seed + pass_id * 1000)
                pass_rng.shuffle(pass_trials)

                for trial in pass_trials:
                    rep = trial["rep"]
                    prof_id = trial["prof_id"]
                    mode = trial["mode"]
                    q_idx = trial["query_id"]

                    latency_accounting["attempted"] += 1
                    try:
                        rows, lat_ms = measure_query_latency(
                            conn, trial["query_vec"], SCHEMA_NAME, trial["table_name"], trial["cast_type"],
                            trial["prof_params"], mode=mode, top_k=top_k
                        )
                        actual_timed_executions += 1
                        latency_accounting["successful"] += 1
                    except Exception as e:
                        latency_accounting["failed"] += 1
                        latency_accounting["errors"].append({
                            "scale": scale, "rep": rep, "profile_id": prof_id, "mode": mode,
                            "pass_id": pass_id, "query_id": q_idx, "error": str(e)
                        })
                        raise

                    # Recall computation against FP32 exact ground truth
                    rec_metrics = calculate_recall_metrics(
                        ground_truths[q_idx], rows, gt_tie_info=ground_truth_ties.get(q_idx), top_k=top_k
                    )
                    conv_recall = rec_metrics["conventional_recall"]
                    tie_recall = rec_metrics["tie_aware_recall"]
                    has_tie = rec_metrics["has_boundary_tie"]
                    b_dist = rec_metrics["boundary_distance"]
                    b_delta = rec_metrics["delta"]

                    rec_bucket = measured_data[(rep, prof_id, mode)]
                    rec_bucket["latencies"].append(lat_ms)
                    rec_bucket["recalls"].append(conv_recall)
                    rec_bucket["tie_aware_recalls"].append(tie_recall)
                    rec_bucket["per_query_recalls"][q_idx].append(conv_recall)
                    rec_bucket["per_query_tie_recalls"][q_idx].append(tie_recall)
                    rec_bucket["tie_flags"].append(has_tie)

                    ts_now = datetime.now(timezone.utc).isoformat()
                    latency_writer.writerow([
                        ts_now, scale, rep, prof_id, mode,
                        pass_id, q_idx, f"{lat_ms:.4f}", f"{conv_recall:.4f}", f"{tie_recall:.4f}",
                        has_tie, f"{b_dist:.6f}" if b_dist is not None else "", f"{b_delta:.8f}" if b_delta is not None else ""
                    ])
                    latency_file.flush()

                    all_trial_records.append({
                        "scale": scale, "rep": rep, "profile_id": prof_id, "mode": mode,
                        "query_id": q_idx, "latency_ms": lat_ms, "recall": conv_recall,
                        "tie_aware_recall": tie_recall
                    })
                    actual_execution_order_log.append({
                        "pass_id": pass_id, "rep": rep, "profile_id": prof_id, "mode": mode, "query_id": q_idx
                    })

            # 6. Phase C: Diagnostic EXPLAIN ANALYZE (Executed across all queries under canonical profile)
            print(f"\n--- Phase C: Diagnostic EXPLAIN ANALYZE (Canonical Profile: {active_canonical_id}, All {len(embeddings)} Queries) ---")
            diagnostic_cache = {}
            for rep in target_reps:
                table_name = f"chunks_{scale}_{rep}"
                cast_type = "halfvec(768)" if "fp16" in rep else "vector(768)"
                expected_index_name = f"{table_name}_hnsw_idx"

                for mode in target_modes:
                    diag_records_for_mode = []
                    for q_idx, q_vec in enumerate(embeddings):
                        diagnostic_accounting["attempted"] += 1
                        actual_diagnostic_executions += 1
                        diag_info = execute_diagnostic_pass(
                            conn, q_vec, SCHEMA_NAME, table_name, cast_type,
                            canonical_prof_params, mode=mode, top_k=top_k
                        )

                        if diag_info.get("status") == "success":
                            if mode == "forced_hnsw" and (
                                diag_info.get("scan_node_type") != "Index Scan" or
                                diag_info.get("index_name") != expected_index_name
                            ):
                                diagnostic_accounting["failed"] += 1
                                diag_err = (
                                    f"DIAGNOSTIC INDEX VERIFICATION FAILURE on '{table_name}': Expected Index Scan "
                                    f"using '{expected_index_name}', got '{diag_info.get('index_name')}'."
                                )
                                diagnostic_accounting["errors"].append({
                                    "scale": scale,
                                    "rep": rep,
                                    "profile_id": active_canonical_id,
                                    "mode": mode,
                                    "diagnostic_query_id": q_idx,
                                    "error": diag_err
                                })
                                diag_info["status"] = "error"
                                diag_info["error_message"] = diag_err
                            else:
                                diagnostic_accounting["successful"] += 1
                        else:
                            diagnostic_accounting["failed"] += 1
                            diagnostic_accounting["errors"].append({
                                "scale": scale,
                                "rep": rep,
                                "profile_id": active_canonical_id,
                                "mode": mode,
                                "diagnostic_query_id": q_idx,
                                "error": diag_info.get("error_message")
                            })

                        diag_record = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "scale": scale,
                            "representation": rep,
                            "profile_id": active_canonical_id,
                            "mode": mode,
                            "diagnostic_query_id": q_idx,
                            "status": diag_info.get("status", "unknown"),
                            "error_message": diag_info.get("error_message"),
                            "actual_total_time_ms": diag_info.get("actual_total_time_ms"),
                            "planning_time_ms": diag_info.get("planning_time_ms"),
                            "execution_time_ms": diag_info.get("execution_time_ms"),
                            "shared_hit_buffers": diag_info.get("shared_hit_buffers"),
                            "shared_read_buffers": diag_info.get("shared_read_buffers"),
                            "shared_dirtied_buffers": diag_info.get("shared_dirtied_buffers"),
                            "shared_written_buffers": diag_info.get("shared_written_buffers"),
                            "scan_node_type": diag_info.get("scan_node_type"),
                            "index_name": diag_info.get("index_name"),
                            "raw_plan": diag_info.get("raw_plan")
                        }
                        diagnostic_jsonl_file.write(json.dumps(diag_record) + "\n")
                        diagnostic_jsonl_file.flush()
                        diag_records_for_mode.append(diag_record)

                    diagnostic_cache[(rep, mode)] = diag_records_for_mode

            # 7. Compile Summary Records
            has_gt_tie_at_scale = any(ground_truth_ties[q_i]["has_tie"] for q_i in range(len(embeddings)))
            for rep in target_reps:
                for prof_id, prof_params in target_profiles.items():
                    for mode in target_modes:
                        plan_info = plan_capture_cache.get((rep, prof_id, mode), {})
                        is_timed = (mode == "planner_selected") or (prof_id == active_canonical_id)

                        if is_timed:
                            b_data = measured_data[(rep, prof_id, mode)]
                            lat_stats = compute_distribution_stats(b_data["latencies"])
                            mean_rec = statistics.mean(b_data["recalls"]) if b_data["recalls"] else 0.0
                            mean_tie_rec = statistics.mean(b_data["tie_aware_recalls"]) if b_data["tie_aware_recalls"] else 0.0

                            # Per-query recall dictionaries
                            per_q_recall_map = {
                                f"q{q_i}": round(statistics.mean(b_data["per_query_recalls"][q_i]), 4)
                                for q_i in range(len(embeddings))
                            }
                            per_q_tie_recall_map = {
                                f"q{q_i}": round(statistics.mean(b_data["per_query_tie_recalls"][q_i]), 4)
                                for q_i in range(len(embeddings))
                            }

                            # Extract canonical diagnostic metrics for this mode
                            canonical_diags = diagnostic_cache.get((rep, mode), [])
                            avg_hits = statistics.mean([d["shared_hit_buffers"] for d in canonical_diags if d["shared_hit_buffers"] is not None]) if canonical_diags else None
                            avg_reads = statistics.mean([d["shared_read_buffers"] for d in canonical_diags if d["shared_read_buffers"] is not None]) if canonical_diags else None
                            avg_exec_time = statistics.mean([d["execution_time_ms"] for d in canonical_diags if d["execution_time_ms"] is not None]) if canonical_diags else None

                            summary_records.append({
                                "scale": scale,
                                "representation": rep,
                                "profile_id": prof_id,
                                "random_page_cost": prof_params["random_page_cost"],
                                "cpu_operator_cost": prof_params["cpu_operator_cost"],
                                "mode": mode,
                                "execution_status": "measured",
                                "trials": len(b_data["latencies"]),
                                "mean_latency_ms": f"{lat_stats.get('mean', 0.0):.4f}",
                                "median_latency_ms": f"{lat_stats.get('median', 0.0):.4f}",
                                "iqr_latency_ms": f"{lat_stats.get('iqr', 0.0):.4f}",
                                "p95_latency_ms": f"{lat_stats.get('p95', 0.0):.4f}",
                                "mean_recall_at_10": f"{mean_rec:.4f}",
                                "mean_tie_aware_recall_at_10": f"{mean_tie_rec:.4f}",
                                "per_query_recall": json.dumps(per_q_recall_map),
                                "per_query_tie_aware_recall": json.dumps(per_q_tie_recall_map),
                                "has_ground_truth_tie": has_gt_tie_at_scale,
                                "selected_scan_type": plan_info.get("scan_node_type"),
                                "selected_index_name": plan_info.get("index_name") or "",
                                "est_startup_cost": plan_info.get("root_startup_cost"),
                                "est_total_cost": plan_info.get("root_total_cost"),
                                "diagnostic_canonical_profile": active_canonical_id if prof_id == active_canonical_id else None,
                                "diagnostic_avg_shared_hit_buffers": round(avg_hits, 1) if avg_hits is not None else None,
                                "diagnostic_avg_shared_read_buffers": round(avg_reads, 1) if avg_reads is not None else None,
                                "diagnostic_avg_server_exec_time_ms": round(avg_exec_time, 3) if avg_exec_time is not None else None
                            })
                            print(f"  [{mode}] Latency: Mean={lat_stats.get('mean', 0.0):.2f}ms, Median={lat_stats.get('median', 0.0):.2f}ms | Recall@10: {mean_rec:.4f} (Tie-Aware: {mean_tie_rec:.4f}) | Plan: {plan_info.get('scan_node_type')}")
                        else:
                            summary_records.append({
                                "scale": scale,
                                "representation": rep,
                                "profile_id": prof_id,
                                "random_page_cost": prof_params["random_page_cost"],
                                "cpu_operator_cost": prof_params["cpu_operator_cost"],
                                "mode": mode,
                                "execution_status": f"plan_only (canonical: {active_canonical_id})",
                                "trials": 0,
                                "mean_latency_ms": None,
                                "median_latency_ms": None,
                                "iqr_latency_ms": None,
                                "p95_latency_ms": None,
                                "mean_recall_at_10": None,
                                "mean_tie_aware_recall_at_10": None,
                                "per_query_recall": None,
                                "per_query_tie_aware_recall": None,
                                "has_ground_truth_tie": has_gt_tie_at_scale,
                                "selected_scan_type": plan_info.get("scan_node_type"),
                                "selected_index_name": plan_info.get("index_name") or "",
                                "est_startup_cost": plan_info.get("root_startup_cost"),
                                "est_total_cost": plan_info.get("root_total_cost"),
                                "diagnostic_canonical_profile": None,
                                "diagnostic_avg_shared_hit_buffers": None,
                                "diagnostic_avg_shared_read_buffers": None,
                                "diagnostic_avg_server_exec_time_ms": None
                            })
                            print(f"  [{mode}] Plan-Only (Canonical: {active_canonical_id}): Est Startup={plan_info.get('root_startup_cost')}, Est Total={plan_info.get('root_total_cost')} | Plan: {plan_info.get('scan_node_type')}")
        finally:
            conn.close()

    latency_file.close()
    plan_jsonl_file.close()
    diagnostic_jsonl_file.close()

    # Finalize runtime dataset status
    if diagnostic_accounting["successful"] == diagnostic_accounting["expected"] and diagnostic_accounting["failed"] == 0:
        diagnostic_accounting["dataset_status"] = "complete"
    else:
        diagnostic_accounting["dataset_status"] = "incomplete"

    if latency_accounting["successful"] == latency_accounting["expected"] and latency_accounting["failed"] == 0:
        latency_accounting["dataset_status"] = "complete"
    else:
        latency_accounting["dataset_status"] = "incomplete"

    # Fail-closed programmatic workload verification
    if actual_timed_executions != workload_spec["total_timed_executions"]:
        raise RuntimeError(
            f"CRITICAL WORKLOAD MISMATCH: Actual timed executions ({actual_timed_executions}) "
            f"did not match declared manifest ({workload_spec['total_timed_executions']})! "
            f"Failing closed to prevent corrupted experimental reporting."
        )

    if actual_diagnostic_executions != workload_spec["total_diagnostic_executions"]:
        raise RuntimeError(
            f"CRITICAL WORKLOAD MISMATCH: Actual diagnostic executions ({actual_diagnostic_executions}) "
            f"did not match declared manifest ({workload_spec['total_diagnostic_executions']})! "
            f"Failing closed to prevent corrupted experimental reporting."
        )

    execution_count_status = "PASSED"
    latency_dataset_status = "COMPLETE" if latency_accounting["dataset_status"] == "complete" else "INCOMPLETE"
    diagnostic_dataset_status = "COMPLETE" if diagnostic_accounting["dataset_status"] == "complete" else "INCOMPLETE"
    overall_status = (
        "COMPLETE" if (latency_dataset_status == "COMPLETE" and diagnostic_dataset_status == "COMPLETE")
        else "PARTIAL" if latency_dataset_status == "COMPLETE"
        else "INCOMPLETE"
    )

    print(f"\n[WORKLOAD VERIFIED] Execution count validation: {execution_count_status} (Timed: {actual_timed_executions}, Diagnostics: {actual_diagnostic_executions})")
    print(f"[STATUS] Latency Dataset Completeness: {latency_dataset_status} ({latency_accounting['successful']}/{latency_accounting['expected']} successful, {latency_accounting['failed']} failed)")
    print(f"[STATUS] Diagnostic Dataset Completeness: {diagnostic_dataset_status} ({diagnostic_accounting['successful']}/{diagnostic_accounting['expected']} successful, {diagnostic_accounting['failed']} failed)")
    print(f"[STATUS] Overall Experiment Status: {overall_status}")

    # Write summary CSV
    if summary_records:
        summary_fields = list(summary_records[0].keys())
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as sf:
            s_writer = csv.DictWriter(sf, fieldnames=summary_fields)
            s_writer.writeheader()
            for row in summary_records:
                s_writer.writerow(row)
        print(f"\n[EXPORTED] Summary statistics written to {summary_csv_path}")

    # Write experiment manifest
    time_completed_iso = datetime.now(timezone.utc).isoformat()
    manifest_data = {
        "experiment_title": "Precision-Aware Query Planning for Vector Retrieval in PostgreSQL",
        "timestamp_start": time_start_iso,
        "timestamp_completed": time_completed_iso,
        "random_seed": random_seed,
        "git_commit": env_snapshot["git_commit"],
        "output_directory": os.path.abspath(output_dir),
        "status_summary": {
            "execution_count_validation_status": execution_count_status,
            "latency_dataset_completeness_status": latency_dataset_status,
            "diagnostic_dataset_completeness_status": diagnostic_dataset_status,
            "overall_experiment_status": overall_status
        },
        "execution_count_validation_status": execution_count_status,
        "latency_dataset_completeness_status": latency_dataset_status,
        "diagnostic_dataset_completeness_status": diagnostic_dataset_status,
        "overall_experiment_status": overall_status,
        "scales_evaluated": target_scales,
        "representations_evaluated": target_reps,
        "profiles_evaluated": list(target_profiles.keys()),
        "canonical_forced_profile": active_canonical_id,
        "modes_evaluated": target_modes,
        "top_k": top_k,
        "numerical_tolerance_policy": {
            "primary_tolerance": COSINE_DISTANCE_TOLERANCE,
            "sensitivity_tolerances": SENSITIVITY_TOLERANCES,
            "description": TOLERANCE_POLICY
        },
        "workload_specification": workload_spec,
        "actual_workload": {
            "timed_executions": actual_timed_executions,
            "warmup_executions": actual_warmup_executions,
            "plan_captures": actual_plan_captures,
            "diagnostic_executions": actual_diagnostic_executions
        },
        "latency_accounting": latency_accounting,
        "diagnostic_accounting": diagnostic_accounting,
        "workload_validation": execution_count_status,
        "total_timing_trials": len(all_trial_records),
        "total_summary_configurations": len(summary_records),
        "sample_execution_sequence": actual_execution_order_log[:20],
        "artifacts": {
            "environment_snapshot": env_snapshot_path,
            "plan_capture_jsonl": plan_jsonl_path,
            "latency_results_csv": latency_csv_path,
            "diagnostic_results_jsonl": diagnostic_jsonl_path,
            "summary_results_csv": summary_csv_path
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)
    print(f"[EXPORTED] Experiment manifest written to {manifest_path}")

    return {
        "status": overall_status.lower(),
        "execution_count_status": execution_count_status,
        "latency_dataset_status": latency_dataset_status,
        "diagnostic_dataset_status": diagnostic_dataset_status,
        "overall_status": overall_status,
        "manifest": manifest_data,
        "summary": summary_records
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Planner Sensitivity Benchmark.")
    parser.add_argument("--scale", choices=["1001", "5000", "10000", "test", "all"], default="all",
                        help="Corpus scale to benchmark (default: all).")
    parser.add_argument("--representation", choices=["fp32_extended", "fp32_plain", "fp16_extended", "all"],
                        default="all", help="Representation to test (default: all).")
    parser.add_argument("--profiles", nargs="+", default=None,
                        help="Specific profile IDs to test (e.g. P1 P5 P9). Default is all 9.")
    parser.add_argument("--canonical-profile", default=DEFAULT_CANONICAL_FORCED_PROFILE,
                        help="Canonical profile for forced-path timing (default: P4).")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed for deterministic trial presentation (default: 42).")
    parser.add_argument("--warmup-passes", type=int, default=2,
                        help="Number of warmup passes per query (default: 2).")
    parser.add_argument("--measured-passes", type=int, default=5,
                        help="Number of timed measurement passes per query (default: 5).")
    parser.add_argument("--pilot", action="store_true", default=False,
                        help="Run minimal pilot benchmark on test database (profiles P1 and P9, 2 queries, 2 passes).")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Validate configurations and database connections without query execution.")
    parser.add_argument("--output-dir", default="results/planner_sensitivity",
                        help="Directory to store experimental artifacts.")
    args = parser.parse_args()

    if args.pilot:
        print("\n>>> Launching Minimal Acceptance Pilot on 'compression_aware_rag_test' <<<")
        pilot_profiles = {
            "P1": PLANNER_COST_PROFILES["P1"],
            "P9": PLANNER_COST_PROFILES["P9"]
        }
        pilot_queries = EVALUATION_QUERIES[:2]
        res = run_planner_sensitivity_benchmark(
            scales=["test"],
            representations=["fp32_extended", "fp32_plain", "fp16_extended"],
            profiles=pilot_profiles,
            canonical_profile_id="P1",  # P1 serves as canonical forced profile in P1/P9 pilot
            modes=["exact", "forced_hnsw", "planner_selected"],
            queries=pilot_queries,
            warmup_passes=2,
            measured_passes=2,
            top_k=FIXED_LIMIT,
            random_seed=args.random_seed,
            output_dir=os.path.join(args.output_dir, "pilot"),
            dry_run=args.dry_run
        )
        print("\nPilot benchmark completed successfully.")
        return

    scales_to_run = [1001, 5000, 10000] if args.scale == "all" else [int(args.scale) if args.scale != "test" else "test"]
    reps_to_run = ["fp32_extended", "fp32_plain", "fp16_extended"] if args.representation == "all" else [args.representation]
    profiles_to_run = PLANNER_COST_PROFILES
    if args.profiles:
        profiles_to_run = {p: PLANNER_COST_PROFILES[p] for p in args.profiles if p in PLANNER_COST_PROFILES}

    run_planner_sensitivity_benchmark(
        scales=scales_to_run,
        representations=reps_to_run,
        profiles=profiles_to_run,
        canonical_profile_id=args.canonical_profile,
        warmup_passes=args.warmup_passes,
        measured_passes=args.measured_passes,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
