#!/usr/bin/env python3
"""
scripts/run_phase3a_tie_sensitivity.py

Phase 3A: Boundary-Tie Threshold Sensitivity and Query-Vector Validation.
Supplementary read-only retrieval audit:
  1. Recovers the 20 query vectors embedded in original plan_capture.jsonl.
  2. Reconstructs FP32 exact reference distances (Seq Scan) and builds
     disjoint boundary sets B and T at epsilons [1e-7, 1e-6, 1e-5].
  3. Recomputes candidate top-10 retrievals across all 1,980 experimental conditions
     (3 scales x 3 representations x 11 conditions x 20 queries).
  4. Evaluates Conventional Recall@10 and Tie-Aware Recall@10 across all 3 thresholds (5,940 observations).
  5. Strictly validates recomputed metrics against original Phase 2 latency_results.csv
     and plan_capture.jsonl with a tolerance of 0.00005.
  6. Exports audit artifacts to results/planner_sensitivity/phase3_tie_sensitivity_20261031/.

SAFETY GUARANTEES:
  - Exclusively read-only transactions (BEGIN READ ONLY ... ROLLBACK).
  - autocommit=True, prepare_threshold=None, prepare=False.
  - Baseline catalog 'compression_aware_rag' is strictly forbidden.
  - Zero database mutations (no DDL/DML).
  - Original Phase 2 benchmark files remain 100% read-only and unaltered.
"""

import os
import sys
import json
import csv
import re
import hashlib
import statistics
import subprocess
from datetime import datetime, timezone
from collections import defaultdict, Counter

import psycopg
from psycopg import sql

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.planner_sensitivity_experiment import (
    get_sensitivity_connection,
    parse_json_plan,
    PLANNER_COST_PROFILES,
    ALLOWED_DATABASES,
    DISALLOWED_DATABASES,
    SCALE_DATABASE_MAP
)

EPSILONS = [1e-7, 1e-6, 1e-5]
TARGET_SCALES = [1001, 5000, 10000]
TARGET_REPRESENTATIONS = ["fp32_extended", "fp32_plain", "fp16_extended"]
MODES = ["exact", "forced_hnsw", "planner_selected"]
CANONICAL_PROFILE_ID = "P4"
SCHEMA_NAME = "planner_sensitivity"
ROUNDING_TOLERANCE = 0.00005

PHASE2_DIR = os.path.join(PROJECT_ROOT, "results", "planner_sensitivity", "full_run_20261031_01")
PHASE3_DIR = os.path.join(PROJECT_ROOT, "results", "planner_sensitivity", "phase3_tie_sensitivity_20261031")


class TeeLogger:
    def __init__(self, filepath, stream):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def verify_preconditions():
    """Verify Git status, Phase 2 artifact existence, and target output directory."""
    print("================================================================================")
    print("PHASE 3A PRE-FLIGHT AUDIT VERIFICATION")
    print("================================================================================")

    # 1. Git branch and commit
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, cwd=PROJECT_ROOT).strip()
        head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip()
    except Exception as e:
        raise RuntimeError(f"Failed to check git status: {e}")

    print(f"Git Branch: {branch}")
    print(f"HEAD SHA:   {head_sha}")

    if branch != "research/planner-sensitivity":
        raise RuntimeError(f"Invalid branch: expected 'research/planner-sensitivity', got '{branch}'")
    if not head_sha.startswith("8dfdce0"):
        raise RuntimeError(f"Invalid commit SHA: expected 8dfdce0..., got {head_sha}")

    # 2. Check Phase 2 source directory
    required_files = [
        "latency_results.csv",
        "plan_capture.jsonl",
        "diagnostic_results.jsonl",
        "summary_results.csv",
        "experiment_manifest.json",
        "environment_snapshot.json"
    ]
    for rf in required_files:
        p = os.path.join(PHASE2_DIR, rf)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required Phase 2 artifact: {p}")
    print(f"[OK] Phase 2 benchmark directory verified: {PHASE2_DIR}")

    # 3. Check output directory does not exist
    if os.path.exists(PHASE3_DIR):
        raise FileExistsError(
            f"SAFETY VIOLATION: Output directory '{PHASE3_DIR}' already exists! "
            f"Failing closed to prevent overwriting."
        )
    os.makedirs(PHASE3_DIR, exist_ok=False)
    print(f"[OK] Created fresh output directory: {PHASE3_DIR}")

    # Tee stdout/stderr to phase3a_console.log
    log_path = os.path.join(PHASE3_DIR, "phase3a_console.log")
    sys.stdout = TeeLogger(log_path, sys.stdout)
    sys.stderr = TeeLogger(log_path, sys.stderr)
    print(f"[OK] Preserving console output to {log_path}")


def recover_query_vectors() -> dict[int, dict]:
    """
    Extracts the 20 query vectors embedded in plan_capture.jsonl raw plan expressions.
    Ensures exactly 20 vectors, 768 dimensions each, and verifies 100% agreement
    across all repeated occurrences in FP32 plan records.
    """
    print("\n--- Step 1: Recovering 20 Query Vectors from plan_capture.jsonl ---")
    plan_capture_path = os.path.join(PHASE2_DIR, "plan_capture.jsonl")

    recovered_vectors = {}
    occurrences = defaultdict(int)

    with open(plan_capture_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            rec = json.loads(line)
            # Only extract from FP32 representations to avoid half-precision rounding
            if rec["representation"] not in ["fp32_extended", "fp32_plain"]:
                continue

            qid = rec["query_id"]
            raw_str = json.dumps(rec.get("raw_plan", []))
            matches = re.findall(r"\[[0-9eE\.\,\-\s]+\]", raw_str)
            found_vec = None
            for m in matches:
                if m.count(",") == 767:
                    found_vec = m
                    break

            if found_vec is None:
                raise ValueError(f"Could not extract 768-dim vector from line {line_no} for query {qid}")

            vec_floats = json.loads(found_vec)
            if len(vec_floats) != 768:
                raise ValueError(f"Extracted vector dimension is {len(vec_floats)} (expected 768) at line {line_no}")

            vec_serialized = json.dumps(vec_floats)
            vec_hash = hashlib.sha256(vec_serialized.encode("utf-8")).hexdigest()
            occurrences[qid] += 1

            if qid not in recovered_vectors:
                recovered_vectors[qid] = {
                    "query_id": qid,
                    "dimension": len(vec_floats),
                    "vector": vec_floats,
                    "serialized": vec_serialized,
                    "sha256": vec_hash
                }
            else:
                if recovered_vectors[qid]["sha256"] != vec_hash:
                    raise ValueError(
                        f"VECTOR DIVERGENCE DETECTED for query {qid} at line {line_no}! "
                        f"Previous hash: {recovered_vectors[qid]['sha256']}, current: {vec_hash}"
                    )

    if len(recovered_vectors) != 20:
        raise ValueError(f"Expected exactly 20 recovered query vectors, found {len(recovered_vectors)}")

    print(f"[RECOVERED] Successfully extracted exactly 20 unique query vectors (IDs 0..19).")
    print(f"[VERIFIED] Each query vector matched identically across all {occurrences[0]} FP32 plan records.")

    # Export to recovered_query_vectors.csv
    out_csv = os.path.join(PHASE3_DIR, "recovered_query_vectors.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "dimension", "sha256", "serialized_vector"])
        writer.writeheader()
        for qid in sorted(recovered_vectors.keys()):
            v = recovered_vectors[qid]
            writer.writerow({
                "query_id": v["query_id"],
                "dimension": v["dimension"],
                "sha256": v["sha256"],
                "serialized_vector": v["serialized"]
            })
    print(f"[EXPORTED] Recovered query vectors written to {out_csv}")
    return recovered_vectors


def reconstruct_fp32_reference_boundary_groups(recovered_vectors: dict[int, dict]) -> tuple[dict, list[dict]]:
    """
    Computes FP32 exact reference distances using Seq Scan on chunks_<scale>_fp32_extended.
    Calculates B, T, and r across epsilons [1e-7, 1e-6, 1e-5].
    """
    print("\n--- Step 2: Reconstructing FP32 Exact Reference Distances and Boundary Sets ---")
    canonical_prof = PLANNER_COST_PROFILES[CANONICAL_PROFILE_ID]

    # Structure: ref_cache[scale][query_id] = { "top10": [...], "D": float, "epsilons": { eps: { "B": set, "T": set, "r": int, "has_tie": bool } } }
    ref_cache = defaultdict(dict)
    boundary_csv_rows = []

    for scale in TARGET_SCALES:
        db_name = SCALE_DATABASE_MAP[scale]
        tbl_name = f"chunks_{scale}_fp32_extended"
        conn = get_sensitivity_connection(db_name)
        print(f"Processing scale {scale} on '{db_name}' ({tbl_name})...")

        for qid in range(20):
            q_vec = recovered_vectors[qid]["vector"]
            with conn.cursor() as cur:
                cur.execute("BEGIN READ ONLY;")
                try:
                    cur.execute(f"SET LOCAL seq_page_cost = {canonical_prof['seq_page_cost']};")
                    cur.execute(f"SET LOCAL random_page_cost = {canonical_prof['random_page_cost']};")
                    cur.execute(f"SET LOCAL cpu_operator_cost = {canonical_prof['cpu_operator_cost']};")
                    cur.execute(f"SET LOCAL cpu_tuple_cost = {canonical_prof['cpu_tuple_cost']};")
                    cur.execute("SET LOCAL enable_indexscan = off;")
                    cur.execute("SET LOCAL enable_bitmapscan = off;")
                    cur.execute("SET LOCAL enable_seqscan = on;")
                    cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")

                    # 1. Verify Seq Scan plan
                    explain_sql = sql.SQL("""
                        EXPLAIN (FORMAT JSON)
                        SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                        FROM {schema}.{table}
                        ORDER BY distance ASC
                        LIMIT 10;
                    """).format(schema=sql.Identifier(SCHEMA_NAME), table=sql.Identifier(tbl_name))
                    cur.execute(explain_sql, (q_vec,), prepare=False)
                    parsed_plan = parse_json_plan(cur.fetchone()[0])
                    if parsed_plan.get("scan_node_type") != "Seq Scan":
                        raise RuntimeError(f"GROUND TRUTH SCAN VERIFICATION FAILED: expected Seq Scan, got {parsed_plan.get('scan_node_type')}")

                    # 2. Retrieve top-10 exact ground truth
                    query_sql = sql.SQL("""
                        SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                        FROM {schema}.{table}
                        ORDER BY distance ASC
                        LIMIT 10;
                    """).format(schema=sql.Identifier(SCHEMA_NAME), table=sql.Identifier(tbl_name))
                    cur.execute(query_sql, (q_vec,), prepare=False)
                    top10_rows = cur.fetchall()
                    top10_ids = [r[0] for r in top10_rows]
                    D = float(top10_rows[9][1])

                    # 3. Retrieve all rows up to D + 1e-5 using exact sequential scan
                    near_sql = sql.SQL("""
                        SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                        FROM {schema}.{table}
                        WHERE (embedding <=> %s::vector(768)) <= %s + 1e-5
                        ORDER BY distance ASC;
                    """).format(schema=sql.Identifier(SCHEMA_NAME), table=sql.Identifier(tbl_name))
                    cur.execute(near_sql, (q_vec, q_vec, D), prepare=False)
                    near_rows = [{"chunk_id": r[0], "distance": float(r[1])} for r in cur.fetchall()]
                finally:
                    cur.execute("ROLLBACK;")

            # Calculate B and T across epsilons
            eps_data = {}
            for eps in EPSILONS:
                B_ids = {r["chunk_id"] for r in near_rows if r["distance"] < D - eps}
                T_ids = {r["chunk_id"] for r in near_rows if abs(r["distance"] - D) <= eps}
                r_slots = max(0, 10 - len(B_ids))
                has_tie = len(T_ids) > r_slots

                # Assert mathematical invariants
                assert B_ids.isdisjoint(T_ids), f"Disjointness invariant violated for scale={scale}, q={qid}, eps={eps}"
                assert r_slots >= 0, f"Negative tie slots: {r_slots}"
                assert len(B_ids) <= 10, f"B set size exceeds 10: {len(B_ids)}"

                eps_data[eps] = {
                    "B": B_ids,
                    "T": T_ids,
                    "r": r_slots,
                    "has_tie": has_tie
                }

                boundary_csv_rows.append({
                    "scale": scale,
                    "query_id": qid,
                    "boundary_distance": D,
                    "epsilon": f"{eps:.1e}",
                    "size_B": len(B_ids),
                    "size_T": len(T_ids),
                    "available_tie_slots": r_slots,
                    "has_boundary_tie": has_tie,
                    "B_ids": json.dumps(sorted(list(B_ids))),
                    "T_ids": json.dumps(sorted(list(T_ids)))
                })

            ref_cache[scale][qid] = {
                "top10_ids": top10_ids,
                "D": D,
                "epsilons": eps_data
            }

        conn.close()

    # Export fp32_reference_boundary_groups.csv
    out_csv = os.path.join(PHASE3_DIR, "fp32_reference_boundary_groups.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "scale", "query_id", "boundary_distance", "epsilon",
            "size_B", "size_T", "available_tie_slots", "has_boundary_tie",
            "B_ids", "T_ids"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in boundary_csv_rows:
            writer.writerow(row)
    print(f"[EXPORTED] FP32 reference boundary sets (180 rows) written to {out_csv}")

    return ref_cache, boundary_csv_rows


def load_phase2_baseline() -> tuple[dict, dict]:
    """
    Loads Phase 2 latency_results.csv and plan_capture.jsonl to establish baseline lookup maps.
    """
    print("\n--- Step 3: Loading Phase 2 Baseline Metrics for Reconciliation ---")
    latency_csv_path = os.path.join(PHASE2_DIR, "latency_results.csv")
    plan_jsonl_path = os.path.join(PHASE2_DIR, "plan_capture.jsonl")

    # phase2_latency[(scale, rep, profile, mode, qid)] = { "recall_at_10": float, "tie_aware_recall_at_10": float, "has_distance_tie": bool }
    phase2_latency = {}
    with open(latency_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            k = (int(r["scale"]), r["representation"], r["profile_id"], r["mode"], int(r["query_id"]))
            if k not in phase2_latency:
                phase2_latency[k] = {
                    "recall_at_10": float(r["recall_at_10"]),
                    "tie_aware_recall_at_10": float(r["tie_aware_recall_at_10"]),
                    "has_distance_tie": r["has_distance_tie"].lower() == "true",
                    "boundary_distance": float(r["boundary_distance"]) if r["boundary_distance"] else None
                }

    # phase2_plans[(scale, rep, profile, mode, qid)] = { "scan_node_type": str, "index_name": str }
    phase2_plans = {}
    with open(plan_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mode_val = rec["mode"]
            k = (int(rec["scale"]), rec["representation"], rec["profile_id"], mode_val, int(rec["query_id"]))
            phase2_plans[k] = {
                "scan_node_type": rec.get("scan_node_type"),
                "index_name": rec.get("index_name")
            }

    print(f"[LOADED] Phase 2 latency lookup units: {len(phase2_latency)} (expected 1,980)")
    print(f"[LOADED] Phase 2 plan lookup entries: {len(phase2_plans)}")
    return phase2_latency, phase2_plans


def recompute_candidate_retrievals_and_audit(
    recovered_vectors: dict[int, dict],
    ref_cache: dict,
    phase2_latency: dict,
    phase2_plans: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Executes all 1,980 candidate queries, verifies execution plans, computes metrics across
    the 3 thresholds (5,940 observations), and reconciles against Phase 2 results.
    """
    print("\n--- Step 4: Recomputing 1,980 Candidate Retrievals & Multi-Threshold Metrics ---")

    # The 11 conditions
    conditions = []
    # 1. exact (P4)
    conditions.append({"mode": "exact", "profile_id": CANONICAL_PROFILE_ID})
    # 2. forced_hnsw (P4)
    conditions.append({"mode": "forced_hnsw", "profile_id": CANONICAL_PROFILE_ID})
    # 3. planner_selected (P1..P9)
    for p in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]:
        conditions.append({"mode": "planner_selected", "profile_id": p})

    assert len(conditions) == 11, f"Expected 11 conditions, got {len(conditions)}"

    detailed_query_results = []  # 5,940 rows
    reconciliation_rows = []     # 1,980 rows
    reconciliation_mismatches = []

    total_units_checked = 0

    for scale in TARGET_SCALES:
        db_name = SCALE_DATABASE_MAP[scale]
        conn = get_sensitivity_connection(db_name)
        print(f"\nProcessing scale {scale} on '{db_name}'...")

        for rep in TARGET_REPRESENTATIONS:
            tbl_name = f"chunks_{scale}_{rep}"
            cast_type = "halfvec(768)" if rep == "fp16_extended" else "vector(768)"

            for cond in conditions:
                mode = cond["mode"]
                prof_id = cond["profile_id"]
                prof = PLANNER_COST_PROFILES[prof_id]

                for qid in range(20):
                    q_vec = recovered_vectors[qid]["vector"]
                    unit_key = (scale, rep, prof_id, mode, qid)
                    total_units_checked += 1

                    # Execute candidate retrieval inside read-only transaction
                    with conn.cursor() as cur:
                        cur.execute("BEGIN READ ONLY;")
                        try:
                            # Apply GUCs
                            cur.execute(f"SET LOCAL seq_page_cost = {prof['seq_page_cost']};")
                            cur.execute(f"SET LOCAL random_page_cost = {prof['random_page_cost']};")
                            cur.execute(f"SET LOCAL cpu_operator_cost = {prof['cpu_operator_cost']};")
                            cur.execute(f"SET LOCAL cpu_tuple_cost = {prof['cpu_tuple_cost']};")
                            cur.execute("SET LOCAL hnsw.ef_search = 40;")
                            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0;")

                            if mode == "exact":
                                cur.execute("SET LOCAL enable_seqscan = on;")
                                cur.execute("SET LOCAL enable_indexscan = off;")
                                cur.execute("SET LOCAL enable_bitmapscan = off;")
                            elif mode == "forced_hnsw":
                                cur.execute("SET LOCAL enable_seqscan = off;")
                                cur.execute("SET LOCAL enable_indexscan = on;")
                                cur.execute("SET LOCAL enable_bitmapscan = off;")
                            else:  # planner_selected
                                cur.execute("SET LOCAL enable_seqscan = on;")
                                cur.execute("SET LOCAL enable_indexscan = on;")
                                cur.execute("SET LOCAL enable_bitmapscan = on;")

                            # Capture plan
                            explain_sql = sql.SQL("""
                                EXPLAIN (FORMAT JSON)
                                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                                FROM {schema}.{table}
                                ORDER BY distance ASC
                                LIMIT 10;
                            """).format(
                                cast=sql.SQL(cast_type),
                                schema=sql.Identifier(SCHEMA_NAME),
                                table=sql.Identifier(tbl_name)
                            )
                            cur.execute(explain_sql, (q_vec,), prepare=False)
                            plan_parsed = parse_json_plan(cur.fetchone()[0])
                            actual_scan_node = plan_parsed.get("scan_node_type")
                            actual_index_name = plan_parsed.get("index_name")

                            # Execute query
                            exec_sql = sql.SQL("""
                                SELECT chunk_id, (embedding <=> %s::{cast}) AS distance
                                FROM {schema}.{table}
                                ORDER BY distance ASC
                                LIMIT 10;
                            """).format(
                                cast=sql.SQL(cast_type),
                                schema=sql.Identifier(SCHEMA_NAME),
                                table=sql.Identifier(tbl_name)
                            )
                            cur.execute(exec_sql, (q_vec,), prepare=False)
                            cand_rows = cur.fetchall()
                            cand_ids = [r[0] for r in cand_rows]
                        finally:
                            cur.execute("ROLLBACK;")

                    # Compare with FP32 reference ground truth
                    ref_data = ref_cache[scale][qid]
                    ref_top10 = ref_data["top10_ids"]
                    S = set(cand_ids)

                    conv_hits = len(S.intersection(set(ref_top10)))
                    conv_recall = conv_hits / 10.0

                    tie_recalls_by_eps = {}
                    for eps in EPSILONS:
                        b_info = ref_data["epsilons"][eps]
                        B = b_info["B"]
                        T = b_info["T"]
                        r = b_info["r"]
                        hits_B = len(S.intersection(B))
                        hits_T = len(S.intersection(T))
                        credited_T = min(hits_T, r)
                        tie_hits = hits_B + credited_T
                        tie_recall = tie_hits / 10.0

                        # Assert boundedness
                        assert 0.0 <= tie_recall <= 1.0, f"Tie recall {tie_recall} out of bounds"
                        assert tie_recall >= conv_recall - 1e-9, f"Tie recall {tie_recall} < conv recall {conv_recall}"

                        tie_recalls_by_eps[eps] = tie_recall

                        detailed_query_results.append({
                            "scale": scale,
                            "representation": rep,
                            "profile_id": prof_id,
                            "mode": mode,
                            "query_id": qid,
                            "epsilon": f"{eps:.1e}",
                            "conventional_recall": round(conv_recall, 4),
                            "tie_aware_recall": round(tie_recall, 4),
                            "size_B": len(B),
                            "size_T": len(T),
                            "available_tie_slots": r,
                            "has_boundary_tie": b_info["has_tie"],
                            "scan_node_type": actual_scan_node,
                            "index_name": actual_index_name or "",
                            "candidate_ids": json.dumps(cand_ids)
                        })

                    # Reconcile against Phase 2 baseline
                    p2_lat = phase2_latency[unit_key]
                    p2_plan = phase2_plans.get(unit_key, {})

                    p2_conv = p2_lat["recall_at_10"]
                    p2_tie_1e6 = p2_lat["tie_aware_recall_at_10"]
                    recomputed_tie_1e6 = tie_recalls_by_eps[1e-6]

                    diff_conv = abs(conv_recall - p2_conv)
                    diff_tie = abs(recomputed_tie_1e6 - p2_tie_1e6)

                    match_conv = diff_conv <= ROUNDING_TOLERANCE
                    match_tie = diff_tie <= ROUNDING_TOLERANCE
                    match_plan = (actual_scan_node == p2_plan.get("scan_node_type")) and (
                        (actual_index_name or "") == (p2_plan.get("index_name") or "")
                    )

                    status = "MATCH" if (match_conv and match_tie and match_plan) else "MISMATCH"
                    if status == "MISMATCH":
                        mismatch_detail = {
                            "unit_key": unit_key,
                            "match_conv": match_conv,
                            "diff_conv": diff_conv,
                            "match_tie": match_tie,
                            "diff_tie": diff_tie,
                            "match_plan": match_plan,
                            "actual_plan": (actual_scan_node, actual_index_name),
                            "phase2_plan": (p2_plan.get("scan_node_type"), p2_plan.get("index_name"))
                        }
                        reconciliation_mismatches.append(mismatch_detail)

                    reconciliation_rows.append({
                        "scale": scale,
                        "representation": rep,
                        "profile_id": prof_id,
                        "mode": mode,
                        "query_id": qid,
                        "phase2_conventional_recall": round(p2_conv, 4),
                        "recomputed_conventional_recall": round(conv_recall, 4),
                        "conventional_diff": round(diff_conv, 6),
                        "conventional_match": match_conv,
                        "phase2_tie_aware_recall_1e6": round(p2_tie_1e6, 4),
                        "recomputed_tie_aware_recall_1e6": round(recomputed_tie_1e6, 4),
                        "tie_aware_diff": round(diff_tie, 6),
                        "tie_aware_match": match_tie,
                        "phase2_scan_node_type": p2_plan.get("scan_node_type", ""),
                        "phase3a_scan_node_type": actual_scan_node,
                        "phase2_index_name": p2_plan.get("index_name") or "",
                        "phase3a_index_name": actual_index_name or "",
                        "plan_match": match_plan,
                        "reconciliation_status": status
                    })

        conn.close()

    print(f"\n[RECONCILIATION COMPLETE] Total units audited: {total_units_checked}")
    print(f"[RECONCILIATION RESULT] Matches: {total_units_checked - len(reconciliation_mismatches)}, Mismatches: {len(reconciliation_mismatches)}")

    if reconciliation_mismatches:
        print("\nCRITICAL WARNING: The following reconciliation mismatches were detected:")
        for m in reconciliation_mismatches[:10]:
            print(f"  - {m}")
        if len(reconciliation_mismatches) > 10:
            print(f"  ... and {len(reconciliation_mismatches) - 10} more.")

    # Export CSVs
    # 1. tie_sensitivity_query_results.csv (5,940 rows)
    out_results_csv = os.path.join(PHASE3_DIR, "tie_sensitivity_query_results.csv")
    with open(out_results_csv, "w", newline="", encoding="utf-8") as f:
        fields = list(detailed_query_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in detailed_query_results:
            writer.writerow(r)
    print(f"[EXPORTED] Detailed query sensitivity results (5,940 rows) written to {out_results_csv}")

    # 2. phase2_recall_reconciliation.csv (1,980 rows)
    out_recon_csv = os.path.join(PHASE3_DIR, "phase2_recall_reconciliation.csv")
    with open(out_recon_csv, "w", newline="", encoding="utf-8") as f:
        fields = list(reconciliation_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in reconciliation_rows:
            writer.writerow(r)
    print(f"[EXPORTED] Phase 2 reconciliation audit (1,980 rows) written to {out_recon_csv}")

    return detailed_query_results, reconciliation_rows, reconciliation_mismatches


def generate_tie_sensitivity_summary(detailed_results: list[dict], ref_cache: dict) -> list[dict]:
    """
    Computes summary statistics grouped by (scale, representation, mode, epsilon)
    and overall by (scale, epsilon).
    """
    print("\n--- Step 5: Generating Tie Sensitivity Summary Statistics ---")

    # Group by (scale, representation, mode, epsilon)
    groups = defaultdict(list)
    for r in detailed_results:
        key = (r["scale"], r["representation"], r["mode"], r["epsilon"])
        groups[key].append(r)

    summary_rows = []
    for (scale, rep, mode, eps), rows in sorted(groups.items()):
        conv_vals = [r["conventional_recall"] for r in rows]
        tie_vals = [r["tie_aware_recall"] for r in rows]
        ties_count = sum(1 for r in rows if r["has_boundary_tie"])

        mean_conv = statistics.mean(conv_vals)
        median_conv = statistics.median(conv_vals)
        mean_tie = statistics.mean(tie_vals)
        median_tie = statistics.median(tie_vals)
        mean_delta = mean_tie - mean_conv

        summary_rows.append({
            "scale": scale,
            "representation": rep,
            "mode": mode,
            "epsilon": eps,
            "num_observations": len(rows),
            "tie_prevalence_count": ties_count,
            "mean_conventional_recall": round(mean_conv, 4),
            "median_conventional_recall": round(median_conv, 4),
            "mean_tie_aware_recall": round(mean_tie, 4),
            "median_tie_aware_recall": round(median_tie, 4),
            "mean_tie_recall_boost": round(mean_delta, 4)
        })

    # Also compute overall per (scale, epsilon)
    scale_eps_groups = defaultdict(list)
    for r in detailed_results:
        scale_eps_groups[(r["scale"], r["epsilon"])].append(r)

    for (scale, eps), rows in sorted(scale_eps_groups.items()):
        conv_vals = [r["conventional_recall"] for r in rows]
        tie_vals = [r["tie_aware_recall"] for r in rows]
        ties_count = sum(1 for r in rows if r["has_boundary_tie"])

        summary_rows.append({
            "scale": scale,
            "representation": "ALL",
            "mode": "ALL",
            "epsilon": eps,
            "num_observations": len(rows),
            "tie_prevalence_count": ties_count,
            "mean_conventional_recall": round(statistics.mean(conv_vals), 4),
            "median_conventional_recall": round(statistics.median(conv_vals), 4),
            "mean_tie_aware_recall": round(statistics.mean(tie_vals), 4),
            "median_tie_aware_recall": round(statistics.median(tie_vals), 4),
            "mean_tie_recall_boost": round(statistics.mean(tie_vals) - statistics.mean(conv_vals), 4)
        })

    out_summary_csv = os.path.join(PHASE3_DIR, "tie_sensitivity_summary.csv")
    with open(out_summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"[EXPORTED] Tie sensitivity summary written to {out_summary_csv}")

    return summary_rows


def export_audit_manifest(
    recovered_vectors: dict,
    ref_cache: dict,
    mismatches: list,
    detailed_results: list
):
    """
    Exports comprehensive phase3a_audit_manifest.json with all metadata.
    """
    manifest_path = os.path.join(PHASE3_DIR, "phase3a_audit_manifest.json")
    print(f"\n--- Step 6: Exporting Audit Manifest to {manifest_path} ---")

    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip()

    # Pre-calculate tie prevalence across unique (scale, qid) pairs for each epsilon
    ties_by_scale_eps = defaultdict(set)
    for sc in TARGET_SCALES:
        for qid in range(20):
            for eps in EPSILONS:
                if ref_cache[sc][qid]["epsilons"][eps]["has_tie"]:
                    ties_by_scale_eps[(sc, f"{eps:.1e}")].add(qid)

    manifest_data = {
        "audit_title": "Phase 3A: Boundary-Tie Threshold Sensitivity and Query-Vector Validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head_sha,
        "base_phase2_directory": os.path.abspath(PHASE2_DIR),
        "audit_output_directory": os.path.abspath(PHASE3_DIR),
        "reconciliation_summary": {
            "total_query_condition_units": 1980,
            "total_reconciled_matches": 1980 - len(mismatches),
            "total_reconciled_mismatches": len(mismatches),
            "reconciliation_status": "PASSED" if len(mismatches) == 0 else "FAILED",
            "rounding_tolerance": ROUNDING_TOLERANCE
        },
        "query_vector_recovery": {
            "total_unique_queries": len(recovered_vectors),
            "dimension": 768,
            "source_artifact": "plan_capture.jsonl (FP32 representations)",
            "verification_status": "IDENTICAL_ACROSS_ALL_OCCURRENCES"
        },
        "threshold_sensitivity": {
            "evaluated_epsilons": [f"{eps:.1e}" for eps in EPSILONS],
            "total_observations": len(detailed_results),
            "tie_prevalence_unique_queries": {
                f"scale_{sc}_eps_{eps}": sorted(list(qids))
                for (sc, eps), qids in sorted(ties_by_scale_eps.items())
            }
        },
        "artifacts_generated": {
            "recovered_query_vectors_csv": "recovered_query_vectors.csv",
            "fp32_reference_boundary_groups_csv": "fp32_reference_boundary_groups.csv",
            "tie_sensitivity_query_results_csv": "tie_sensitivity_query_results.csv",
            "phase2_recall_reconciliation_csv": "phase2_recall_reconciliation.csv",
            "tie_sensitivity_summary_csv": "tie_sensitivity_summary.csv",
            "phase3a_audit_manifest_json": "phase3a_audit_manifest.json",
            "phase3a_console_log": "phase3a_console.log"
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[EXPORTED] Audit manifest successfully generated.")


def main():
    start_time = datetime.now(timezone.utc)
    print(f"Phase 3A Audit Execution Started at {start_time.isoformat()}")

    # 1. Preconditions
    verify_preconditions()

    # 2. Recover Query Vectors
    recovered_vectors = recover_query_vectors()

    # 3. Reconstruct FP32 Reference Boundary Sets
    ref_cache, boundary_csv_rows = reconstruct_fp32_reference_boundary_groups(recovered_vectors)

    # 4. Load Phase 2 Baseline
    phase2_latency, phase2_plans = load_phase2_baseline()

    # 5. Recompute Candidate Retrievals & Multi-Threshold Metrics
    detailed_results, reconciliation_rows, mismatches = recompute_candidate_retrievals_and_audit(
        recovered_vectors, ref_cache, phase2_latency, phase2_plans
    )

    # 6. Generate Summary Statistics
    generate_tie_sensitivity_summary(detailed_results, ref_cache)

    # 7. Export Manifest
    export_audit_manifest(recovered_vectors, ref_cache, mismatches, detailed_results)

    end_time = datetime.now(timezone.utc)
    elapsed = (end_time - start_time).total_seconds()
    print("\n================================================================================")
    print(f"PHASE 3A AUDIT COMPLETE in {elapsed:.2f}s")
    print("================================================================================")


if __name__ == "__main__":
    main()
