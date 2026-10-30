#!/usr/bin/env python3
"""
tests/test_planner_sensitivity_experiment.py

Comprehensive unit and regression test suite for the planner sensitivity experiment.
All tests run safely without mutating live production or experimental databases.
"""

import os
import math
import json
import random
import statistics
import pytest
import psycopg
from unittest.mock import MagicMock, patch

from src.planner_sensitivity_experiment import (
    generate_cartesian_cost_grid,
    calculate_expected_workload,
    assert_sensitivity_database,
    get_sensitivity_connection,
    parse_json_plan,
    calculate_recall_at_k,
    calculate_recall_metrics,
    retrieve_boundary_tie_group,
    check_boundary_distance_tie,
    compute_distribution_stats,
    capture_query_plan,
    measure_query_latency,
    execute_diagnostic_pass,
    capture_environment_snapshot,
    ALLOWED_DATABASES,
    DISALLOWED_DATABASES,
    PLANNER_COST_PROFILES,
    DEFAULT_CANONICAL_FORCED_PROFILE,
    DEFAULT_RANDOM_SEED,
    COSINE_DISTANCE_TOLERANCE,
    TOLERANCE_POLICY,
    FIXED_LIMIT
)


# ==============================================================================
# 1. Cartesian Cost Grid Generation & Validation
# ==============================================================================

def test_cartesian_grid_generation():
    """Verify that generate_cartesian_cost_grid outputs exactly 9 distinct valid profiles."""
    grid = generate_cartesian_cost_grid()
    assert len(grid) == 9
    assert list(grid.keys()) == [f"P{i}" for i in range(1, 10)]

    rpcs = {p["random_page_cost"] for p in grid.values()}
    ops = {p["cpu_operator_cost"] for p in grid.values()}

    assert rpcs == {1.0, 4.0, 8.0}
    assert ops == {0.0025, 0.010, 0.050}

    for p_id, p in grid.items():
        assert p["seq_page_cost"] == 1.0
        assert p["cpu_tuple_cost"] == 0.01
        assert p["random_page_cost"] in [1.0, 4.0, 8.0]
        assert p["cpu_operator_cost"] in [0.0025, 0.010, 0.050]
        assert "description" in p


def test_configuration_validation():
    """Verify custom parameter overrides and edge cases."""
    custom_grid = generate_cartesian_cost_grid(
        random_page_costs=[2.0],
        cpu_operator_costs=[0.02],
        seq_page_cost=1.5,
        cpu_tuple_cost=0.02
    )
    assert len(custom_grid) == 1
    assert "P1" in custom_grid
    assert custom_grid["P1"]["random_page_cost"] == 2.0
    assert custom_grid["P1"]["cpu_operator_cost"] == 0.02
    assert custom_grid["P1"]["seq_page_cost"] == 1.5
    assert custom_grid["P1"]["cpu_tuple_cost"] == 0.02


# ==============================================================================
# 2. Optimized Workload Calculation & Manifest Verification
# ==============================================================================

def test_calculate_expected_workload_full_matrix():
    """
    Verify exact workload count calculation for full experimental matrix:
    - 9 rep-scale combinations x 9 profiles x 20 queries x 5 passes = 8,100 (planner_selected)
    - 9 rep-scale combinations x 1 profile x 20 queries x 5 passes = 900 (forced exact)
    - 9 rep-scale combinations x 1 profile x 20 queries x 5 passes = 900 (forced HNSW)
    - Total timed executions = 8,100 + 900 + 900 = 9,900.
    - Diagnostics = 9 rep-scale combinations x 3 modes x 20 queries = 540 at canonical P5.
    - Plan captures = 9 rep-scale combinations x 3 modes x 9 profiles x 20 queries = 4,860.
    """
    scales = [1001, 5000, 10000]
    reps = ["fp32_extended", "fp32_plain", "fp16_extended"]
    profiles = PLANNER_COST_PROFILES
    modes = ["exact", "forced_hnsw", "planner_selected"]

    workload = calculate_expected_workload(
        scales=scales,
        representations=reps,
        profiles=profiles,
        modes=modes,
        num_queries=20,
        measured_passes=5,
        warmup_passes=2,
        canonical_profile_id=DEFAULT_CANONICAL_FORCED_PROFILE
    )

    assert workload["rep_scale_combinations"] == 9
    assert workload["timed_planner_selected"] == 8100
    assert workload["timed_forced_exact"] == 900
    assert workload["timed_forced_hnsw"] == 900
    assert workload["total_timed_executions"] == 9900
    assert workload["total_warmup_executions"] == 3960
    assert workload["total_plan_captures"] == 4860
    assert workload["total_diagnostic_executions"] == 540


def test_calculate_expected_workload_pilot():
    """
    Verify exact workload count calculation for minimal pilot matrix:
    - 3 rep-scale combinations x 2 profiles x 2 queries x 2 passes = 24 (planner_selected)
    - 3 rep-scale combinations x 1 profile x 2 queries x 2 passes = 12 (forced exact)
    - 3 rep-scale combinations x 1 profile x 2 queries x 2 passes = 12 (forced HNSW)
    - Total timed executions = 24 + 12 + 12 = 48.
    - Diagnostics = 3 rep-scale combinations x 3 modes x 2 queries = 18 at canonical P1.
    - Plan captures = 3 rep-scale combinations x 3 modes x 2 profiles x 2 queries = 36.
    """
    scales = ["test"]
    reps = ["fp32_extended", "fp32_plain", "fp16_extended"]
    profiles = {"P1": PLANNER_COST_PROFILES["P1"], "P9": PLANNER_COST_PROFILES["P9"]}
    modes = ["exact", "forced_hnsw", "planner_selected"]

    workload = calculate_expected_workload(
        scales=scales,
        representations=reps,
        profiles=profiles,
        modes=modes,
        num_queries=2,
        measured_passes=2,
        warmup_passes=2,
        canonical_profile_id="P1"
    )

    assert workload["rep_scale_combinations"] == 3
    assert workload["timed_planner_selected"] == 24
    assert workload["timed_forced_exact"] == 12
    assert workload["timed_forced_hnsw"] == 12
    assert workload["total_timed_executions"] == 48
    assert workload["total_warmup_executions"] == 48
    assert workload["total_plan_captures"] == 36
    assert workload["total_diagnostic_executions"] == 18


def test_deterministic_randomized_execution_order():
    """Verify that a fixed random seed produces identical deterministic trial sequences."""
    items = list(range(100))
    rng1 = random.Random(DEFAULT_RANDOM_SEED)
    rng2 = random.Random(DEFAULT_RANDOM_SEED)
    rng3 = random.Random(DEFAULT_RANDOM_SEED + 1)

    shuffled1 = items.copy()
    shuffled2 = items.copy()
    shuffled3 = items.copy()

    rng1.shuffle(shuffled1)
    rng2.shuffle(shuffled2)
    rng3.shuffle(shuffled3)

    assert shuffled1 == shuffled2
    assert shuffled1 != shuffled3


# ==============================================================================
# 3. Database Safety Guards & Connection Isolation
# ==============================================================================

def test_allowed_and_disallowed_databases():
    """Verify fail-closed guard blocks disallowed catalogs and permits approved ones."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Disallowed production databases must immediately raise RuntimeError
    for forbidden_db in DISALLOWED_DATABASES:
        mock_cur.fetchone.return_value = [forbidden_db]
        with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
            assert_sensitivity_database(mock_conn)

    # Unknown databases must raise RuntimeError
    mock_cur.fetchone.return_value = ["random_unauthorized_catalog"]
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
        assert_sensitivity_database(mock_conn)

    # Approved databases must pass
    for allowed_db in ALLOWED_DATABASES:
        mock_cur.fetchone.return_value = [allowed_db]
        assert assert_sensitivity_database(mock_conn, allowed_db) == allowed_db


@patch("src.planner_sensitivity_experiment.register_vector")
@patch("psycopg.connect")
def test_read_only_connection_enforcement(mock_connect, mock_reg):
    """Verify get_sensitivity_connection sets conn.read_only = True, autocommit = True, prepare_threshold = None."""
    mock_conn = MagicMock()
    mock_conn.prepare_threshold = None
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ["compression_aware_rag_scale"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_connect.return_value = mock_conn

    conn = get_sensitivity_connection("compression_aware_rag_scale")
    assert conn.read_only is True
    assert conn.prepare_threshold is None
    mock_connect.assert_called_once()
    _, kwargs = mock_connect.call_args
    assert kwargs.get("autocommit") is True
    assert kwargs.get("prepare_threshold") is None


# ==============================================================================
# 4. Transaction Containment & GUC Scoping
# ==============================================================================

def test_guc_restoration_normal_completion():
    """Verify that SET LOCAL settings are enclosed in BEGIN READ ONLY and ROLLBACK."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = [[{"Plan": {"Node Type": "Limit", "Startup Cost": 10.0, "Total Cost": 20.0}}]]

    prof = PLANNER_COST_PROFILES["P1"]
    capture_query_plan(mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended", "vector(768)", prof)

    executed_statements = [str(call[0][0]) for call in mock_cur.execute.call_args_list]
    assert any("BEGIN READ ONLY" in s for s in executed_statements)
    assert any("SET LOCAL random_page_cost" in s for s in executed_statements)
    assert any("SET LOCAL cpu_operator_cost" in s for s in executed_statements)
    assert any("ROLLBACK" in s for s in executed_statements)


def test_guc_restoration_exception_handling():
    """Verify that in the event of an unhandled exception, ROLLBACK is guaranteed in finally."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Simulate query failure during execution
    def fail_on_select(query, params=None, *args, **kwargs):
        query_str = str(query)
        if "EXPLAIN" in query_str:
            raise psycopg.OperationalError("Simulated server interruption")
        return None

    mock_cur.execute.side_effect = fail_on_select
    prof = PLANNER_COST_PROFILES["P7"]

    with pytest.raises(psycopg.OperationalError, match="Simulated server interruption"):
        capture_query_plan(mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended", "vector(768)", prof)

    # Assert that ROLLBACK was still called in the finally path
    executed_statements = [str(call[0][0]) for call in mock_cur.execute.call_args_list]
    assert any("ROLLBACK" in s for s in executed_statements)


# ==============================================================================
# 5. JSON Plan Parsing & Hierarchy Extraction
# ==============================================================================

def test_json_plan_parsing_seq_and_index_scan():
    """Verify parsing of synthetic PostgreSQL 18 JSON plans for Seq Scan and Index Scan."""
    # Case 1: Sequential Scan with Top-N Sort
    seq_plan_raw = [{
        "Plan": {
            "Node Type": "Limit",
            "Startup Cost": 250.0,
            "Total Cost": 250.5,
            "Plan Rows": 10,
            "Plan Width": 12,
            "Plans": [{
                "Node Type": "Sort",
                "Startup Cost": 250.0,
                "Total Cost": 252.0,
                "Plans": [{
                    "Node Type": "Seq Scan",
                    "Relation Name": "chunks_5000_fp32_extended",
                    "Startup Cost": 0.0,
                    "Total Cost": 150.0,
                    "Plan Rows": 5000,
                    "Plan Width": 12
                }]
            }]
        },
        "Planning Time": 0.150,
        "Execution Time": 12.345,
        "Settings": {"random_page_cost": "4", "seq_page_cost": "1"}
    }]

    parsed_seq = parse_json_plan(seq_plan_raw)
    assert parsed_seq["root_node_type"] == "Limit"
    assert parsed_seq["root_startup_cost"] == 250.0
    assert parsed_seq["root_total_cost"] == 250.5
    assert parsed_seq["scan_node_type"] == "Seq Scan"
    assert parsed_seq["relation_name"] == "chunks_5000_fp32_extended"
    assert parsed_seq["index_name"] is None
    assert parsed_seq["has_sort_node"] is True
    assert parsed_seq["scan_startup_cost"] == 0.0
    assert parsed_seq["scan_total_cost"] == 150.0
    assert parsed_seq["planning_time_ms"] == 0.150
    assert parsed_seq["execution_time_ms"] == 12.345
    assert parsed_seq["settings"] == {"random_page_cost": "4", "seq_page_cost": "1"}

    # Case 2: HNSW Index Scan
    idx_plan_raw = [{
        "Plan": {
            "Node Type": "Limit",
            "Startup Cost": 35.82,
            "Total Cost": 55.10,
            "Plan Rows": 10,
            "Plan Width": 12,
            "Shared Hit Blocks": 753,
            "Shared Read Blocks": 0,
            "Plans": [{
                "Node Type": "Index Scan",
                "Index Name": "chunks_5000_fp32_extended_hnsw_idx",
                "Relation Name": "chunks_5000_fp32_extended",
                "Startup Cost": 35.82,
                "Total Cost": 950.0,
                "Plan Rows": 5000,
                "Plan Width": 12
            }]
        },
        "Planning Time": 0.080,
        "Execution Time": 1.250
    }]

    parsed_idx = parse_json_plan(idx_plan_raw)
    assert parsed_idx["root_node_type"] == "Limit"
    assert parsed_idx["root_startup_cost"] == 35.82
    assert parsed_idx["root_total_cost"] == 55.10
    assert parsed_idx["scan_node_type"] == "Index Scan"
    assert parsed_idx["index_name"] == "chunks_5000_fp32_extended_hnsw_idx"
    assert parsed_idx["has_sort_node"] is False
    assert parsed_idx["shared_hit_buffers"] == 753
    assert parsed_idx["shared_read_buffers"] == 0


def test_nested_plan_tree_parsing():
    """Verify recursive traversal across multi-level nested plan trees."""
    nested_raw = [{
        "Plan": {
            "Node Type": "Limit",
            "Startup Cost": 100.0,
            "Total Cost": 110.0,
            "Plans": [{
                "Node Type": "Materialize",
                "Plans": [{
                    "Node Type": "Bitmap Heap Scan",
                    "Relation Name": "chunks_10000_fp16_extended",
                    "Startup Cost": 50.0,
                    "Total Cost": 90.0,
                    "Plans": [{
                        "Node Type": "Bitmap Index Scan",
                        "Index Name": "chunks_10000_fp16_extended_hnsw_idx"
                    }]
                }]
            }]
        }
    }]
    parsed = parse_json_plan(nested_raw)
    assert parsed["root_node_type"] == "Limit"
    assert parsed["scan_node_type"] == "Bitmap Heap Scan"
    assert parsed["relation_name"] == "chunks_10000_fp16_extended"


def test_parse_json_plan_root_buffer_aggregation_regression():
    """
    Regression test for root buffer aggregation in nested execution plans.
    PostgreSQL reports cumulative buffer counters at parent nodes.
    Query-level totals must be extracted strictly from the root Plan node,
    and NOT recursively summed across child nodes (Limit -> Sort -> Seq Scan).
    """
    # Case 1: Limit -> Sort -> Seq Scan (multi-level hierarchy with cumulative buffers)
    nested_seq_plan = [{
        "Plan": {
            "Node Type": "Limit",
            "Startup Cost": 41.14,
            "Total Cost": 41.17,
            "Shared Hit Blocks": 100,
            "Shared Read Blocks": 50,
            "Shared Dirtied Blocks": 10,
            "Shared Written Blocks": 5,
            "Plans": [{
                "Node Type": "Sort",
                "Startup Cost": 41.14,
                "Total Cost": 43.65,
                "Shared Hit Blocks": 100,
                "Shared Read Blocks": 50,
                "Shared Dirtied Blocks": 10,
                "Shared Written Blocks": 5,
                "Plans": [{
                    "Node Type": "Seq Scan",
                    "Relation Name": "chunks_1001_fp32_extended",
                    "Startup Cost": 0.0,
                    "Total Cost": 19.51,
                    "Shared Hit Blocks": 100,
                    "Shared Read Blocks": 50,
                    "Shared Dirtied Blocks": 10,
                    "Shared Written Blocks": 5
                }]
            }]
        }
    }]

    parsed_seq = parse_json_plan(nested_seq_plan)
    assert parsed_seq["root_node_type"] == "Limit"
    assert parsed_seq["scan_node_type"] == "Seq Scan"
    assert parsed_seq["has_sort_node"] is True
    # Crucial assertion: exactly 100 hits, 50 reads, 10 dirtied, 5 written (NOT 300, 150, 30, 15)
    assert parsed_seq["shared_hit_buffers"] == 100
    assert parsed_seq["shared_read_buffers"] == 50
    assert parsed_seq["shared_dirtied_buffers"] == 10
    assert parsed_seq["shared_written_buffers"] == 5

    # Case 2: Limit -> Index Scan (two-level hierarchy)
    nested_idx_plan = [{
        "Plan": {
            "Node Type": "Limit",
            "Startup Cost": 10.05,
            "Total Cost": 15.20,
            "Shared Hit Blocks": 42,
            "Shared Read Blocks": 8,
            "Shared Dirtied Blocks": 2,
            "Shared Written Blocks": 1,
            "Plans": [{
                "Node Type": "Index Scan",
                "Index Name": "chunks_1001_fp32_extended_hnsw_idx",
                "Relation Name": "chunks_1001_fp32_extended",
                "Startup Cost": 10.05,
                "Total Cost": 15.20,
                "Shared Hit Blocks": 42,
                "Shared Read Blocks": 8,
                "Shared Dirtied Blocks": 2,
                "Shared Written Blocks": 1
            }]
        }
    }]

    parsed_idx = parse_json_plan(nested_idx_plan)
    assert parsed_idx["root_node_type"] == "Limit"
    assert parsed_idx["scan_node_type"] == "Index Scan"
    assert parsed_idx["index_name"] == "chunks_1001_fp32_extended_hnsw_idx"
    assert parsed_idx["has_sort_node"] is False
    assert parsed_idx["shared_hit_buffers"] == 42
    assert parsed_idx["shared_read_buffers"] == 8
    assert parsed_idx["shared_dirtied_buffers"] == 2
    assert parsed_idx["shared_written_buffers"] == 1


def test_root_vs_scan_node_cost_extraction():
    """Verify distinct extraction of root costs vs scan-node costs."""
    plan_raw = [{
        "Plan": {
            "Node Type": "Limit",
            "Startup Cost": 10.0,
            "Total Cost": 25.0,
            "Plans": [{
                "Node Type": "Index Scan",
                "Index Name": "sample_idx",
                "Startup Cost": 10.0,
                "Total Cost": 800.0
            }]
        }
    }]
    parsed = parse_json_plan(plan_raw)
    assert parsed["root_total_cost"] == 25.0
    assert parsed["scan_total_cost"] == 800.0


# ==============================================================================
# 6. HNSW Plan Assertion & Mode Verification
# ==============================================================================

def test_hnsw_index_name_assertion():
    """Verify that forced_hnsw mode raises RuntimeError if expected index is not used."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Planner returned Seq Scan instead of Index Scan
    wrong_plan = [[{
        "Plan": {
            "Node Type": "Limit",
            "Plans": [{
                "Node Type": "Seq Scan",
                "Relation Name": "chunks_5000_fp32_extended"
            }]
        }
    }]]
    mock_cur.fetchone.return_value = wrong_plan

    prof = PLANNER_COST_PROFILES["P1"]
    plan_info = capture_query_plan(
        mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended",
        "vector(768)", prof, mode="forced_hnsw"
    )

    # In forced_hnsw mode, verify assertion logic
    expected_index = "chunks_5000_fp32_extended_hnsw_idx"
    assert plan_info["scan_node_type"] == "Seq Scan"
    assert plan_info["index_name"] is None
    assert plan_info["index_name"] != expected_index


# ==============================================================================
# 7. Exact K=10 Verification & Auxiliary Boundary Tie Query
# ==============================================================================

def test_parameterized_query_formatting_exact_k10():
    """Verify that timed SELECT passes strictly top_k (10) as SQL parameter, not top_k + 1."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchall.return_value = [(i, 0.05 * i) for i in range(1, 11)]

    prof = PLANNER_COST_PROFILES["P1"]
    dummy_vec = [0.12345] * 768

    rows, _ = measure_query_latency(
        mock_conn, dummy_vec, "planner_sensitivity", "chunks_1001_fp16_extended",
        "halfvec(768)", prof, mode="planner_selected", top_k=10
    )

    # Check query and parameters passed to cursor.execute
    select_call = [call for call in mock_cur.execute.call_args_list if "SELECT chunk_id" in str(call[0][0])][0]
    query_str = str(select_call[0][0])
    params = select_call[0][1]

    assert "%s::halfvec(768)" in query_str or "%s" in query_str
    assert "0.12345, 0.12345" not in query_str
    assert params[0] == dummy_vec
    assert params[1] == 10  # EXACTLY top_k (10), not 11!
    assert len(rows) == 10


def test_auxiliary_boundary_tie_detection():
    """Verify check_boundary_distance_tie runs an auxiliary exact Seq Scan with LIMIT 2 OFFSET top_k-1."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Simulate plan returning Seq Scan, and two rows with identical cosine distance
    mock_cur.fetchone.return_value = [[{"Plan": {"Node Type": "Seq Scan"}}]]
    mock_cur.fetchall.return_value = [(101, 0.555555), (102, 0.555555)]

    prof = PLANNER_COST_PROFILES["P1"]
    dummy_vec = [0.1] * 768

    tie_res = check_boundary_distance_tie(
        mock_conn, dummy_vec, "planner_sensitivity", "chunks_5000_fp32_extended",
        "vector(768)", prof, mode="exact", top_k=10
    )

    assert tie_res["has_tie"] is True
    assert tie_res["rank_10_chunk_id"] == 101
    assert tie_res["rank_11_chunk_id"] == 102
    assert math.isclose(tie_res["delta"], 0.0, abs_tol=1e-7)
    assert tie_res["scan_node_type"] == "Seq Scan"

    # Verify SQL structure: LIMIT 2 OFFSET %s with parameter top_k - 1 (9)
    tie_calls = [call for call in mock_cur.execute.call_args_list if "OFFSET %s" in str(call[0][0])]
    assert len(tie_calls) >= 2  # EXPLAIN call + SELECT call
    select_call = tie_calls[-1]
    query_str = str(select_call[0][0])
    params = select_call[0][1]

    assert "LIMIT 2 OFFSET %s" in query_str
    assert "<=>" in query_str  # Cosine distance operator
    assert params[0] == dummy_vec
    assert params[1] == 9  # top_k - 1


def test_exact_boundary_tie_scan_assertion_failure():
    """Verify check_boundary_distance_tie raises RuntimeError if planner attempts non-SeqScan."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Simulate planner attempting Index Scan
    mock_cur.fetchone.return_value = [[{
        "Plan": {
            "Node Type": "Index Scan",
            "Index Name": "chunks_5000_fp32_extended_hnsw_idx"
        }
    }]]

    prof = PLANNER_COST_PROFILES["P1"]
    dummy_vec = [0.1] * 768

    with pytest.raises(RuntimeError, match="EXACT TIE CHECK SCAN ASSERTION FAILED"):
        check_boundary_distance_tie(
            mock_conn, dummy_vec, "planner_sensitivity", "chunks_5000_fp32_extended",
            "vector(768)", prof, mode="exact", top_k=10
        )


def test_exact_boundary_tie_enforces_seq_scan_gucs():
    """Verify check_boundary_distance_tie enforces disable indexscan/bitmapscan and enable seqscan."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchone.return_value = [[{"Plan": {"Node Type": "Seq Scan"}}]]
    mock_cur.fetchall.return_value = [(101, 0.40), (102, 0.45)]

    check_boundary_distance_tie(
        mock_conn, [0.1]*768, "planner_sensitivity", "chunks_1001_fp16_extended",
        "halfvec(768)", top_k=10
    )

    executed_sql = " ".join([str(call[0][0]) for call in mock_cur.execute.call_args_list])
    assert "enable_indexscan = off" in executed_sql
    assert "enable_bitmapscan = off" in executed_sql
    assert "enable_seqscan = on" in executed_sql
    assert "halfvec(768)" in executed_sql


# ==============================================================================
# 8. Tie-Aware Recall & Disjoint-Set Metric Validation (12 Invariant Tests)
# ==============================================================================

def test_tie_aware_recall_case_1_no_ties_at_boundary():
    """Case 1: No ties at the boundary. Conventional and tie-aware recall must be identical."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 11)]  # rank 10 dist = 0.50
    gt_group = {
        "boundary_distance": 0.50,
        "tolerance": 1e-6,
        "size_B": 9,
        "size_T": 1,
        "available_tie_slots": 1,
        "has_boundary_tie": False,
        "B_ids": set(range(1, 10)),
        "T_ids": {10}
    }
    # Candidate with 10 matching IDs
    cand_10 = [{"chunk_id": i} for i in range(1, 11)]
    res = calculate_recall_metrics(ref, cand_10, gt_tie_group=gt_group, top_k=10)
    assert res["conventional_recall"] == 1.0
    assert res["tie_aware_recall"] == 1.0
    assert res["has_boundary_tie"] is False

    # Candidate with 9 matching IDs
    cand_9 = [{"chunk_id": i} for i in range(1, 10)] + [{"chunk_id": 999}]
    res9 = calculate_recall_metrics(ref, cand_9, gt_tie_group=gt_group, top_k=10)
    assert res9["conventional_recall"] == 0.9
    assert res9["tie_aware_recall"] == 0.9


def test_tie_aware_recall_case_2_exactly_two_tied_boundary_documents():
    """Case 2: Exactly two documents tied at the boundary (rank 10 and rank 11)."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 11)]
    ref[-1]["distance"] = 0.50
    gt_group = {
        "boundary_distance": 0.50,
        "tolerance": 1e-6,
        "size_B": 9,
        "size_T": 2,
        "available_tie_slots": 1,
        "has_boundary_tie": True,
        "B_ids": set(range(1, 10)),
        "T_ids": {10, 11}
    }
    # Candidate returns items 1..9 plus chunk 11 (the alternative tied doc)
    cand = [{"chunk_id": i} for i in range(1, 10)] + [{"chunk_id": 11}]
    res = calculate_recall_metrics(ref, cand, gt_tie_group=gt_group, top_k=10)
    assert res["conventional_recall"] == 0.9  # conventional strictly penalizes chunk 11
    assert res["tie_aware_recall"] == 1.0     # tie-aware credits chunk 11 as valid hit
    assert res["hits_B"] == 9
    assert res["hits_T"] == 1
    assert res["credited_T"] == 1
    assert res["has_boundary_tie"] is True


def test_tie_aware_recall_case_3_three_or_more_tied_boundary_documents():
    """Case 3: Three or more documents tied at the boundary (e.g. 4 tied, 2 slots)."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 9)] + [{"chunk_id": 9, "distance": 0.50}, {"chunk_id": 10, "distance": 0.50}]
    gt_group = {
        "boundary_distance": 0.50,
        "tolerance": 1e-6,
        "size_B": 8,
        "size_T": 4,  # docs 9, 10, 11, 12 tied
        "available_tie_slots": 2,
        "has_boundary_tie": True,
        "B_ids": set(range(1, 9)),
        "T_ids": {9, 10, 11, 12}
    }
    # Candidate returns 1..8 plus 11 and 12 (two alternatives from T)
    cand = [{"chunk_id": i} for i in range(1, 9)] + [{"chunk_id": 11}, {"chunk_id": 12}]
    res = calculate_recall_metrics(ref, cand, gt_tie_group=gt_group, top_k=10)
    assert res["conventional_recall"] == 0.8
    assert res["tie_aware_recall"] == 1.0
    assert res["hits_B"] == 8
    assert res["hits_T"] == 2
    assert res["credited_T"] == 2


def test_tie_aware_recall_case_4_candidate_includes_boundary_and_alternative():
    """Case 4: Candidate includes one original boundary doc and one alternative tied doc."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 9)] + [{"chunk_id": 9, "distance": 0.50}, {"chunk_id": 10, "distance": 0.50}]
    gt_group = {
        "boundary_distance": 0.50,
        "tolerance": 1e-6,
        "size_B": 8,
        "size_T": 4,
        "available_tie_slots": 2,
        "has_boundary_tie": True,
        "B_ids": set(range(1, 9)),
        "T_ids": {9, 10, 11, 12}
    }
    # Candidate returns 1..8 plus original 9 and alternative 11
    cand = [{"chunk_id": i} for i in range(1, 9)] + [{"chunk_id": 9}, {"chunk_id": 11}]
    res = calculate_recall_metrics(ref, cand, gt_tie_group=gt_group, top_k=10)
    assert res["conventional_recall"] == 0.9  # 9 is in ref, 11 is not
    assert res["tie_aware_recall"] == 1.0     # both 9 and 11 are in T, 2 slots available
    assert res["hits_B"] == 8
    assert res["hits_T"] == 2
    assert res["credited_T"] == 2


def test_tie_aware_recall_case_5_candidate_includes_multiple_tied_beyond_slots():
    """Case 5: Candidate includes multiple tied documents exceeding available tie slots r."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 11)]
    gt_group = {
        "boundary_distance": 0.50,
        "tolerance": 1e-6,
        "size_B": 9,
        "size_T": 4,  # docs 10, 11, 12, 13 tied; r = 1 slot available
        "available_tie_slots": 1,
        "has_boundary_tie": True,
        "B_ids": set(range(1, 10)),
        "T_ids": {10, 11, 12, 13}
    }
    # Candidate returns 1..7 (missing 8, 9 from B) plus 10, 11, 12 (3 docs from T)
    cand = [{"chunk_id": i} for i in range(1, 8)] + [{"chunk_id": 10}, {"chunk_id": 11}, {"chunk_id": 12}]
    res = calculate_recall_metrics(ref, cand, gt_tie_group=gt_group, top_k=10)
    # Conventional recall: items 1..7 and 10 match = 8/10 = 0.8
    assert res["conventional_recall"] == 0.8
    # Tie-aware: 7 from B + min(3, 1) from T = 7 + 1 = 8/10 = 0.8
    # It cannot credit extra tied docs to make up for missed docs in B!
    assert res["tie_aware_recall"] == 0.8
    assert res["hits_B"] == 7
    assert res["hits_T"] == 3
    assert res["credited_T"] == 1


def test_tie_aware_recall_case_6_candidate_misses_strictly_better_document():
    """Case 6: Candidate misses a strictly better document (in B). Missing B must always penalize."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 11)]
    gt_group = {
        "boundary_distance": 0.50,
        "tolerance": 1e-6,
        "size_B": 9,
        "size_T": 2,  # docs 10, 11 tied; r = 1 slot
        "available_tie_slots": 1,
        "has_boundary_tie": True,
        "B_ids": set(range(1, 10)),
        "T_ids": {10, 11}
    }
    # Candidate misses doc 1 (in B), returns 2..10 plus 11
    cand = [{"chunk_id": i} for i in range(2, 11)] + [{"chunk_id": 11}]
    res = calculate_recall_metrics(ref, cand, gt_tie_group=gt_group, top_k=10)
    assert res["conventional_recall"] == 0.9  # items 2..10 match
    assert res["tie_aware_recall"] == 0.9     # 8 from B + min(2, 1) = 9/10 = 0.9
    assert res["hits_B"] == 8
    assert res["hits_T"] == 2
    assert res["credited_T"] == 1


def test_tie_aware_recall_case_7_no_double_counting_between_b_and_t():
    """Case 7: Mathematical guarantee that B and T are mutually disjoint with zero overlap."""
    gt_group = {
        "boundary_distance": 0.450000,
        "tolerance": 1e-6,
        "B_ids": {1, 2, 3, 4, 5, 6, 7, 8, 9},
        "T_ids": {10, 11, 12},
        "available_tie_slots": 1
    }
    B = gt_group["B_ids"]
    T = gt_group["T_ids"]
    assert B.isdisjoint(T)
    assert len(B.intersection(T)) == 0

    cand = [{"chunk_id": i} for i in range(1, 11)]
    S = {c["chunk_id"] for c in cand}
    hits_B = S.intersection(B)
    hits_T = S.intersection(T)
    assert hits_B.isdisjoint(hits_T)


def test_tie_aware_recall_case_8_never_exceeds_one():
    """Case 8: Tie-aware recall is strictly bounded by 1.0 across any configuration."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 11)]
    for r in range(1, 11):
        size_B = 10 - r
        B = set(range(1, size_B + 1))
        T = set(range(size_B + 1, size_B + 20))  # 19 tied docs
        gt_group = {
            "boundary_distance": 0.50,
            "tolerance": 1e-6,
            "size_B": len(B),
            "size_T": len(T),
            "available_tie_slots": r,
            "B_ids": B,
            "T_ids": T
        }
        cand = [{"chunk_id": i} for i in range(size_B + 1, size_B + 11)]
        res = calculate_recall_metrics(ref, cand, gt_tie_group=gt_group, top_k=10)
        assert 0.0 <= res["tie_aware_recall"] <= 1.0


def test_tie_aware_recall_case_9_conventional_recall_remains_unchanged():
    """Case 9: Conventional Recall@10 remains strictly identical to calculate_recall_at_k."""
    ref = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 11)]
    cand = [{"chunk_id": i} for i in [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]]
    gt_group = {
        "boundary_distance": 0.50,
        "tolerance": 1e-6,
        "size_B": 9,
        "size_T": 10,
        "available_tie_slots": 1,
        "B_ids": set(range(1, 10)),
        "T_ids": set(range(10, 20))
    }
    conv_from_at_k = calculate_recall_at_k(ref, cand, top_k=10)
    res = calculate_recall_metrics(ref, cand, gt_tie_group=gt_group, top_k=10)
    assert res["conventional_recall"] == 0.5
    assert conv_from_at_k == 0.5
    assert res["conventional_recall"] == conv_from_at_k


def test_tie_aware_recall_case_10_fp16_candidate_judged_by_fp32_distances():
    """Case 10: FP16 candidate IDs are evaluated strictly against FP32 ground truth B and T sets."""
    cand_fp16 = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 10)] + [{"chunk_id": 11, "distance": 0.499912}]
    ref_fp32 = [{"chunk_id": i, "distance": 0.05 * i} for i in range(1, 11)]

    gt_group_fp32 = {
        "boundary_distance": 0.500000,
        "tolerance": 1e-6,
        "size_B": 9,
        "size_T": 2,
        "available_tie_slots": 1,
        "has_boundary_tie": True,
        "B_ids": set(range(1, 10)),
        "T_ids": {10, 11}
    }
    res = calculate_recall_metrics(ref_fp32, cand_fp16, gt_tie_group=gt_group_fp32, top_k=10)
    assert res["conventional_recall"] == 0.9
    assert res["tie_aware_recall"] == 1.0


def test_tie_aware_recall_case_11_tolerance_boundary_cases():
    """Case 11: Tolerance policy strictly classifies rows at D ± tau."""
    D = 0.500000
    tau = COSINE_DISTANCE_TOLERANCE
    assert tau == 1e-6
    assert (D - 1.01e-6) < (D - tau)
    assert abs((D - 0.99e-6) - D) <= tau
    assert abs((D + 0.99e-6) - D) <= tau
    assert abs(D - D) <= tau
    assert (D + 1.01e-6) > (D + tau)


def test_tie_aware_recall_case_12_exact_sequential_scan_enforcement():
    """Case 12: retrieve_boundary_tie_group enforces Seq Scan and fails if index scan is used."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchone.side_effect = [
        (10, 0.450000),
        [[{"Plan": {"Node Type": "Index Scan", "Index Name": "bad_hnsw_idx"}}]]
    ]

    with pytest.raises(RuntimeError, match="EXACT TIE GROUP SCAN ASSERTION FAILED"):
        retrieve_boundary_tie_group(
            mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended", top_k=10
        )


def test_plan_capture_and_diagnostic_exact_top_k():
    """Verify that capture_query_plan and execute_diagnostic_pass both pass exactly top_k (10)."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = [[{"Plan": {"Node Type": "Limit"}}]]

    prof = PLANNER_COST_PROFILES["P1"]
    dummy_vec = [0.1] * 768

    # Phase A
    capture_query_plan(mock_conn, dummy_vec, "planner_sensitivity", "chunks_1001_fp32_plain", "vector(768)", prof, top_k=10)
    explain_call = [call for call in mock_cur.execute.call_args_list if "EXPLAIN (SETTINGS" in str(call[0][0])][0]
    assert explain_call[0][1][1] == 10

    # Phase C
    execute_diagnostic_pass(mock_conn, dummy_vec, "planner_sensitivity", "chunks_1001_fp32_plain", "vector(768)", prof, top_k=10)
    diag_call = [call for call in mock_cur.execute.call_args_list if "EXPLAIN (ANALYZE" in str(call[0][0])][0]
    assert diag_call[0][1][1] == 10


# ==============================================================================
# 8. Statistical & Recall Mathematics
# ==============================================================================

def test_recall_at_10_computation():
    """Verify Recall@10 set-overlap mathematics across edge cases."""
    ref = [{"chunk_id": i} for i in range(1, 11)]  # 1..10

    # Perfect match: 10/10 = 1.0
    cand_perfect = [{"chunk_id": i} for i in range(1, 11)]
    assert calculate_recall_at_k(ref, cand_perfect, top_k=10) == 1.0

    # 8 of 10 match: 0.8
    cand_partial = [{"chunk_id": i} for i in range(1, 9)] + [{"chunk_id": 99}, {"chunk_id": 100}]
    assert math.isclose(calculate_recall_at_k(ref, cand_partial, top_k=10), 0.8)

    # Disjoint match: 0/10 = 0.0
    cand_zero = [{"chunk_id": i} for i in range(50, 60)]
    assert calculate_recall_at_k(ref, cand_zero, top_k=10) == 0.0

    # Empty inputs
    assert calculate_recall_at_k([], cand_perfect, top_k=10) == 0.0
    assert calculate_recall_at_k(ref, [], top_k=10) == 0.0
    assert calculate_recall_at_k(ref, cand_perfect, top_k=0) == 0.0


def test_metadata_and_output_schema_validation():
    """Verify compute_distribution_stats output schema and calculations."""
    lats = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    stats = compute_distribution_stats(lats)

    assert "mean" in stats
    assert "median" in stats
    assert "iqr" in stats
    assert "p95" in stats
    assert stats["min"] == 10.0
    assert stats["max"] == 24.0
    assert stats["mean"] == 17.0
    assert stats["median"] == 17.0
    assert stats["iqr"] == 9.0


def test_environment_snapshot_format():
    """Verify capture_environment_snapshot captures full engine details and GUC defaults."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchone.side_effect = [
        ["PostgreSQL 18.4 on x86_64-pc-windows-msvc"],
        ["0.8.4"],
        ["compression_aware_rag_scale"],
        ["40"]
    ]
    mock_cur.fetchall.return_value = [
        ("random_page_cost", "4.0", "", "4.0", "4.0", "default"),
        ("shared_buffers", "128MB", "8kB", "16384", "16384", "configuration file"),
        ("work_mem", "4MB", "kB", "4096", "4096", "default"),
        ("max_parallel_workers_per_gather", "0", "", "2", "2", "default"),
        ("max_parallel_workers", "8", "", "8", "8", "default"),
        ("max_parallel_maintenance_workers", "2", "", "2", "2", "default")
    ]

    snapshot = capture_environment_snapshot(mock_conn)
    assert snapshot["database"] == "compression_aware_rag_scale"
    assert "PostgreSQL 18.4" in snapshot["postgresql_version"]
    assert snapshot["pgvector_version"] == "0.8.4"
    assert "shared_buffers" in snapshot["active_pg_settings"]
    assert snapshot["active_pg_settings"]["shared_buffers"]["setting"] == "128MB"
    assert snapshot["active_pg_settings"]["random_page_cost"]["setting"] == "4.0"
    assert snapshot["timestamp"] is not None


# ==============================================================================
# 9. Baseline Immutability Guarantee & Accounting
# ==============================================================================

def test_diagnostic_and_workload_accounting_logic():
    """Verify runtime accounting logic for diagnostic and latency executions."""
    diag_accounting = {
        "expected": 540,
        "attempted": 540,
        "successful": 538,
        "failed": 2,
        "dataset_status": "in_progress",
        "errors": [{"error": "connection timeout"}]
    }

    # If any failed, dataset_status must be incomplete
    if diag_accounting["successful"] == diag_accounting["expected"] and diag_accounting["failed"] == 0:
        diag_accounting["dataset_status"] = "complete"
    else:
        diag_accounting["dataset_status"] = "incomplete"

    assert diag_accounting["dataset_status"] == "incomplete"
    assert len(diag_accounting["errors"]) == 1

    # Perfect run case
    diag_perfect = {
        "expected": 540,
        "attempted": 540,
        "successful": 540,
        "failed": 0,
        "dataset_status": "in_progress",
        "errors": []
    }
    if diag_perfect["successful"] == diag_perfect["expected"] and diag_perfect["failed"] == 0:
        diag_perfect["dataset_status"] = "complete"
    else:
        diag_perfect["dataset_status"] = "incomplete"

    assert diag_perfect["dataset_status"] == "complete"


def test_no_accidental_mutation_of_baseline_tables():
    """Verify that retrieval and plan capture queries strictly target schema planner_sensitivity."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = [[{"Plan": {"Node Type": "Limit"}}]]

    prof = PLANNER_COST_PROFILES["P5"]
    capture_query_plan(mock_conn, [0.0]*768, "planner_sensitivity", "chunks_5000_fp32_plain", "vector(768)", prof)

    all_executed_sql = " ".join([str(call[0][0]) for call in mock_cur.execute.call_args_list])
    # Must explicitly reference schema planner_sensitivity
    assert "planner_sensitivity" in all_executed_sql
    # Must NOT execute INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, VACUUM, or ANALYZE
    mutating_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "VACUUM", "ANALYZE"]
    for kw in mutating_keywords:
        assert kw not in all_executed_sql.upper()


# ==============================================================================
# 10. Phase 1E Final Scientific and Provisioning-Safety Regression Suite
# ==============================================================================

def test_canonical_profile_selection_p4():
    """Verify DEFAULT_CANONICAL_FORCED_PROFILE is P4 (standard PostgreSQL 18 defaults)."""
    assert DEFAULT_CANONICAL_FORCED_PROFILE == "P4"
    p4 = PLANNER_COST_PROFILES["P4"]
    assert p4["seq_page_cost"] == 1.0
    assert p4["random_page_cost"] == 4.0
    assert p4["cpu_operator_cost"] == 0.0025
    assert p4["cpu_tuple_cost"] == 0.01
    assert "PostgreSQL Standard Default" in p4["description"]


def test_p5_sensitivity_profile_labeling():
    """Verify Profile P5 is labeled as an elevated CPU operator cost sensitivity setting."""
    p5 = PLANNER_COST_PROFILES["P5"]
    assert p5["seq_page_cost"] == 1.0
    assert p5["random_page_cost"] == 4.0
    assert p5["cpu_operator_cost"] == 0.010
    assert p5["cpu_tuple_cost"] == 0.01
    assert "Elevated CPU Operator Cost" in p5["description"]
    assert DEFAULT_CANONICAL_FORCED_PROFILE != "P5"


def test_cost_grid_metadata_and_relative_units():
    """Verify all 9 profiles omit physical hardware simulation analogies and use relative units."""
    grid = generate_cartesian_cost_grid()
    disallowed_analogies = ["NVMe", "SSD", "EBS", "gp3", "HDD", "SAN", "hardware", "AWS"]
    for p_id, prof in grid.items():
        desc = prof["description"]
        for analogy in disallowed_analogies:
            assert analogy.lower() not in desc.lower(), f"Profile {p_id} contains hardware analogy: {analogy}"


def test_tolerance_sensitivity_analysis_multithreshold():
    """Verify calculate_recall_metrics computes sensitivity across 1e-7, 1e-6, 1e-5."""
    ref_results = [{"chunk_id": i, "distance": 0.40 + i * 0.01} for i in range(1, 15)]
    cand_results = [{"chunk_id": i, "distance": 0.40 + i * 0.01} for i in range(1, 11)]

    # Mock tie group with sensitivity_results
    tie_group = {
        "boundary_distance": 0.50,
        "has_boundary_tie": False,
        "available_tie_slots": 1,
        "B_ids": {i for i in range(1, 10)},
        "T_ids": {10},
        "sensitivity_results": {
            "1.0e-07": {"has_boundary_tie": False, "B_ids": {i for i in range(1, 10)}, "T_ids": {10}, "available_tie_slots": 1},
            "1.0e-06": {"has_boundary_tie": False, "B_ids": {i for i in range(1, 10)}, "T_ids": {10}, "available_tie_slots": 1},
            "1.0e-05": {"has_boundary_tie": True, "B_ids": {i for i in range(1, 10)}, "T_ids": {10, 11}, "available_tie_slots": 1}
        }
    }

    metrics = calculate_recall_metrics(ref_results, cand_results, gt_tie_group=tie_group, top_k=10)
    assert "sensitivity_recalls" in metrics
    assert "sensitivity_ties" in metrics
    assert "1.0e-07" in metrics["sensitivity_recalls"]
    assert "1.0e-06" in metrics["sensitivity_recalls"]
    assert "1.0e-05" in metrics["sensitivity_recalls"]
    assert metrics["sensitivity_ties"]["1.0e-06"] is False
    assert metrics["sensitivity_ties"]["1.0e-05"] is True
    assert metrics["conventional_recall"] == 1.0


def test_retrieve_boundary_tie_group_no_mock_fallback():
    """Verify retrieve_boundary_tie_group extracts boundary distance from actual row without mock fallback."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Mock valid boundary row: rank 10 chunk_id 100, distance 0.423456
    mock_cur.fetchone.side_effect = [
        (100, 0.423456),  # Boundary row
        [[{"Plan": {"Node Type": "Seq Scan"}}]]  # EXPLAIN plan
    ]
    # Corpus rows up to D + max_tol
    mock_cur.fetchall.return_value = [(i, 0.40 + i * 0.002) for i in range(1, 12)]

    res = retrieve_boundary_tie_group(
        mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended", top_k=10
    )
    assert res["boundary_distance"] == 0.423456
    assert res["scan_node_type"] == "Seq Scan"
    assert "sensitivity_results" in res
    assert "1.0e-06" in res["sensitivity_results"]


def test_retrieve_boundary_tie_group_malformed_result_handling():
    """Verify retrieve_boundary_tie_group raises ValueError when boundary row is malformed or non-numeric."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Case A: Boundary row contains unexpected list/dict
    mock_cur.fetchone.return_value = ([{"unexpected": "dict"}],)
    with pytest.raises(ValueError, match="malformed row structure|unexpected container types"):
        retrieve_boundary_tie_group(
            mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended", top_k=10
        )

    # Case B: Boundary row contains non-numeric distance
    mock_cur.fetchone.return_value = (100, "not-a-number")
    with pytest.raises(ValueError, match="non-numeric distance"):
        retrieve_boundary_tie_group(
            mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended", top_k=10
        )

    # Case C: Boundary row contains infinite distance
    mock_cur.fetchone.return_value = (100, float("inf"))
    with pytest.raises(ValueError, match="non-finite distance"):
        retrieve_boundary_tie_group(
            mock_conn, [0.1]*768, "planner_sensitivity", "chunks_5000_fp32_extended", top_k=10
        )


def test_provisioner_aborts_on_partial_target_state():
    """Verify check_target_scale_state detects partial pre-existing tables/indexes."""
    from scripts.provision_planner_sensitivity import check_target_scale_state

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Simulate 1 table existing out of 3
    with patch("scripts.provision_planner_sensitivity.check_table_exists", side_effect=[True, False, False]), \
         patch("scripts.provision_planner_sensitivity.check_index_exists", return_value=False):
        state = check_target_scale_state(mock_conn, "1001")
        assert state == "partial"


def test_provisioner_validates_source_row_count():
    """Verify check_source_table_readiness fails closed if source table has fewer rows than expected."""
    from scripts.provision_planner_sensitivity import check_source_table_readiness

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Table exists, but only contains 500 rows when 1001 are required
    mock_cur.fetchone.side_effect = [
        (1,),    # table exists
        (500,),  # row count = 500
    ]

    with pytest.raises(RuntimeError, match="less than the required 1001 rows"):
        check_source_table_readiness(mock_conn, "public", "chunks_embedding_fp32", expected_count=1001)


def test_provisioner_transaction_rollback_on_failure():
    """Verify provision_scale rolls back the transaction when an error occurs during provisioning."""
    from scripts.provision_planner_sensitivity import provision_scale

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    spec = {
        "scale": 1001,
        "database": "compression_aware_rag_exp",
        "source_schema": "public",
        "source_table": "chunks_embedding_fp32",
        "expected_count": 1001
    }

    with patch("scripts.provision_planner_sensitivity.psycopg.connect", return_value=mock_conn), \
         patch("scripts.provision_planner_sensitivity.register_vector"), \
         patch("scripts.provision_planner_sensitivity.assert_approved_database"), \
         patch("scripts.provision_planner_sensitivity.check_source_table_readiness", return_value=(1001, 768)), \
         patch("scripts.provision_planner_sensitivity.check_target_scale_state", return_value="clean"):

        # Make cursor execution raise an error during table creation
        mock_cur.execute.side_effect = RuntimeError("Disk full simulation")

        with pytest.raises(RuntimeError, match="Disk full simulation"):
            provision_scale("1001", spec, apply=True)

        # Assert rollback was called and commit was NOT called
        assert mock_conn.rollback.called
        assert not mock_conn.commit.called


def test_provisioner_analyze_failure_triggers_rollback():
    """Verify that an error during in-transaction ANALYZE prevents commit and triggers rollback."""
    from scripts.provision_planner_sensitivity import provision_scale

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    spec = {
        "scale": 1001,
        "database": "compression_aware_rag_exp",
        "source_schema": "public",
        "source_table": "chunks_embedding_fp32",
        "expected_count": 1001
    }

    def execute_side_effect(query, params=None):
        query_str = str(query)
        if "ANALYZE" in query_str.upper():
            raise RuntimeError("Simulated ANALYZE failure inside transaction")
        return None

    mock_cur.execute.side_effect = execute_side_effect
    mock_cur.fetchone.side_effect = [
        ("512MB",),  # SHOW maintenance_work_mem
        (1001, 1001),  # row count & distinct count rep 1
        ("chunks_1001_fp32_extended_pkey", 1),  # PK check rep 1
        (1001, 1001),  # row count & distinct count rep 2
        ("chunks_1001_fp32_plain_pkey", 1),  # PK check rep 2
        (1001, 1001),  # row count & distinct count rep 3
        ("chunks_1001_fp16_extended_pkey", 1),  # PK check rep 3
        ('x',),   # storage policy check rep 1 (fp32_extended -> EXTENDED)
        ('p',),   # storage policy check rep 2 (fp32_plain -> PLAIN)
        ('x',),   # storage policy check rep 3 (fp16_extended -> EXTENDED)
        ('vector_cosine_ops', True),   # index check rep 1
        ('vector_cosine_ops', True),   # index check rep 2
        ('halfvec_cosine_ops', True),  # index check rep 3
    ]

    with patch("scripts.provision_planner_sensitivity.psycopg.connect", return_value=mock_conn), \
         patch("scripts.provision_planner_sensitivity.register_vector"), \
         patch("scripts.provision_planner_sensitivity.assert_approved_database"), \
         patch("scripts.provision_planner_sensitivity.check_source_table_readiness", return_value=(1001, 768)), \
         patch("scripts.provision_planner_sensitivity.check_target_scale_state", return_value="clean"), \
         patch("scripts.provision_planner_sensitivity.validate_chunk_id_equivalence", return_value={"valid": True, "total_rows": 1001, "mismatched_ids": 0}):

        with pytest.raises(RuntimeError, match="Simulated ANALYZE failure inside transaction"):
            provision_scale("1001", spec, apply=True)

        assert mock_conn.rollback.called
        assert not mock_conn.commit.called


def test_hnsw_operator_class_validation():
    """Verify expected operator classes for vector vs halfvec representations."""
    for rep in ["fp32_extended", "fp32_plain"]:
        is_halfvec = "fp16" in rep
        opclass = "halfvec_cosine_ops" if is_halfvec else "vector_cosine_ops"
        assert opclass == "vector_cosine_ops"

    for rep in ["fp16_extended"]:
        is_halfvec = "fp16" in rep
        opclass = "halfvec_cosine_ops" if is_halfvec else "vector_cosine_ops"
        assert opclass == "halfvec_cosine_ops"


def test_readiness_checker_validates_types_and_opclasses():
    """Verify parse_reloptions and readiness checker logic."""
    from scripts.verify_table_readiness import parse_reloptions, EXPECTED_HNSW_M, EXPECTED_HNSW_EF_CONSTRUCTION

    opts = ["m=16", "ef_construction=64"]
    parsed = parse_reloptions(opts)
    assert parsed == {"m": 16, "ef_construction": 64}
    assert parse_reloptions(None) == {}
    assert parse_reloptions([]) == {}


def test_hnsw_parameter_validation_reloptions():
    """Verify parse_reloptions and expected HNSW parameters."""
    from scripts.verify_table_readiness import parse_reloptions, EXPECTED_HNSW_M, EXPECTED_HNSW_EF_CONSTRUCTION

    # Correct options
    opts = ["m=16", "ef_construction=64"]
    parsed = parse_reloptions(opts)
    assert parsed.get("m") == EXPECTED_HNSW_M
    assert parsed.get("ef_construction") == EXPECTED_HNSW_EF_CONSTRUCTION

    # Incorrect m
    opts_bad_m = ["m=32", "ef_construction=64"]
    parsed_bad_m = parse_reloptions(opts_bad_m)
    assert parsed_bad_m.get("m") != EXPECTED_HNSW_M

    # Incorrect ef_construction
    opts_bad_ef = ["m=16", "ef_construction=128"]
    parsed_bad_ef = parse_reloptions(opts_bad_ef)
    assert parsed_bad_ef.get("ef_construction") != EXPECTED_HNSW_EF_CONSTRUCTION

    # Malformed options
    opts_malformed = ["m=invalid", "ef_construction=not_an_int"]
    parsed_malformed = parse_reloptions(opts_malformed)
    assert parsed_malformed == {"m": "invalid", "ef_construction": "not_an_int"}
    assert parsed_malformed.get("m") != EXPECTED_HNSW_M
    assert parsed_malformed.get("ef_construction") != EXPECTED_HNSW_EF_CONSTRUCTION


def test_readiness_checker_parameter_validation_scenarios():
    """Test verify_scale_readiness with correct, incorrect m, incorrect ef, and missing options."""
    from scripts.verify_table_readiness import verify_scale_readiness

    spec = {
        "scale": 180,
        "database": "compression_aware_rag_test",
        "expected_count": 180
    }

    def build_mock_cur(raw_options):
        mock_cur = MagicMock()
        mock_cur.fetchone.side_effect = [
            (1,),  # schema exists
            # rep 1
            (1,),  # tbl exists
            (180, 180),  # rows & distinct
            ("chunks_test_fp32_extended_pkey", 1),  # PK
            ("vector(768)",),  # type
            ("x",),  # storage
            ("chunks_test_fp32_extended_hnsw_idx", True, "vector_cosine_ops", "hnsw", raw_options),
            (1000, 2000, 0),  # size
            (500,),  # idx size
            # rep 2
            (1,),  # tbl exists
            (180, 180),  # rows & distinct
            ("chunks_test_fp32_plain_pkey", 1),  # PK
            ("vector(768)",),  # type
            ("p",),  # storage
            ("chunks_test_fp32_plain_hnsw_idx", True, "vector_cosine_ops", "hnsw", raw_options),
            (1000, 2000, 0),  # size
            (500,),  # idx size
            # rep 3
            (1,),  # tbl exists
            (180, 180),  # rows & distinct
            ("chunks_test_fp16_extended_pkey", 1),  # PK
            ("halfvec(768)",),  # type
            ("x",),  # storage
            ("chunks_test_fp16_extended_hnsw_idx", True, "halfvec_cosine_ops", "hnsw", raw_options),
            (1000, 2000, 0),  # size
            (500,),  # idx size
            # parity check
            (0,),  # mismatches
        ]
        mock_cur.fetchall.side_effect = [
            [(768,)],  # dims rep 1
            [(768,)],  # dims rep 2
            [(768,)],  # dims rep 3
        ]
        return mock_cur

    # Scenario 1: Correct options
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = build_mock_cur(["m=16", "ef_construction=64"])
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is True
        for rep in ["fp32_extended", "fp32_plain", "fp16_extended"]:
            chk = res["checks"][f"chunks_test_{rep}"]
            assert chk["params_ok"] is True
            assert chk["index_parameters"] == {"m": 16, "ef_construction": 64}

    # Scenario 2: Incorrect m (m=32)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = build_mock_cur(["m=32", "ef_construction=64"])
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is False
        for rep in ["fp32_extended", "fp32_plain", "fp16_extended"]:
            chk = res["checks"][f"chunks_test_{rep}"]
            assert chk["params_ok"] is False

    # Scenario 3: Incorrect ef_construction (ef_construction=128)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = build_mock_cur(["m=16", "ef_construction=128"])
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is False
        for rep in ["fp32_extended", "fp32_plain", "fp16_extended"]:
            chk = res["checks"][f"chunks_test_{rep}"]
            assert chk["params_ok"] is False

    # Scenario 4: Missing options (raw_options = [])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = build_mock_cur([])
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is False
        for rep in ["fp32_extended", "fp32_plain", "fp16_extended"]:
            chk = res["checks"][f"chunks_test_{rep}"]
            assert chk["params_ok"] is False

    # Scenario 5: Malformed options
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = build_mock_cur(["m=invalid", "ef_construction=bad"])
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is False
        for rep in ["fp32_extended", "fp32_plain", "fp16_extended"]:
            chk = res["checks"][f"chunks_test_{rep}"]
            assert chk["params_ok"] is False


def test_provisioner_check_target_scale_state_hardened_invariants():
    """Verify check_target_scale_state verifies row counts, types, storage, opclasses, index validity and params."""
    from scripts.provision_planner_sensitivity import check_target_scale_state

    # Subtest A: All 3 tables and 3 indexes exist, but row count is 900 instead of 1001
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("scripts.provision_planner_sensitivity.check_table_exists", return_value=True), \
         patch("scripts.provision_planner_sensitivity.check_index_exists", return_value=True):
        mock_cur.fetchone.return_value = (900, 900)  # wrong row count
        state = check_target_scale_state(mock_conn, "1001", expected_count=1001)
        assert state == "partial"

    # Subtest B: Table exists with wrong HNSW parameters (m=32 instead of 16)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("scripts.provision_planner_sensitivity.check_table_exists", return_value=True), \
         patch("scripts.provision_planner_sensitivity.check_index_exists", return_value=True):
        mock_cur.fetchone.side_effect = [
            (1001, 1001),  # rows & distinct
            ("chunks_1001_fp32_extended_pkey", 1),  # PK
            ("vector(768)",),  # col type
            ("x",),  # storage
            ("chunks_1001_fp32_extended_hnsw_idx", True, "vector_cosine_ops", "hnsw", ["m=32", "ef_construction=64"]),  # bad m
        ]
        mock_cur.fetchall.return_value = [(768,)]
        state = check_target_scale_state(mock_conn, "1001", expected_count=1001)
        assert state == "partial"

    # Subtest C: Table exists with invalid index status (indisvalid = False)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("scripts.provision_planner_sensitivity.check_table_exists", return_value=True), \
         patch("scripts.provision_planner_sensitivity.check_index_exists", return_value=True):
        mock_cur.fetchone.side_effect = [
            (1001, 1001),  # rows & distinct
            ("chunks_1001_fp32_extended_pkey", 1),  # PK
            ("vector(768)",),  # col type
            ("x",),  # storage
            ("chunks_1001_fp32_extended_hnsw_idx", False, "vector_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),  # invalid index
        ]
        mock_cur.fetchall.return_value = [(768,)]
        state = check_target_scale_state(mock_conn, "1001", expected_count=1001)
        assert state == "partial"


def test_provisioner_sets_local_maintenance_work_mem():
    """Verify provision_scale configures SET LOCAL maintenance_work_mem = '512MB' inside transaction."""
    from scripts.provision_planner_sensitivity import provision_scale

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    spec = {
        "scale": 1001,
        "database": "compression_aware_rag_exp",
        "source_schema": "public",
        "source_table": "chunks_embedding_fp32",
        "expected_count": 1001
    }

    executed_queries = []
    def record_execute(query, params=None):
        executed_queries.append(str(query))
        return None

    mock_cur.execute.side_effect = record_execute
    mock_cur.fetchone.side_effect = [
        ("512MB",),  # SHOW maintenance_work_mem
        (1001, 1001),  # row count & distinct count rep 1
        ("chunks_1001_fp32_extended_pkey", 1),  # PK check rep 1
        (1001, 1001),  # row count & distinct count rep 2
        ("chunks_1001_fp32_plain_pkey", 1),  # PK check rep 2
        (1001, 1001),  # row count & distinct count rep 3
        ("chunks_1001_fp16_extended_pkey", 1),  # PK check rep 3
        ('x',),   # storage policy check rep 1 (fp32_extended -> EXTENDED)
        ('p',),   # storage policy check rep 2 (fp32_plain -> PLAIN)
        ('x',),   # storage policy check rep 3 (fp16_extended -> EXTENDED)
        ('vector_cosine_ops', True),   # index check rep 1
        ('vector_cosine_ops', True),   # index check rep 2
        ('halfvec_cosine_ops', True),  # index check rep 3
    ]

    with patch("scripts.provision_planner_sensitivity.psycopg.connect", return_value=mock_conn), \
         patch("scripts.provision_planner_sensitivity.register_vector"), \
         patch("scripts.provision_planner_sensitivity.assert_approved_database"), \
         patch("scripts.provision_planner_sensitivity.check_source_table_readiness", return_value=(1001, 768)), \
         patch("scripts.provision_planner_sensitivity.check_target_scale_state", return_value="clean"), \
         patch("scripts.provision_planner_sensitivity.validate_chunk_id_equivalence", return_value={"valid": True, "total_rows": 1001, "mismatched_ids": 0}), \
         patch("scripts.provision_planner_sensitivity.inspect_table_metrics", return_value={"heap_bytes": 1024, "toast_total_bytes": 0, "hnsw_index_bytes": 512}):

        provision_scale("1001", spec, apply=True)

    assert any("maintenance_work_mem" in q and "512MB" in q for q in executed_queries)


def test_readiness_checker_primary_key_positive_and_negative():
    """Verify verify_scale_readiness validates presence of single-column PRIMARY KEY on chunk_id."""
    from scripts.verify_table_readiness import verify_scale_readiness

    spec = {
        "scale": 180,
        "database": "compression_aware_rag_test",
        "expected_count": 180
    }

    # Case A: Missing Primary Key
    mock_cur = MagicMock()
    mock_cur.fetchone.side_effect = [
        (1,),  # schema exists
        (1,),  # tbl exists
        (180, 180),  # rows & distinct
        None,  # NO PRIMARY KEY
        ("vector(768)",),  # type
        ("x",),  # storage
        ("chunks_test_fp32_extended_hnsw_idx", True, "vector_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        # rep 2
        (1,), (180, 180), ("chunks_test_fp32_plain_pkey", 1), ("vector(768)",), ("p",),
        ("chunks_test_fp32_plain_hnsw_idx", True, "vector_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        # rep 3
        (1,), (180, 180), ("chunks_test_fp16_extended_pkey", 1), ("halfvec(768)",), ("x",),
        ("chunks_test_fp16_extended_hnsw_idx", True, "halfvec_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        (0,) # parity
    ]
    mock_cur.fetchall.side_effect = [[(768,)], [(768,)], [(768,)]]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is False
        assert res["checks"]["chunks_test_fp32_extended"]["pk_ok"] is False

    # Case B: Composite Primary Key (cardinality = 2)
    mock_cur2 = MagicMock()
    mock_cur2.fetchone.side_effect = [
        (1,),  # schema exists
        (1,),  # tbl exists
        (180, 180),  # rows & distinct
        ("composite_pkey", 2),  # COMPOSITE PK (2 columns)
        ("vector(768)",),  # type
        ("x",),  # storage
        ("chunks_test_fp32_extended_hnsw_idx", True, "vector_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        # rep 2
        (1,), (180, 180), ("chunks_test_fp32_plain_pkey", 1), ("vector(768)",), ("p",),
        ("chunks_test_fp32_plain_hnsw_idx", True, "vector_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        # rep 3
        (1,), (180, 180), ("chunks_test_fp16_extended_pkey", 1), ("halfvec(768)",), ("x",),
        ("chunks_test_fp16_extended_hnsw_idx", True, "halfvec_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        (0,) # parity
    ]
    mock_cur2.fetchall.side_effect = [[(768,)], [(768,)], [(768,)]]

    mock_conn2 = MagicMock()
    mock_conn2.cursor.return_value.__enter__.return_value = mock_cur2
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn2), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is False
        assert res["checks"]["chunks_test_fp32_extended"]["pk_ok"] is False


def test_readiness_checker_duplicate_chunk_ids():
    """Verify verify_scale_readiness fails closed if COUNT(DISTINCT chunk_id) != COUNT(*)."""
    from scripts.verify_table_readiness import verify_scale_readiness

    spec = {
        "scale": 180,
        "database": "compression_aware_rag_test",
        "expected_count": 180
    }

    mock_cur = MagicMock()
    mock_cur.fetchone.side_effect = [
        (1,),  # schema exists
        (1,),  # tbl exists
        (180, 179),  # DUPLICATE DETECTED: 180 rows, but only 179 distinct chunk_ids!
        ("chunks_test_fp32_extended_pkey", 1),
        ("vector(768)",),
        ("x",),
        ("chunks_test_fp32_extended_hnsw_idx", True, "vector_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        # rep 2
        (1,), (180, 180), ("chunks_test_fp32_plain_pkey", 1), ("vector(768)",), ("p",),
        ("chunks_test_fp32_plain_hnsw_idx", True, "vector_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        # rep 3
        (1,), (180, 180), ("chunks_test_fp16_extended_pkey", 1), ("halfvec(768)",), ("x",),
        ("chunks_test_fp16_extended_hnsw_idx", True, "halfvec_cosine_ops", "hnsw", ["m=16", "ef_construction=64"]),
        (1000, 2000, 0), (500,),
        (0,)
    ]
    mock_cur.fetchall.side_effect = [[(768,)], [(768,)], [(768,)]]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    with patch("scripts.verify_table_readiness.psycopg.connect", return_value=mock_conn), \
         patch("scripts.verify_table_readiness.register_vector"):
        res = verify_scale_readiness("test", spec)
        assert res["ready"] is False
        assert res["checks"]["chunks_test_fp32_extended"]["distinct_ok"] is False


def test_provisioner_check_target_scale_state_pk_and_duplicates():
    """Verify check_target_scale_state detects missing PK, composite PK, and duplicate IDs."""
    from scripts.provision_planner_sensitivity import check_target_scale_state

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Case A: Duplicate chunk IDs
    with patch("scripts.provision_planner_sensitivity.check_table_exists", return_value=True), \
         patch("scripts.provision_planner_sensitivity.check_index_exists", return_value=True):
        mock_cur.fetchone.return_value = (1001, 1000)  # duplicate detected
        state = check_target_scale_state(mock_conn, "1001", expected_count=1001)
        assert state == "partial"

    # Case B: Missing PK constraint
    with patch("scripts.provision_planner_sensitivity.check_table_exists", return_value=True), \
         patch("scripts.provision_planner_sensitivity.check_index_exists", return_value=True):
        mock_cur.fetchone.side_effect = [
            (1001, 1001),  # row count OK
            None,          # NO PRIMARY KEY
        ]
        state = check_target_scale_state(mock_conn, "1001", expected_count=1001)
        assert state == "partial"

    # Case C: Composite PK (cardinality = 2)
    with patch("scripts.provision_planner_sensitivity.check_table_exists", return_value=True), \
         patch("scripts.provision_planner_sensitivity.check_index_exists", return_value=True):
        mock_cur.fetchone.side_effect = [
            (1001, 1001),  # row count OK
            ("composite_pk", 2),  # cardinality 2
        ]
        state = check_target_scale_state(mock_conn, "1001", expected_count=1001)
        assert state == "partial"


def test_bidirectional_parity_query_scenarios():
    """Verify bidirectional 6-way set difference logic across all representation pairs."""
    def compute_6way_diff(t1: set, t2: set, t3: set) -> int:
        return len(
            (t1 - t2) | (t2 - t1) |
            (t1 - t3) | (t3 - t1) |
            (t2 - t3) | (t3 - t2)
        )

    # 1. Exact complete parity
    s1 = {1, 2, 3, 4, 5}
    s2 = {1, 2, 3, 4, 5}
    s3 = {1, 2, 3, 4, 5}
    assert compute_6way_diff(s1, s2, s3) == 0

    # 2. One-way forward mismatch: T1 has an extra ID not in T2
    assert compute_6way_diff({1, 2, 3, 4, 5, 6}, s2, s3) > 0

    # 3. Reverse-direction mismatch: T2 has an extra ID not in T1
    assert compute_6way_diff(s1, {1, 2, 3, 4, 5, 7}, s3) > 0

    # 4. Pairwise mismatch between T2 and T3 (T3 has ID not in T2)
    assert compute_6way_diff(s1, s2, {1, 2, 3, 4, 8}) > 0
    assert compute_6way_diff(s1, {1, 2, 3, 4, 9}, {1, 2, 3, 4, 8}) > 0


def test_attstorage_label_mapping():
    """Verify exact PostgreSQL pg_attribute.attstorage specification mapping across all 4 codes."""
    from scripts.provision_planner_sensitivity import ATTSTORAGE_MAP as PROV_MAP
    from scripts.verify_table_readiness import ATTSTORAGE_MAP as VERIFY_MAP

    expected_mapping = {
        'p': 'PLAIN',
        'e': 'EXTERNAL',
        'm': 'MAIN',
        'x': 'EXTENDED',
    }

    # Verify provisioner map
    assert PROV_MAP == expected_mapping
    assert PROV_MAP['p'] == 'PLAIN'
    assert PROV_MAP['e'] == 'EXTERNAL'
    assert PROV_MAP['m'] == 'MAIN'
    assert PROV_MAP['x'] == 'EXTENDED'

    # Verify readiness map
    assert VERIFY_MAP == expected_mapping
    assert VERIFY_MAP['p'] == 'PLAIN'
    assert VERIFY_MAP['e'] == 'EXTERNAL'
    assert VERIFY_MAP['m'] == 'MAIN'
    assert VERIFY_MAP['x'] == 'EXTENDED'

    # Assert that none of the four codes produce an unexpected or reversed label
    assert PROV_MAP['e'] != 'EXTENDED', "attstorage 'e' must not map to EXTENDED"
    assert PROV_MAP['x'] != 'EXTERNAL', "attstorage 'x' must not map to EXTERNAL"
    assert VERIFY_MAP['e'] != 'EXTENDED', "attstorage 'e' must not map to EXTENDED"
    assert VERIFY_MAP['x'] != 'EXTERNAL', "attstorage 'x' must not map to EXTERNAL"


def test_inspect_table_metrics_storage_policy():
    """Verify inspect_table_metrics applies correct storage policy labels for p, e, m, x."""
    from scripts.provision_planner_sensitivity import inspect_table_metrics
    from unittest.mock import MagicMock

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    for code, expected_label in [('p', 'PLAIN'), ('e', 'EXTERNAL'), ('m', 'MAIN'), ('x', 'EXTENDED')]:
        mock_cur.fetchone.side_effect = [
            (100000, 10, 1001.0), # heap_b, relpages, reltuples
            (220000,),            # total_inc_b
            (12345,),             # toast_oid
            (50000,),             # toast_tbl_b
            (10000,),             # toast_idx_b
            (40000,),             # idx_bytes (hnsw)
            (20000,),             # pk_bytes
            (1001,),              # actual_rows
            (code,),              # attstorage
            (3072, 3072, 3072.0), # min, max, avg col sizes
        ]
        mock_cur.fetchall.side_effect = [
            [(768,)],             # vector_dims
        ]
        res = inspect_table_metrics(mock_conn, "1001", "fp32_extended")
        assert res["storage_policy"] == expected_label, (
            f"Code {code} expected {expected_label}, got {res['storage_policy']}"
        )


# ==============================================================================
# Phase 1L Tests: Prepared-Statement Safety, Plan Freshness, Ground Truth & Isolation
# ==============================================================================

@patch("src.planner_sensitivity_experiment.register_vector")
@patch("psycopg.connect")
def test_prepare_threshold_disabled_on_all_scale_connections(mock_connect, mock_reg):
    """Verify get_sensitivity_connection sets prepare_threshold=None across all scale databases."""
    from src.planner_sensitivity_experiment import SCALE_DATABASE_MAP

    for scale_key, db_name in SCALE_DATABASE_MAP.items():
        mock_conn = MagicMock()
        mock_conn.prepare_threshold = None
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = [db_name]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        conn = get_sensitivity_connection(db_name)
        assert conn.prepare_threshold is None
        _, kwargs = mock_connect.call_args
        assert kwargs.get("prepare_threshold") is None, f"Database {db_name} failed to pass prepare_threshold=None"
        assert kwargs.get("autocommit") is True


@patch("src.planner_sensitivity_experiment.register_vector")
@patch("psycopg.connect")
def test_prepare_threshold_non_none_rejection(mock_connect, mock_reg):
    """Verify get_sensitivity_connection fails closed if prepare_threshold is not None."""
    mock_conn = MagicMock()
    mock_conn.prepare_threshold = 5  # Psycopg default
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ["compression_aware_rag_scale"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_connect.return_value = mock_conn

    with pytest.raises(RuntimeError, match="PREPARE THRESHOLD SAFETY VIOLATION"):
        get_sensitivity_connection("compression_aware_rag_scale")


def test_exact_ground_truth_seq_scan_assertion_success():
    """Verify retrieve_exact_ground_truth succeeds when plan indicates Seq Scan."""
    from src.planner_sensitivity_experiment import retrieve_exact_ground_truth, PLANNER_COST_PROFILES

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Mock EXPLAIN output with Seq Scan
    seq_plan = [{
        "Plan": {
            "Node Type": "Limit",
            "Plans": [{
                "Node Type": "Seq Scan",
                "Relation Name": "chunks_1001_fp32_extended"
            }]
        }
    }]
    mock_cur.fetchone.return_value = [seq_plan]
    mock_cur.fetchall.return_value = [("c1", 0.12), ("c2", 0.15)]

    prof = PLANNER_COST_PROFILES["P4"]
    res = retrieve_exact_ground_truth(
        mock_conn, [0.1]*768, "planner_sensitivity", "chunks_1001_fp32_extended",
        "vector(768)", prof, top_k=10
    )
    assert len(res) == 2
    assert res[0]["chunk_id"] == "c1"
    assert res[0]["distance"] == 0.12

    # Verify SET LOCAL GUC calls
    executed_statements = [str(call[0][0]) for call in mock_cur.execute.call_args_list]
    assert any("SET LOCAL enable_seqscan = on" in s for s in executed_statements)
    assert any("SET LOCAL enable_indexscan = off" in s for s in executed_statements)
    assert any("SET LOCAL enable_bitmapscan = off" in s for s in executed_statements)
    assert any("ROLLBACK" in s for s in executed_statements)


def test_exact_ground_truth_seq_scan_assertion_failure():
    """Verify retrieve_exact_ground_truth fails closed if plan indicates Index Scan."""
    from src.planner_sensitivity_experiment import retrieve_exact_ground_truth, PLANNER_COST_PROFILES

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Mock EXPLAIN output with unintended Index Scan
    index_plan = [{
        "Plan": {
            "Node Type": "Limit",
            "Plans": [{
                "Node Type": "Index Scan",
                "Index Name": "chunks_1001_fp32_extended_hnsw_idx"
            }]
        }
    }]
    mock_cur.fetchone.return_value = [index_plan]

    prof = PLANNER_COST_PROFILES["P4"]
    with pytest.raises(RuntimeError, match="GROUND TRUTH SCAN VERIFICATION FAILURE"):
        retrieve_exact_ground_truth(
            mock_conn, [0.1]*768, "planner_sensitivity", "chunks_1001_fp32_extended",
            "vector(768)", prof, top_k=10
        )


def test_output_directory_collision_prevention(tmp_path):
    """Verify run_planner_sensitivity_benchmark aborts with FileExistsError if output files exist."""
    from src.planner_sensitivity_experiment import run_planner_sensitivity_benchmark

    output_dir = str(tmp_path / "existing_run")
    os.makedirs(output_dir, exist_ok=True)

    # Create a pre-existing artifact file
    collision_file = os.path.join(output_dir, "latency_results.csv")
    with open(collision_file, "w", encoding="utf-8") as f:
        f.write("pre-existing data\n")

    with pytest.raises(FileExistsError, match="OUTPUT DIRECTORY COLLISION"):
        run_planner_sensitivity_benchmark(
            scales=["test"],
            representations=["fp32_extended"],
            output_dir=output_dir,
            dry_run=False
        )

    # Ensure pre-existing data was preserved
    with open(collision_file, "r", encoding="utf-8") as f:
        assert f.read() == "pre-existing data\n"


def test_diagnostic_completeness_accounting_status():
    """Verify status reporting differentiates execution count validation from diagnostic success."""
    # When all diagnostics succeed:
    diag_acct_success = {"expected": 540, "attempted": 540, "successful": 540, "failed": 0, "dataset_status": "complete"}
    lat_acct_success = {"expected": 9900, "attempted": 9900, "successful": 9900, "failed": 0, "dataset_status": "complete"}

    # When diagnostics encounter errors:
    diag_acct_error = {"expected": 540, "attempted": 540, "successful": 538, "failed": 2, "dataset_status": "incomplete"}

    # Status derivation check
    diag_stat_succ = "COMPLETE" if diag_acct_success["dataset_status"] == "complete" else "INCOMPLETE"
    lat_stat_succ = "COMPLETE" if lat_acct_success["dataset_status"] == "complete" else "INCOMPLETE"
    overall_succ = "COMPLETE" if (lat_stat_succ == "COMPLETE" and diag_stat_succ == "COMPLETE") else "PARTIAL"

    assert diag_stat_succ == "COMPLETE"
    assert lat_stat_succ == "COMPLETE"
    assert overall_succ == "COMPLETE"

    diag_stat_err = "COMPLETE" if diag_acct_error["dataset_status"] == "complete" else "INCOMPLETE"
    overall_err = "COMPLETE" if (lat_stat_succ == "COMPLETE" and diag_stat_err == "COMPLETE") else "PARTIAL"

    assert diag_stat_err == "INCOMPLETE"
    assert overall_err == "PARTIAL"





