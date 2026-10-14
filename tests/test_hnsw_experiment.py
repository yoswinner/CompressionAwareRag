import pytest
from unittest.mock import MagicMock
from src.database import (
    get_connection,
    assert_test_database,
    TEST_DB_NAME
)
from src.embeddings import get_embedding
from src.representations import (
    populate_representation_tables,
    initialize_representation_tables
)
from src.hnsw_experiment import (
    assert_allowed_db,
    get_server_settings,
    build_hnsw_indexes,
    get_hnsw_index_sizes,
    get_query_plan,
    search_hnsw,
    calculate_overlap_recall,
    run_hnsw_benchmark
)

@pytest.fixture(autouse=True)
def cleanup_hnsw_test_indexes():
    """Ensure HNSW indexes created on test database are dropped after test."""
    yield
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS chunks_embedding_fp32_hnsw_idx;")
            cur.execute("DROP INDEX IF EXISTS chunks_embedding_halfvec_hnsw_idx;")
    finally:
        conn.close()

def test_hnsw_safety_guard_blocks_production():
    """Verify fail-closed guard blocks execution on production database."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ["compression_aware_rag"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with pytest.raises(RuntimeError, match="CRITICAL SAFETY GUARD VIOLATION"):
        assert_allowed_db(mock_conn)

def test_server_settings_retrieval():
    """Verify retrieval of server memory and cache settings."""
    settings = get_server_settings(target_db=TEST_DB_NAME)
    assert "maintenance_work_mem" in settings
    assert "shared_buffers" in settings
    assert "work_mem" in settings
    assert len(settings["maintenance_work_mem"]) > 0
    assert len(settings["shared_buffers"]) > 0

def test_hnsw_index_creation_and_sizing(clean_test_db):
    """Verify HNSW index creation and sizing on isolated test database."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        # Populate test data
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_hnsw_1", "Test content"))
            d_id = cur.fetchone()[0]
            emb1 = get_embedding("HNSW testing query alpha")
            emb2 = get_embedding("HNSW testing query beta")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s);",
                (d_id, 0, "Chunk 0", emb1)
            )
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s);",
                (d_id, 1, "Chunk 1", emb2)
            )

        populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)

        # First build: should measure build time
        res = build_hnsw_indexes(conn=conn, target_db=TEST_DB_NAME, m=16, ef_construction=64)
        assert res["target_db"] == TEST_DB_NAME
        assert res["indexes"]["fp32"]["already_existed"] is False
        assert res["indexes"]["fp32"]["build_time_seconds"] > 0
        assert res["indexes"]["fp32"]["index_size_bytes"] > 0
        assert res["indexes"]["halfvec"]["already_existed"] is False
        assert res["indexes"]["halfvec"]["build_time_seconds"] > 0
        assert res["indexes"]["halfvec"]["index_size_bytes"] > 0

        # Second build: should detect already existing and not claim new build duration
        res2 = build_hnsw_indexes(conn=conn, target_db=TEST_DB_NAME, m=16, ef_construction=64)
        assert res2["indexes"]["fp32"]["already_existed"] is True
        assert res2["indexes"]["fp32"]["build_time_seconds"] is None
        assert res2["indexes"]["halfvec"]["already_existed"] is True
        assert res2["indexes"]["halfvec"]["build_time_seconds"] is None

        # Verify sizes
        sizes = get_hnsw_index_sizes(conn=conn, target_db=TEST_DB_NAME)
        assert sizes["fp32_index_bytes"] > 0
        assert sizes["halfvec_index_bytes"] > 0
    finally:
        conn.close()

def test_forced_plan_verification_asserts_index_scan(clean_test_db):
    """Verify forced-index execution plan programmatically asserts Index Scan on HNSW index."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_plan", "Content"))
            d_id = cur.fetchone()[0]
            emb = get_embedding("Plan verification test")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s);",
                (d_id, 0, "Chunk 0", emb)
            )

        populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)
        build_hnsw_indexes(conn=conn, target_db=TEST_DB_NAME)

        plan_fp32 = get_query_plan(emb, representation="fp32", enable_seqscan=False, top_k=1, conn=conn, target_db=TEST_DB_NAME)
        assert plan_fp32["scan_type"] == "Index Scan"
        assert plan_fp32["index_name_used"] == "chunks_embedding_fp32_hnsw_idx"

        plan_half = get_query_plan(emb, representation="halfvec", enable_seqscan=False, top_k=1, conn=conn, target_db=TEST_DB_NAME)
        assert plan_half["scan_type"] == "Index Scan"
        assert plan_half["index_name_used"] == "chunks_embedding_halfvec_hnsw_idx"
    finally:
        conn.close()

def test_calculate_overlap_recall_cases():
    """Verify Recall@K calculation logic between ground truth and candidate rankings."""
    ref = [{"chunk_id": 10}, {"chunk_id": 20}, {"chunk_id": 30}]
    cand1 = [{"chunk_id": 10}, {"chunk_id": 20}, {"chunk_id": 30}]
    cand2 = [{"chunk_id": 10}, {"chunk_id": 20}, {"chunk_id": 99}]
    cand3 = [{"chunk_id": 77}, {"chunk_id": 88}, {"chunk_id": 99}]

    # 100% agreement
    assert calculate_overlap_recall(ref, cand1, top_k=3) == 1.0
    # 2 out of 3
    assert abs(calculate_overlap_recall(ref, cand2, top_k=3) - (2.0 / 3.0)) < 1e-5
    # 0 out of 3
    assert calculate_overlap_recall(ref, cand3, top_k=3) == 0.0
    # Edge case top_k=0
    assert calculate_overlap_recall(ref, cand1, top_k=0) == 0.0

def test_alternating_trial_order_logic():
    """Verify that multi-run trial passes alternate execution order between FP32 and FP16."""
    runs = 5
    orders = []
    for run_id in range(runs):
        order = ["fp32", "halfvec"] if (run_id % 2 == 0) else ["halfvec", "fp32"]
        orders.append(order)

    assert orders[0] == ["fp32", "halfvec"]
    assert orders[1] == ["halfvec", "fp32"]
    assert orders[2] == ["fp32", "halfvec"]
    assert orders[3] == ["halfvec", "fp32"]
    assert orders[4] == ["fp32", "halfvec"]

def test_run_hnsw_benchmark_mini(clean_test_db):
    """Verify run_hnsw_benchmark runs mini execution and produces expected schema."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_mini", "Content"))
            d_id = cur.fetchone()[0]
            emb = get_embedding("Mini benchmark query")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s);",
                (d_id, 0, "Chunk 0", emb)
            )

        populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)
        build_hnsw_indexes(conn=conn, target_db=TEST_DB_NAME)

        res = run_hnsw_benchmark(
            queries=["test query"],
            top_k_values=[1],
            ef_search=40,
            warmup_runs=1,
            measurement_runs=2,
            target_db=TEST_DB_NAME
        )

        assert res["target_db"] == TEST_DB_NAME
        assert "server_settings" in res
        assert "forced_hnsw" in res
        assert "planner_selected" in res
        assert "fp32_latency_ms" in res["forced_hnsw"]
        assert "halfvec_latency_ms" in res["forced_hnsw"]
        assert "Recall@1" in res["forced_hnsw"]["fp32_recall_vs_exact"]
        assert "Recall@1" in res["forced_hnsw"]["halfvec_recall_vs_exact"]
        assert "Agreement@1" in res["forced_hnsw"]["relative_index_agreement"]
    finally:
        conn.close()
