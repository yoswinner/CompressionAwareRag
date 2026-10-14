import pytest
from unittest.mock import MagicMock
from src.database import (
    get_connection,
    assert_test_database,
    TEST_DB_NAME
)
from src.embeddings import get_embedding
from src.representations import (
    assert_allowed_db,
    initialize_representation_tables,
    populate_representation_tables,
    validate_representations,
    exact_search_fp32,
    exact_search_halfvec,
    get_exact_plan_fp32,
    get_exact_plan_halfvec,
    calculate_agreement_recall,
    measure_storage_metrics,
    run_representation_benchmark
)

def test_safety_guard_blocks_production_database():
    """Verify fail-closed guard raises RuntimeError if connected to production database."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ["compression_aware_rag"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with pytest.raises(RuntimeError, match="CRITICAL SAFETY GUARD VIOLATION"):
        assert_allowed_db(mock_conn)

def test_representation_tables_initialization(clean_test_db):
    """Verify representation tables are correctly created in test database."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        initialize_representation_tables(conn=conn, target_db=TEST_DB_NAME)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('chunks_embedding_fp32', 'chunks_embedding_halfvec');
            """)
            tables = {r[0] for r in cur.fetchall()}
            assert "chunks_embedding_fp32" in tables
            assert "chunks_embedding_halfvec" in tables
    finally:
        conn.close()

def test_populate_and_validate_representations(clean_test_db):
    """Verify populating FP32 and halfvec representations preserves correspondence."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_rep_1", "Sample content"))
            d_id = cur.fetchone()[0]

            emb1 = get_embedding("Sample content 1")
            emb2 = get_embedding("Sample content 2")

            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
                (d_id, 0, "Chunk 1", emb1)
            )
            c1_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
                (d_id, 1, "Chunk 2", emb2)
            )
            c2_id = cur.fetchone()[0]

        pop_res = populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)
        assert pop_res["source_count"] == 2
        assert pop_res["fp32_count"] == 2
        assert pop_res["halfvec_count"] == 2

        val_res = validate_representations(conn=conn, target_db=TEST_DB_NAME)
        assert val_res["is_valid"] is True
        assert val_res["fp32_dims"] == [768]
        assert val_res["halfvec_dims"] == [768]
        assert val_res["missing_fp32"] == 0
        assert val_res["missing_halfvec"] == 0
        assert val_res["mismatched_ids"] == 0
        assert val_res["source_null_embeddings"] == 0
    finally:
        conn.close()

def test_exact_search_fp32_and_halfvec(clean_test_db):
    """Verify exhaustive exact search returns ranked nearest neighbors."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_search", "Content"))
            d_id = cur.fetchone()[0]
            emb = get_embedding("Quantum computing and databases")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
                (d_id, 0, "Chunk 0", emb)
            )
        populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)

        res_fp32 = exact_search_fp32(emb, top_k=1, conn=conn, target_db=TEST_DB_NAME)
        res_half = exact_search_halfvec(emb, top_k=1, conn=conn, target_db=TEST_DB_NAME)

        assert len(res_fp32) == 1
        assert len(res_half) == 1
        assert "chunk_id" in res_fp32[0] and "distance" in res_fp32[0]
        assert "chunk_id" in res_half[0] and "distance" in res_half[0]
        # Querying identical embedding should yield distance close to 0
        assert abs(res_fp32[0]["distance"]) < 1e-4
        assert abs(res_half[0]["distance"]) < 1e-3
    finally:
        conn.close()

def test_calculate_agreement_recall():
    """Verify Recall@K calculation logic between reference and candidate rankings."""
    fp32_res = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}, {"chunk_id": 4}]
    half_res = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 5}, {"chunk_id": 6}]

    # Recall@2: {1, 2} vs {1, 2} -> 2/2 = 1.0
    assert calculate_agreement_recall(fp32_res, half_res, top_k=2) == 1.0

    # Recall@3: {1, 2, 3} vs {1, 2, 5} -> 2/3 ≈ 0.6667
    assert abs(calculate_agreement_recall(fp32_res, half_res, top_k=3) - (2.0 / 3.0)) < 1e-5

    # Recall@4: {1, 2, 3, 4} vs {1, 2, 5, 6} -> 2/4 = 0.5
    assert calculate_agreement_recall(fp32_res, half_res, top_k=4) == 0.5

    # Edge cases
    assert calculate_agreement_recall([], [], top_k=0) == 0.0

def test_storage_metrics_structure(clean_test_db):
    """Verify storage measurement structure and column size differences."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_stor", "Content"))
            d_id = cur.fetchone()[0]
            emb = get_embedding("Storage measurement sample")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
                (d_id, 0, "Chunk 0", emb)
            )
        populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)

        storage = measure_storage_metrics(conn=conn, target_db=TEST_DB_NAME)
        assert "tables" in storage
        assert "chunks_embedding_fp32" in storage["tables"]
        assert "chunks_embedding_halfvec" in storage["tables"]

        fp32_info = storage["tables"]["chunks_embedding_fp32"]
        half_info = storage["tables"]["chunks_embedding_halfvec"]

        # Measured column sizes
        assert fp32_info["column_size_min_bytes"] == 3076
        assert half_info["column_size_min_bytes"] == 1544
        assert fp32_info["theoretical_payload_per_row_bytes"] == 3072
        assert half_info["theoretical_payload_per_row_bytes"] == 1536
    finally:
        conn.close()

def test_query_plans_use_sequential_scan(clean_test_db):
    """Verify execution plans confirm exhaustive Sequential Scan on representation tables."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_plan", "Content"))
            d_id = cur.fetchone()[0]
            emb = get_embedding("Query plan verification")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
                (d_id, 0, "Chunk 0", emb)
            )
        populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)

        plan_fp32 = get_exact_plan_fp32(emb, top_k=1, conn=conn, target_db=TEST_DB_NAME)
        plan_half = get_exact_plan_halfvec(emb, top_k=1, conn=conn, target_db=TEST_DB_NAME)

        plan_fp32_text = " ".join(plan_fp32)
        plan_half_text = " ".join(plan_half)

        assert "Seq Scan on chunks_embedding_fp32" in plan_fp32_text
        assert "Seq Scan on chunks_embedding_halfvec" in plan_half_text
    finally:
        conn.close()

def test_run_representation_benchmark_mini(clean_test_db):
    """Verify run_representation_benchmark runs and returns complete structure."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_bm", "Content"))
            d_id = cur.fetchone()[0]
            emb = get_embedding("Mini benchmark test")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
                (d_id, 0, "Chunk 0", emb)
            )
        populate_representation_tables(conn=conn, target_db=TEST_DB_NAME)

        res = run_representation_benchmark(
            queries=["test query alpha"],
            top_k_values=[1],
            warmup_runs=1,
            measurement_runs=2,
            target_db=TEST_DB_NAME
        )

        assert res["target_db"] == TEST_DB_NAME
        assert "latency_ms" in res
        assert "fp32_exact" in res["latency_ms"]
        assert "halfvec_exact" in res["latency_ms"]
        assert "Recall@1" in res["recall"]
        assert "storage" in res
        assert "sample_plans" in res
    finally:
        conn.close()
