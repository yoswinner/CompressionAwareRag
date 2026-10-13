import os
import pytest
from unittest.mock import MagicMock
from src.config import get_db_name, get_db_connection_string, EMBEDDING_DIM
from src.database import (
    get_connection,
    initialize_schema,
    clear_database,
    assert_test_database,
    TEST_DB_NAME
)
from src.embeddings import get_embedding
from src.preprocessing import chunk_text
from src.search import (
    exact_search,
    create_hnsw_index,
    hnsw_search,
    verify_hnsw_index,
    get_search_plan,
    verify_execution_plan
)
from src.benchmark import calculate_recall, run_benchmark

def test_db_config_and_connection():
    """Verify database configuration targets compression_aware_rag_test and connects successfully."""
    assert get_db_name() == TEST_DB_NAME
    assert TEST_DB_NAME in get_db_connection_string()

    conn = get_connection()
    try:
        assert conn is not None
        with conn.cursor() as cur:
            cur.execute("SELECT current_database();")
            current_db = cur.fetchone()[0]
            assert current_db == TEST_DB_NAME
    finally:
        conn.close()

def test_safety_guard_blocks_non_test_db():
    """Verify fail-closed safety guard raises RuntimeError when connected to non-test database."""
    # Test with a mock cursor that returns a production database name
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ["compression_aware_rag"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with pytest.raises(RuntimeError, match="CRITICAL SAFETY GUARD VIOLATION"):
        assert_test_database(conn=mock_conn)

def test_clear_database_safety_guard():
    """Verify clear_database refuses to clear non-test database without explicit override."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ["production_db"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with pytest.raises(RuntimeError, match="Destructive operations against non-test databases are blocked"):
        clear_database(conn=mock_conn, allow_non_test=False)

def test_embedding_dimension():
    """Verify embedding generation produces vectors with exact expected dimension (768)."""
    emb = get_embedding("test query text")
    assert isinstance(emb, list)
    assert len(emb) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in emb)

def test_chunking():
    """Verify text chunking respects chunk_size and overlap."""
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 0
    assert len(chunks[0]) == 300
    # Second chunk should start after chunk_size - overlap = 250 characters
    assert len(chunks) == 4

def test_document_and_chunk_insertion_in_test_db(clean_test_db):
    """Verify insertion and retrieval inside the isolated test database."""
    assert_test_database()
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("test_doc_1", "Sample content for testing"))
            doc_id = cur.fetchone()[0]
            assert doc_id > 0
            
            emb = get_embedding("Sample content for testing")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
                (doc_id, 0, "Sample content for testing", emb)
            )
            chunk_id = cur.fetchone()[0]
            assert chunk_id > 0

            cur.execute("SELECT count(*) FROM document_chunks;")
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()

def test_exact_search_with_and_without_connection(clean_test_db):
    """Verify exact_search functionality both with caller-supplied connection and auto-managed connection."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_search_1", "Alpha text content"))
            d1 = cur.fetchone()[0]
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_search_2", "Beta text content"))
            d2 = cur.fetchone()[0]

            emb1 = get_embedding("Alpha text content")
            emb2 = get_embedding("Beta text content")

            cur.execute("INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, 0, %s, %s);", (d1, "Alpha text content", emb1))
            cur.execute("INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, 0, %s, %s);", (d2, "Beta text content", emb2))
    finally:
        conn.close()

    query_emb = get_embedding("Alpha query")
    
    # Auto connection mode
    results_auto = exact_search(query_emb, top_k=2)
    assert len(results_auto) == 2
    assert "chunk_id" in results_auto[0]
    assert "distance" in results_auto[0]
    
    # Caller supplied connection mode
    test_conn = get_connection()
    try:
        results_supplied = exact_search(query_emb, top_k=2, conn=test_conn)
        assert len(results_supplied) == 2
        assert results_supplied[0]["chunk_id"] == results_auto[0]["chunk_id"]
    finally:
        test_conn.close()

def test_hnsw_index_creation_and_search(clean_test_db):
    """Verify HNSW index creation and hnsw_search on test database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_hnsw", "HNSW content"))
            doc_id = cur.fetchone()[0]
            emb = get_embedding("HNSW content")
            cur.execute("INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, 0, %s, %s);", (doc_id, "HNSW content", emb))
    finally:
        conn.close()

    # Create HNSW index
    create_hnsw_index(m=16, ef_construction=64)
    assert verify_hnsw_index() is True

    # Search with HNSW
    query_emb = get_embedding("HNSW content")
    res = hnsw_search(query_emb, top_k=1, ef_search=40)
    assert len(res) == 1
    assert res[0]["chunk_text"] == "HNSW content"

def test_execution_plan_verification(clean_test_db):
    """Verify that execution plans distinguish natural sequential scan from forced HNSW index scan."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("doc_plan", "Plan verification content"))
            doc_id = cur.fetchone()[0]
            emb = get_embedding("Plan verification content")
            cur.execute("INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, 0, %s, %s);", (doc_id, "Plan verification content", emb))

        create_hnsw_index(m=16, ef_construction=64, conn=conn)

        # Exact plan (natural) should use Seq Scan
        exact_plan = get_search_plan(emb, top_k=1, force_hnsw=False, conn=conn)
        assert verify_execution_plan(exact_plan, "Seq Scan")
        assert not verify_execution_plan(exact_plan, "document_chunks_embedding_hnsw_idx")

        # Forced HNSW plan should use Index Scan
        forced_plan = get_search_plan(emb, top_k=1, force_hnsw=True, ef_search=40, conn=conn)
        assert verify_execution_plan(forced_plan, "Index Scan using document_chunks_embedding_hnsw_idx")
    finally:
        conn.close()

def test_recall_calculation_cases():
    """Verify Recall@K math across edge cases: perfect match, partial match, zero match, and empty inputs."""
    # Perfect match (2 of 2)
    exact = [{"chunk_id": 10}, {"chunk_id": 20}]
    hnsw = [{"chunk_id": 10}, {"chunk_id": 20}]
    assert calculate_recall(exact, hnsw) == 1.0

    # Partial match (1 of 2)
    hnsw_partial = [{"chunk_id": 10}, {"chunk_id": 99}]
    assert calculate_recall(exact, hnsw_partial) == 0.5

    # Zero match (0 of 2)
    hnsw_none = [{"chunk_id": 88}, {"chunk_id": 99}]
    assert calculate_recall(exact, hnsw_none) == 0.0

    # Empty exact results
    assert calculate_recall([], [{"chunk_id": 1}]) == 0.0

def test_benchmark_result_structure(clean_test_db):
    """Verify run_benchmark outputs complete result structure on test database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", ("bench_doc", "Benchmark content"))
            doc_id = cur.fetchone()[0]
            emb = get_embedding("Benchmark content")
            cur.execute("INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, 0, %s, %s);", (doc_id, "Benchmark content", emb))
        create_hnsw_index(m=16, ef_construction=64, conn=conn)
    finally:
        conn.close()

    results = run_benchmark(
        queries=["Benchmark content"],
        top_k=1,
        m=16,
        ef_construction=64,
        ef_search=40,
        warmup_runs=1,
        num_runs=1,
        rebuild_index=False
    )

    assert results is not None
    assert results["database"] == TEST_DB_NAME
    assert results["num_vectors"] == 1
    assert results["top_k"] == 1
    assert results["num_queries"] == 1
    assert results["recall"] == 1.0
    assert "exact_latency_avg_ms" in results
    assert "exact_latency_median_ms" in results
    assert "hnsw_latency_avg_ms" in results
    assert "hnsw_latency_median_ms" in results
    assert "forced_hnsw_plan" in results
    assert "exact_plan" in results
    assert results["metric_description"] == "client-observed retrieval query latency with a persistent connection"
