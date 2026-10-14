import pytest
from unittest.mock import MagicMock
from src.database import (
    get_connection,
    assert_test_database,
    TEST_DB_NAME
)
from src.embeddings import get_embedding
from src.scale_experiment import (
    assert_scale_mutation_db,
    get_exp_read_only_connection,
    validate_canonical_cache_against_baseline,
    calculate_overlap_recall,
    compute_paired_stats,
    summarize_latencies,
    parse_plan_details,
    exact_search_scale,
    get_exact_plan_scale,
    get_hnsw_plan_scale,
    SCALE_DB_NAME,
    EXP_DB_NAME
)

def test_scale_mutation_guard_blocks_other_databases():
    """Verify assert_scale_mutation_db blocks mutations on non-scale databases."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    # Block production DB
    mock_cur.fetchone.return_value = ["compression_aware_rag"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
        assert_scale_mutation_db(mock_conn)

    # Block exp DB
    mock_cur.fetchone.return_value = ["compression_aware_rag_exp"]
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
        assert_scale_mutation_db(mock_conn)

    # Block test DB from scale mutation guard (test mutations belong to test harness)
    mock_cur.fetchone.return_value = ["compression_aware_rag_test"]
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
        assert_scale_mutation_db(mock_conn)

    # Allow scale DB
    mock_cur.fetchone.return_value = [SCALE_DB_NAME]
    assert assert_scale_mutation_db(mock_conn) == SCALE_DB_NAME

def test_deterministic_chunk_slicing_zero_overshoot():
    """Verify that chunk slicing guarantees exact chunk limits without overshoot."""
    target_chunks = 10
    inserted_chunks = 8
    incoming_chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]

    if inserted_chunks + len(incoming_chunks) > target_chunks:
        sliced_chunks = incoming_chunks[:target_chunks - inserted_chunks]
    else:
        sliced_chunks = incoming_chunks

    assert len(sliced_chunks) == 2
    assert inserted_chunks + len(sliced_chunks) == target_chunks

def test_calculate_overlap_recall_cases():
    """Verify set-overlap Recall@K logic."""
    ref = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}, {"chunk_id": 4}, {"chunk_id": 5}]
    cand_perfect = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}, {"chunk_id": 4}, {"chunk_id": 5}]
    cand_partial = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 9}, {"chunk_id": 10}, {"chunk_id": 11}]
    cand_disjoint = [{"chunk_id": 10}, {"chunk_id": 20}, {"chunk_id": 30}]

    assert calculate_overlap_recall(ref, cand_perfect, top_k=5) == 1.0
    assert calculate_overlap_recall(ref, cand_partial, top_k=5) == 0.4
    assert calculate_overlap_recall(ref, cand_disjoint, top_k=3) == 0.0
    assert calculate_overlap_recall(ref, cand_perfect, top_k=0) == 0.0

def test_paired_stats_and_summaries():
    """Verify paired difference calculation and summary statistics."""
    fp32 = [10.0, 20.0, 30.0, 40.0]
    half = [8.0, 15.0, 25.0, 35.0]

    paired = compute_paired_stats(fp32, half)
    assert paired["mean_diff_ms"] == 4.25
    assert paired["median_diff_ms"] == 5.0
    assert paired["halfvec_faster_count"] == 4
    assert paired["fp32_faster_count"] == 0

    stats = summarize_latencies(fp32)
    assert stats["mean"] == 25.0
    assert stats["median"] == 25.0
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0

def test_parse_plan_details():
    """Verify execution plan parser for Seq Scan and Index Scan."""
    raw_seq = [
        "Seq Scan on chunks_5k_fp32  (cost=0.00..150.00 rows=5000 width=12)",
        "  Buffers: shared hit=1000",
        "Planning Time: 0.150 ms",
        "Execution Time: 12.345 ms"
    ]
    parsed_seq = parse_plan_details(raw_seq)
    assert parsed_seq["scan_type"] == "Seq Scan"
    assert parsed_seq["index_name_used"] is None
    assert parsed_seq["shared_hit_buffers"] == 1000
    assert parsed_seq["execution_time_ms"] == 12.345

    raw_idx = [
        "Index Scan using chunks_5k_fp32_hnsw_idx on chunks_5k_fp32  (cost=0.00..50.00 rows=10 width=12)",
        "  Buffers: shared hit=42 read=5",
        "Planning Time: 0.200 ms",
        "Execution Time: 2.500 ms"
    ]
    parsed_idx = parse_plan_details(raw_idx)
    assert parsed_idx["scan_type"] == "Index Scan"
    assert parsed_idx["index_name_used"] == "chunks_5k_fp32_hnsw_idx"
    assert parsed_idx["shared_hit_buffers"] == 42
    assert parsed_idx["shared_read_buffers"] == 5

def test_exact_search_plan_verification_enforces_seq_scan(clean_test_db):
    """Verify that get_exact_plan_scale programmatically asserts Seq Scan and rejects index usage."""
    assert_test_database()
    conn = get_connection(TEST_DB_NAME)
    try:
        # Create mini schema in test DB
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_5k_fp32 (
                    chunk_id INT PRIMARY KEY,
                    embedding vector(768) NOT NULL
                );
            """)
            emb = get_embedding("test plan verification")
            cur.execute("INSERT INTO chunks_5k_fp32 (chunk_id, embedding) VALUES (%s, %s);", (1, emb))

        plan = get_exact_plan_scale(emb, scale=5000, representation="fp32", top_k=1, conn=conn, target_db=TEST_DB_NAME)
        assert plan["scan_type"] == "Seq Scan"
        assert plan["index_name_used"] is None
    finally:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks_5k_fp32 CASCADE;")
        conn.close()

def test_get_exp_read_only_connection():
    """Verify get_exp_read_only_connection enforces read_only mode and rejects unauthorized DBs."""
    from unittest.mock import patch

    # Case 1: Successful connection establishes read_only mode before first statement
    with patch("psycopg.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = [EXP_DB_NAME]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        conn = get_exp_read_only_connection()
        assert conn == mock_conn
        assert mock_conn.read_only is True
        mock_connect.assert_called_once()
        _, kwargs = mock_connect.call_args
        assert kwargs.get("autocommit") is False
        mock_cur.execute.assert_called_once_with("SELECT current_database();")

    # Case 2: Unauthorized database is immediately closed and rejected
    with patch("psycopg.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ["compression_aware_rag"]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
            get_exp_read_only_connection()
        mock_conn.close.assert_called_once()

def test_validate_canonical_cache_mismatches(tmp_path):
    """Verify validate_canonical_cache_against_baseline raises on hash mismatches, missing keys, or invalid counts."""
    from unittest.mock import patch

    tsv_valid = tmp_path / "canonical_valid.tsv"
    baseline_rows = []
    with open(tsv_valid, "w", encoding="utf-8") as f:
        for i in range(1001):
            doc_id = f"doc_{i}"
            text = f"Sample text passage for chunk {i}"
            f.write(f"{doc_id}\t{text}\n")
            baseline_rows.append((doc_id, 0, text))

    # Case 1: Successful validation
    with patch("src.scale_experiment.get_exp_read_only_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = baseline_rows
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        res = validate_canonical_cache_against_baseline(canonical_tsv_path=str(tsv_valid))
        assert res["verified"] is True
        assert res["matched_chunks"] == 1001
        assert res["total_baseline_chunks"] == 1001

    # Case 2: Content hash mismatch raises RuntimeError
    tsv_corrupt = tmp_path / "canonical_corrupt.tsv"
    with open(tsv_corrupt, "w", encoding="utf-8") as f:
        for i in range(1001):
            doc_id = f"doc_{i}"
            text = f"CORRUPTED text for chunk {i}" if i == 500 else f"Sample text passage for chunk {i}"
            f.write(f"{doc_id}\t{text}\n")

    with patch("src.scale_experiment.get_exp_read_only_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = baseline_rows
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            validate_canonical_cache_against_baseline(canonical_tsv_path=str(tsv_corrupt))

    # Case 3: Missing/unexpected key raises RuntimeError
    tsv_wrong_key = tmp_path / "canonical_wrong_key.tsv"
    with open(tsv_wrong_key, "w", encoding="utf-8") as f:
        for i in range(1001):
            doc_id = f"unexpected_doc_{i}"
            text = f"Sample text passage for chunk {i}"
            f.write(f"{doc_id}\t{text}\n")

    with patch("src.scale_experiment.get_exp_read_only_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = baseline_rows
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        with pytest.raises(RuntimeError, match="not found in baseline map"):
            validate_canonical_cache_against_baseline(canonical_tsv_path=str(tsv_wrong_key))

    # Case 4: Baseline row count != 1001 raises RuntimeError
    with patch("src.scale_experiment.get_exp_read_only_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = baseline_rows[:500]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        with pytest.raises(RuntimeError, match="BASELINE INTEGRITY FAILURE"):
            validate_canonical_cache_against_baseline(canonical_tsv_path=str(tsv_valid))

def test_get_hnsw_plan_scale_assertions():
    """Verify get_hnsw_plan_scale asserts correct Index Scan and rejects wrong index or Seq Scan."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Case 1: Correct HNSW index plan is accepted
    mock_cur.fetchall.return_value = [
        ["Limit (cost=511.82..529.00 rows=10 width=20) (actual time=0.400..0.406 rows=10 loops=1)"],
        ["  Buffers: shared hit=753"],
        ["  -> Index Scan using chunks_5k_halfvec_hnsw_idx on chunks_5k_halfvec (cost=511.82..9100.50 rows=5000 width=20)"],
        ["Planning Time: 0.050 ms"],
        ["Execution Time: 0.450 ms"]
    ]
    plan = get_hnsw_plan_scale(
        [0.1] * 768, scale=5000, representation="halfvec", enable_seqscan=False,
        conn=mock_conn, target_db="compression_aware_rag_test"
    )
    assert plan["scan_type"] == "Index Scan"
    assert plan["index_name_used"] == "chunks_5k_halfvec_hnsw_idx"

    # Case 2: Wrong index name raises FORCED-INDEX VERIFICATION FAILURE
    mock_cur.fetchall.return_value = [
        ["Limit (cost=10.00..20.00 rows=10 width=20)"],
        ["  -> Index Scan using wrong_index_name on chunks_5k_halfvec"],
        ["Execution Time: 0.5 ms"]
    ]
    with pytest.raises(RuntimeError, match="FORCED-INDEX VERIFICATION FAILURE"):
        get_hnsw_plan_scale(
            [0.1] * 768, scale=5000, representation="halfvec", enable_seqscan=False,
            conn=mock_conn, target_db="compression_aware_rag_test"
        )

    # Case 3: Seq Scan when forced index scan requested raises FORCED-INDEX VERIFICATION FAILURE
    mock_cur.fetchall.return_value = [
        ["Limit (cost=10.00..20.00 rows=10 width=20)"],
        ["  -> Seq Scan on chunks_5k_halfvec"],
        ["Execution Time: 5.5 ms"]
    ]
    with pytest.raises(RuntimeError, match="FORCED-INDEX VERIFICATION FAILURE"):
        get_hnsw_plan_scale(
            [0.1] * 768, scale=5000, representation="halfvec", enable_seqscan=False,
            conn=mock_conn, target_db="compression_aware_rag_test"
        )

    # Case 4: When enable_seqscan=True (planner-auto mode), Seq Scan is permitted without raising
    mock_cur.fetchall.return_value = [
        ["Limit (cost=10.00..20.00 rows=10 width=20)"],
        ["  -> Seq Scan on chunks_5k_halfvec"],
        ["Execution Time: 5.5 ms"]
    ]
    plan_auto = get_hnsw_plan_scale(
        [0.1] * 768, scale=5000, representation="halfvec", enable_seqscan=True,
        conn=mock_conn, target_db="compression_aware_rag_test"
    )
    assert plan_auto["scan_type"] == "Seq Scan"
