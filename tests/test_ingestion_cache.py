import os
import json
import pytest
from src.database import get_connection, assert_test_database, TEST_DB_NAME
from src.ingestion import (
    load_cached_dataset,
    download_and_cache_dataset,
    ingest_dataset_to_db,
    DEFAULT_CACHE_PATH,
    DEFAULT_MANIFEST_PATH
)
from src.preprocessing import chunk_text

def test_load_cached_dataset_valid():
    """Verify loading records from the default cached TSV dataset."""
    assert os.path.exists(DEFAULT_CACHE_PATH), f"Cache file does not exist at {DEFAULT_CACHE_PATH}"
    docs = load_cached_dataset(DEFAULT_CACHE_PATH, max_docs=5)
    assert len(docs) == 5
    for doc_id, text in docs:
        assert isinstance(doc_id, str) and len(doc_id) > 0
        assert isinstance(text, str) and len(text) > 0

def test_load_cached_dataset_limit():
    """Verify max_docs restriction on load_cached_dataset."""
    docs_all = load_cached_dataset(DEFAULT_CACHE_PATH, max_docs=10)
    assert len(docs_all) == 10
    docs_subset = load_cached_dataset(DEFAULT_CACHE_PATH, max_docs=3)
    assert len(docs_subset) == 3
    assert docs_subset == docs_all[:3]

def test_load_cached_dataset_missing_file(tmp_path):
    """Verify load_cached_dataset raises FileNotFoundError for non-existent path."""
    non_existent = tmp_path / "non_existent.tsv"
    with pytest.raises(FileNotFoundError):
        load_cached_dataset(str(non_existent))

def test_manifest_metadata_validity():
    """Verify the cached dataset manifest metadata schema and completeness."""
    assert os.path.exists(DEFAULT_MANIFEST_PATH), f"Manifest file missing at {DEFAULT_MANIFEST_PATH}"
    with open(DEFAULT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta.get("complete") is True
    assert meta.get("source") == "BeIR/msmarco"
    assert meta.get("config") == "corpus"
    assert meta.get("actual_documents") == 550
    assert meta.get("actual_chunks") == 1001
    assert meta.get("actual_chunks") >= meta.get("target_chunks", 1000)

def test_chunk_counting_with_limit():
    """Verify that chunk counting accurately enforces the target chunk limit."""
    # Each 400-char string produces exactly 2 chunks (0..300 and 250..400) with size=300, overlap=50
    sample_texts = [
        "A" * 400,  # 2 chunks
        "B" * 400,  # 2 chunks
        "C" * 400,  # 2 chunks
    ]
    target_chunks = 3
    accumulated_chunks = 0
    selected_docs = 0

    for text in sample_texts:
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        accumulated_chunks += len(chunks)
        selected_docs += 1
        if accumulated_chunks >= target_chunks:
            break

    assert selected_docs == 2
    assert accumulated_chunks == 4  # Stopped at >= target_chunks

def test_cache_skip_when_complete(tmp_path):
    """Verify that download_and_cache_dataset skips download when valid cache exists."""
    cached_tsv = tmp_path / "test_cache.tsv"
    cached_tsv.write_text("doc1\tSample text one\ndoc2\tSample text two\n", encoding="utf-8")
    cached_meta = tmp_path / "test_cache.meta.json"
    manifest_data = {
        "complete": True,
        "actual_documents": 2,
        "actual_chunks": 4,
        "source": "BeIR/msmarco"
    }
    cached_meta.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Should immediately return cached counts without attempting HF download
    docs, chunks = download_and_cache_dataset(
        target_chunks=4,
        output_path=str(cached_tsv),
        manifest_path=str(cached_meta),
        force_download=False
    )
    assert docs == 2
    assert chunks == 4

def test_ingest_safety_guard_blocks_production():
    """Verify fail-closed guard prevents ingestion into production database."""
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY GUARD VIOLATION"):
        ingest_dataset_to_db(target_db="compression_aware_rag")

def test_ingest_dataset_into_test_database(clean_test_db, tmp_path):
    """Verify ingestion pipeline end-to-end against the isolated test database."""
    assert_test_database()

    # Create a small sample cache file
    tsv_content = (
        "test_doc_alpha\tArtificial intelligence and vector databases provide efficient search capabilities.\n"
        "test_doc_beta\tDatabase index compression minimizes memory footprint while preserving recall.\n"
    )
    sample_tsv = tmp_path / "mini_cache.tsv"
    sample_tsv.write_text(tsv_content, encoding="utf-8")

    result = ingest_dataset_to_db(
        target_db=TEST_DB_NAME,
        tsv_path=str(sample_tsv),
        chunk_size=300,
        overlap=50
    )

    assert result["target_db"] == TEST_DB_NAME
    assert result["documents_ingested"] == 2
    assert result["chunks_inserted"] >= 2

    # Verify rows in test database
    conn = get_connection(TEST_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()[0]
            assert doc_count == 2

            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            chunk_count = cur.fetchone()[0]
            assert chunk_count >= 2

            cur.execute("SELECT DISTINCT vector_dims(embedding) FROM document_chunks;")
            dims = cur.fetchone()[0]
            assert dims == 768
    finally:
        conn.close()
