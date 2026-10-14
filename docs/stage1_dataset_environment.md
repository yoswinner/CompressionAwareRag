# Stage 1: Experimental Dataset Environment and Ingestion Specification

**Project:** Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency Retrieval-Augmented Generation (RAG)<br>
**Stage:** 1 — Dataset Scaling and Environment Preparation<br>
**Status:** Completed and Verified<br>
**Date:** 2026-10-14

---

## 1. Overview and Objectives

Stage 1 establishes a reproducible, scaled experimental dataset environment while preserving the committed Review 2 stabilization baseline intact.

Prior to Stage 1, the repository utilized an HTTP REST row-pagination endpoint from Hugging Face that only fetched a 100-document subset (yielding 180 chunks). This small corpus size was suitable for initial functional demonstration, but was too small for realistic vector index benchmarking, cache-eviction analysis, or index compression evaluation.

Stage 1 replaces the REST endpoint with the supported Hugging Face `datasets` streaming API, caches a validated MS MARCO corpus subset targeting ~1,000 chunks, and populates a dedicated experimental database (`compression_aware_rag_exp`) with FP32 embeddings.

---

## 2. Dataset Provenance and Configuration

| Parameter | Specification |
|---|---|
| **Dataset Repository** | [`BeIR/msmarco`](https://huggingface.co/datasets/BeIR/msmarco) |
| **Configuration** | `corpus` |
| **Split** | `corpus` |
| **Access Mechanism** | Hugging Face `datasets.load_dataset(..., streaming=True)` |
| **Stable Document Identifier** | Source `_id` field (sequential string IDs: `'0'`, `'1'`, ... `'549'`) |
| **Target Chunk Count** | $\approx 1,000$ chunks (dynamically evaluated during streaming) |
| **Actual Document Count** | **550 documents** |
| **Actual Chunk Count** | **1,001 chunks** |
| **Chunking Parameters** | `chunk_size = 300`, `overlap = 50` (using `src.preprocessing.chunk_text`) |
| **Embedding Model** | Ollama `nomic-embed-text` |
| **Embedding Dimension** | 768-dimensional FP32 vectors (`vector(768)`) |

---

## 3. Local Dataset Caching Methodology

To avoid non-deterministic network dependency, rate-limiting, and partial downloads, the ingestion pipeline implements atomic disk caching:

1. **Atomic Write Protocol:**
   - As records are streamed from Hugging Face, they are parsed and written to a temporary file (`msmarco_1k_chunks.tsv.tmp`).
   - The streaming generator tracks generated chunk counts per document in real-time.
   - Upon reaching the target chunk limit, the file stream is flushed and closed, and atomically renamed to `data/raw/msmarco_1k_chunks.tsv`.
2. **Metadata Manifest:**
   - A companion JSON manifest (`data/raw/msmarco_1k_chunks.meta.json`) is generated recording:
     - `source`: `"BeIR/msmarco"`
     - `config`: `"corpus"`
     - `split`: `"corpus"`
     - `target_chunks`: `1000`
     - `actual_documents`: `550`
     - `actual_chunks`: `1001`
     - `chunk_size`: `300`
     - `overlap`: `50`
     - `complete`: `true`
     - `timestamp`: UTC ISO timestamp
3. **Idempotent Re-use:**
   - Subsequent invocations of `download_and_cache_dataset()` verify `complete: true` in the manifest and validate existing chunk counts before skipping unnecessary re-downloads.

---

## 4. Database Isolation and Safety Boundaries

Three distinct PostgreSQL databases are maintained under PostgreSQL 18.4 with `pgvector 0.8.4`:

1. **Production Database (`compression_aware_rag`):**
   - **Status:** **UNMODIFIED BASELINE**
   - Documents: 100
   - Chunks: 180
   - Indexes: B-Tree on `id`, B-Tree on `(document_id, chunk_index)`, HNSW index on `embedding` (`m=16`, `ef_construction=64`).
   - Guarded by explicit fail-closed checks preventing truncation or accidental re-ingestion.
2. **Experiment Database (`compression_aware_rag_exp`):**
   - **Status:** **STAGE 1 EXPERIMENTAL BASELINE**
   - Documents: 550
   - Chunks: 1,001
   - Storage Type: FP32 `vector(768)`
   - Indexes: Minimum required schema (B-tree primary key and unique constraint). No compression or ANN indexes created in Stage 1.
3. **Test Database (`compression_aware_rag_test`):**
   - **Status:** **ISOLATED TEST HARNESS**
   - Controlled by `tests/conftest.py`.
   - Cleared and populated per-test for unit/integration verification.

---

## 5. Reproduction Commands

### A. Download and Cache Dataset
```bash
python -c "from src.ingestion import download_and_cache_dataset; download_and_cache_dataset(target_chunks=1000)"
```

### B. Ingest into Experiment Database
```bash
python -c "from src.ingestion import ingest_dataset_to_db; ingest_dataset_to_db(target_db='compression_aware_rag_exp')"
```

### C. Verify Database Counts and Integrity
```bash
python -c "from src.database import get_connection; conn = get_connection('compression_aware_rag_exp'); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM documents;'); print('Docs:', cur.fetchone()[0]); cur.execute('SELECT COUNT(*) FROM document_chunks;'); print('Chunks:', cur.fetchone()[0]); cur.execute('SELECT DISTINCT vector_dims(embedding) FROM document_chunks;'); print('Dims:', cur.fetchall()); conn.close()"
```

### D. Run Automated Test Suite
```bash
pytest -v
```

---

## 6. Verification and Validation Summary

- **Tests Passing:** 19/19 pytest cases passed (11 baseline stabilization tests + 8 ingestion/cache tests).
- **Document Count in `compression_aware_rag_exp`:** 550
- **Chunk Count in `compression_aware_rag_exp`:** 1,001
- **Null Fields:** 0 null values across documents and chunks.
- **Orphan Chunks:** 0 orphaned chunk records.
- **Vector Dimensions:** 100% verified as 768-dimensional float32.
- **Production Baseline:** Confirmed unchanged (100 docs, 180 chunks, HNSW index preserved).
