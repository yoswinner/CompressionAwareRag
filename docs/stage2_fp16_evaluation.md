# Stage 2: FP16 (`halfvec`) Vector Representation and Exact-Search Evaluation

**Project:** Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency Retrieval-Augmented Generation (RAG)<br>
**Stage:** 2 — Numerical Representation Compression (FP32 vs FP16/`halfvec`)<br>
**Status:** Completed and Verified<br>
**Date:** 2026-10-14

---

## 1. Experiment Objectives

The primary objective of Stage 2 is to evaluate the impact of numerical precision reduction on vector storage footprint, retrieval latency, and nearest-neighbor agreement without the confounding effects of approximate nearest-neighbor (ANN) graph indexing.

Specifically, this stage compares:
1. **Full-Precision (FP32):** Standard IEEE 754 32-bit floating point vectors (`vector(768)`).
2. **Half-Precision (FP16):** IEEE 754 16-bit floating point vectors (`halfvec(768)`), supported natively in PostgreSQL 18 with `pgvector 0.8.4`.

Both representations are evaluated using **exhaustive sequential scans (exact cosine-distance search)** across the 1,001-chunk MS MARCO experimental corpus in `compression_aware_rag_exp`.

---

## 2. Representation Schemas and Data Provenance

To isolate representation-level storage and I/O characteristics from text storage, two dedicated vector-only representation tables were created in `compression_aware_rag_exp`:

```sql
-- FP32 Vector-Only Representation
CREATE TABLE chunks_embedding_fp32 (
    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL
);

-- FP16 Vector-Only Representation
CREATE TABLE chunks_embedding_halfvec (
    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding halfvec(768) NOT NULL
);
```

### Why the Original FP32 Source Remains Unchanged
The `document_chunks` table retains its original `embedding vector(768)` column to preserve the uncompressed ground-truth baseline established in Stage 1. Populating `chunks_embedding_halfvec` directly casts stored FP32 vectors via PostgreSQL's native `embedding::halfvec(768)` type conversion.

**No document embeddings were regenerated via Ollama.** This guarantees that all observed changes in ranking and retrieval time stem strictly from the numerical compression format rather than stochastic variation in embedding models.

---

## 3. Storage Measurement Methodology and Findings

Storage was inspected directly using PostgreSQL system catalog queries (`pg_relation_size`, `pg_total_relation_size`, `pg_indexes_size`, `pg_column_size`, and TOAST relation metadata) on `compression_aware_rag_exp`.

### A. Measured vs. Theoretical Payload

| Metric | `chunks_embedding_fp32` | `chunks_embedding_halfvec` | Empirical Change |
|---|---:|---:|---:|
| **Row Count** | 1,001 | 1,001 | 1:1 match |
| **Theoretical Vector Payload** | 3,072 bytes ($768 \times 4$) | 1,536 bytes ($768 \times 2$) | 50.00% reduction (2.00x) |
| **Actual Measured Column Datum Size** | 3,076 bytes | 1,544 bytes | **49.80% reduction (1.992x)** |
| **Datum Header / Alignment Overhead** | 4 bytes | 8 bytes | — |

### B. Relation Storage Footprint and TOAST Inversion

| Storage Component | `chunks_embedding_fp32` | `chunks_embedding_halfvec` | Comparison |
|---|---:|---:|---|
| **Primary Heap Size (`pg_relation_size`)** | 57,344 bytes (56 KB) | 1,646,592 bytes (1.57 MB) | Inlined heap storage |
| **TOAST Relation Size** | 4,104,192 bytes (4.01 MB) | **0 bytes** | **100% TOAST elimination** |
| **TOAST Index Size** | 65,536 bytes (64 KB) | 8,192 bytes (8 KB) | Unused index shell |
| **Primary Key B-Tree Index Size** | 40,960 bytes (40 KB) | 40,960 bytes (40 KB) | Identical chunk IDs |
| **Total Relation Size (`pg_total_relation_size`)** | **4,333,568 bytes (4.13 MB)** | **1,728,512 bytes (1.65 MB)** | **60.11% reduction (2.507x)** |

### C. Critical DBMS Storage Insight
In PostgreSQL, heap pages are 8 KB (`BLCKSZ = 8192`), and tuples exceeding the TOAST threshold ($\approx 2\text{ KB}$) trigger out-of-line storage in a separate TOAST relation:
- For **FP32**, the 3,076-byte vector exceeds 2 KB. PostgreSQL stores a 24-byte TOAST pointer in the heap and writes the vector payload into out-of-line TOAST chunks. Consequently, the heap is tiny (56 KB), but total relation size swells to 4.13 MB.
- For **FP16 (`halfvec`)**, the 1,544-byte vector falls safely below the 2 KB threshold. PostgreSQL stores the vector **inline within the heap page**, completely eliminating TOAST storage ($0\text{ bytes}$).
- This results in a **2.507x total relation footprint reduction**, exceeding the theoretical 2.0x scalar ratio due to the elimination of TOAST tuple headers and index overhead.

> [!IMPORTANT]
> **Storage Boundary Limitation:** This comparison isolates the vector-only representation tables. Total database size cannot be claimed to have shrunk overall because the parent `document_chunks` table still retains its original text and FP32 embeddings. RAM savings cannot be directly inferred from disk relations; buffer-cache residency will be evaluated in the indexed benchmarking stage.

---

## 4. Exact-Search Benchmark Methodology

The benchmark followed the stabilized methodology:
1. **Evaluation Set:** 20 fixed, domain-diverse queries (covering MS MARCO topics including the Manhattan Project, science, history, medicine, and databases).
2. **Pre-Computed Query Embeddings:** Generated once using Ollama `nomic-embed-text` (768 dimensions) before latency timing. Embedding generation (~0.48s for 20 queries) was strictly excluded from retrieval timing.
3. **Persistent Connection:** Executed over a single persistent `psycopg` connection with `register_vector` enabled.
4. **Warm-Up Cycles:** 2 full warm-up passes across all 20 queries prior to timing.
5. **Measurement Trials:** 5 measurement runs per query (100 timed trials per representation).
6. **Timing Precision:** Client-observed retrieval round-trip measured using `time.perf_counter()`.
7. **Query Semantics:** Exhaustive sequential scan using cosine distance (`<=>`), evaluated at $K \in \{3, 5, 10\}$.

---

## 5. Measured Retrieval Performance and Ranking Agreement

### A. Client-Observed Latency (100 Trials per Method)

| Metric | FP32 Exact Search | FP16 (`halfvec`) Exact Search | Relative Difference |
|---|---:|---:|---:|
| **Mean Latency** | **3.320 ms** | **1.784 ms** | **46.27% lower latency (1.86x faster)** |
| **Median Latency** | **3.120 ms** | **1.644 ms** | **47.31% lower latency (1.90x faster)** |
| **Minimum Latency** | 2.889 ms | 1.523 ms | — |
| **Maximum Latency** | 5.911 ms | 3.622 ms | — |
| **Standard Deviation** | 0.503 ms | 0.388 ms | Lower variance in FP16 |

### B. Agreement Recall Against FP32 Ground Truth

Agreement Recall measures how faithfully the compressed FP16 representation preserves the top-$K$ nearest neighbors identified by the uncompressed FP32 reference:
$$\text{Recall@K} = \frac{|\text{Top-K}_{FP32} \cap \text{Top-K}_{FP16}|}{K}$$

| Metric | Agreement Recall | Interpretation |
|---|---:|---|
| **Recall@3** | **0.9833 (98.33%)** | Identical top-3 items on 19 of 20 queries; minor order perturbation on 1 query |
| **Recall@5** | **1.0000 (100.0%)** | 100% identical top-5 items across all 20 queries |
| **Recall@10** | **1.0000 (100.0%)** | 100% identical top-10 items across all 20 queries |

> [!NOTE]
> Agreement Recall evaluates numerical fidelity relative to the FP32 mathematical reference. It is not labeled "semantic relevance" or "answer accuracy" because ground-truth relevance labels are not available for arbitrary test queries.

---

## 6. Execution Plan and Buffer Access Verification

Diagnostic `EXPLAIN (ANALYZE, BUFFERS)` plans confirmed exhaustive sequential scans without index lookups:

### FP32 Execution Plan Sample
```text
Limit  (cost=41.14..41.17 rows=10 width=12) (actual time=1.794..1.795 rows=10.00 loops=1)
  Buffers: shared hit=3010
  ->  Sort  (cost=41.14..43.65 rows=1001 width=12) (actual time=1.794..1.794 rows=10.00 loops=1)
        Sort Key: ((embedding <=> '[...]'::vector(768)))
        Sort Method: top-N heapsort  Memory: 25kB
        Buffers: shared hit=3010
        ->  Seq Scan on chunks_embedding_fp32  (cost=0.00..19.51 rows=1001 width=12) (actual time=0.014..1.723 rows=1001.00 loops=1)
              Buffers: shared hit=3010
Planning Time: 0.031 ms
Execution Time: 1.801 ms
```

### FP16 (`halfvec`) Execution Plan Sample
```text
Limit  (cost=235.14..235.17 rows=10 width=12) (actual time=0.397..0.398 rows=10.00 loops=1)
  Buffers: shared hit=201
  ->  Sort  (cost=235.14..237.65 rows=1001 width=12) (actual time=0.397..0.397 rows=10.00 loops=1)
        Sort Key: ((embedding <=> '[...]'::halfvec(768)))
        Sort Method: top-N heapsort  Memory: 25kB
        Buffers: shared hit=201
        ->  Seq Scan on chunks_embedding_halfvec  (cost=0.00..213.51 rows=1001 width=12) (actual time=0.005..0.351 rows=1001.00 loops=1)
              Buffers: shared hit=201
Planning Time: 0.045 ms
Execution Time: 0.404 ms
```

### Buffer Access Comparison
- **FP32 Sequential Scan:** Accessed **3,010 shared buffer pages** because every row required reading TOAST chunks out-of-line.
- **FP16 Sequential Scan:** Accessed only **201 shared buffer pages** because vectors reside inline within the heap pages.
- **Result:** A **14.98x reduction in shared buffer touches**, directly explaining why server-side execution time dropped from 1.80 ms to 0.40 ms.

---

## 7. Automated Test Suite Verification

8 new unit and integration tests were added in [`tests/test_representations.py`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/tests/test_representations.py):
- Fail-closed guard preventing operations on production database `compression_aware_rag`.
- Schema initialization of `chunks_embedding_fp32` and `chunks_embedding_halfvec`.
- Correspondence and parity validation between source chunks and representation rows.
- Exact search execution and cosine distance monotonicity.
- Agreement Recall calculation logic.
- Storage metrics schema and column size verification.
- Sequential scan query plan verification.
- Representation benchmark execution structure.

All **27 tests across the repository pass cleanly** in `compression_aware_rag_test`:
```text
tests/test_basic.py .............. (11 passed)
tests/test_ingestion_cache.py .... (8 passed)
tests/test_representations.py .... (8 passed)
============================== 27 passed in 7.67s ==============================
```

---

## 8. Remaining Work for Next Stages

1. **Stage 3:** HNSW Approximate Nearest Neighbor indexing on FP16 (`halfvec(768)`) using `halfvec_cosine_ops` vs. FP32 HNSW.
2. **Stage 4:** Quantization exploration (Binary Quantization / Scalar Quantization / Product Quantization) supported by PostgreSQL extensions or post-retrieval re-ranking.
3. **Stage 5:** End-to-end RAG pipeline memory efficiency analysis and final project report.
