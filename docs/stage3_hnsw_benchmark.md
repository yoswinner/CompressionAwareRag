# Stage 3: FP32 HNSW vs. FP16 HNSW Indexing Benchmark Report

**Project:** Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency Retrieval-Augmented Generation (RAG)<br>
**Stage:** 3 — Approximate Nearest Neighbor (ANN) Graph Indexing on Compressed Vectors<br>
**Status:** Completed and Verified<br>
**Date:** 2026-10-14

---

## 1. Experiment Overview and Objectives

Stage 3 investigates the interaction between **numerical vector compression** (`vector(768)` vs. `halfvec(768)`) and **graph-based approximate nearest neighbor (ANN) indexing** using Hierarchical Navigable Small World (HNSW) graphs.

In Stage 2, unindexed exhaustive scanning established that FP16 halves column datum payload ($3,076 \to 1,544$ bytes) and eliminates TOAST storage ($4.10\text{ MB} \to 0\text{ bytes}$). Stage 3 tests whether these storage and memory advantages transfer to HNSW index graph construction, physical index footprint, graph traversal latency, buffer pool access, and approximate retrieval accuracy (ANN Recall@K).

---

## 2. Server Configuration and Memory Budget

The benchmark environment was inspected directly via PostgreSQL system queries:

| Configuration Setting | Configured Value | Research Interpretation |
|---|:---:|---|
| **PostgreSQL Version** | PostgreSQL 18.4 | Host DBMS engine |
| **pgvector Extension** | Version 0.8.4 | Native support for `vector` and `halfvec` HNSW indexing |
| **`maintenance_work_mem`** | **64 MB** | **Configured graph-build memory budget**. Bounds memory allocated to the HNSW construction worker; if the in-memory graph exceeds 64 MB, pgvector flushes partial graphs to temporary disk storage. (Not reported as measured peak process RAM). |
| **`shared_buffers`** | **128 MB** | Shared buffer cache pool dedicated to relational table and index pages |
| **`work_mem`** | **4 MB** | Query-level working memory allocated for sort operations and Top-N heaps |

---

## 3. Index Specifications and Construction Performance

Both HNSW indexes were built in `compression_aware_rag_exp` on the 1,001-chunk representation tables using matching graph construction parameters:

```sql
-- FP32 HNSW Index
CREATE INDEX chunks_embedding_fp32_hnsw_idx
ON chunks_embedding_fp32
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- FP16 (halfvec) HNSW Index
CREATE INDEX chunks_embedding_halfvec_hnsw_idx
ON chunks_embedding_halfvec
USING hnsw (embedding halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Empirical Build Time and Physical Index Footprint

| Metric | FP32 HNSW (`vector`) | FP16 HNSW (`halfvec`) | Empirical Difference |
|---|---:|---:|---:|
| **Hyperparameters** | $m=16, ef_{construction}=64$ | $m=16, ef_{construction}=64$ | Identical graph topology |
| **Operator Class** | `vector_cosine_ops` | `halfvec_cosine_ops` | Typed cosine distance |
| **Wall-Clock Build Time** | **0.2334 s** | **0.1606 s** | **31.18% faster construction** |
| **Physical Index Size (`pg_relation_size`)** | **4,087,808 bytes (3.90 MB)** | **2,048,000 bytes (1.95 MB)** | **49.90% footprint reduction (1.996x)** |

> [!NOTE]
> **Build Duration Limitation:** At this corpus size (1,001 chunks), index build times are under 0.25 seconds and subject to operating system process scheduling variance. While FP16 consistently built faster due to writing fewer 8 KB index pages, build duration should be interpreted as a single-run indicative measurement rather than a statistically bounded SLA.

---

## 4. Query Form and Dual-Path Execution Verification

### A. Canonical Query Form with Direct Distance Expression
To ensure full index compatibility and eliminate ambiguity in parameter binding, queries bound the vector parameter directly in the distance operator within both `SELECT` and `ORDER BY`:

```sql
-- Forced FP16 HNSW Query Structure
BEGIN;
SET LOCAL enable_seqscan = off;
SET LOCAL hnsw.ef_search = 40;
SELECT chunk_id, (embedding <=> %s::halfvec(768)) AS distance
FROM chunks_embedding_halfvec
ORDER BY embedding <=> %s::halfvec(768) ASC
LIMIT %s;
COMMIT;
```

### B. Execution Plan Verification (Forced vs. Planner-Selected)

Capturing `EXPLAIN (ANALYZE, BUFFERS)` revealed a major architectural insight into PostgreSQL's cost optimizer:

#### 1. Forced-Index Diagnostic Mode (`enable_seqscan = off`, $ef_{search} = 40$)
- **FP32 Forced HNSW:** Programmatically verified `Index Scan using chunks_embedding_fp32_hnsw_idx`.
  - Server execution time: **1.267 ms**
  - Buffers: `shared hit=595`, `read=0`
- **FP16 Forced HNSW:** Programmatically verified `Index Scan using chunks_embedding_halfvec_hnsw_idx`.
  - Server execution time: **0.886 ms** (**30.07% faster server execution**)
  - Buffers: `shared hit=514`, `read=0` (**13.61% fewer buffer hits**)

#### 2. Planner-Selected Mode (`enable_seqscan = on`, default optimizer cost model)
- **FP32 Table:** The optimizer chooses `Index Scan using chunks_embedding_fp32_hnsw_idx`. Because sequential scanning on `chunks_embedding_fp32` requires reading 3,010 TOAST blocks, the optimizer models the HNSW index scan as lower cost.
- **FP16 Table:** The optimizer chooses `Seq Scan on chunks_embedding_halfvec`! Because all 1,001 FP16 vectors reside inline across only 201 heap blocks ($0\text{ TOAST blocks}$), the optimizer determines that sequentially reading 201 cached heap pages has lower estimated cost than traversing index graph pages.
- **Reporting Boundary:** As mandated by benchmark integrity principles, the planner-selected sequential scan on FP16 is **strictly isolated and never reported as an HNSW result**.

---

## 5. Retrieval Benchmark Results

Evaluated across the 20 fixed evaluation queries over 5 measurement passes (100 trials total per method). Paired execution order was alternated across passes (Runs 1, 3, 5: FP32 then FP16; Runs 2, 4: FP16 then FP32) to eliminate systematic CPU cache and garbage collection bias.

### A. Client-Observed Retrieval Latency (100 Trials per Mode)

| Execution Mode | Representation | Mean Latency | Median Latency | Min Latency | Max Latency | StDev |
|---|---|---:|---:|---:|---:|---:|
| **Forced HNSW Index Scan** | **FP32 (`vector`)** | **3.165 ms** | **2.866 ms** | 2.665 ms | 5.481 ms | 0.618 ms |
| **Forced HNSW Index Scan** | **FP16 (`halfvec`)** | **2.984 ms** | **2.788 ms** | 2.589 ms | 4.836 ms | 0.467 ms |
| *Forced Comparison* | *FP16 vs. FP32* | *-5.73%* | *-2.72%* | — | — | *Lower latency & variance* |
| **Planner-Selected Scan** | **FP32 (HNSW Index)** | **3.111 ms** | **2.852 ms** | 2.599 ms | 6.461 ms | 0.628 ms |
| **Planner-Selected Scan** | **FP16 (Heap Seq Scan)** | **3.171 ms** | **2.984 ms** | 2.812 ms | 5.941 ms | 0.516 ms |

---

## 6. Approximate Nearest Neighbor Recall Analysis

ANN Recall measures the retrieval fidelity of approximate HNSW graph traversal against the uncompressed, exhaustive **FP32 exact-search reference rankings** established in Stage 2:
$$\text{Recall@K}_{\text{ANN}} = \frac{|\text{Top-K}_{\text{FP32 Exact}} \cap \text{Top-K}_{\text{Forced HNSW}}|}{|K|}$$

### Measured Recall Across 20 Evaluation Queries ($ef_{search} = 40$)

| Evaluation Metric | Forced FP32 HNSW | Forced FP16 HNSW | Relative Index Agreement |
|---|---:|---:|---:|
| **Recall@3** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** |
| **Recall@5** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** |
| **Recall@10** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** |

### Findings on Accuracy:
- At $ef_{search} = 40$, neither FP32 HNSW nor FP16 HNSW suffered any retrieval recall degradation relative to the exhaustive FP32 ground truth across all 20 test queries.
- Relative index agreement between FP32 HNSW and FP16 HNSW was 100.0%, confirming that the 16-bit half-precision representation does not distort the top-10 graph neighborhood.

---

## 7. Database Integrity & Safety Verification

Direct inspection of all database environments confirmed strict adherence to safety boundaries:

1. **Production Database (`compression_aware_rag`):**
   - Documents count: **100** (intact)
   - Chunks count: **180** (intact)
   - Indexes: `document_chunks_pkey`, `document_chunks_document_id_chunk_index_key`, and original baseline index `document_chunks_embedding_hnsw_idx` (`m=16, ef_construction=64`).
   - Representation tables: **0** (none created in production).
2. **Experimental Database (`compression_aware_rag_exp`):**
   - Documents count: **550**
   - Chunks count: **1,001**
   - `chunks_embedding_fp32`: 1,001 rows with `chunks_embedding_fp32_hnsw_idx`
   - `chunks_embedding_halfvec`: 1,001 rows with `chunks_embedding_halfvec_hnsw_idx`
3. **Smart Document Chatbot (`smart_document_chatbot`):**
   - Completely untouched (public schema table count: 3).
4. **Test Database (`compression_aware_rag_test`):**
   - Cleared and isolated via test fixtures; 0 rows post-test.

---

## 8. Automated Test Suite Results

The full test suite was expanded to include 7 new Stage 3 tests in [`tests/test_hnsw_experiment.py`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/tests/test_hnsw_experiment.py):
- Fail-closed guard blocking operations against production database.
- Server configuration retrieval (`maintenance_work_mem`, `shared_buffers`, `work_mem`).
- HNSW index creation, sizing, and detection of pre-existing indexes.
- Programmatic plan verification of `Index Scan using ...`.
- Overlap recall logic and boundary conditions.
- Alternating paired execution order logic.
- Mini end-to-end benchmark run.

**Execution Result:**
```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
collected 34 items

tests/test_basic.py ...........                                          [ 32%]
tests/test_hnsw_experiment.py .......                                    [ 52%]
tests/test_ingestion_cache.py ........                                   [ 76%]
tests/test_representations.py ........                                   [100%]

============================== 34 passed in 8.85s ==============================
```

---

## 9. Academic Conclusions and Cumulative Progress

| Dimension | Review 2 Baseline | Stage 2 (FP16 Exact) | Stage 3 (FP16 HNSW) | Cumulative Stage 3 Gain |
|---|---|---|---|---|
| **Corpus Scale** | 100 docs / 180 chunks | 550 docs / 1,001 chunks | 550 docs / 1,001 chunks | 5.56x scaled corpus |
| **Vector Payload** | 3,072 B (FP32) | 1,536 B (FP16) | 1,536 B (FP16) | **50.00% coordinate reduction** |
| **Datum Column Size** | 3,076 B (TOASTed) | 1,544 B (Inline) | 1,544 B (Inline) | **49.80% datum reduction** |
| **Representation Storage** | ~4.5 MB (with text) | 1.73 MB (`halfvec` table) | 1.73 MB (`halfvec` table) | **60.11% table footprint reduction** |
| **Index Size** | 0.82 MB (180 chunks) | N/A (unindexed) | **1.95 MB** (1,001 chunks) | **49.90% smaller than FP32 HNSW (3.90 MB)** |
| **Forced Traversal Latency** | 2.270 ms (180 chunks) | 1.784 ms (Exact Seq Scan) | **2.984 ms** (1,001 chunks) | **5.73% faster than FP32 HNSW (3.165 ms)** |
| **Server-Side Exec Time** | 1.012 ms (180 chunks) | 0.404 ms (Exact Seq Scan) | **0.886 ms** (1,001 chunks) | **30.07% faster than FP32 HNSW (1.267 ms)** |
| **Recall@10 vs Exact** | 1.0000 | 1.0000 | **1.0000** | **0.00% recall loss** |

---

## 10. Reproducibility Commands

To reproduce the Stage 3 experimental findings from scratch:

```bash
# 1. Inspect server configuration settings
python -c "from src.hnsw_experiment import get_server_settings; print(get_server_settings())"

# 2. Build and verify HNSW index sizes on experimental database
python -c "from src.hnsw_experiment import build_hnsw_indexes, get_hnsw_index_sizes; print(build_hnsw_indexes()); print(get_hnsw_index_sizes())"

# 3. Execute full 100-trial benchmark across forced and planner-selected modes
python -c "from src.hnsw_experiment import run_hnsw_benchmark; res = run_hnsw_benchmark(); print('Forced FP32:', res['forced_hnsw']['fp32_latency_ms']); print('Forced FP16:', res['forced_hnsw']['halfvec_latency_ms'])"

# 4. Run automated test suite
pytest -v
```
