# Baseline Benchmark Report: Verified Review 2 Stabilization

**Project:** Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency Retrieval-Augmented Generation (RAG)<br>
**Status:** Verified Stabilization Baseline<br>
**Date:** 2026-10-14

---

## 1. Overview and Provenance

This document records the **verified stabilization baseline benchmark** executed following the remediation of the test-isolation and benchmark methodology issues identified in the codebase audit.

These figures represent previously reported, verified stabilization results. They establish an uncompressed baseline against which future compression techniques (e.g., FP16/`halfvec`, Scalar Quantization, Product Quantization) will be evaluated once the corpus is scaled to experimental size.

---

## 2. Experimental Setup and Methodology

| Parameter | Configuration |
|---|---|
| **Database Engine** | PostgreSQL 18 with pgvector 0.8.4 |
| **Active Database** | `compression_aware_rag` |
| **Document Corpus** | MS MARCO subset (100 documents) |
| **Chunks / Embeddings** | 180 text chunks (chunk size: 300, overlap: 50) |
| **Embedding Model** | Ollama `nomic-embed-text` (768 dimensions) |
| **Distance Metric** | Cosine distance (`<=>` operator, `vector_cosine_ops`) |
| **Top-K ($K$)** | 3 |
| **Query Set** | 5 unique domain questions |
| **Warm-up Cycles** | 2 full warmup runs across all queries |
| **Measurement Runs** | 5 measurement runs per query (25 timed trials per method) |
| **Connection Strategy** | Single persistent database connection reused for all queries |
| **Embedding Timing** | Query embeddings pre-generated once beforehand (excluded from latency) |
| **Timer** | High-resolution `time.perf_counter()` isolating SQL execution and fetch |
| **HNSW Index Parameters** | $m=16$, $ef_{construction}=64$, $ef_{search}=40$ |

---

## 3. Verified Benchmark Results

The following measurements reflect the verified stabilization baseline:

| Metric | Exact Search | HNSW Indexed Search |
|---|---:|---:|
| **Average Client-Observed Latency** | **2.180 ms** | **2.270 ms** |
| **Median Client-Observed Latency** | **2.049 ms** | **2.125 ms** |
| **Single-Sample Server Execution Time** | **0.644 ms** | **1.012 ms** |
| **Single-Sample Server Planning Time** | 0.048 ms | 0.080 ms |
| **Recall@3** | Ground truth | **1.0000** |

---

## 4. Distinction Between Latency Metrics

- **Client-Observed Retrieval Latency:**
  Measured with `time.perf_counter()` over 25 individual trials (5 queries $\times$ 5 runs) using a persistent database connection. This metric captures the client-observed round-trip time for statement execution and row retrieval. Network connection establishment and embedding generation are strictly excluded.
- **Server-Side Execution Time:**
  Captured independently via PostgreSQL `EXPLAIN (ANALYZE, BUFFERS)`. **Crucially, the server execution times (0.644 ms for exact, 1.012 ms for HNSW) are diagnostic single-query sample measurements from the first query (`sample_emb`), not aggregate averages across all 25 trials.** `EXPLAIN ANALYZE` was intentionally not executed during multi-run latency timing to avoid engine profiling overhead.

---

## 5. Execution Plan Verification & Planner Analysis

Both search methods were verified independently via query execution plans:

### Exact Search (Natural Sequential Scan)
```text
Limit  (cost=10.58..10.58 rows=3 width=213)
  ->  Sort  (cost=10.58..11.03 rows=180 width=213)
        Sort Key: ((embedding <=> '[...]'::vector))
        ->  Seq Scan on document_chunks c  (cost=0.00..8.25 rows=180 width=213)
```
Exact search executes an exhaustive sequential scan over all 180 chunks with top-N heapsort. No index is used.

### HNSW Search (Forced Diagnostic Index Scan)
```text
Limit  (cost=365.35..365.79 rows=3 width=213)
  ->  Index Scan using document_chunks_embedding_hnsw_idx on document_chunks c  (cost=365.35..391.60 rows=180 width=213)
        Order By: (embedding <=> '[...]'::vector)
```
To measure HNSW indexing behavior at 180 vectors, the session temporarily applied `SET enable_seqscan = off;` and `SET hnsw.ef_search = 40;`, with settings safely restored in a `finally` block via `RESET enable_seqscan; RESET hnsw.ef_search;`.

### Why PostgreSQL Naturally Prefers Sequential Scanning at this Scale
At 180 rows, PostgreSQL's query optimizer estimates:
- **Sequential scan cost:** $8.25$
- **HNSW index scan startup cost:** $365.35$

The entire table occupies approximately $560\text{ KB}$ across fewer than 75 shared buffer pages (8 KB each). Streaming contiguous memory pages sequentially into CPU cache takes less time ($0.644\text{ ms}$) than following non-contiguous graph pointers and priority-queue traversal in HNSW ($1.012\text{ ms}$).

**Important:** The observation that exact search is slightly faster than HNSW at 180 vectors is an expected property of small-scale database caching. **It must not be claimed that HNSW is generally slower than sequential scanning.** ANN indexes are designed for sub-linear asymptotic scaling on corpora with tens of thousands to millions of vectors, where sequential scanning scales linearly $\mathcal{O}(N)$.

---

## 6. Recall@3 and Dataset Scale Limitations

Recall@3 was evaluated against actual chunk IDs retrieved from verified plans:
$$\text{Recall@3} = \frac{|\text{exact\_ids} \cap \text{hnsw\_ids}|}{|\text{exact\_ids}|}$$

Every query achieved $\text{Recall@3} = 1.0000$ (identical top-3 sets between exact and forced HNSW scans).

### Scope Limitations:
1. **Sample Size:** 180 vectors are insufficient to draw conclusions about ANN recall degradation or trade-offs.
2. **Search Beam vs Graph Size:** With `ef_search=40` and $N=180$, the search queue evaluates ~22% of all nodes in the graph. With $m=16$, graph diameter is 2–3 hops, allowing greedy beam search to reliably locate true nearest neighbors.
3. **Future Scaling Requirement:** To rigorously evaluate vector compression (FP16, SQ, PQ) and HNSW trade-offs, the corpus must be scaled to at least 10,000 to 50,000+ vectors in subsequent phases, where true index speedups and recall trade-offs emerge.
