# Comprehensive Final Submission Handoff & Research Dossier

- **Project Title:** Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency Retrieval-Augmented Generation (RAG)
- **Course:** BCSE302P — Database Systems Lab, Department of Computer Science and Engineering, Vellore Institute of Technology, Vellore
- **Authors:** Yash Pradhan, Anjini Pandey
- **Document Role:** Final Submission Handoff Package for LLM-Assisted Report Generation (Claude Chat) and Technical Review 3 Presentation
- **Baseline Git Commit:** `ac25c88` (`Add corpus scaling experiment`)
- **Status:** Verified, Fully Reconciled, and Read-Only Audited

---

## 1. Executive Summary & Research Framing

### A. Problem Statement
Retrieval-Augmented Generation (RAG) architectures increasingly rely on Relational Database Management Systems (RDBMS) augmented with vector extensions—specifically PostgreSQL with `pgvector`—to unify relational metadata management with dense semantic retrieval. Standard dense embedding models (e.g., `nomic-embed-text` with 768 dimensions) produce vectors encoded in IEEE 754 32-bit single-precision floating-point format (`vector`).

In PostgreSQL, high-dimensional FP32 vectors trigger a severe **storage-engine pathology**:
1. A 768-dimensional FP32 vector occupies **3,076 bytes** of datum storage as measured by PostgreSQL's `pg_column_size()` (with the documented pgvector storage formula defined as $4 \times 768 + 8 = 3,080\text{ bytes}$, leaving an unresolved 4-byte empirical discrepancy).
2. PostgreSQL manages storage in 8 KiB page blocks (`BLCKSZ = 8192`). When an inserted or updated tuple exceeds `TOAST_TUPLE_THRESHOLD` (normally approximately 2 KiB, where the exact value follows PostgreSQL's tuple/page-size calculation), PostgreSQL triggers TOAST processing and attempts compression or out-of-line storage until tuple length falls below `TOAST_TUPLE_TARGET` (normally also approximately 2 KiB).
3. Because dense floating-point arrays do not compress effectively with byte-level LZ algorithms (`pglz`/`lz4`), every uncompressed FP32 vector is evicted out-of-line from the relational heap table into secondary **TOAST (The Oversized-Attribute Storage Technique)** tables (`pg_toast`) to reach `TOAST_TUPLE_TARGET`.
4. TOAST eviction induces massive buffer-access amplification: scanning an FP32 vector requires reading pointer descriptors in the heap, traversing TOAST B-tree indexes, and assembling multiple 2 KiB chunks.
5. In Hierarchical Navigable Small World (HNSW) graph indexing, FP32 vectors double the physical index footprint relative to 16-bit representations, placing higher eviction pressure on PostgreSQL's shared buffer cache (`shared_buffers`).

### B. Core Research Question
*Does coordinate-level half-precision representation (`halfvec` / IEEE 754 Float16) resolve the PostgreSQL TOAST eviction pathology, reduce HNSW index footprint, and improve retrieval latency across varying corpus scales, and what are the empirical trade-offs in nearest-neighbor recall overlap (Recall@K)?*

### C. Primary Objectives
1. **Quantify Storage Compaction:** Measure physical storage across coordinate datums, table heaps, secondary TOAST relations, and HNSW graph indexes from 1,001 up to 10,000 chunks.
2. **Benchmark Retrieval Latency:** Compare exact sequential search and approximate HNSW index search under controlled database conditions.
3. **Evaluate Retrieval Fidelity:** Measure Nearest-Neighbor Overlap (Recall@K against uncompressed FP32 exact-search references) and relative index agreement (pairwise overlap between FP32 HNSW and FP16 HNSW).
4. **Analyze Query Optimizer Plan Transitions:** Track planner cost modeling and plan selection across varying corpus sizes.
5. **Enforce Database Isolation & Safety:** Guarantee zero mutation of baseline production tables during experimentation.

### D. Key Contributions
- **Empirical Demonstration of the TOAST Inversion:** Proved that FP32 vectors create a relational storage inversion where secondary TOAST tables dwarf the primary heap by over 70x.
- **Relational Footprint Compaction Beyond Coordinate Scaling:** Demonstrated that FP16 achieves a **55.35%–55.37% inclusive relation footprint reduction** at 5k and 10k scales, exceeding the theoretical 50.0% coordinate ratio due to the complete elimination of out-of-line TOAST data tables.
- **Buffer-Access and Latency Reductions in Exact Search:** Established that inlining FP16 vectors into heap pages reduces Query 0 (Q0) shared-buffer accesses by **93.35%** (1,000 vs. 15,032 at 5k; 2,000 vs. 30,064 at 10k), which is associated with a **6.22x mean (7.12x median) speedup at 5k** and a **7.24x mean (8.03x median) speedup at 10k** in repeated trials.
- **High-Fidelity Approximate Retrieval:** Confirmed that FP16 HNSW achieves **96.0% Recall@10** at both 5k and 10k chunks relative to the uncompressed FP32 exact reference at $ef_{search} = 40$.
- **HNSW Index Footprint Reduction:** HNSW Index Footprint is **49.90% smaller at 1,001 chunks and 49.98% smaller at 5,000 and 10,000 chunks** relative to FP32 HNSW.
- **Optimizer Crossover Mapping:** Documented the planner's transition from Sequential Scan (at 1,001 chunks / 201 heap blocks) to HNSW Index Scan (at 5,000 chunks / 1,000 heap blocks) under default cost parameters.

### E. Master Empirical Findings Summary (1,001 to 10,000 Chunks)
- **Coordinate Datum Storage:** FP16 reduces datum size from 3,076 B to 1,544 B (49.80% measured reduction; 49.87% formula reduction), bringing vectors below the ~2 KiB TOAST threshold.
- **TOAST Data Elimination:** 100% eliminated on FP16 (0 bytes TOAST data table payload across all tested scales, eliminating 39.06 MiB of TOAST data at 10k).
- **Relation Total Compaction:** 60.11% reduction in pre-HNSW relation total at 1,001 chunks (4.13 MiB to 1.65 MiB); 55.35% inclusive relation reduction at 5k (39.68 MiB to 17.72 MiB) and 55.37% at 10k (79.20 MiB to 35.35 MiB).
- **HNSW Index Footprint:** 49.90% smaller at 1,001 chunks and 49.98% smaller at 5,000 and 10,000 chunks (3.90 MiB to 1.95 MiB at 1k; 19.48 MiB to 9.74 MiB at 5k; 38.89 MiB to 19.45 MiB at 10k).
- **Exact Search Latency (Sequential Scan):** 6.22x mean (7.12x median) speedup at 5k (42.29 ms vs. 6.80 ms mean); 7.24x mean (8.03x median) speedup at 10k (89.00 ms vs. 12.30 ms mean), which is consistent with the observed 93.35% reduction in single-query Q0 shared-buffer accesses (1,000 vs. 15,032 hits at 5k; 2,000 vs. 30,064 hits at 10k).
- **HNSW Traversal Latency:** Traversal latencies remain similar (~3.3–3.8 ms for both representations at ef_search = 40).
- **Index Construction Efficiency:** FP16 HNSW built 13.52% faster at 5k (1.46 s vs. 1.26 s) and 29.79% faster at 10k (3.76 s vs. 2.64 s) in single-run wall-clock measurements.
- **Retrieval Fidelity:** 96.0% Recall@10 at both 5k and 10k scales relative to uncompressed FP32 exact reference rankings; pairwise relative index agreement between FP32 HNSW and FP16 HNSW was 98.5% at 5k and 95.0% at 10k.
- **Planner Plan Selection:** Planner selected Seq Scan on FP16 at 1,001 chunks (201 heap blocks) and transitioned to HNSW Index Scan at 5,000 chunks (1,000 heap blocks) and 10,000 chunks (2,000 heap blocks).

---

## 2. Core DBMS & Vector Retrieval Concepts

### A. The PostgreSQL Storage Engine & TOAST Mechanics
- **Page Layout:** PostgreSQL stores tables in 8 KiB page blocks. Each page contains a page header, an array of line pointers (`ItemIdData`), free space, and tuple data.
- **TOAST Architecture:** TOAST prevents tuples from spanning physical pages.
  - `TOAST_TUPLE_THRESHOLD` triggers out-of-line storage for oversized tuples.
  - `TOAST_TUPLE_TARGET` defines the target tuple length after compression or out-of-line eviction attempts.
  - Both parameters evaluate to approximately 2 KiB under standard 8 KiB pages.
  - When an FP32 vector (3,076 B) is inserted, PostgreSQL creates an out-of-line TOAST pointer in the heap and chunks the vector payload into a separate relation (`pg_toast_<oid>`).
- **The Inversion:** In a dedicated vector-only table at 10,000 chunks, the FP32 table heap occupies only 512 KiB, while its TOAST data relation consumes 39.06 MiB across 5,000 TOAST pages.

### B. `vector` vs. `halfvec` Numerical Formats
- **`vector(768)` (IEEE 754 FP32):** 4 bytes per dimension (1 sign bit, 8 exponent bits, 23 mantissa bits). Dynamic range $\approx 10^{\pm 38}$, precision $\approx 7.2$ decimal digits. Column datum size = 3,076 bytes (measured via `pg_column_size`).
- **`halfvec(768)` (IEEE 754 FP16):** 2 bytes per dimension (1 sign bit, 5 exponent bits, 10 mantissa bits). Dynamic range $\approx 10^{\pm 4}$, precision $\approx 3.3$ decimal digits. Column datum size = 1,544 bytes (measured via `pg_column_size`).
- **Coordinate Reduction:** Reduces datum size by **49.80% (1.992x)** empirically (and 49.87% theoretically under the documented formula).

### C. Exact Exhaustive Search vs. Approximate HNSW Graph Traversal
- **Exact Search:** Performs an exhaustive sequential scan across the table heap (and secondary TOAST tables for FP32), evaluating cosine distance (`<=>`) against every stored vector, followed by top-N sorting.
- **HNSW Approximate Nearest Neighbor (ANN) Search:** Constructs a multilayer geometric graph where greedy routing finds nearest neighbors with sub-linear traversal overhead. Graph density and search quality are controlled by hyperparameters:
  - $m = 16$: Maximum bidirectional connection links per node (up to $2m = 32$ on Layer 0).
  - $ef_{construction} = 64$: Size of the dynamic candidate list during graph construction.
  - $ef_{search} = 40$: Size of the dynamic candidate priority queue during query traversal.

### D. Retrieval Overlap Evaluation (ANN Recall@K)
- Ground-truth nearest neighbors are established independently at each scale using exhaustive, sequential scans on the uncompressed FP32 representation (`enable_seqscan = on`, index scans disabled).
- **Mathematical Definition:**
  $$\text{Recall@K}_{\text{ANN}} = \frac{|\text{Top-K}_{\text{FP32 Exact}} \cap \text{Top-K}_{\text{Candidate HNSW}}|}{K}$$
- **Scope Limitation:** Recall@K measures mathematical neighborhood overlap against the uncompressed exact nearest-neighbor reference. It does not measure semantic relevance, downstream LLM generation quality, or end-to-end RAG answer correctness.

---

## 3. Experimental Progression & Methodological Comparability

### A. Stage-by-Stage Evolution
1. **Review 2 Baseline (180 Chunks):** Evaluated 100 documents / 180 chunks on `compression_aware_rag`. Stored full text and vectors in a monolithic table (`document_chunks`). Benchmark included REST round-trips and variable text overhead.
2. **Stage 1: Setup & Reproducibility (1,001 Chunks):** Ingested a 550-document subset of `BeIR/msmarco` into `compression_aware_rag_exp`. Established cryptographic manifests and baseline isolation.
3. **Stage 2: Representation Evaluation (1,001 Chunks):** Introduced dedicated vector-only tables (`chunks_embedding_fp32` and `chunks_embedding_halfvec`). Demonstrated 60.11% reduction in pre-HNSW relation total and 100% TOAST data table elimination.
4. **Stage 3: HNSW Indexing Benchmark (1,001 Chunks):** Constructed HNSW indexes on 1,001 chunks. Benchmark evaluated queries inside per-trial `BEGIN; SET LOCAL ...; COMMIT;` transaction blocks. Observed query planner selecting `Seq Scan` on FP16 due to compact heap size (201 blocks).
5. **Stage 4: Corpus Scaling Benchmark (5,000 & 10,000 Chunks):** Scaled corpus on dedicated `compression_aware_rag_scale`. Corrected timing harness to Stage 4 v2 (GUC configuration outside timer; timing strictly around client statement execute and fetch; ordering by `distance` alias for exact search). Confirmed planner crossover to `Index Scan` for FP16 at scale.

### B. Methodological Boundaries: What is and is Not Cross-Comparable
- **Valid Comparisons:**
  - Within-scale paired comparisons at 1,001, 5,000, and 10,000 chunks (identical queries, identical hardware, identical alternating execution order).
  - Storage footprint scaling (`pg_relation_size`, `pg_total_relation_size`) across all stages.
  - Stage 4 5k vs. 10k scaling trends under the corrected v2 timing harness.
- **Inadmissible Cross-Stage Comparisons (Must NOT be Conflated):**
  - *Review 2 (180 chunks) vs. Subsequent Stages:* Monolithic text+vector schema vs. dedicated vector-only schema.
  - *Stage 3 (1,001 chunks) Latency vs. Stage 4 v2 (5k & 10k) Latency:* Stage 3 timed transaction-block dispatch (`BEGIN; SET LOCAL; ...; COMMIT;`), whereas Stage 4 v2 timed strictly client statement `execute` and `fetchall`. Comparing these raw latencies directly would confound corpus size with timing harness changes.
  - *Single-Run Index Build Times Across Scales:* Represent indicative wall-clock engineering observations rather than statistically bounded distributions.

---

## 4. Experimental Environment & Parameter Control

All experiments were executed on a dedicated, local research host with strict hardware and software controls:

- **Host Hardware:** Intel Core i7-13700HX (16 physical cores, 24 threads, max turbo 5.0 GHz), 16 GB DDR5 RAM (4800 MT/s), 1 TB NVMe PCIe Gen 4 SSD.
- **Operating System:** Microsoft Windows 11 Home (Build 22631).
- **Database Engine:** PostgreSQL 18.0 (x86_64-pc-windows-msvc) with `pgvector 0.8.4`.
- **Runtime Environment:** Python 3.13.2, `psycopg 3.3.4` (C-extension client), Ollama 0.4.7.
- **PostgreSQL Server Tuning Parameters:**
  - `shared_buffers = 128 MB` (default shared buffer pool)
  - `work_mem = 64 MB` (in-memory sort space)
  - `maintenance_work_mem = 64 MB` (index build memory)
  - `random_page_cost = 1.1` (SSD-optimized random access)
  - `effective_cache_size = 4 GB` (planner OS cache heuristic)
- **Embedding Model:** `nomic-embed-text` via Ollama (768 dimensions, cosine similarity).
- **Chunking Pipeline:** Character-based chunking with `chunk_size = 300` characters and `overlap = 50` characters.
- **Dataset Provenance:** Microsoft MARCO Passage Ranking dataset (`BeIR/msmarco`). Canonical manifest tracked at [`data/raw/msmarco_10k_canonical.meta.json`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/data/raw/msmarco_10k_canonical.meta.json) with verified SHA-256 integrity.
- **Database Isolation Architecture:**
  - `compression_aware_rag`: Production baseline (100 documents / 180 chunks). Zero modifications; read-only verification queries only.
  - `smart_document_chatbot`: Application database. Never modified.
  - `compression_aware_rag_exp`: Experimental baseline for Stages 1–3 (550 documents / 1,001 chunks).
  - `compression_aware_rag_scale`: Dedicated database for Stage 4 scaling (5,000 and 10,000 chunks).
- **Benchmark Execution Controls:**
  - 20 fixed evaluation queries pre-embedded to eliminate inference variance.
  - 2 unmeasured warm-up passes to prime buffer cache.
  - 5 measurement passes (100 timed trials per configuration).
  - Alternating paired execution order (FP32 then FP16 on even passes; FP16 then FP32 on odd passes) to eliminate systemic cache and GC bias.

---

## 5. Verified Master Cross-Stage Results Matrix

The table below provides complete cross-stage metrics. All binary storage values are reported in mebibytes (MiB, $1\text{ MiB} = 1,048,576\text{ bytes}$) and kibibytes (KiB, $1\text{ KiB} = 1,024\text{ bytes}$).

| Experimental Dimension | Review 2 Baseline | Stage 1 (Setup) | Stage 2 (FP16 Exact) | Stage 3 (FP16 HNSW) | Stage 4 (5k Chunks) | Stage 4 (10k Chunks) |
|---|---|---|---|---|---|---|
| **Target Database** | `compression_aware_rag` | `compression_aware_rag_exp` | `compression_aware_rag_exp` | `compression_aware_rag_exp` | `compression_aware_rag_scale` | `compression_aware_rag_scale` |
| **Corpus Scale** | 100 docs / 180 chunks | 550 docs / 1,001 chunks | 550 docs / 1,001 chunks | 550 docs / 1,001 chunks | 2,729 docs / 5,000 chunks | 5,457 docs / 10,000 chunks |
| **Table Schema** | Monolithic (`content`+vec) | Monolithic (Stage 1 base) | Dedicated Vector-Only | Dedicated Vector-Only | Dedicated Vector-Only | Dedicated Vector-Only |
| **Evaluation Queries** | 5 domain queries | N/A (ingestion only) | 20 evaluation queries | 20 evaluation queries | 20 evaluation queries | 20 evaluation queries |
| **Timed Trials / Config** | 25 trials (5q × 5r) | N/A | 100 trials (20q × 5r) | 100 trials (20q × 5r) | 100 trials (20q × 5r) | 100 trials (20q × 5r) |
| **Timing Boundary** | Client SQL + Fetch | N/A | Client SQL + Fetch | Session Tx + GUC + Fetch | **Client Exec + Fetch (v2)** | **Client Exec + Fetch (v2)** |
| **Column Datum (FP32)** | 3,076 B | 3,076 B | 3,076 B | 3,076 B | 3,076 B | 3,076 B |
| **Column Datum (FP16)** | N/A | N/A | 1,544 B (-49.80%) | 1,544 B (-49.80%) | 1,544 B (-49.80%) | 1,544 B (-49.80%) |
| **TOAST Data Table (FP32)** | ~3.8 MiB (in monolithic) | ~3.91 MiB | 3.91 MiB (4,104,192 B) | 3.91 MiB (4,104,192 B) | 19.53 MiB (20,480,000 B) | 39.06 MiB (40,960,000 B) |
| **TOAST Data Table (FP16)** | N/A | N/A | **0 B (100% eliminated)** | **0 B (100% eliminated)** | **0 B (100% eliminated)** | **0 B (100% eliminated)** |
| **Pre-HNSW Relation Total (`pg_total_relation_size`)** | ~4.3 MiB (with text) | ~4.4 MiB | 4.13 MiB (4,333,568 B) [FP32]<br>**1.65 MiB (1,728,512 B) [FP16]**<br>*(60.11% reduction; includes PK index)* | 4.13 MiB (4,333,568 B) [FP32]<br>**1.65 MiB (1,728,512 B) [FP16]**<br>*(pre-HNSW relation total)* | N/A (indexed directly) | N/A (indexed directly) |
| **HNSW Index Footprint (`pg_relation_size`)** | 0.80 MiB (835,584 B) | N/A | N/A (unindexed) | 3.90 MiB (4,087,808 B) [FP32]<br>**1.95 MiB (2,048,000 B) [FP16]**<br>*(49.90% reduction)* | 19.48 MiB (20,422,656 B) [FP32]<br>**9.74 MiB (10,215,424 B) [FP16]**<br>*(49.98% reduction)* | 38.89 MiB (40,779,776 B) [FP32]<br>**19.45 MiB (20,398,080 B) [FP16]**<br>*(49.98% reduction)* |
| **Inclusive Relation Total (`pg_total_relation_size`)** | Unmeasured inclusive total | N/A | 4.13 MiB [FP32] / 1.65 MiB [FP16]<br>*(pre-HNSW relation; no HNSW index)* | Unrecorded inclusive post-index total<br>*(HNSW measured separately)* | 39.68 MiB (41,607,168 B) [FP32]<br>**17.72 MiB (18,579,456 B) [FP16]**<br>*(55.35% reduction)* | 79.20 MiB (83,050,496 B) [FP32]<br>**35.35 MiB (37,068,800 B) [FP16]**<br>*(55.37% reduction)* |
| **HNSW Build Duration** | Not recorded | N/A | N/A | 0.233 s [FP32] / **0.161 s [FP16]**<br>*(~30.9% faster; 31.19% rec.)* | 1.462 s [FP32] / **1.264 s [FP16]**<br>*(~13.5% faster; 13.52% rec.)* | 3.762 s [FP32] / **2.641 s [FP16]**<br>*(~29.8% faster; 29.79% rec.)* |
| **Exact Search Mean Latency** | 2.180 ms (180c) | N/A | 3.320 ms [FP32] / 1.784 ms [FP16] | N/A | 42.285 ms [FP32] / **6.802 ms [FP16]**<br>*(6.22x speedup / -83.91%)* | 89.004 ms [FP32] / **12.301 ms [FP16]**<br>*(7.24x speedup / -86.18%)* |
| **Exact Search Median Latency** | 2.049 ms (180c) | N/A | 3.120 ms [FP32] / 1.644 ms [FP16] | ~2.2 ms [FP32] / ~1.8 ms [FP16]<br>*(historical benchmark)* | 44.694 ms [FP32] / **6.277 ms [FP16]**<br>*(7.12x speedup / -85.96%)* | 92.664 ms [FP32] / **11.541 ms [FP16]**<br>*(8.03x speedup / -87.55%)* |
| **Forced HNSW Mean Latency** | 2.270 ms (180c) | N/A | N/A | 3.165 ms [FP32] / 2.984 ms [FP16] | 4.129 ms [FP32] / **3.372 ms [FP16]**<br>*(18.32% lower)* | 3.674 ms [FP32] / **3.856 ms [FP16]**<br>*(similar traversal times)* |
| **Forced HNSW Median Latency** | 2.125 ms (180c) | N/A | N/A | 2.866 ms [FP32] / 2.788 ms [FP16] | 3.371 ms [FP32] / **2.923 ms [FP16]**<br>*(13.28% lower)* | 3.415 ms [FP32] / **3.820 ms [FP16]**<br>*(both medians within ~0.4 ms)* |
| **Diagnostic Q0 Buffer Hits** | Not recorded | N/A | 3,010 [FP32] / 201 [FP16]<br>*(Exact search)* | 595 [FP32] / 514 [FP16]<br>*(Forced HNSW)* | 15,032 [FP32] / 1,000 [FP16] (Exact)<br>830 [FP32] / 753 [FP16] (HNSW) | 30,064 [FP32] / 2,000 [FP16] (Exact)<br>930 [FP32] / 847 [FP16] (HNSW) |
| **Planner Choice (FP16)** | `Seq Scan` (75 blks) | N/A | `Seq Scan` (unindexed) | `Seq Scan` (201 blks) | **`Index Scan` (753 hits)** | **`Index Scan` (847 hits)** |
| **Recall@10 vs Exact Ref** | 1.0000 (Recall@3) | N/A | 1.0000 (Agreement) | 1.0000 (Forced HNSW) | **0.9600 (96.0%)** | **0.9600 (96.0%)** |

---

## 6. Chart Specifications & Source Data Assets

The chart datasets are stored as clean CSV files in [`docs/submission_assets/`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/submission_assets/):

### Chart 1: Physical Storage Footprint Scaling Across Corpus Scales
- **File:** [`docs/submission_assets/chart1_storage.csv`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/submission_assets/chart1_storage.csv)
- **Chart Type:** Grouped Bar Chart (Clustered by Scale)
- **X-axis:** Corpus Scale (1,001, 5,000, 10,000 chunks)
- **Y-axis:** Storage Footprint in Mebibytes (MiB)
- **Series:**
  1. `FP32 Table Base Storage` (Heap + TOAST + PK Index + forks)
  2. `FP16 Table Base Storage` (Heap + TOAST + PK Index + forks)
  3. `FP32 HNSW Index Footprint`
  4. `FP16 HNSW Index Footprint`
- **Caption:** *Physical storage footprint comparison showing base table storage and HNSW index size across corpus scales. Base table storage is kept distinct from HNSW index size to prevent double-counting. FP16 eliminates TOAST tables, achieving ~55.4% inclusive relation compaction, while HNSW index footprint is 49.90% smaller at 1k chunks and 49.98% smaller at 5k and 10k chunks.*
- **Methodological Limitation:** At 1,001 chunks, inclusive post-index total was not recorded as a single catalog query in Stage 3; component sizes are plotted from Stage 2 and Stage 3 measurements.

### Chart 2: Diagnostic Query 0 Shared-Buffer Accesses vs. Server Execution Time
- **File:** [`docs/submission_assets/chart2_q0_diagnostics.csv`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/submission_assets/chart2_q0_diagnostics.csv)
- **Chart Type:** Scatter Plot with Connected Diagnostic Trendlines / Dual Bar Chart
- **X-axis:** Q0 Shared-Buffer Accesses (Buffer Hits from `EXPLAIN (ANALYZE, BUFFERS)`)
- **Y-axis:** Q0 Server Execution Time (ms)
- **Plotted Points:**
  - 5k Exact FP32: 15,032 hits, 24.54 ms
  - 5k Exact FP16: 1,000 hits, 3.00 ms
  - 10k Exact FP32: 30,064 hits, 54.49 ms
  - 10k Exact FP16: 2,000 hits, 7.81 ms
- **Caption:** *Relationship between shared-buffer accesses and server execution time during diagnostic Query 0 exact-search executions. Inlining FP16 vectors into the heap reduces single-query Q0 buffer accesses by 93.35%, a reduction strongly associated with and consistent with the observed 6.2x–8.0x speedups measured across 100 timed trials, rather than serving as standalone proof of causality.*
- **Methodological Limitation:** Reflects single diagnostic query executions (Q0) under `EXPLAIN (ANALYZE, BUFFERS)`; buffer hits represent runtime page accesses (including repeated accesses) rather than unique block counts, and must be distinguished from the 100-trial repeated client latency distributions reported in Chart 4.

### Chart 3: Approximate Nearest Neighbor Overlap (ANN Recall@K) & Pairwise Agreement
- **Files:** [`docs/submission_assets/chart3_hnsw_recall.csv`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/submission_assets/chart3_hnsw_recall.csv) (Representation Recall vs. Ground Truth) and [`docs/submission_assets/chart3_pairwise_agreement.csv`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/submission_assets/chart3_pairwise_agreement.csv) (Pairwise FP32 vs. FP16 Agreement)
- **Chart Type:** Multi-Bar Grouped Chart
- **X-axis:** Corpus Scale (5,000 and 10,000 chunks)
- **Y-axis:** Recall Ratio / Overlap Fraction (0.85 to 1.00)
- **Groupings:** Recall@3, Recall@5, Recall@10 for FP32 HNSW vs. FP16 HNSW (relative to uncompressed FP32 exact ground truth), alongside pairwise relative index agreement (0.9667, 0.9800, 0.9850 at 5k; 0.9667, 0.9600, 0.9500 at 10k).
- **Key Observation:** At $ef_{search} = 40$, FP16 HNSW achieves 96.0% Recall@10 at both 5k and 10k chunks, matching or slightly exceeding FP32 HNSW (95.5% at 5k, 94.5% at 10k). Pairwise relative index agreement remains $\ge 95\%$ (98.5% at 5k, 95.0% at 10k).
- **Metric Verification Note:** Relative index agreement is defined as the mean top-$K$ set intersection overlap fraction ($\frac{|S_{\text{FP32}} \cap S_{\text{FP16}}|}{K}$) averaged across the 20 evaluation queries. It is a symmetric set intersection proportion, not Jaccard similarity. Pairwise agreement is tracked in a dedicated dataset (`chart3_pairwise_agreement.csv`) to prevent conflating pairwise cross-index consistency with representation-specific ground-truth recall.
- **Methodological Limitation:** Evaluates overlap against uncompressed FP32 exact retrieval; does not measure semantic downstream RAG generation quality.

### Chart 4: Retrieval Latency Scaling (Exact Scan vs. Forced HNSW)
- **File:** [`docs/submission_assets/chart4_latency.csv`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/submission_assets/chart4_latency.csv)
- **Chart Type:** Dual-Panel Line / Bar Chart (Panel A: Exact Search; Panel B: Forced HNSW)
- **X-axis:** Corpus Scale (5k and 10k chunks under Stage 4 v2 timing)
- **Y-axis:** Client-Observed Retrieval Latency (ms)
- **Metrics Plotted:** Mean and Median latencies for FP32 and FP16 under Exact Search and Forced HNSW
- **Key Observation:** Across the two tested Stage 4 v2 scales (5,000 and 10,000 chunks), exact search mean latency increased from 42.29 ms to 89.00 ms for FP32 and from 6.80 ms to 12.30 ms for FP16, maintaining a 6.22x to 7.24x mean speedup (7.12x to 8.03x median speedup). Forced HNSW traversal mean latencies were measured at 4.13 ms (5k) and 3.67 ms (10k) for FP32, and 3.37 ms (5k) and 3.86 ms (10k) for FP16 (consistently within 3.3–4.2 ms across both scales). Because only two scales were benchmarked under this harness, these values represent discrete empirical comparisons rather than verified linear or asymptotic scaling trends.
- **Methodological Limitation:** Stage 3 1,001-chunk latency is strictly excluded because it timed a full transaction block (`BEGIN; SET LOCAL; ...; COMMIT;`), which would confound harness changes with corpus scale. Linear or asymptotic scaling laws must not be extrapolated from two data points.

---

## 7. Master Source-Evidence Map

The table below maps every major reported finding to its supporting repository artifact and JSON key:

| Finding / Metric Category | Reported Value | Supporting Document | Primary Source File & Key / Artifact Path |
|---|---|---|---|
| **FP32 Column Datum Size** | 3,076 B | [`stage2_fp16_evaluation.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage2_fp16_evaluation.md#L57) | `pg_column_size(embedding)` catalog query in Stage 2 & 4 audits |
| **FP16 Column Datum Size** | 1,544 B | [`stage2_fp16_evaluation.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage2_fp16_evaluation.md#L57) | `pg_column_size(embedding)` catalog query in Stage 2 & 4 audits |
| **1k Pre-HNSW Relation Size** | 4.13 MiB (FP32) / 1.65 MiB (FP16) | [`stage2_fp16_evaluation.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage2_fp16_evaluation.md#L68) | `pg_total_relation_size()` catalog queries on `compression_aware_rag_exp` |
| **1k HNSW Index Size** | 3.90 MiB (FP32) / 1.95 MiB (FP16) | [`stage3_hnsw_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage3_hnsw_benchmark.md#L58) | `pg_relation_size()` on `chunks_embedding_fp32_hnsw_idx` |
| **5k Sizing & Forks** | FP32: 41,607,168 B / FP16: 18,579,456 B | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L104-L118) | `benchmark_scale_5k_v2.json` -> `storage` dictionary |
| **10k Sizing & Forks** | FP32: 83,050,496 B / FP16: 37,068,800 B | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L104-L118) | `benchmark_scale_10k_v2.json` -> `storage` dictionary |
| **5k Exact Search Latencies** | FP32: 42.285 ms / FP16: 6.802 ms | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L138-L140) | `benchmark_scale_5k_v2.json` -> `exact_search.fp32_latency_ms.mean` / `halfvec_...` |
| **10k Exact Search Latencies** | FP32: 89.004 ms / FP16: 12.301 ms | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L151-L153) | `benchmark_scale_10k_v2.json` -> `exact_search.fp32_latency_ms.mean` / `halfvec_...` |
| **5k Forced HNSW Latencies** | FP32: 4.129 ms / FP16: 3.372 ms | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L141-L143) | `benchmark_scale_5k_v2.json` -> `forced_hnsw.fp32_latency_ms.mean` / `halfvec_...` |
| **10k Forced HNSW Latencies** | FP32: 3.674 ms / FP16: 3.856 ms | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L154-L156) | `benchmark_scale_10k_v2.json` -> `forced_hnsw.fp32_latency_ms.mean` / `halfvec_...` |
| **5k & 10k HNSW Recall@10** | 96.0% Recall@10 at both scales | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L197-L200) | `benchmark_scale_5k_v2.json` and `10k_v2.json` -> `forced_hnsw.halfvec_recall_vs_exact` |
| **Q0 Plans & Buffer Hits** | 5k: 15,032 vs 1,000 / 10k: 30,064 vs 2,000 | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L138-L159) | `benchmark_scale_..._v2.json` -> `sample_plans` -> `shared hit=...` |
| **Dataset Provenance** | 10,000 canonical chunks, 300/50 | [`stage4_scale_benchmark.md`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/docs/stage4_scale_benchmark.md#L43-L78) | [`data/raw/msmarco_10k_canonical.meta.json`](file:///c:/Coding/GenAI/Projects/CompressionAwareRag/data/raw/msmarco_10k_canonical.meta.json) |

---

## 8. Unresolved Discrepancies & Negative Claims (What NOT to Claim)

### A. Documented Discrepancies Retained Without Speculative Inventions
1. **The 4-Byte FP32 Column Datum Discrepancy:**
   - Catalog queries across Stage 2 and Stage 4 recorded **3,076 bytes** for `pg_column_size(vector(768))`.
   - The documented pgvector formula defines storage as $4 \times 768 + 8 = \mathbf{3,080\text{ bytes}}$.
   - *Status:* Retained as an unresolved empirical measurement discrepancy. Neither figure is altered, and no unverified internal storage cause is assigned.
2. **Stage 3 Inclusive Relation Footprint:**
   - In Stage 3, pre-HNSW base relation size was recorded (4.13 MiB / 1.65 MiB), and separate HNSW index sizes were recorded (3.90 MiB / 1.95 MiB), but post-index `pg_total_relation_size` was not captured as a single catalog query.
   - *Status:* Left unrecorded rather than inventing a synthetic total.
3. **Stage 3 Planner Internal Cost Numbers:**
   - Stage 3 recorded verified scan types (`Seq Scan` on FP16, `Index Scan` on FP32), but did not preserve numerical cost estimates (`cost=...`) in JSON artifacts.
   - *Status:* Acknowledged as a hypothesis rather than an established measurement.

### B. Prohibited Claims in Report & Presentation
- ❌ **Do NOT claim:** *"FP16 makes HNSW graph traversal 10x faster."* (Traversal times are 3.3–3.8 ms for both representations; speedups apply strictly to exact exhaustive search).
- ❌ **Do NOT claim:** *"The database shrank overall."* (The parent `document_chunks` table retains raw text and FP32 source embeddings; savings apply to vector representation tables and indexes).
- ❌ **Do NOT claim:** *"HNSW build durations were proven with statistical significance or measured WAL reduction."* (Index build times represent single-run wall-clock measurements; WAL volume was not measured).
- ❌ **Do NOT claim:** *"Production databases were never accessed."* (State accurately: *zero modifications were made; read-only metadata verification checks only*).
- ❌ **Do NOT claim:** *"Recall@K proves semantic answer accuracy or downstream RAG generation quality."* (State accurately: *Recall@K measures mathematical overlap against the uncompressed exact nearest-neighbor reference*).
- ❌ **Do NOT claim:** *"Results prove O(log N) asymptotic complexity for HNSW or that exact search was proven to scale linearly."* (Report observed latency measurements across the two tested Stage 4 v2 sizes (5k and 10k) without asserting theoretical asymptotic proofs or extrapolating linear/flat scaling curves from two points).
- ❌ **Do NOT claim:** *"Diagnostic Q0 buffer hits prove exact-search latency causality."* (State accurately: *Single-query Q0 buffer access reductions are consistent with and associated with repeated-trial latency speedups, but do not prove individual-trial causality*).
- ❌ **Do NOT claim:** *"Relative index agreement is Jaccard similarity."* (State accurately: *It is the mean top-K set intersection overlap fraction $|S_{\text{FP32}} \cap S_{\text{FP16}}| / K$, not the Jaccard index $|S_{\text{FP32}} \cap S_{\text{FP16}}| / |S_{\text{FP32}} \cup S_{\text{FP16}}|$*).
- ❌ **Do NOT claim:** *"Exceeding RAM automatically guarantees disk paging."* (State accurately: *induces memory pressure and cache eviction rather than automatically guaranteeing disk paging under every OS configuration*).
- ❌ **Do NOT present:** Superseded v1 latency numbers (e.g., 76.76 ms for 5k FP32) or interpolated 1,001-chunk latency figures.

---

## 9. Academic Report Blueprint (BCSE302P Final Report Structure)

Recommended structure for the comprehensive final report:

1. **Title, Abstract & Keywords:**
   - Focused title highlighting compression-aware vector database indexing in PostgreSQL.
   - Abstract summarizing the TOAST pathology, FP16 inlining, 55.4% relation reduction, 6.2x–8.0x exact search speedup, and 96.0% Recall@10.
2. **Introduction & Motivation:**
   - Proliferation of RAG architectures and enterprise adoption of PostgreSQL/pgvector.
   - The memory and storage bottleneck of high-dimensional FP32 embeddings.
   - Formal statement of research question and primary objectives.
3. **DBMS Storage Engine Pathology & Related Work:**
   - PostgreSQL page block architecture (`BLCKSZ = 8192`).
   - Detailed analysis of `TOAST_TUPLE_THRESHOLD` and `TOAST_TUPLE_TARGET` (~2 KiB).
   - Coordinate quantization (`halfvec`) vs. lossy codebook quantization (PQ/SQ).
   - HNSW graph indexing fundamentals ($m$, $ef_{construction}$, $ef_{search}$).
4. **Experimental Design & Methodology:**
   - Architecture: database isolation, schema design (monolithic vs. dedicated vector-only).
   - Dataset provenance: MS MARCO passage ranking, chunking pipeline (300/50), cryptographic verification.
   - Benchmark harness: timing boundaries (v1 vs. v2), GUC isolation, warm-up passes, alternating paired trials.
5. **Empirical Results & Comparative Analysis:**
   - Storage footprint compaction: coordinate, heap, TOAST elimination, auxiliary fork reconciliation.
   - Exact retrieval latency: buffer hits, mean/median speedups, halving of expression evaluations via `ORDER BY distance`.
   - Approximate HNSW retrieval: latency parity, build time observations, Recall@K overlap.
   - Query optimizer cost modeling: crossover analysis from `Seq Scan` (1k) to `Index Scan` (5k/10k).
6. **Threats to Validity & Methodological Boundaries:**
   - Timing harness variations across stages.
   - Single-run build time limitations.
   - Host caching hierarchy (buffer pool vs. OS page cache).
7. **Conclusion & Future Work:**
   - Summary of verified findings.
   - Roadmap: 100k–1M scaling under memory pressure, parallel index workers, compound quantization, downstream LLM fidelity evaluation.
8. **References & Appendix:**
   - Schema DDL, master results matrix, and verified SQL queries.

---

## 10. Review 3 Technical Viva Voce Presentation Deck (12-Slide Outline)

- **Slide 1: Title Slide & Team Identification**
  - Project Title, Course Code (BCSE302P), Guide/Department, Authors (Yash Pradhan, Anjini Pandey).
- **Slide 2: The Core DBMS Problem: The TOAST Storage Inversion**
  - Visual diagram showing 8 KiB PostgreSQL page, 2 KiB TOAST threshold, 3,076 B FP32 vector eviction, and secondary TOAST relation swelling.
- **Slide 3: Research Question & Proposed Architecture**
  - Research question; FP16 `halfvec` coordinate compression (1,544 B); vector-only representation schema; architectural database isolation across 4 environments.
- **Slide 4: Experimental Methodology & Scaling Roadmap**
  - Scaling trajectory: 180 → 1,001 → 5,000 → 10,000 chunks; MS MARCO dataset provenance; 20 evaluation queries; 100 timed trials per config with alternating paired order.
- **Slide 5: Storage Footprint Results (Chart 1)**
  - Grouped bar chart comparing Table Base Storage and HNSW Index Footprint across 1k, 5k, and 10k scales.
  - Key finding: 100% TOAST data table elimination; 55.4% inclusive relation compaction; HNSW Index Footprint is 49.90% smaller at 1,001 chunks and 49.98% smaller at 5,000 and 10,000 chunks.
- **Slide 6: Exact Retrieval Performance: Speedup & Buffer Hits (Chart 2 & Chart 4A)**
  - Exact search latency comparisons (42.3 ms vs 6.8 ms at 5k; 89.0 ms vs 12.3 ms at 10k).
  - Diagnostic Q0 buffer hits (15,032 vs 1,000 at 5k; 30,064 vs 2,000 at 10k — 93.35% fewer buffer hits; associated with observed speedups).
- **Slide 7: HNSW Approximate Retrieval & Construction Performance**
  - HNSW traversal latencies (3.3–3.8 ms for both representations; architectural context on graph traversal overhead).
  - Single-run build time observations (13.5% faster at 5k; 29.8% faster at 10k).
- **Slide 8: Retrieval Fidelity & Recall@K (Chart 3)**
  - Multi-bar chart showing Recall@3, Recall@5, and Recall@10 at 5k and 10k scales.
  - Key finding: 96.0% Recall@10 parity; $\ge 95\%$ relative index agreement (0.985 at 5k, 0.950 at 10k).
- **Slide 9: Query Optimizer Cost Modeling & Plan Crossover**
  - Explanation of PostgreSQL planner behavior: `Seq Scan` at 1,001 chunks (201 heap blocks) vs. `Index Scan` at 5,000 chunks (1,000 heap blocks) and 10,000 chunks (2,000 heap blocks).
  - Showing recorded EXPLAIN costs (Limit cost 529.00 vs 1170.57 at 5k).
- **Slide 10: Methodological Rigor & Audit Corrections**
  - Transparency slide detailing Stage 4 audit remediation: moving GUCs outside timer, using `distance` alias, fork reconciliation, and correcting "never accessed" claims.
- **Slide 11: Experimental Limitations & Future Research Roadmap**
  - Limitations: 10k scale boundary, single-run build times, single-host memory hierarchy.
  - Future work: 100k–1M+ scale under memory pressure, multi-worker indexing, compound quantization.
- **Slide 12: Conclusion & Viva Voce Defense Summary**
  - Bulleted summary of verified contributions; QR code / link to GitHub repository; opening for examiner Q&A.
