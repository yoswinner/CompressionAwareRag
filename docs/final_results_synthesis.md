# Comprehensive Cross-Stage Results Synthesis & Final Research Evidence

- **Project:** Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency Retrieval-Augmented Generation (RAG)
- **Course:** BCSE302P — Database Systems Lab, Department of Computer Science and Engineering, VIT Vellore
- **Authors:** Yash Pradhan, Anjini Pandey
- **Target Milestones:** Final Project Report & Review 3 Technical Viva Voce
- **Document Purpose:** Unified Cross-Stage Research Synthesis, Historical Methodology Reconciliation, Slide/Report Guidance, and Viva Defense Blueprint
- **Baseline Git Checkpoint:** `ac25c88` (`Add corpus scaling experiment`)
- **Status:** Verified, Fully Reconciled, and Read-Only Audited

---

## 1. Executive Abstract & Research Framing

### A. Research Motivation and Problem Statement
Retrieval-Augmented Generation (RAG) architectures frequently utilize relational database management systems (RDBMS) extended with vector capabilities, such as PostgreSQL with `pgvector`, to unify relational metadata processing with dense vector retrieval. However, standard dense embedding models generate high-dimensional vectors (e.g., 768 dimensions in `nomic-embed-text`) encoded in IEEE 754 32-bit floating-point format (`vector`).

In PostgreSQL, high-dimensional FP32 vectors create a distinct **storage-engine pathology**:
1. A 768-dimensional FP32 vector occupies **3,076 bytes** of datum storage as measured by PostgreSQL's `pg_column_size()` (with the documented pgvector storage formula defined as 3,080 bytes, retaining 3,076 bytes as an unresolved empirical measurement).
2. PostgreSQL divides storage into 8 KiB page blocks (`BLCKSZ = 8192`). When an inserted or updated tuple exceeds `TOAST_TUPLE_THRESHOLD` (normally approximately 2 KiB, where the exact value follows PostgreSQL's tuple/page-size calculation), PostgreSQL triggers TOAST processing and attempts compression or out-of-line storage until the tuple length falls below `TOAST_TUPLE_TARGET` (normally also approximately 2 KiB). These parameters are distinct: `TOAST_TUPLE_THRESHOLD` triggers TOAST processing, while `TOAST_TUPLE_TARGET` defines the target tuple length after compression or eviction attempts.
3. Because 3,076 bytes exceeds `TOAST_TUPLE_THRESHOLD` (~2 KiB) and dense float arrays do not compress effectively with byte-level LZ compressors (`pglz`/`lz4`), every uncompressed FP32 vector payload is evicted from the relational table heap into secondary **TOAST (The Oversized-Attribute Storage Technique)** tables to reach `TOAST_TUPLE_TARGET`.
4. TOAST eviction induces substantial buffer-access amplification: scanning an FP32 vector requires reading out-of-line TOAST pointer descriptors in the heap, navigating TOAST indexes, and assembling multiple 2 KiB TOAST chunk rows.
5. In Hierarchical Navigable Small World (HNSW) graph indexing, FP32 vectors double the physical index footprint relative to 16-bit representations, placing higher pressure on PostgreSQL's shared buffer pool (`shared_buffers`).

### B. The Core Research Question
*Does coordinate-level half-precision representation (`halfvec` / IEEE 754 Float16) resolve the PostgreSQL TOAST eviction pathology, reduce HNSW index footprint, and improve retrieval latency across varying corpus scales, and what are the empirical trade-offs in nearest-neighbor recall overlap (Recall@K)?*

### C. Summary of Cumulative Empirical Findings
Across four experimentally isolated stages scaling from 180 chunks up to **10,000 chunks**:
- **Storage Compaction:** FP16 reduces coordinate datum size from 3,076 bytes to 1,544 bytes, bringing the vector **below the 2 KiB TOAST threshold**. This eliminates out-of-line TOAST data tables entirely (0 bytes in TOAST table payload, with only an 8,192-byte index shell remaining), yielding a **60.11% reduction in the pre-HNSW relation total** at 1,001 chunks (Stage 2, comprising primary heap, TOAST data, TOAST index, primary-key B-tree index, and auxiliary forks) and an **inclusive relation footprint reduction of 55.35% at 5k chunks and 55.37% at 10k chunks** (Stage 4) once primary-key and HNSW indexes are included. The HNSW index footprint alone is reduced by **49.90% at 1k chunks, 49.98% at 5k chunks, and 49.98% at 10k chunks**.
- **Exact Retrieval Speedup:** Eliminating TOAST page traversal substantially reduces shared-buffer accesses during exhaustive sequential nearest-neighbor scans:
  - **At 5,000 chunks:** Mean latency drops from 42.285 ms to 6.802 ms (**6.22x speedup / 83.91% lower**); median latency drops from 44.694 ms to 6.277 ms (**7.12x speedup / 85.96% lower**). In the recorded diagnostic Query 0 (Q0) plan, shared-buffer accesses were reduced by **93.35%** (1,000 vs. 15,032 buffer hits).
  - **At 10,000 chunks:** Mean latency drops from 89.004 ms to 12.301 ms (**7.24x speedup / 86.18% lower**); median latency drops from 92.664 ms to 11.541 ms (**8.03x speedup / 87.55% lower**). In the recorded diagnostic Q0 plan, shared-buffer accesses were reduced by **93.35%** (2,000 vs. 30,064 buffer hits).
- **Index Construction Efficiency:** In single-run builds, FP16 HNSW construction took **approximately 13.5% less wall-clock time at 5k** (1.264 s vs. 1.462 s displayed, ~13.54%; 13.52% using verified full-precision inputs 1.2640003 s vs. 1.4615468 s) and **approximately 29.8% less wall-clock time at 10k** (2.641 s vs. 3.762 s displayed, ~29.80%; 29.79% using verified full-precision inputs 2.6411371 s vs. 3.7616984 s) compared to FP32.
- **Query Optimizer Plan Selection:** At 1,001 chunks (201 heap blocks for FP16), the PostgreSQL query planner selected a Sequential Scan on FP16 under default settings. At 5,000 chunks (1,000 heap blocks) and 10,000 chunks (2,000 heap blocks), the planner selected an Index Scan using the HNSW index. This confirms that the planner transitions from sequential scan to index scan between 1,001 and 5,000 chunks under this configuration.
- **Nearest-Neighbor Overlap (Recall@K):** At `ef_search = 40`, FP16 HNSW achieved **96.0% Recall@10 at 5,000 chunks** and **96.0% Recall@10 at 10,000 chunks** relative to the verified uncompressed FP32 exact nearest-neighbor reference, showing similar observed overlap to FP32 HNSW (95.5% at 5k, 94.5% at 10k).

---

## 2. Master Cross-Stage Experimental Evolution Matrix

The table below traces the experimental progression across all project milestones. **Crucially, dimensions are categorized by methodological comparability** to prevent scientific mischaracterization. All binary storage metrics are reported in mebibytes (MiB), where 1 MiB = 1,048,576 bytes, and kibibytes (KiB), where 1 KiB = 1,024 bytes.

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

## 3. Methodological Boundaries: What is and is Not Cross-Comparable

A cornerstone of sound database systems research is strictly distinguishing directly comparable empirical axes from non-comparable baseline contexts.

### A. Valid Direct Comparisons
1. **Within-Scale Paired Comparisons (Stages 2, 3, and 4):**
   - At each scale (1,001, 5,000, and 10,000 chunks), FP32 and FP16 were benchmarked over the exact same queries, identical warm-up cycles, identical connection handling, and alternating paired trial execution order. Within each scale, relative latency differences and percentage changes are strictly valid.
2. **Storage and Index Footprint Scaling:**
   - Sizing metrics (`pg_relation_size`, `pg_total_relation_size`) were collected via identical PostgreSQL 18 catalog queries across all scales. Physical sizing curves (KiB per node and TOAST allocations) are directly comparable across stages.
3. **Stage 4 5k vs. 10k Scaling Trends:**
   - Both 5k and 10k experiments share the exact same dedicated database (`compression_aware_rag_scale`), the exact same canonical cache stream (`BeIR/msmarco`), identical hyperparameter tuning (`m = 16`, `ef_construction = 64`, `ef_search = 40`), and the **exact same corrected v2 timing harness**.

### B. Inadmissible Cross-Stage Comparisons (Why Raw Latencies Must Not Be Conflated)
1. **Review 2 (180 Chunks) vs. Subsequent Stages:**
   - Review 2 used a legacy REST-paginated subset on `compression_aware_rag` storing full document text in the same table (`document_chunks`). Later stages used dedicated vector-only representation tables to isolate vector I/O from variable-length text overhead.
2. **Stage 3 (1,001 Chunks) Latency vs. Stage 4 v2 (5k & 10k Chunks) Latency:**
   - Stage 3 evaluated HNSW retrieval inside a `BEGIN; SET LOCAL ...; COMMIT;` transaction block per query trial.
   - Stage 4 v2 applies GUC configuration once before the measurement passes and times strictly client-side statement `cur.execute()` and `cur.fetchall()`.
   - Naively plotting Stage 3 1,001-chunk latency against Stage 4 5k/10k latency would compare transaction-block dispatch overhead against pure query execution, confounding corpus size with timing harness changes.
3. **Index Build Times Across Scales:**
   - Index creation was executed on single runs using `maintenance_work_mem = 64 MB`. While build durations scale monotonically (0.161 s → 1.264 s → 2.641 s for FP16), single-run wall-clock build times reflect operating system I/O scheduling and thread dispatch, representing engineering benchmarks rather than formal statistical distributions.

---

## 4. Reconciliation of Superseded Figures vs. Current Verified Results

During the iterative audit of Stage 4, several measurement anomalies and documentation discrepancies were corrected. The table below provides complete academic transparency regarding superseded numbers:

| Metric / Parameter | Superseded / Inaccurate Value | Current Verified v2 Value | Root Cause & Resolution |
|---|---|---|---|
| **5k Exact FP32 Mean Latency** | 76.765 ms | **42.285 ms** | In v1, GUC `SET` was executed per trial, and Q0 buffer hits were 30,032. In v2, GUCs were moved outside the timed block and `ORDER BY distance` alias was used, halving Q0 buffer hits to 15,032. Avoiding duplicate expression evaluation and detoasting during the scan is an interpretation consistent with this exact halving. |
| **5k Exact FP16 Mean Latency** | 11.983 ms | **6.802 ms** | In v1, two GUC queries (`SET enable_indexscan`, `SET enable_bitmapscan`) were dispatched inside the timer. Moving GUCs out eliminated ~5.18 ms of client/server round-trips. |
| **5k Forced HNSW FP32 Mean** | 4.775 ms | **4.129 ms** | v1 included per-trial GUC configuration overhead. v2 isolates query execution. |
| **5k Forced HNSW FP16 Mean** | 4.644 ms | **3.372 ms** | v1 included per-trial GUC configuration overhead. v2 isolates query execution. |
| **10k Exact FP32 Mean Latency**| 157.901 ms | **89.004 ms** | Moving GUCs outside timer and ordering by `distance` alias reduced Q0 buffer hits from 60,064 to 30,064 (consistent with avoiding duplicate detoasting). |
| **10k Exact FP16 Mean Latency**| 21.321 ms | **12.301 ms** | Corrected per-trial GUC dispatch overhead. |
| **10k Forced HNSW FP32 Mean**| 5.305 ms | **3.674 ms** | Corrected per-trial GUC dispatch overhead. |
| **10k Forced HNSW FP16 Mean**| 5.484 ms | **3.856 ms** | Corrected per-trial GUC dispatch overhead. |
| **Stage 4 Chunking Configuration** | `chunk_size=1000, overlap=200` | **`chunk_size=300, overlap=50`** | Corrected typographical progress summary claim. Cryptographic manifest and TSV cache confirmed 300/50 throughout. |
| **Planner Cost Estimation** | Unsupported synthetic cost formulas | **Direct EXPLAIN plan costs** | Removed unsupported analytical cost models from documentation. Plans report actual engine costs (e.g., cost=202.55, cost=511.82). |
| **1,001-Chunk Scaling Latency** | Interpolated latency from 5k results | **No interpolation permitted** | Preserved historic Stage 3 measurements; clearly documented why cross-stage timing boundaries differ. |
| **Production DB Access Claims**| Claimed "never accessed" | **Zero modifications; read-only verification queries only** | Updated Section 4 to accurately reflect that read-only metadata checks verified baseline counts. |

---

## 5. Detailed Technical Synthesis by Architectural Dimension

### A. Coordinate-Level Numerical Compression (`vector` vs. `halfvec`)
- **Mathematical Specification:** Full-precision `vector(768)` allocates 4 bytes per coordinate (IEEE 754 single precision: 1 sign bit, 8 exponent bits, 23 mantissa bits). Half-precision `halfvec(768)` allocates 2 bytes per coordinate (IEEE 754 half precision: 1 sign bit, 5 exponent bits, 10 mantissa bits).
- **Datum Size Measurement and Formula Comparison:**
  - **Empirical Measurement:** Catalog queries using `pg_column_size(embedding)` across Stage 2 and Stage 4 recorded **3,076 bytes** for `vector(768)` and **1,544 bytes** for `halfvec(768)`. Based on these recorded empirical measurements, the coordinate datum reduction is **49.80% (1.992x)**.
  - **Documented Storage Formula:** The documented pgvector 0.8.4 formula is defined as 4 × dimensions + 8 bytes header = 3,080 bytes for `vector(768)`, and 2 × dimensions + 8 bytes header = 1,544 bytes for `halfvec(768)`. Under this theoretical formula, the coordinate reduction is **49.87% (1.995x)**.
  - **Unresolved Measurement/Formula Discrepancy:** The 4-byte difference between the 3,076-byte empirical result and the 3,080-byte documented formula is an unresolved measurement/formula discrepancy that cannot be resolved from existing benchmark artifacts without live database instrumentation. Neither figure should be altered silently, nor should an unverified internal storage cause be assigned without direct evidence. Both the measured reduction (49.80%) and theoretical reduction (49.87%) are retained as clearly separated figures.
- **Precision Trade-Off:** Reducing mantissa bits from 23 to 10 limits coordinate dynamic range and precision (~3.3 decimal digits). However, high-dimensional vector spaces exhibit significant dimensional redundancy. Cosine similarity calculates dot products normalized by L2-norm; coordinate truncation differences across 768 dimensions resulted in similar observed overlap, with Recall@10 remaining ≥ 96.0% at scale.

### B. PostgreSQL Storage Engine: The 2 KiB TOAST Threshold and Buffer Cache
- **The Relational Storage Inversion:** PostgreSQL divides heap files into 8 KiB blocks. To keep tuples within individual pages, PostgreSQL checks `TOAST_TUPLE_THRESHOLD` (normally approximately 2 KiB, where the exact value follows PostgreSQL's tuple/page-size calculation) to trigger TOAST processing when a tuple is sufficiently large, and then attempts compression or out-of-line storage to reduce the tuple below `TOAST_TUPLE_TARGET` (also normally approximately 2 KiB). These parameters are distinct and not interchangeable: `TOAST_TUPLE_THRESHOLD` determines whether TOAST processing is initiated, whereas `TOAST_TUPLE_TARGET` sets the target tuple length after compression or out-of-line storage attempts.
- **The FP32 Pathology:** At 3,076 bytes (or 3,080 bytes theoretically), every FP32 vector exceeds `TOAST_TUPLE_THRESHOLD` (~2 KiB) and is uncompressible with byte-level LZ compressors. PostgreSQL moves the vector payload out-of-line to secondary `pg_toast` tables to reduce the tuple below `TOAST_TUPLE_TARGET`, storing an out-of-line TOAST pointer in the heap tuple. For 10,000 chunks, TOAST data tables occupy **39.06 MiB (40,960,000 B)** across exactly **5,000 TOAST blocks** (40,960,000 / 8,192 = 5,000).
- **The FP16 Solution:** At 1,544 bytes, `halfvec` falls safely below `TOAST_TUPLE_THRESHOLD` (~2 KiB). In our evaluated representation schema, FP16 vectors remained inline within standard heap pages and the TOAST data-table payload measured zero bytes across all tested scales, with observed Stage 4 block counts establishing an average density of **approximately 5 rows per 8 KiB block** (1,000 blocks for 5,000 rows; 2,000 blocks for 10,000 rows). Only an empty 8 KiB (8,192-byte) initial TOAST index shell remains in the catalog.
- **Observed Shared-Buffer Accesses:** In the single recorded diagnostic Q0 plan, scanning FP16 required accessing 1,000 heap blocks at 5k (vs. 15,032 buffer accesses for FP32) and 2,000 heap blocks at 10k (vs. 30,064 buffer accesses for FP32), achieving a **93.35% reduction in shared-buffer accesses**. In the recorded Q0 diagnostic `EXPLAIN (ANALYZE, BUFFERS)` plans, `shared read = 0` was observed, indicating that pages for those diagnostic query executions were served from PostgreSQL's shared buffer pool in memory. This observation is specific to the diagnostic plans and should not be generalized to claim that physical disk reads were absent across every timed benchmark trial or workload execution. The reduction in shared-buffer accesses in the diagnostic plans is consistent with the observed 6.2x–8.0x exact retrieval speedups.

### C. HNSW Graph Index Footprint and Construction Dynamics
- **Graph Node Topology:** In pgvector, an HNSW index allocates an entry node, multilayer graph neighbor lists (maximum m = 16 connections per node, 2m = 32 on Layer 0), and stores the vector coordinate payload within index pages.
- **Observed Footprint Scaling Across Tested Sizes:**
  - FP32 HNSW index size: 3.90 MiB at 1k → 19.48 MiB at 5k → 38.89 MiB at 10k (~3.9 KiB per node).
  - FP16 HNSW index size: 1.95 MiB at 1k → 9.74 MiB at 5k → 19.45 MiB at 10k (~1.95 KiB per node).
  - HNSW index footprint reduction: **49.90% at 1k, 49.98% at 5k, and 49.98% at 10k**.
- **Construction Wall-Clock Durations:**
  - 1,001 Chunks: 0.233 s (FP32) vs. 0.161 s (FP16) (31.19% faster calculated from recorded values 0.2334 s vs. 0.1606 s; ~30.90% based on displayed 3-decimal values; raw full-precision floats were not preserved in the Stage 3 JSON artifact).
  - 5,000 Chunks: 1.462 s (FP32) vs. 1.264 s (FP16) (13.52% faster calculated from verified full-precision JSON inputs 1.4615468 s vs. 1.2640003 s; ~13.54% based on displayed 3-decimal values; 13.51% if calculated from 4-decimal rounded values 1.4615 s vs. 1.2640 s).
  - 10,000 Chunks: 3.762 s (FP32) vs. 2.641 s (FP16) (29.79% faster calculated from verified full-precision JSON inputs 3.7616984 s vs. 2.6411371 s; ~29.80% based on displayed 3-decimal values; 29.79% from 4-decimal rounded values 3.7617 s vs. 2.6411 s).
  - *Interpretation:* The physical index written to disk is 49.98% smaller, meaning pgvector formats and writes approximately half as many 8 KiB index pages. While fewer index pages are a plausible contributor to lower build times, WAL volume, dirty page write rates, and CPU cycles were not directly instrumented, and these single-run timings represent indicative engineering observations rather than statistically bounded distributions.

### D. Exact vs. Approximate Retrieval Latency Across Scales
- **Exact Search Scaling:**
  - Exact search involves an exhaustive sequential scan that evaluates distance against every row in the table.
  - At 5,000 chunks: Mean latency was 42.285 ms (FP32) vs. 6.802 ms (FP16) (**6.22x speedup / 83.91% lower**); median latency was 44.694 ms (FP32) vs. 6.277 ms (FP16) (**7.12x speedup / 85.96% lower**).
  - At 10,000 chunks: Mean latency was 89.004 ms (FP32) vs. 12.301 ms (FP16) (**7.24x speedup / 86.18% lower**); median latency was 92.664 ms (FP32) vs. 11.541 ms (FP16) (**8.03x speedup / 87.55% lower**).
- **HNSW Graph Traversal Latency:**
  - Across the methodologically equivalent Stage 4 scales (5k and 10k under the v2 harness), client-observed retrieval latency (measured strictly around statement `execute` and `fetchall`, with GUC configuration excluded) for forced HNSW traversal showed similar medians:
    - At 5,000 chunks: Mean latency was 4.129 ms (FP32) vs. 3.372 ms (FP16) (**18.32% lower**); median latency was 3.371 ms (FP32) vs. 2.923 ms (FP16) (**13.28% lower**).
    - At 10,000 chunks: Mean latency was 3.674 ms (FP32) vs. 3.856 ms (FP16) (**4.95% difference**); median latency was 3.415 ms (FP32) vs. 3.820 ms (FP16) (**both medians within ~0.4 ms**).
  - *Interpretation of Similar HNSW Latencies:* In HNSW, the approximate search evaluates a bounded candidate set controlled by `ef_search = 40`. In the recorded Q0 diagnostic plans, both representations accessed a small number of buffer pages in memory (830 vs. 753 at 5k; 930 vs. 847 at 10k). While CPU distance evaluations and priority queue operations are plausible primary contributors to elapsed time once buffer I/O is small and uniform, CPU cycles and queue operations were not directly instrumented; the similar client-observed traversal times are reported as an empirical finding across these configurations.

### E. Retrieval Fidelity & Nearest-Neighbor Overlap (Recall@K)
- **Ground-Truth Reference:** All recall calculations evaluate retrieved top-K chunk IDs against an independent, exhaustive uncompressed FP32 sequential scan on the exact same database.
- **Scope Limitation:** Recall@K strictly measures mathematical overlap against the uncompressed exact nearest-neighbor reference. It does not measure semantic relevance, downstream LLM generation quality, answer correctness, or end-to-end RAG task performance, which were not evaluated in this benchmark.
- **FP16 Exhaustive Search Overlap:**
  - 5,000 Chunks: Recall@3 = 100.0%, Recall@5 = 99.0%, Recall@10 = 99.5%.
  - 10,000 Chunks: Recall@3 = 100.0%, Recall@5 = 99.0%, Recall@10 = 100.0%.
  - Half-precision coordinate quantization introduces minimal ranking deviation relative to the FP32 exact reference.
- **HNSW Approximate Retrieval Overlap (`ef_search = 40`):**
  - At 5,000 Chunks: FP32 HNSW Recall@10 = 95.5% | FP16 HNSW Recall@10 = **96.0%**.
  - At 10,000 Chunks: FP32 HNSW Recall@10 = 94.5% | FP16 HNSW Recall@10 = **96.0%**.
  - Relative Index Agreement (overlap between FP32 and FP16 HNSW result sets): 98.5% at 5k, 95.0% at 10k.
  - **Verdict:** FP16 HNSW exhibits similar observed overlap compared to FP32 HNSW under the evaluated benchmark queries.

### F. Query Optimizer Cost Modeling & Plan Selection
- **Observed Plan Selection Across Scales:**
  - **At 1,001 Chunks (Stage 3):** Under default planner cost settings (`enable_seqscan = on`), the PostgreSQL query optimizer selected `Index Scan using chunks_embedding_fp32_hnsw_idx` for FP32, but selected `Seq Scan on chunks_embedding_halfvec` for FP16.
  - **At 5,000 Chunks (Stage 4 v2):** The planner selected `Index Scan using chunks_5k_fp32_hnsw_idx` for FP32 (Limit cost: 35.82..56.17; forced seq scan cost: 202.55..202.57) and selected **`Index Scan using chunks_5k_halfvec_hnsw_idx`** for FP16 (Limit cost: 511.82..529.00; forced seq scan cost: 1170.55..1170.57; Q0 buffer hits: 753).
  - **At 10,000 Chunks (Stage 4 v2):** The planner maintained `Index Scan using chunks_10k_fp32_hnsw_idx` for FP32 (Limit cost: 68.10..88.43; forced seq scan cost: 405.10..405.12) and maintained **`Index Scan using chunks_10k_halfvec_hnsw_idx`** for FP16 (Limit cost: 548.68..566.31; forced seq scan cost: 2341.10..2341.12; Q0 buffer hits: 847).
- **Separation of Observations from Hypotheses on Plan Selection:**
  - *Planner Cost Mechanics:* PostgreSQL calculates sequential-scan cost from main heap relation pages (`relpages`) and tuple processing (`reltuples`). Crucially, PostgreSQL's planner generally does not directly account for fetching out-of-line TOAST values during sequential-scan cost estimation, and it does not count secondary TOAST-table blocks as sequential-scan pages. Therefore, the planner's plan selection cannot be explained by claiming the planner directly counted TOAST blocks.
  - *Hypothesis on the 1,001-Chunk Plan Selection:* In PostgreSQL, the planner compares the estimated startup and total cost of traversing an index against sequentially scanning the table heap and sorting candidates. For FP16 at 1,001 chunks (201 heap blocks), the optimizer estimated that sequential table scanning had a lower cost than HNSW index traversal, leading to the selection of `Seq Scan`. For FP32 at 1,001 chunks, the planner selected the HNSW index scan. Because internal costing details for the unforced Stage 3 plans were not instrumented beyond the chosen scan type, the precise internal cost differential that tipped FP32 to HNSW remains a hypothesis rather than an established measurement. Note that buffer-hit metrics (such as the 3,010 Q0 shared buffer hits during execution) represent runtime buffer accesses, not physical heap block counts or planner `relpages`.
  - *The Plan Transition at Scale:* As the FP16 table grew from 201 heap blocks (at 1,001 chunks) to 1,000 heap blocks (at 5k) and 2,000 heap blocks (at 10k), sequential scan base cost scaled proportionally (cost=1062.50 at 5k; cost=2125.00 at 10k). The recorded EXPLAIN costs confirm that at 5k and 10k, the total estimated cost for the HNSW index scan with Limit (529.00 at 5k; 566.31 at 10k) was substantially lower than for the sequential scan and sort path (1170.57 at 5k; 2341.12 at 10k), causing the optimizer to definitively select the HNSW index scan.

---

## 6. Recommendations for Final Report & Review 3 Presentation Slides

### A. Core Findings to Feature in the Final Report & Slides
1. **Differentiated Storage Reductions:**
   - Feature the exact 49.98% HNSW index reduction and distinguish the 60.11% reduction in the pre-HNSW relation total (Stage 2, including the primary key index) from the ~55.35%–55.37% inclusive relation footprint reduction (Stage 4). Explain that total relation reduction exceeds 50% due to eliminating out-of-line TOAST data tables.
2. **Differentiated Exact Search Speedups:**
   - Highlight exact sequential search speedups with clear separation of metrics:
     - 5k Chunks: Mean speedup 6.22x (42.29 ms vs. 6.80 ms); Median speedup 7.12x (44.69 ms vs. 6.28 ms).
     - 10k Chunks: Mean speedup 7.24x (89.00 ms vs. 12.30 ms); Median speedup 8.03x (92.66 ms vs. 11.54 ms).
   - Explain the database engine mechanics: 93.35% fewer shared-buffer accesses in Q0 plans because vectors reside inline within heap pages.
3. **Retrieval Overlap Parity at 96% Recall@10:**
   - Report that FP16 HNSW achieved 96.0% Recall@10 at both 5k and 10k scales relative to the verified uncompressed FP32 exact reference, showing similar observed overlap to FP32 HNSW (95.5% at 5k, 94.5% at 10k).
4. **Planner Plan Transition:**
   - Explain why PostgreSQL preferred `Seq Scan` at 1,001 chunks (201 heap blocks) and transitioned to `Index Scan` at 5,000 chunks (1,000 heap blocks) and 10,000 chunks (2,000 heap blocks).

### B. High-Impact Charts to Generate for Review 3
1. **Chart 1 (Grouped Bar Chart):** *Physical Storage Footprint Across Scales (1k, 5k, 10k Chunks)*
   - Series: FP32 Table Base Storage (Heap + TOAST + PK Index, MiB), FP16 Table Base Storage (Heap + TOAST + PK Index, MiB), FP32 HNSW Index (MiB), FP16 HNSW Index (MiB).
   - *Note:* HNSW index size is plotted separately from table base storage to avoid double-counting index bytes. Shows observed scaling trends across the tested corpus sizes.
2. **Chart 2 (Diagnostic Scatter/Bar Plot):** *Diagnostic Query 0 Shared-Buffer Accesses vs. Query 0 Server Execution Time*
   - X-axis: Q0 Shared-Buffer Accesses (1,000 vs. 15,032 at 5k; 2,000 vs. 30,064 at 10k).
   - Y-axis: Q0 Server Execution Time (ms, from EXPLAIN ANALYZE: 3.00 ms vs. 24.54 ms at 5k; 7.81 ms vs. 54.49 ms at 10k).
   - *Note:* Both axes share the exact same single-query diagnostic plan basis to illustrate the relationship between buffer accesses and server execution time.
3. **Chart 3 (Multi-Bar Chart):** *ANN Recall@K Overlap Comparison at Scale (5k and 10k)*
   - Groupings: Recall@3, Recall@5, Recall@10 for FP32 HNSW vs. FP16 HNSW.
   - Visualizes similar observed nearest-neighbor overlap at ≥ 95%.
4. **Chart 4 (Line Chart):** *Retrieval Latency Trends Across Tested Sizes (Exact Scan vs. Forced HNSW)*
   - X-axis: Corpus Scale (5k, 10k chunks under Stage 4 v2).
   - Y-axis: Client-Observed Retrieval Latency (ms).
   - Shows observed growth in exact sequential search latency vs. flat trajectory for forced HNSW traversal across the tested sizes.

### C. Claims to Strictly Exclude or Qualify
- ❌ **Do NOT claim:** *"FP16 makes HNSW graph traversal 10x faster."* (Empirical traversal times are 3.3–3.8 ms for both; the 6.2x–8.0x speedups apply strictly to exact exhaustive search).
- ❌ **Do NOT claim:** *"The database shrank overall."* (The parent `document_chunks` table retains raw text and FP32 source embeddings; savings apply to the vector representation tables and indexes).
- ❌ **Do NOT claim:** *"HNSW index build duration was proven with statistical significance or measured WAL reduction."* (Index build times represent single-run wall-clock measurements; WAL volume was not measured).
- ❌ **Do NOT claim:** *"Production databases were never accessed."* (State accurately: *zero modifications were made; read-only metadata verification checks only*).
- ❌ **Do NOT claim:** *"Recall@K proves semantic answer quality or RAG accuracy."* (State accurately: *Recall@K measures overlap against the verified FP32 exact nearest-neighbor reference*).
- ❌ **Do NOT claim:** *"Results prove O(log N) asymptotic complexity for HNSW."* (Report observed latency trends across tested sizes without asserting theoretical asymptotic proofs).
- ❌ **Do NOT present:** Superseded v1 latency numbers (e.g., 76.76 ms for 5k FP32) or interpolated 1,001-chunk latency figures.

---

## 7. Viva Voce Defense Blueprint: 10 Technical Questions & Answers

### Q1: Why did you focus on representation compression (FP16) instead of product quantization (PQ) or scalar quantization (SQ)?
**Examiner-Ready Answer:**
> "Product Quantization (PQ) and Scalar Quantization (SQ) are lossy compression techniques typically implemented in standalone vector libraries (like FAISS) that require codebook training and separate vector reconstruction steps. In contrast, FP16 (`halfvec`) is natively supported in PostgreSQL 18 with pgvector 0.8.4. It operates at the physical relational storage layer without requiring offline training, supports standard transactional SQL updates, and directly avoids PostgreSQL's ~2 KiB TOAST eviction threshold (`TOAST_TUPLE_THRESHOLD`) while maintaining 96% Recall@10 overlap."

### Q2: Why does FP16 reduce total table storage by 55.4% at scale, which is greater than the theoretical 50% coordinate reduction?
**Examiner-Ready Answer:**
> "A 768-dimensional FP32 vector occupies 3,076 bytes (measured via `pg_column_size`), which exceeds PostgreSQL's `TOAST_TUPLE_THRESHOLD` (normally ~2 KiB). This triggers TOAST processing, evicting the uncompressible vector out-of-line to a secondary TOAST relation to satisfy `TOAST_TUPLE_TARGET` (also normally ~2 KiB), storing an out-of-line TOAST pointer in the heap and incurring TOAST tuple headers, 2 KiB chunking overhead, and a B-tree TOAST index. Both parameters are distinct: `TOAST_TUPLE_THRESHOLD` triggers processing for sufficiently large tuples, while `TOAST_TUPLE_TARGET` defines the target tuple length after eviction. In contrast, FP16 vectors occupy 1,544 bytes, falling safely below `TOAST_TUPLE_THRESHOLD`; in our evaluated representation schema, FP16 vectors remained inline within standard 8 KiB heap pages, and the TOAST data-table payload measured zero bytes. Eliminating the out-of-line TOAST data table eliminates this structural relational overhead, expanding the storage savings from the 50.0% theoretical coordinate ratio to 55.35%–55.37% inclusive relation compaction at scale."

### Q3: What explains the discrepancy between the sum of heap, TOAST, and index sizes and `pg_total_relation_size`?
**Examiner-Ready Answer:**
> "`pg_relation_size(oid, 'main')` reports strictly the main data fork of a table or index. `pg_total_relation_size(oid)` additionally aggregates auxiliary relation forks, specifically the Free Space Map (`_fsm`) and Visibility Map (`_vm`) across the table heap, its TOAST table, and all attached indexes. In our Stage 4 audit, these auxiliary forks accounted for exactly 64 KiB (5k FP32), 32 KiB (5k FP16), 72 KiB (10k FP32), and 32 KiB (10k FP16). Accounting for these forks reconciled the components to the byte with zero double-counting."

### Q4: In Stage 3, the optimizer preferred Sequential Scan for FP16 at 1,001 chunks. Why did that happen, and why did it change at 5,000 chunks?
**Examiner-Ready Answer:**
> "Under default planner settings at 1,001 chunks, the optimizer selected `Index Scan` for FP32 but `Seq Scan` for FP16. In PostgreSQL, sequential-scan cost is estimated from the table's main heap pages (`relpages`) and tuple processing costs; the planner generally does not directly account for fetching out-of-line TOAST values, so plan choice was not caused by the planner counting TOAST blocks. For FP16 at 1,001 chunks, the planner estimated sequentially scanning the 201 heap blocks to be lower cost than HNSW index traversal. At 5,000 and 10,000 chunks, the FP16 heap grew to 1,000 and 2,000 heap blocks, respectively. The recorded EXPLAIN plans confirm that the estimated sequential-scan-plus-sort cost rose to 1,170.57 (at 5k) and 2,341.12 (at 10k), well above the estimated HNSW index scan costs of 529.00 and 566.31. Consequently, the optimizer transitioned to selecting the HNSW index scan for FP16 at scale."

### Q5: Why was FP16 exact search 6.2x–8.0x faster than FP32 exact search, while forced HNSW showed similar traversal times?
**Examiner-Ready Answer:**
> "Exact search is an exhaustive sequential scan that evaluates every row. For FP32, reading 5,000 or 10,000 rows required accessing 15,032 and 30,064 shared-buffer pages in Q0 diagnostic plans to assemble out-of-line TOAST chunks. For FP16, inlined heap storage required only 1,000 and 2,000 shared-buffer accesses—a 93.35% reduction in buffer pool accesses in diagnostic plans. In HNSW, however, the search is an approximate graph traversal that visits only a small candidate set (~750–930 buffer hits in Q0 diagnostic plans for both representations). Because HNSW accesses far fewer pages in memory, the buffer-access differential is much smaller; while CPU distance evaluations and priority queue operations are plausible contributors to the remaining time, they were not directly profiled, and the similar client-observed traversal times are reported as an empirical finding."

### Q6: How was ground truth established for Recall@K, and why is it labeled 'ANN Recall' rather than 'Accuracy'?
**Examiner-Ready Answer:**
> "Ground truth was computed independently at each scale using exhaustive, sequential scans on the uncompressed FP32 representation (`chunks_..._fp32`) with index scans disabled. We label this 'ANN Recall' or 'Nearest-Neighbor Overlap' because it mathematically quantifies the overlap between the candidate retrieved set and the exact mathematical nearest neighbors. It should not be termed 'semantic accuracy' because our benchmark queries do not have human relevance labels."

### Q7: Why did you modify the exact search SQL query to use `ORDER BY distance ASC` in Stage 4 v2?
**Examiner-Ready Answer:**
> "In Stage 4 v1, the query repeated the distance expression in both the SELECT list and the ORDER BY clause (`ORDER BY embedding <=> %s::{cast_type}`), logging 30,032 buffer hits in the 5k Q0 plan. In v2, ordering by the computed alias `distance` reduced Q0 buffer hits by exactly half to 15,032. While internal engine detoasting calls were not directly traced in C, avoiding duplicate expression evaluation and detoasting during the scan is an interpretation consistent with this observed halving of Q0 buffer accesses, isolating the intended query execution cost."

### Q8: Why did you retain the direct distance expression in the ORDER BY clause for HNSW queries instead of using the alias?
**Examiner-Ready Answer:**
> "PostgreSQL's query planner matches queries to HNSW indexes using operator class expressions. The HNSW index is defined on `embedding vector_cosine_ops`. Specifying the direct indexed operator expression (`embedding <=> %s`) in the ORDER BY clause makes the HNSW index path eligible for selection by the planner, whereas using an alias can prevent index path matching and fall back to sequential scan. Actual utilization of the index was directly confirmed by the recorded Index Scan in the EXPLAIN execution plans."

### Q9: What database safety mechanisms did you implement to protect production data during experimentation?
**Examiner-Ready Answer:**
> "We implemented architectural database isolation across four distinct databases: `compression_aware_rag` (production baseline), `smart_document_chatbot` (application database), `compression_aware_rag_exp` (Stage 1–3 experimental baseline), and `compression_aware_rag_scale` (Stage 4 scaling target). Dedicated programmatic guards (`assert_scale_mutation_db`) verified `current_database()` and blocked operations on any non-target database. Furthermore, vector reuse queries against the baseline database entered `read_only = True` before beginning transactions."

### Q10: What are the primary experimental limitations of this study?
**Examiner-Ready Answer:**
> "First, the maximum scale evaluated was 10,000 chunks; industrial RAG deployments often scale to millions of vectors where physical RAM exhaustion occurs. Second, index construction was evaluated on single wall-clock runs bounded by `maintenance_work_mem = 64 MB` rather than repeated SLA benchmarks. Third, shared buffer caching was maintained on a single host; distributed sharding and multi-tenant cache eviction under concurrent query loads were not evaluated. Fourth, embedding models were fixed to 768-dimensional `nomic-embed-text`; higher dimensionalities (e.g., 1536 or 3072) may exhibit different compression characteristics."

---

## 8. Experimental Limitations & Future Research Roadmap

1. **Corpus Scale Boundaries & Cache Hierarchy:**
   - 10,000 chunks (~5,457 documents) successfully demonstrated the optimizer plan transition and linear storage scaling. However, 10,000 768-dim FP16 vectors occupy only 19.45 MiB of index space, which fits comfortably within the 128 MB `shared_buffers`. In PostgreSQL, if an index exceeds `shared_buffers`, the operating system filesystem page cache in physical RAM continues to buffer pages; exceeding `shared_buffers` alone does not imply physical RAM exhaustion or guaranteed disk paging. Testing larger corpora and working sets (e.g., 100,000 to 1,000,000+ chunks) under explicit memory pressure, with physical disk I/O directly measured via OS and storage counters, is a valuable prospective avenue for future research (noting that exceeding host RAM capacity induces memory pressure and cache eviction rather than automatically guaranteeing disk paging under every OS configuration).
2. **Single-Worker Index Build Instrumentation:**
   - Index creation utilized a single worker thread bounded by `maintenance_work_mem = 64 MB`. Parallel index build worker scaling (`max_parallel_maintenance_workers`), WAL volume generation, and peak resident set size (RSS) tracking should be instrumented in future work.
3. **Compound Quantization Pipelines:**
   - Future research should explore combining coordinate compression with graph pruning: combining `halfvec` with scalar quantization or matryoshka representation learning (e.g., truncating 768d embeddings to 256d or 512d before FP16 indexing).
4. **End-to-End RAG Evaluation:**
   - Integrating the retrieval harness with a downstream Large Language Model (e.g., Llama-3 or Mistral) to evaluate generation fidelity metrics (Faithfulness, Answer Relevance via RAGAS) when retrieved contexts originate from FP16 vs. FP32 indexes.
