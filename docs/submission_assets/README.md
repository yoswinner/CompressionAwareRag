# Submission Assets Data Dictionary & Benchmark Methodology Guide

This directory contains verified, disaggregated CSV datasets supporting the charts, academic report, and Review 3 technical defense for the research project:

**Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency Retrieval-Augmented Generation (RAG)**.

All metrics are directly derived from PostgreSQL 18 catalog queries and benchmark execution artifacts without synthetic interpolation or extrapolation.

---

## Table of Contents
- [Quick Reference Asset Index](#quick-reference-asset-index)
- [1. `chart1_storage.csv` — Physical Storage Footprint Scaling](#1-chart1_storagecsv--physical-storage-footprint-scaling)
- [2. `chart2_q0_diagnostics.csv` — Single-Query Execution Plan Diagnostics](#2-chart2_q0_diagnosticscsv--single-query-execution-plan-diagnostics)
- [3. Retrieval Fidelity & Neighborhood Agreement Assets](#3-retrieval-fidelity--neighborhood-agreement-assets)
  - [3.A. `chart3_hnsw_recall.csv` — Nearest-Neighbor Overlap (ANN Recall@K)](#3a-chart3_hnsw_recallcsv--nearest-neighbor-overlap-ann-recallk)
  - [3.B. `chart3_pairwise_agreement.csv` — Pairwise Top-K Neighborhood Agreement](#3b-chart3_pairwise_agreementcsv--pairwise-top-k-neighborhood-agreement)
- [4. `chart4_latency.csv` — Client-Observed Retrieval Latency Distributions](#4-chart4_latencycsv--client-observed-retrieval-latency-distributions)
- [Important Methodological Boundaries & Negative Claims](#important-methodological-boundaries--negative-claims)

---

## Quick Reference Asset Index

| Asset File | Target Figure | Primary Metrics | Tested Scales | Key DBMS Dimension |
|---|---|---|:---:|---|
| [`chart1_storage.csv`](chart1_storage.csv) | Chart 1 | Heap, TOAST table, TOAST index, HNSW index, total relation | 1k, 5k, 10k | Physical disk footprint & TOAST elimination |
| [`chart2_q0_diagnostics.csv`](chart2_q0_diagnostics.csv) | Chart 2 | Shared-buffer hits, server execution time, limit actual time | 5k, 10k | Memory buffer access in single-query plans |
| [`chart3_hnsw_recall.csv`](chart3_hnsw_recall.csv) | Chart 3A | Recall@3, Recall@5, Recall@10 vs. uncompressed FP32 exact ref | 5k, 10k | Mathematical overlap against uncompressed ground truth |
| [`chart3_pairwise_agreement.csv`](chart3_pairwise_agreement.csv) | Chart 3B | Agreement@3, Agreement@5, Agreement@10 (FP32 HNSW vs FP16 HNSW) | 5k, 10k | Cross-representation neighborhood consistency |
| [`chart4_latency.csv`](chart4_latency.csv) | Chart 4 | Mean, median, IQR, min, max, stdev, speedup factor | 5k, 10k | 100-trial client latency distributions |

---

## 1. `chart1_storage.csv` — Physical Storage Footprint Scaling

### Purpose
Documents physical on-disk storage scaling across 1,001, 5,000, and 10,000 indexed chunks under standard 8 KiB page blocks (`BLCKSZ = 8192`).

### Key DBMS Invariant & Storage Scopes
PostgreSQL storage must never be treated as a single undifferentiated number. This asset strictly disaggregates base table storage from index storage to prevent double-counting:
- **Pre-HNSW Base Relation Total (`pg_total_relation_size`):** Captures main table heap + TOAST data table + TOAST index + primary key B-tree index + auxiliary relation forks (Free Space Map `_fsm` and Visibility Map `_vm`).
- **HNSW Vector Index Footprint (`pg_relation_size`):** Captures only the main data fork of the HNSW vector index.
- **Inclusive Total Relation Storage (`pg_total_relation_size`):** Captures the entire table inclusive of all indexes, auxiliary forks, and TOAST storage.

> [!WARNING]
> **Do NOT interchange storage scopes.** At 1,001 chunks, inclusive post-index total was not recorded as a single catalog query in Stage 3; base relation and HNSW index sizes were recorded independently. At 5k and 10k, base table storage is derived by subtracting the HNSW index footprint from the total inclusive relation size.

### Schema & Columns
| Column Name | Type | Description & Unit |
|---|---|---|
| `corpus_scale_chunks` | Integer | Evaluated chunk count (`1001`, `5000`, `10000`). |
| `representation` | String | Coordinate precision: `FP32 (vector)` or `FP16 (halfvec)`. |
| `table_base_storage_bytes` | Integer | Table base storage in bytes (heap + TOAST table + TOAST idx + PK idx + aux forks). |
| `table_base_storage_mib` | Float | Table base storage in mebibytes (`bytes / 1,048,576`). |
| `hnsw_index_bytes` | Integer | Main data fork of the HNSW vector index in bytes (`pg_relation_size`). |
| `hnsw_index_mib` | Float | HNSW index footprint in mebibytes (`bytes / 1,048,576`). |
| `total_inclusive_bytes` | Integer / Empty | Full inclusive relation size in bytes (`pg_total_relation_size`). Empty at 1k. |
| `total_inclusive_mib` | Float / Empty | Full inclusive relation size in mebibytes. Empty at 1k. |
| `notes` | String | Provenance, catalog queries, and component-level breakdown. |

### Valid Comparisons & Findings
- **TOAST Data Elimination:** FP16 eliminates 100% of the out-of-line TOAST data table payload across all tested scales in the evaluated schema (0 bytes vs. 4.10 MiB at 1k, 20.48 MiB at 5k, and 40.96 MiB at 10k).
- **Inclusive Compaction:** FP16 achieves a **55.35% inclusive relation reduction** at 5k (39.68 MiB $\to$ 17.72 MiB) and **55.37% reduction** at 10k (79.20 MiB $\to$ 35.35 MiB), exceeding the theoretical 50.0% coordinate reduction due to elimination of the out-of-line TOAST data payload.
- **HNSW Index Footprint:** FP16 HNSW index footprint is **49.90% smaller at 1k** (3.90 MiB $\to$ 1.95 MiB) and **49.98% smaller at 5k and 10k** (19.48 MiB $\to$ 9.74 MiB at 5k; 38.89 MiB $\to$ 19.45 MiB at 10k).

---

## 2. `chart2_q0_diagnostics.csv` — Single-Query Execution Plan Diagnostics

### Purpose
Documents server-side execution diagnostics for single-query Query 0 (Q0) runs captured via PostgreSQL's `EXPLAIN (ANALYZE, BUFFERS)` engine.

### Key DBMS Invariant & Interpretation
- **Shared-Buffer Accesses:** Shared-buffer hits in `EXPLAIN ANALYZE` represent **runtime page accesses** (including repeated accesses to the same page block), not unique relation block counts.
- **Diagnostic vs. Repeated-Trial Latency:** Single-query Q0 diagnostics provide isolated query-plan verification under controlled cache conditions. The observed 93.35% reduction in shared-buffer accesses is **consistent with** the observed 6.2x–8.0x exact search speedups. However, because Q0 diagnostics represent single-query samples while latency statistics are derived from 100 timed trials, the Q0 observation is consistent with the latency trend but does not establish the causal mechanism of every measured latency improvement.

### Schema & Columns
| Column Name | Type | Description & Unit |
|---|---|---|
| `corpus_scale_chunks` | Integer | Evaluated chunk count (`5000`, `10000`). |
| `retrieval_mode` | String | Retrieval mode: `Exact Search` (sequential scan) or `Forced HNSW` (index scan). |
| `representation` | String | Coordinate precision: `FP32 (vector)` or `FP16 (halfvec)`. |
| `q0_shared_buffer_hits` | Integer | Total shared-buffer hits reported in the top-level plan. |
| `q0_server_execution_time_ms` | Float | Server execution time reported by PostgreSQL in milliseconds. |
| `q0_limit_actual_time_ms` | Float | Node execution time on the top-level `Limit` node in milliseconds. |
| `plan_type_verified` | String | Verified execution scan node (`Seq Scan` or `Index Scan`). |
| `notes` | String | Architectural explanation of buffer accesses and TOAST assembly. |

### Valid Comparisons & Findings
- **Sequential Scan Buffer Reduction:** Inlining FP16 vectors into heap pages reduces Q0 shared-buffer accesses by **93.35%** (1,000 vs. 15,032 hits at 5k; 2,000 vs. 30,064 hits at 10k). FP32 requires accessing secondary TOAST chunks out-of-line for every tuple, whereas FP16 reads tuples inline within primary heap blocks.
- **Forced HNSW Buffer Access:** HNSW index scans require significantly fewer buffer accesses (~753–930 hits across both representations; FP16 is 9.28% lower at 5k and 8.92% lower at 10k) because traversal navigates graph neighborhood links rather than scanning all relation blocks.

---

## 3. Retrieval Fidelity & Neighborhood Agreement Assets

Retrieval fidelity is documented through two distinct, non-interchangeable CSV assets to prevent conflating representation-specific recall with pairwise cross-index agreement.

```
Ground Truth Reference: Uncompressed FP32 Exact Sequential Scan
       │                                     │
       ├── Recall@K (chart3_hnsw_recall.csv)   ├── Recall@K (chart3_hnsw_recall.csv)
       ▼                                     ▼
 [FP32 HNSW Index] ◄── Pairwise Agreement ──► [FP16 HNSW Index]
                  (chart3_pairwise_agreement.csv)
```

### 3.A. `chart3_hnsw_recall.csv` — Nearest-Neighbor Overlap (ANN Recall@K)

#### Purpose
Records Nearest-Neighbor Overlap (ANN Recall@K) at $ef_{search} = 40$ across 20 fixed evaluation queries relative to an independent, uncompressed ground truth.

#### Ground Truth Definition
Ground truth is defined independently for each corpus scale by exhaustive, uncompressed FP32 sequential scan on the exact same database (`ORDER BY embedding <=> query_embedding LIMIT K`).

#### Schema & Columns
| Column Name | Type | Description & Unit |
|---|---|---|
| `corpus_scale_chunks` | Integer | Evaluated chunk count (`5000`, `10000`). |
| `representation` | String | Index evaluated: `FP32 HNSW (vector)` or `FP16 HNSW (halfvec)`. |
| `recall_at_3` | Float | Overlap ratio at K=3 relative to exact FP32 ground truth. |
| `recall_at_5` | Float | Overlap ratio at K=5 relative to exact FP32 ground truth. |
| `recall_at_10` | Float | Overlap ratio at K=10 relative to exact FP32 ground truth. |
| `reference_ground_truth` | String | Description of the baseline ground truth reference. |
| `notes` | String | Configuration details ($ef_{search}=40$, 20 queries). |

#### Valid Comparisons & Findings
- **High-Fidelity Parity:** At $ef_{search} = 40$, FP16 HNSW achieves **96.0% Recall@10** at both 5,000 chunks (0.9600 vs. 0.9550 for FP32) and 10,000 chunks (0.9600 vs. 0.9450 for FP32).
- **Non-Monotonicity Note (10k FP32):** In the 10,000-chunk FP32 HNSW benchmark, recorded Recall values are 0.9667 at K=3, 0.9500 at K=5, and 0.9450 at K=10. This empirical behavior is preserved exactly as recorded; the metric reflects set overlap fraction divided by $K$ averaged over queries.
- **Scope Limitation:** Recall@K measures mathematical nearest-neighbor rank overlap; it does **not** evaluate downstream LLM generation quality or semantic relevance.

---

### 3.B. `chart3_pairwise_agreement.csv` — Pairwise Top-K Neighborhood Agreement

#### Purpose
Records pairwise neighborhood agreement between the candidate sets returned by FP32 HNSW and FP16 HNSW index traversals at $ef_{search} = 40$.

#### Exact Mathematical Formula
For each evaluation query $q \in \{1, \dots, 20\}$ and top-$K \in \{3, 5, 10\}$:
$$\text{Agreement@K}(q) = \frac{|\text{Top-}K_{\text{FP32}}(q) \cap \text{Top-}K_{\text{FP16}}(q)|}{K}$$

Aggregated across all 20 fixed evaluation queries:
$$\text{Mean Relative Agreement@K} = \frac{1}{20} \sum_{q=1}^{20} \text{Agreement@K}(q)$$

> [!IMPORTANT]
> **Methodological Distinction:** This metric is the **symmetric top-$K$ set intersection overlap fraction** (divided by $K$). It is **NOT** the Jaccard similarity index (which divides by $|S_{\text{FP32}} \cup S_{\text{FP16}}| = 2K - |S_{\text{FP32}} \cap S_{\text{FP16}}|$).

#### Schema & Columns
| Column Name | Type | Description & Unit |
|---|---|---|
| `corpus_scale_chunks` | Integer | Evaluated chunk count (`5000`, `10000`). |
| `comparison_pair` | String | Evaluated pair: `FP32 HNSW vs FP16 HNSW`. |
| `agreement_at_3` | Float | Mean top-3 set intersection fraction (0.9667 at 5k; 0.9667 at 10k). |
| `agreement_at_5` | Float | Mean top-5 set intersection fraction (0.9800 at 5k; 0.9600 at 10k). |
| `agreement_at_10` | Float | Mean top-10 set intersection fraction (0.9850 at 5k; 0.9500 at 10k). |
| `aggregation_method` | String | Formal aggregation and averaging definition. |
| `notes` | String | Provenance and evaluation configuration. |

#### Valid Comparisons & Findings
- **Candidate-Set Consistency:** Pairwise agreement remains $\ge 95\%$ across all evaluated top-$K$ cutoffs (98.5% at 5k; 95.0% at 10k for $K=10$), demonstrating high overlap between the top-$K$ candidate sets returned by FP32 HNSW and FP16 HNSW at $ef_{search} = 40$. Note that this metric measures candidate-set overlap between returned result lists and does not measure internal HNSW graph-edge connectivity or graph topology.

---

## 4. `chart4_latency.csv` — Client-Observed Retrieval Latency Distributions

### Purpose
Documents client-observed query execution latency distributions across 100 timed trials (20 queries × 5 measurement passes) under the corrected Stage 4 v2 timing harness.

### Key DBMS Invariant & Harness Specifications
- **Strict Timing Boundaries:** Latencies are measured using `time.perf_counter()` strictly bounding `cur.execute()` and `cur.fetchall()`.
- **GUC Statement Isolation:** Database configuration statements (`SET enable_indexscan ...`, `SET hnsw.ef_search ...`) are executed **outside** the measurement timer.
- **Alternating Paired Trials:** Measurement passes alternate execution order (`FP32`, `FP16` vs. `FP16`, `FP32`) to prevent systemic cache bias.
- **Single-Host Warm-up:** Warm-up iterations are executed prior to timed trials to ensure relations are resident in shared buffers.
- **Exclusion of Stage 3 Timings:** Stage 3 1,001-chunk latency is strictly excluded from Chart 4 because it timed a full transaction block (`BEGIN; SET LOCAL; ...; COMMIT;`), which would confound harness changes with corpus scale.

### Schema & Columns
| Column Name | Type | Description & Unit |
|---|---|---|
| `corpus_scale_chunks` | Integer | Evaluated chunk count (`5000`, `10000`). |
| `retrieval_mode` | String | Search mode: `Exact Search` (Seq Scan) or `Forced HNSW` (Index Scan). |
| `representation` | String | Coordinate precision: `FP32 (vector)` or `FP16 (halfvec)`. |
| `mean_latency_ms` | Float | Arithmetic mean latency across 100 trials (ms). |
| `median_latency_ms` | Float | Median latency across 100 trials (ms). |
| `iqr_ms` | Float | Interquartile range ($Q_3 - Q_1$) in milliseconds. |
| `min_latency_ms` | Float | Minimum recorded trial latency (ms). |
| `max_latency_ms` | Float | Maximum recorded trial latency (ms). |
| `stdev_ms` | Float | Sample standard deviation in milliseconds. |
| `speedup_vs_fp32_mean` | String | Speedup factor on mean latency (`FP32 mean / FP16 mean`). |
| `speedup_vs_fp32_median` | String | Speedup factor on median latency (`FP32 median / FP16 median`). |
| `latency_reduction_pct_mean` | String | Percentage reduction in mean latency. |
| `latency_reduction_pct_median` | String | Percentage reduction in median latency. |
| `notes` | String | Benchmark harness details and paired win rates. |

### Valid Comparisons & Findings
- **Exact Search Speedup:**
  - **At 5,000 chunks:** FP32 mean 42.285 ms vs. FP16 mean 6.802 ms (**6.22x speedup / 83.91% lower mean latency**); median 44.694 ms vs. 6.277 ms (**7.12x speedup / 85.96% lower median latency**). FP16 was faster on 100/100 paired trials.
  - **At 10,000 chunks:** FP32 mean 89.004 ms vs. FP16 mean 12.301 ms (**7.24x speedup / 86.18% lower mean latency**); median 92.664 ms vs. 11.541 ms (**8.03x speedup / 87.55% lower median latency**). FP16 was faster on 100/100 paired trials.
- **HNSW Traversal Latency Parity:**
  - Traversal latencies remain similar across representations (mean latencies span approximately 3.37–4.13 ms; medians span 2.92–3.82 ms): individual means measure 4.129 ms (FP32) vs. 3.372 ms (FP16) at 5k, and 3.674 ms (FP32) vs. 3.856 ms (FP16) at 10k (medians measure 3.371 ms vs. 2.923 ms at 5k; 3.415 ms vs. 3.820 ms at 10k).
  - *Plausible Explanation (Unverified):* A plausible explanation for similar traversal latencies is that HNSW query time may be dominated by graph navigation overhead rather than coordinate distance calculations once data is cached; however, this hypothesis was not directly verified through profiling.

---

## Important Methodological Boundaries & Negative Claims

When generating charts or writing reports from these assets, adhere to the following negative claims:

1. ❌ **Do NOT claim linear or asymptotic scaling laws from two points:** Exact search and forced HNSW latencies in Chart 4 are reported for two discrete scales (5k and 10k). Do not extrapolate linear $\mathcal{O}(N)$ or logarithmic $\mathcal{O}(\log N)$ asymptotic scaling curves from two points.
2. ❌ **Do NOT claim Q0 buffer hits prove trial latency causality:** Single-query Q0 buffer access reductions are consistent with repeated-trial latency speedups, but do not establish the causal mechanism of every measured latency improvement across repeated trials.
3. ❌ **Do NOT conflate pairwise agreement with Recall@K:** Recall@K measures overlap against uncompressed FP32 exact ground truth; pairwise agreement measures consistency between the two HNSW graph representations.
4. ❌ **Do NOT call pairwise agreement Jaccard overlap:** The metric is the mean top-$K$ set intersection overlap fraction ($\frac{|A \cap B|}{K}$), not the Jaccard index.
5. ❌ **Do NOT claim HNSW traversal is 10x faster:** Traversal times are similar across both representations (mean latencies span approximately 3.37–4.13 ms; medians span 2.92–3.82 ms); speedups apply strictly to exact sequential scans.
6. ❌ **Do NOT interchange storage scopes:** Pre-index table storage, HNSW index-only storage, and inclusive total storage represent different database components and must not be conflated.
