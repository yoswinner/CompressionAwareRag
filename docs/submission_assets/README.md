# Submission Assets Data Dictionary & Provenance Notes

This directory contains clean, verified CSV datasets supporting the charts, academic report, and Review 3 presentation for the project:
**Compression-Aware Vector Database Indexing for Memory-Efficient and Low-Latency RAG**.

All metrics are extracted from verified benchmark artifacts and PostgreSQL 18 catalog queries without synthetic interpolation.

---

## 1. `chart1_storage.csv`

- **Purpose:** Tracks physical storage footprint scaling across 1,001, 5,000, and 10,000 chunks.
- **Key DBMS Invariant:** Table base storage is reported separately from the HNSW vector index to prevent double-counting.
- **Columns:**
  - `corpus_scale_chunks` (integer): Number of indexed chunks (1001, 5000, 10000).
  - `representation` (string): Coordinate format (`FP32 (vector)` or `FP16 (halfvec)`).
  - `table_base_storage_bytes` (integer): Main heap + TOAST data relation + TOAST index + primary key B-tree index + auxiliary relation forks (FSM and VM), in bytes.
    - *At 1,001 chunks:* Stage 2 `pg_total_relation_size` pre-HNSW total (4,333,568 B for FP32; 1,728,512 B for FP16).
    - *At 5,000 chunks:* Derived as total inclusive relation size (41,607,168 B) minus HNSW index size (20,422,656 B) = 21,184,512 B for FP32; 18,579,456 B minus 10,215,424 B = 8,364,032 B for FP16.
    - *At 10,000 chunks:* Derived as total inclusive relation size (83,050,496 B) minus HNSW index size (40,779,776 B) = 42,270,720 B for FP32; 37,068,800 B minus 20,398,080 B = 16,670,720 B for FP16.
  - `table_base_storage_mib` (float): Table base storage in mebibytes (`bytes / 1,048,576`).
  - `hnsw_index_bytes` (integer): Main data fork of the HNSW index (`pg_relation_size`), in bytes.
  - `hnsw_index_mib` (float): HNSW index footprint in mebibytes (`bytes / 1,048,576`).
  - `total_inclusive_bytes` (integer or empty): `pg_total_relation_size` inclusive of all indexes and forks. Left empty at 1,001 chunks because inclusive post-index total was not recorded as a single catalog query in Stage 3.
  - `total_inclusive_mib` (float or empty): Total inclusive relation size in mebibytes.
  - `notes` (string): Detailed breakdown of individual components.

---

## 2. `chart2_q0_diagnostics.csv`

- **Purpose:** Documents execution diagnostics for single-query Query 0 (Q0) runs captured via `EXPLAIN (ANALYZE, BUFFERS)`.
- **Key DBMS Invariant & Interpretation:** Shared-buffer hits reflect runtime page accesses (including repeated accesses), not unique relation block counts. Single-query Q0 diagnostic runs provide isolated query-plan verification; their 93.35% reduction in buffer accesses is consistent with and associated with the observed 6.2x–8.0x speedups measured across 100 timed trials, rather than serving as standalone proof of causality.
- **Columns:**
  - `corpus_scale_chunks` (integer): Corpus scale (5000, 10000).
  - `retrieval_mode` (string): Search strategy (`Exact Search` or `Forced HNSW`).
  - `representation` (string): Coordinate format (`FP32 (vector)` or `FP16 (halfvec)`).
  - `q0_shared_buffer_hits` (integer): Total shared-buffer hits reported in the Q0 plan.
  - `q0_server_execution_time_ms` (float): PostgreSQL server execution time in milliseconds reported in the `EXPLAIN ANALYZE` summary.
  - `q0_limit_actual_time_ms` (float): Node execution time on the top-level `Limit` node.
  - `plan_type_verified` (string): Verified scan type (`Seq Scan` or `Index Scan`).
  - `notes` (string): Architectural explanation of buffer access dynamics and qualification of diagnostic vs. trial-distribution data.

---

## 3. Nearest-Neighbor Overlap & Agreement Assets

Retrieval fidelity is documented using two distinct, complementary CSV assets to prevent conflating representation-specific recall with pairwise index agreement.

### 3.A. `chart3_hnsw_recall.csv`
- **Purpose:** Records Nearest-Neighbor Overlap (ANN Recall@K) at $ef_{search} = 40$ across 20 fixed evaluation queries relative to independent uncompressed ground truth.
- **Key DBMS Invariant:** Ground truth is defined by exhaustive, uncompressed FP32 sequential scan on the exact same database.
- **Columns:**
  - `corpus_scale_chunks` (integer): Corpus scale (5000, 10000).
  - `representation` (string): `FP32 HNSW (vector)` or `FP16 HNSW (halfvec)`.
  - `recall_at_3` (float): Overlap ratio at K=3 relative to exact FP32 reference.
  - `recall_at_5` (float): Overlap ratio at K=5 relative to exact FP32 reference.
  - `recall_at_10` (float): Overlap ratio at K=10 relative to exact FP32 reference.
  - `reference_ground_truth` (string): Description of the independent baseline reference.
  - `notes` (string): Provenance details.

### 3.B. `chart3_pairwise_agreement.csv`
- **Purpose:** Documents pairwise neighborhood agreement between FP32 HNSW and FP16 HNSW index traversals at $ef_{search} = 40$.
- **Exact Calculation Formula:**
  - For each evaluation query $q \in \{1, \dots, 20\}$ and top-$K \in \{3, 5, 10\}$:
    $$\text{Agreement@K}(q) = \frac{|\text{Top-}K_{\text{FP32}}(q) \cap \text{Top-}K_{\text{FP16}}(q)|}{K}$$
  - Aggregated across the 20 evaluation queries:
    $$\text{Mean Agreement@K} = \frac{1}{20} \sum_{q=1}^{20} \text{Agreement@K}(q)$$
  - *Methodological Distinction:* This calculation is the symmetric top-$K$ set intersection overlap fraction divided by $K$. It is **not** Jaccard index (which divides by $|S_{\text{FP32}} \cup S_{\text{FP16}}| = 2K - |S_{\text{FP32}} \cap S_{\text{FP16}}|$).
- **Columns:**
  - `corpus_scale_chunks` (integer): Corpus scale (5000, 10000).
  - `comparison_pair` (string): Pair evaluated (`FP32 HNSW vs FP16 HNSW`).
  - `agreement_at_3` (float): Mean top-3 overlap fraction (0.9667 at both scales).
  - `agreement_at_5` (float): Mean top-5 overlap fraction (0.9800 at 5k; 0.9600 at 10k).
  - `agreement_at_10` (float): Mean top-10 overlap fraction (0.9850 at 5k; 0.9500 at 10k).
  - `aggregation_method` (string): Exact description of set intersection and averaging.
  - `notes` (string): Benchmark provenance and configuration details.

---

## 4. `chart4_latency.csv`

- **Purpose:** Records client-observed retrieval latencies across 100 timed trials (20 queries × 5 measurement runs) using the corrected Stage 4 v2 timing harness.
- **Key DBMS Invariant & Interpretation:** Timing is strictly bounded around `cur.execute()` and `cur.fetchall()`. GUC configuration statements are executed outside the timer. Observations are strictly limited to the two evaluated scales (5,000 and 10,000 chunks) under this harness. Two discrete data points demonstrate concrete empirical measurements across 5k and 10k, and must not be used to infer linear growth or asymptotic scaling laws.
- **Columns:**
  - `corpus_scale_chunks` (integer): Corpus scale (5000, 10000).
  - `retrieval_mode` (string): `Exact Search` or `Forced HNSW`.
  - `representation` (string): `FP32 (vector)` or `FP16 (halfvec)`.
  - `mean_latency_ms` (float): Arithmetic mean latency across 100 trials.
  - `median_latency_ms` (float): Median latency across 100 trials.
  - `iqr_ms` (float): Interquartile range ($Q_3 - Q_1$).
  - `min_latency_ms` (float): Minimum recorded trial latency.
  - `max_latency_ms` (float): Maximum recorded trial latency.
  - `stdev_ms` (float): Sample standard deviation.
  - `speedup_vs_fp32_mean` (string): Mean speedup factor (`FP32 mean / FP16 mean`).
  - `speedup_vs_fp32_median` (string): Median speedup factor (`FP32 median / FP16 median`).
  - `latency_reduction_pct_mean` (string): Percentage reduction in mean latency.
  - `latency_reduction_pct_median` (string): Percentage reduction in median latency.
  - `notes` (string): Benchmark methodology details.
