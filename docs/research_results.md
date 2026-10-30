# Precision-Aware Query Planning for Vector Retrieval in PostgreSQL
## Comprehensive Empirical Research Results & Statistical Synthesis

**Author:** Yash Pradhan  
**Document Type:** Version-Controlled Empirical Research Synthesis  
**Target Venue:** IEEE Transactions on Knowledge and Data Engineering (TKDE) / IEEE International Conference on Data Engineering (ICDE)  
**Repository Branch:** `research/planner-sensitivity`  
**Base Benchmark Commit:** `3f2beee66cecd67aa8fc4cec69c364b92c07b884`  
**Audit Baseline:** Phase 3 Verification Complete

---

## 1. Research Questions & Hypotheses

This research investigates how vector precision quantization and relational physical storage policies alter PostgreSQL query planning and execution for Approximate Nearest Neighbor (ANN) vector retrieval:

- **RQ1 (Planner Indifference Frontier):** Under what combinations of planner cost settings ($random\_page\_cost \in \{1.0, 4.0, 8.0\}$ and $cpu\_operator\_cost \in \{0.0025, 0.010, 0.050\}$) does PostgreSQL elect a brute-force sequential scan with top-$N$ sort over an HNSW index for vector retrieval ($K=10$)?
- **RQ2 (Physical Storage Policy & Quantization Impact):** How does physical tuple layout (`FP32-EXTENDED` vs. `FP32-PLAIN` vs. `FP16-EXTENDED`) alter heap and TOAST page counts, shifting the optimizer's startup cost calculations and path selections across dataset scales ($N \in \{1{,}001, 5{,}000, 10{,}000\}$)?
- **RQ3 (Latency, Buffer Footprint, and Recall Accuracy):** What are the empirical execution time penalties ($P_{50}, P_{95}$), shared buffer page hits, and Recall@10 consequences of planner-selected paths compared to forced HNSW index scans?

### Hypotheses
1. **$H_1$ (Storage-Induced Plan Divergence):** Out-of-line TOASTing of `vector(768)` under `STORAGE EXTENDED` lowers main heap pages ($relpages$). Based on inspection of PostgreSQL's `hnswcostestimate` implementation, this represents a supported explanatory mechanism for altered startup-cost calculations, while inline storage (`STORAGE PLAIN` or `halfvec(768)`) retains higher heap page counts.
2. **$H_2$ (Scale-Dependent Plan Convergence):** At small corpus scales ($N=1{,}001$), planner cost profiles heavily influence path choice (HNSW vs. Seq Scan). At larger evaluated corpus scales ($N \ge 5{,}000$), HNSW index scans are selected 100% of the time across all evaluated cost configurations.
3. **$H_3$ (Storage and Retrieval Parity of FP16):** Half-precision 16-bit quantization (`halfvec(768)`) yields $\approx 55\%$ total storage savings. Under canonical Profile P4 (forced-HNSW), FP16 Extended achieves Recall@10 of 0.985 / 0.995 (1k), 0.960 / 0.980 (5k), and 0.935 / 0.975 (10k) (conventional / tie-aware), maintaining $\ge 0.975$ tie-aware recall across all evaluated scales. Across all 660 query-condition units per scale combined, scale-level mean Recall@10 reaches 0.9914 / 0.9605 / 0.9394 (conventional) and 0.9971 / 0.9847 / 0.9758 (tie-aware).

---

## 2. Experimental Design Matrix

The experimental evaluation decouples execution-free plan capture from timed query latency:

- **Planning-Only Plan Captures (Fully Crossed Matrix):** Captured across 3 scales $\times$ 3 representations $\times$ 3 execution modes $\times$ 9 planner cost profiles $\times$ 20 queries = **4,860 plan captures**.
- **Timed Workload Measurements (Optimized Partial Factorial Design):** Forced-path timing is restricted to the canonical profile (P4) to evaluate representation and scale effects under PostgreSQL default cost parameters, while planner-selected execution is measured across all 9 profiles:
  - Planner-selected: 3 scales $\times$ 3 representations $\times$ 9 profiles $\times$ 20 queries $\times$ 5 passes = **8,100 timed queries**.
  - Forced exact: 3 scales $\times$ 3 representations $\times$ Profile P4 $\times$ 20 queries $\times$ 5 passes = **900 timed queries**.
  - Forced HNSW: 3 scales $\times$ 3 representations $\times$ Profile P4 $\times$ 20 queries $\times$ 5 passes = **900 timed queries**.
  - Total: **9,900 timed executions**.
- **Diagnostic EXPLAIN ANALYZE Runs:** 3 scales $\times$ 3 representations $\times$ 3 modes $\times$ Profile P4 $\times$ 20 queries = **540 diagnostic runs**.

### 2.1 The Three Vector Representations
1. **`FP32 Extended` (`vector(768)`, `STORAGE EXTENDED`):** Default PostgreSQL representation. Coordinates stored as 32-bit single-precision floats. Observed `pg_column_size` is 3,076 bytes per datum, while theoretical uncompressed datum size is $4d + 8 = 3{,}080$ bytes. While the 3,076-byte measurement is the recorded physical datum size in PostgreSQL, the precise architectural reason for this 4-byte difference was not established by the experimental evidence. Tuples exceed the 2 KiB page threshold and are stored out-of-line in TOAST relations.
2. **`FP32 Plain` (`vector(768)`, `STORAGE PLAIN`):** Inline storage policy enforced via `ALTER TABLE ... SET STORAGE PLAIN`. Observed `pg_column_size` is 3,080 bytes per datum ($4d + 8$). Forces vectors into main heap blocks (fitting exactly 2 tuples per 8 KiB page), avoiding TOAST overhead while expanding main heap pages.
3. **`FP16 Extended` (`halfvec(768)`, `STORAGE EXTENDED`):** IEEE 754 16-bit half-precision floating-point coordinates. Observed `pg_column_size` is 1,544 bytes per datum ($2d + 8$). Sits naturally below the 2 KiB page threshold, remaining 100% inline within main heap blocks.

### 2.2 The Nine Planner Cost Profiles

Cost parameters are dimensionless relative units normalized to $seq\_page\_cost = 1.0$:

| Profile ID | $random\_page\_cost$ | $cpu\_operator\_cost$ | $seq\_page\_cost$ | $cpu\_tuple\_cost$ | Description |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | 1.0 | 0.0025 | 1.0 | 0.01 | Low I/O cost, pgvector default operator cost |
| **P2** | 1.0 | 0.0100 | 1.0 | 0.01 | Low I/O cost, elevated CPU operator cost |
| **P3** | 1.0 | 0.0500 | 1.0 | 0.01 | Low I/O cost, high CPU operator cost |
| **P4** | 4.0 | 0.0025 | 1.0 | 0.01 | **Canonical Profile** (PostgreSQL default settings) |
| **P5** | 4.0 | 0.0100 | 1.0 | 0.01 | Standard I/O cost, elevated CPU operator cost |
| **P6** | 4.0 | 0.0500 | 1.0 | 0.01 | Standard I/O cost, high CPU operator cost |
| **P7** | 8.0 | 0.0025 | 1.0 | 0.01 | High I/O cost, pgvector default operator cost |
| **P8** | 8.0 | 0.0100 | 1.0 | 0.01 | High I/O cost, elevated CPU operator cost |
| **P9** | 8.0 | 0.0500 | 1.0 | 0.01 | High I/O cost, high CPU operator cost |

- **Invariants:** `hnsw.ef_search = 40`, `max_parallel_workers_per_gather = 0`, `LIMIT = 10`, `HNSW parameters: m=16, ef_construction=64`.
- **Query Set:** 20 pre-generated MS MARCO semantic queries with precomputed 768-dimensional embeddings.
- **Passes:** 2 unmeasured warm-up passes, 5 measured passes per query.

---

## 3. Environment & Provenance Information

### 3.1 Host & Database Environment
- **Operating System:** Windows 11 (x86_64, 64-bit).
- **Database Engine:** PostgreSQL 18.4 on `x86_64-pc-windows-msvc`.
- **Vector Extension:** `pgvector` v0.8.4 (with `vector` and `halfvec` support).
- **Client Driver:** `psycopg` 3.3.4 (binary distribution).
- **Language Runtime:** Python 3.13.14.
- **Embedding Model:** `nomic-embed-text:latest` (768 dimensions) served via Ollama local daemon.
- **Database Catalogs:** Dedicated experimental databases `compression_aware_rag_exp` ($N=1{,}001$) and `compression_aware_rag_scale` ($N=5{,}000; 10{,}000$). Baseline production catalog `compression_aware_rag` remained strictly read-only and unaccessed.

### 3.2 Runtime vs. Post-Run Provenance Distinctions
- **Pre-execution Snapshot:** The primary benchmark captured environment metadata into `environment_snapshot.json` and `experiment_manifest.json`.
- **Post-run Verification:** The Phase 2A and Phase 3A audits independently verified:
  1. No automatic prepared statement caching occurred (`prepare_threshold=None`, `prepare=False`).
  2. Root-plan buffer consumption was parsed from top-level plan nodes to eliminate cumulative recursive child double-counting.
  3. Query vectors were completely recovered from execution plans and re-verified.
  4. Ground-truth reference document IDs were exported and verified under exact sequential scans.

---

## 4. Physical Storage Results

Physical table layout and disk consumption were measured using PostgreSQL system catalog inspection functions (`pg_relation_size`, `pg_table_size`, `pg_indexes_size`, `pg_total_relation_size`):

### 4.1 Storage Footprint Breakdown Across Scales

| Scale ($N$) | Representation | Storage Policy | Heap Size (Bytes) | TOAST Total (Bytes) | HNSW Index (Bytes) | Total Size (Bytes) | Total Reduction vs. FP32 Ext |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,001** | `fp32_extended` | `EXTENDED` | 57,344 | 4,169,728 | 4,087,808 | 8,413,184 | Baseline (0.00%) |
| **1,001** | `fp32_plain` | `PLAIN` | 4,104,192 | 8,192 | 4,087,808 | 8,265,728 | **+1.75%** |
| **1,001** | `fp16_extended` | `EXTENDED` | 1,646,592 | 8,192 | 2,048,000 | 3,768,320 | **+55.21%** |
| **5,000** | `fp32_extended` | `EXTENDED` | 262,144 | 20,725,760 | 20,422,656 | 41,607,168 | Baseline (0.00%) |
| **5,000** | `fp32_plain` | `PLAIN` | 20,480,000 | 8,192 | 20,430,848 | 41,082,880 | **+1.26%** |
| **5,000** | `fp16_extended` | `EXTENDED` | 8,192,000 | 8,192 | 10,215,424 | 18,579,456 | **+55.35%** |
| **10,000** | `fp32_extended` | `EXTENDED` | 524,288 | 41,426,944 | 40,779,776 | 83,050,496 | Baseline (0.00%) |
| **10,000** | `fp32_plain` | `PLAIN` | 40,960,000 | 8,192 | 40,779,776 | 82,034,688 | **+1.22%** |
| **10,000** | `fp16_extended` | `EXTENDED` | 16,384,000 | 8,192 | 20,398,080 | 37,068,800 | **+55.37%** |

### 4.2 Storage Analysis Observations
1. **TOAST Inversion:** Under `STORAGE EXTENDED`, `fp32_extended` stores $41{,}426{,}944$ bytes in TOAST relations versus $524{,}288$ bytes in heap blocks at $N=10{,}000$, representing **98.75%** of table storage ($41{,}426{,}944 / 41{,}951{,}232$, heap + TOAST, excluding indexes). In contrast, `fp32_plain` forces all 40.96 MB into heap blocks.
2. **HNSW Graph Size Reduction:** FP16 representation halves the HNSW index graph size (e.g. 20.40 MB for FP16 vs 40.78 MB for FP32 at 10k, a $49.98\%$ index storage reduction).
3. **Total Footprint Reduction:** Across all three scales, `halfvec(768)` delivers a consistent $\approx \mathbf{55.3\%}$ total physical storage reduction compared to FP32 extended.

---

## 5. Planner Path Selection Results

Out of 1,620 planner-selected execution plans (3 scales $\times$ 3 representations $\times$ 9 profiles $\times$ 20 queries):
- **HNSW Index Scan:** 1,400 plans (**86.4%**)
- **Sequential Scan (Seq Scan):** 220 plans (**13.6%**)

### 5.1 Plan Choices by Scale & Profile

| Scale | Profile | $random\_page\_cost$ | $cpu\_operator\_cost$ | HNSW Index Scan Count | Seq Scan Count | Total Plans | HNSW Scan % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,001** | P1 | 1.0 | 0.0025 | 60 | 0 | 60 | 100.0% |
| **1,001** | P2 | 1.0 | 0.0100 | 60 | 0 | 60 | 100.0% |
| **1,001** | P3 | 1.0 | 0.0500 | 60 | 0 | 60 | 100.0% |
| **1,001** | P4 | 4.0 | 0.0025 | 20 | 40 | 60 | 33.3% |
| **1,001** | P5 | 4.0 | 0.0100 | 20 | 40 | 60 | 33.3% |
| **1,001** | P6 | 4.0 | 0.0500 | 60 | 0 | 60 | 100.0% |
| **1,001** | P7 | 8.0 | 0.0025 | 0 | 60 | 60 | 0.0% |
| **1,001** | P8 | 8.0 | 0.0100 | 20 | 40 | 60 | 33.3% |
| **1,001** | P9 | 8.0 | 0.0500 | 20 | 40 | 60 | 33.3% |
| **5,000** | P1 – P9 | All combinations | All combinations | **540** | **0** | **540** | **100.0%** |
| **10,000** | P1 – P9 | All combinations | All combinations | **540** | **0** | **540** | **100.0%** |

### 5.2 Representation Influence at Scale 1,001
At $N=1{,}001$, representation storage layout directly alters path selection:
- `fp32_extended`: Selected HNSW Index Scan in **520 of 540** cases (96.3%) and Seq Scan in only 20 cases.
- `fp32_plain`: Selected HNSW Index Scan in **440 of 540** cases (81.5%) and Seq Scan in 100 cases.
- `fp16_extended`: Selected HNSW Index Scan in **440 of 540** cases (81.5%) and Seq Scan in 100 cases.

**Key Finding:** Sequential scans were chosen **exclusively** at the 1,001-row scale. At scales 5,000 and 10,000, the PostgreSQL query planner selected HNSW Index Scan 100% of the time across all 9 profiles and all representations within the evaluated matrix.

---

## 6. Execution Latency Results

Descriptive statistics are computed over the 100 timed executions per condition (5 measured passes $\times$ 20 queries) under canonical Profile P4 ($random\_page\_cost = 4.0, cpu\_operator\_cost = 0.0025$). P50 corresponds to the sample median, P95 is calculated using nearest-rank indexing, and sample standard deviation ($n-1$) reflects within-condition variability, reconciled exactly against `summary_results.csv`:

### 6.1 Profile P4 Latency Summary

| Scale | Representation | Mode | $N$ | P50 (ms) | P95 (ms) | Mean (ms) | Sample SD (ms) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1,001** | `fp16_extended` | Exact (Seq Scan) | 100 | 1.699 | 2.788 | 1.845 | 0.430 |
| **1,001** | `fp16_extended` | Forced HNSW | 100 | 1.591 | 2.627 | 1.736 | 0.420 |
| **1,001** | `fp16_extended` | Planner Selected | 100 | 1.674 | 3.045 | 1.857 | 0.513 |
| **1,001** | `fp32_extended` | Exact (Seq Scan) | 100 | 5.127 | 10.475 | 6.171 | 6.575 |
| **1,001** | `fp32_extended` | Forced HNSW | 100 | 1.779 | 2.964 | 1.998 | 1.134 |
| **1,001** | `fp32_extended` | Planner Selected | 100 | 1.701 | 2.625 | 1.825 | 0.374 |
| **1,001** | `fp32_plain` | Exact (Seq Scan) | 100 | 1.899 | 3.420 | 2.131 | 0.606 |
| **1,001** | `fp32_plain` | Forced HNSW | 100 | 1.670 | 3.142 | 1.917 | 0.677 |
| **1,001** | `fp32_plain` | Planner Selected | 100 | 1.870 | 3.411 | 2.103 | 0.585 |
| **5,000** | `fp16_extended` | Exact (Seq Scan) | 100 | 4.504 | 9.176 | 4.883 | 2.021 |
| **5,000** | `fp16_extended` | Forced HNSW | 100 | 1.971 | 3.835 | 2.241 | 0.803 |
| **5,000** | `fp16_extended` | Planner Selected | 100 | 1.824 | 3.728 | 2.164 | 0.910 |
| **5,000** | `fp32_extended` | Exact (Seq Scan) | 100 | 29.416 | 53.056 | 36.027 | 13.128 |
| **5,000** | `fp32_extended` | Forced HNSW | 100 | 1.866 | 3.480 | 2.223 | 1.140 |
| **5,000** | `fp32_extended` | Planner Selected | 100 | 1.853 | 3.358 | 2.080 | 0.588 |
| **5,000** | `fp32_plain` | Exact (Seq Scan) | 100 | 6.182 | 10.475 | 6.968 | 2.752 |
| **5,000** | `fp32_plain` | Forced HNSW | 100 | 1.867 | 3.582 | 2.128 | 0.660 |
| **5,000** | `fp32_plain` | Planner Selected | 100 | 1.870 | 3.244 | 2.127 | 0.639 |
| **10,000** | `fp16_extended` | Exact (Seq Scan) | 100 | 12.770 | 28.163 | 14.252 | 6.974 |
| **10,000** | `fp16_extended` | Forced HNSW | 100 | 2.479 | 4.499 | 2.776 | 1.053 |
| **10,000** | `fp16_extended` | Planner Selected | 100 | 2.450 | 6.125 | 2.906 | 1.456 |
| **10,000** | `fp32_extended` | Exact (Seq Scan) | 100 | 91.954 | 126.147 | 93.638 | 24.100 |
| **10,000** | `fp32_extended` | Forced HNSW | 100 | 2.716 | 8.466 | 3.847 | 4.127 |
| **10,000** | `fp32_extended` | Planner Selected | 100 | 2.811 | 5.596 | 3.150 | 1.500 |
| **10,000** | `fp32_plain` | Exact (Seq Scan) | 100 | 53.556 | 70.385 | 52.800 | 10.927 |
| **10,000** | `fp32_plain` | Forced HNSW | 100 | 2.881 | 7.020 | 3.550 | 1.889 |
| **10,000** | `fp32_plain` | Planner Selected | 100 | 2.785 | 7.157 | 3.433 | 1.919 |

### 6.2 Latency Insights
1. **TOAST Overhead in Sequential Scan:** In exact sequential scan, `fp32_extended` incurs substantial latency penalties due to out-of-line TOAST pointer dereferencing (91.95 ms P50 at 10k vs 53.56 ms P50 for `fp32_plain` and 12.77 ms P50 for `fp16_extended`).
2. **HNSW Convergence:** Under HNSW index scans, median latency remains between 1.59 ms and 2.88 ms across all representations and scales.
3. **FP16 Performance at Scale:** At 10,000 rows, `fp16_extended` achieved the lowest observed P50 latency (2.45 ms planner-selected, 2.48 ms forced HNSW).

---

## 7. Retrieval Accuracy & Boundary-Tie Sensitivity

Retrieval accuracy was audited using the complete set of 1,980 query-condition units against the ground-truth top-10 FP32 reference set:

### 7.1 Canonical Profile P4 Forced-HNSW FP16 Extended Retrieval Accuracy

Under the canonical planner profile (P4) with forced HNSW index scan, `FP16 Extended` achieves the following retrieval accuracy across corpus scales:

| Scale ($N$) | Conventional Recall@10 | Boundary Tie-Aware Recall@10 |
| :---: | :---: | :---: |
| **1,001** | 0.985 | 0.995 |
| **5,000** | 0.960 | 0.980 |
| **10,000** | 0.935 | 0.975 |

*Retrieval Observation:* FP16 retrieval accuracy exhibits a mild scale-dependent decline as corpus size increases (0.985 $\to$ 0.960 $\to$ 0.935 conventional; 0.995 $\to$ 0.980 $\to$ 0.975 tie-aware). Boundary tie-aware adjustment substantially recovers nearest-neighbor parity across all evaluated scales ($\ge 0.975$ across all scales).

### 7.2 Overall Scale-Level Means Across All 660 Query-Condition Units Per Scale

The table below summarizes retrieval metrics computed across all 660 query-condition units evaluated per scale (spanning all 9 planner profiles, 3 representations, and 3 execution modes across 20 queries):

| Scale | Epsilon ($\tau$) | Observations | Tied Obs | Mean Conventional Recall@10 | Mean Tie-Aware Recall@10 | Recall Boost ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,001** | $1.0 \times 10^{-5}$ | 660 | 99 | 0.9914 | 0.9971 | +0.0058 |
| **1,001** | $1.0 \times 10^{-6}$ | 660 | 99 | 0.9914 | 0.9971 | +0.0058 |
| **1,001** | $1.0 \times 10^{-7}$ | 660 | 99 | 0.9914 | 0.9971 | +0.0058 |
| **5,000** | $1.0 \times 10^{-5}$ | 660 | 99 | 0.9605 | 0.9864 | +0.0259 |
| **5,000** | $1.0 \times 10^{-6}$ | 660 | 66 | 0.9605 | 0.9847 | +0.0242 |
| **5,000** | $1.0 \times 10^{-7}$ | 660 | 66 | 0.9605 | 0.9847 | +0.0242 |
| **10,000** | $1.0 \times 10^{-5}$ | 660 | 33 | 0.9394 | 0.9758 | +0.0364 |
| **10,000** | $1.0 \times 10^{-6}$ | 660 | 33 | 0.9394 | 0.9758 | +0.0364 |
| **10,000** | $1.0 \times 10^{-7}$ | 660 | 33 | 0.9394 | 0.9758 | +0.0364 |

### 7.3 Boundary-Tie Analysis & Numerical Tolerance Sensitivity
1. **Tolerance Sensitivity Across Evaluated Settings:** Evaluated tolerances ($\tau \in \{10^{-7}, 10^{-6}, 10^{-5}\}$) span 2 orders of magnitude (a factor of 100). Retrieval results were strictly identical between $\tau = 10^{-7}$ and $\tau = 10^{-6}$ across all 1,980 query-condition units. Widening the tolerance to $\tau = 10^{-5}$ identified one additional boundary-tie case at Scale 5,000 (Query ID 17), shifting scale-5,000 mean tie-aware recall from 0.9847 to 0.9864.
2. **Deterministic Query Attribution (0-Based Query IDs):** Exact sequential scan ground-truth analysis identifies specific queries exhibiting boundary distance ties across the 20-query suite:
   - **Scale 1,001:** Query IDs 5, 13, 16 (e.g. Query 5 boundary distance $0.473267$ with 4 chunks tied for 1 boundary slot).
   - **Scale 5,000:** Query IDs 3, 9 (e.g. Query 3 boundary distance $0.406328$ with 12 chunks tied for 8 slots; Query 9 with 12 chunks tied for 10 slots). Widening to $\tau = 10^{-5}$ introduces Query ID 17 (boundary distance $0.452952$ with 2 chunks tied for 1 slot).
   - **Scale 10,000:** Query ID 9 (boundary distance $0.443279$ with 28 chunks tied for 8 boundary slots).
3. **Relational Mechanism of Boundary Ties:** PostgreSQL executes vector top-$K$ queries via `ORDER BY distance ASC LIMIT 10` without a secondary deterministic tie-breaking key (such as `chunk_id`). When multiple candidate tuples share identical floating-point distance to the query vector at the rank-10 boundary, the relational executor arbitrates ties based on physical page order. Conventional set-intersection Recall@10 penalizes valid nearest neighbors that happen to be arbitrated out of the top 10; tie-aware Recall@10 credits any candidate whose distance matches the rank-10 boundary distance within tolerance $\tau$.

---

## 8. Diagnostic Root Buffer Results

Following Phase 2A audit corrections, buffer statistics are extracted exclusively from the root plan node, avoiding recursive parent/child over-counting:

| Scale | Representation | Mode | Reconstructed Mean Shared Hits | Reconstructed Mean Shared Reads |
| :---: | :--- | :--- | :---: | :---: |
| **1,001** | `fp16_extended` | Exact (Seq Scan) | 201.0 | 0.00 |
| **1,001** | `fp16_extended` | Forced HNSW | 486.6 | 0.00 |
| **1,001** | `fp16_extended` | Planner Selected | 201.0 | 0.00 |
| **1,001** | `fp32_extended` | Exact (Seq Scan) | 3,010.0 | 0.00 |
| **1,001** | `fp32_extended` | Forced HNSW | 508.5 | 0.00 |
| **1,001** | `fp32_extended` | Planner Selected | 508.5 | 0.00 |
| **1,001** | `fp32_plain` | Exact (Seq Scan) | 501.0 | 0.00 |
| **1,001** | `fp32_plain` | Forced HNSW | 484.9 | 0.00 |
| **1,001** | `fp32_plain` | Planner Selected | 501.0 | 0.00 |
| **5,000** | `fp16_extended` | Exact (Seq Scan) | 1,000.0 | 0.00 |
| **5,000** | `fp16_extended` | Forced HNSW | 671.2 | 0.00 |
| **5,000** | `fp16_extended` | Planner Selected | 671.2 | 0.00 |
| **5,000** | `fp32_extended` | Exact (Seq Scan) | 15,032.0 | 0.00 |
| **5,000** | `fp32_extended` | Forced HNSW | 697.0 | 0.00 |
| **5,000** | `fp32_extended` | Planner Selected | 697.0 | 0.00 |
| **5,000** | `fp32_plain` | Exact (Seq Scan) | 2,500.0 | 0.00 |
| **5,000** | `fp32_plain` | Forced HNSW | 684.1 | 0.00 |
| **5,000** | `fp32_plain` | Planner Selected | 684.1 | 0.00 |
| **10,000** | `fp16_extended` | Exact (Seq Scan) | 1,906.5 | 93.45 |
| **10,000** | `fp16_extended` | Forced HNSW | 679.5 | 36.40 |
| **10,000** | `fp16_extended` | Planner Selected | 715.9 | 0.00 |
| **10,000** | `fp32_extended` | Exact (Seq Scan) | 30,064.0 | 0.00 |
| **10,000** | `fp32_extended` | Forced HNSW | 715.5 | 29.35 |
| **10,000** | `fp32_extended` | Planner Selected | 744.9 | 0.00 |
| **10,000** | `fp32_plain` | Exact (Seq Scan) | 1,117.0 | 3,883.00 |
| **10,000** | `fp32_plain` | Forced HNSW | 643.0 | 88.05 |
| **10,000** | `fp32_plain` | Planner Selected | 731.0 | 0.00 |

**Buffer Observations:** In exact sequential scan at $N=10{,}000$, `fp32_extended` accesses 30,064 buffer pages (due to accessing both heap and TOAST relation pages). In contrast, HNSW index scans require only $\approx 640\text{--}745$ buffer page touches across all representations.

---

## 9. Non-Parametric Statistical Inference

Statistical testing is conducted using the 20 query-level experimental units ($N=20$).

### 9.1 Friedman Test Across 9 Cost Profiles (Benjamini-Hochberg Adjusted)
Tests whether planner cost profiles produce statistically significant latency differences under planner-selected execution, evaluated across the 9 experimental strata with standard Friedman tie correction and Benjamini-Hochberg (BH) false discovery rate control:

| Scale | Representation | Friedman $Q$ Stat | $C_{\text{ties}}$ | $df$ | Raw $p$-value | BH-Adjusted $p$ | Outcome ($\alpha = 0.05$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,001** | `fp32_extended` | 50.547 | 1.0000 | 8 | $3.21 \times 10^{-8}$ | **$< 0.001$** ($2.89 \times 10^{-7}$) | **Statistically Significant** |
| **1,001** | `fp32_plain` | 25.267 | 1.0000 | 8 | $1.40 \times 10^{-3}$ | **$0.0063$** | **Statistically Significant** |
| **1,001** | `fp16_extended` | 15.547 | 1.0000 | 8 | $4.93 \times 10^{-2}$ | $0.1480$ | Not Significant |
| **5,000** | `fp32_extended` | 8.160 | 1.0000 | 8 | 0.418 | $0.6270$ | Not Significant |
| **5,000** | `fp32_plain` | 6.000 | 1.0000 | 8 | 0.647 | $0.6860$ | Not Significant |
| **5,000** | `fp16_extended` | 9.467 | 1.0000 | 8 | 0.304 | $0.5480$ | Not Significant |
| **10,000** | `fp32_extended` | 5.653 | 1.0000 | 8 | 0.686 | $0.6860$ | Not Significant |
| **10,000** | `fp32_plain` | 6.720 | 1.0000 | 8 | 0.567 | $0.6860$ | Not Significant |
| **10,000** | `fp16_extended` | 10.507 | 1.0000 | 8 | 0.231 | $0.5203$ | Not Significant |

*Note on Ties:* In the empirical query-level latency data, all 9 profile latencies per query are distinct continuous values, resulting in $C_{\text{ties}} = 1.0000$. After controlling the false discovery rate across all 9 strata via Benjamini-Hochberg adjustment, cost profile sensitivity remains statistically significant only for `FP32 Extended` ($p_{\text{adj}} < 0.001$) and `FP32 Plain` ($p_{\text{adj}} = 0.0063$) at 1,001 rows. For `FP16 Extended` at 1,001 rows, the unadjusted raw $p = 0.0493$ fails to survive multiple comparison correction ($p_{\text{adj}} = 0.1480$). At $N \ge 5{,}000$, cost profile changes produce no plan alterations (HNSW index scan is chosen 100% of the time), resulting in no statistically significant latency variance across profiles ($p_{\text{adj}} \ge 0.52$).

### 9.2 Paired Wilcoxon Signed-Rank Tests (Profile P4, Forced HNSW)
Comparing FP16 vs. FP32 representations across the 20 queries under canonical Profile P4 with zero-difference handling, continuity correction, tie-corrected variance, and Holm-Bonferroni step-down correction ($\alpha = 0.05$). Monotonicity is enforced via forward cumulative maximum ($p_{\text{adj}, i} = \min(1.0, \max_{k \le i} q_k)$). Effect sizes are quantified using the matched-pairs rank-biserial correlation $r_{\text{rb}}$, where negative values indicate FP16 latency is lower than FP32:

| Scale | Comparison | $W$ Stat | Raw $p$-value | Holm-Adjusted $p$ | Rank-Biserial $r_{\text{rb}}$ | Outcome ($\alpha = 0.05$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10,000** | FP16 Ext vs. FP32 Plain | 28.0 | $4.29 \times 10^{-3}$ | **$0.0257$** | $-0.733$ | **Statistically Significant** |
| **10,000** | FP16 Ext vs. FP32 Ext | 41.0 | $1.78 \times 10^{-2}$ | $0.0888$ | $-0.610$ | Not Significant |
| **1,001** | FP16 Ext vs. FP32 Plain | 43.0 | $2.17 \times 10^{-2}$ | $0.0888$ | $-0.590$ | Not Significant |
| **1,001** | FP16 Ext vs. FP32 Ext | 55.0 | $6.46 \times 10^{-2}$ | $0.1938$ | $-0.476$ | Not Significant |
| **5,000** | FP16 Ext vs. FP32 Plain | 91.0 | $6.14 \times 10^{-1}$ | 1.0000 | $+0.133$ | Not Significant |
| **5,000** | FP16 Ext vs. FP32 Ext | 98.0 | $8.08 \times 10^{-1}$ | 1.0000 | $+0.067$ | Not Significant |

**Inference:** Only the comparison between FP16 Extended and FP32 Plain at the 10,000-row scale achieves statistical significance after Holm adjustment ($p_{\text{adj}} = 0.0257, r_{\text{rb}} = -0.733$). At smaller scales and against FP32 Extended, latency differences between FP16 and FP32 under forced HNSW scan do not survive family-wise error rate control.

### 9.3 Major Exact vs. Forced-HNSW Latency Inference under Profile P4
To quantify the latency difference between exact sequential scans and forced HNSW index scans under canonical Profile P4, paired differences were analyzed across queries.

*Methodological Distinction:* The 95% Confidence Intervals for Mean Difference (ms) and Speedup Factor are estimated via **30,000 query-cluster bootstrap resamples**. Inferential $p$-values are computed from **paired query-level Wilcoxon signed-rank tests ($N=20$)** with **Benjamini-Hochberg (BH) False Discovery Rate control**. Reported BH-adjusted $p$-values are derived from the Wilcoxon tests and are NOT bootstrap-derived $p$-values:

| Scale | Representation | Mean Diff (ms) | 95% Bootstrap CI (ms) | Speedup Factor | 95% Bootstrap CI (Speedup) | BH-Adjusted Wilcoxon $p$ | Outcome ($\alpha = 0.05$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,001** | `fp32_extended` | $+4.173$ | $[+3.064, +5.867]$ | $3.09\times$ | $[2.39\times, 4.13\times]$ | **$1.23 \times 10^{-4}$** | **Statistically Significant** |
| **1,001** | `fp32_plain` | $+0.215$ | $[+0.072, +0.363]$ | $1.11\times$ | $[1.04\times, 1.20\times]$ | **$0.0244$** | **Statistically Significant** |
| **1,001** | `fp16_extended` | $+0.108$ | $[-0.014, +0.236]$ | $1.06\times$ | $[0.99\times, 1.14\times]$ | $0.1978$ | Not Significant |
| **5,000** | `fp32_extended` | $+33.804$ | $[+31.043, +36.741]$ | $16.20\times$ | $[14.07\times, 18.53\times]$ | **$1.23 \times 10^{-4}$** | **Statistically Significant** |
| **5,000** | `fp32_plain` | $+4.840$ | $[+4.238, +5.522]$ | $3.27\times$ | $[2.92\times, 3.70\times]$ | **$1.23 \times 10^{-4}$** | **Statistically Significant** |
| **5,000** | `fp16_extended` | $+2.642$ | $[+2.262, +3.123]$ | $2.18\times$ | $[1.95\times, 2.48\times]$ | **$1.23 \times 10^{-4}$** | **Statistically Significant** |
| **10,000** | `fp32_extended` | $+89.791$ | $[+84.921, +94.979]$ | $24.34\times$ | $[19.13\times, 30.72\times]$ | **$1.23 \times 10^{-4}$** | **Statistically Significant** |
| **10,000** | `fp32_plain` | $+49.250$ | $[+47.323, +51.086]$ | $14.87\times$ | $[13.30\times, 16.81\times]$ | **$1.23 \times 10^{-4}$** | **Statistically Significant** |
| **10,000** | `fp16_extended` | $+11.476$ | $[+9.864, +13.102]$ | $5.13\times$ | $[4.45\times, 5.86\times]$ | **$1.23 \times 10^{-4}$** | **Statistically Significant** |

**Inference:** 8 of 9 exact-vs-forced-HNSW comparisons survive Benjamini-Hochberg adjustment ($p_{\text{adj}} < 0.05$). The sole exception is `FP16 Extended` at 1,001 rows (mean difference $+0.108$ ms, 95% CI $[-0.014, +0.236]$ ms, speedup $1.06\times$, 95% CI $[0.99\times, 1.14\times]$, BH $p_{\text{adj}} = 0.1978$). No recall comparison between exact and forced HNSW is statistically tested because exact sequential scan defines the reference truth (Recall = 1.0).

---

## 10. Research Limitations & Validity Threats

1. **Query Sample Size ($N=20$):** Evaluation utilized 20 representative MS MARCO semantic queries. While multi-pass repeated timing yielded 100 observations per condition, semantic variance across broader query workloads requires further evaluation.
2. **Warm-Cache Operating Regime:** Benchmark executions evaluated warmed buffer-cache behavior to measure algorithmic and planner efficiency. Cold-cache I/O from spinning disks or cold cloud volumes was not benchmarked.
3. **Single-Host Environment:** Experiments ran on a dedicated Windows workstation under PostgreSQL 18.4 on `x86_64`. Distributed PostgreSQL architectures (e.g. Citus) were not evaluated.
4. **Corpus Scale Scope:** The largest evaluated corpus size in this study is $N=10{,}000$ tuples. Multi-million-row production-scale corpora may encounter secondary memory or index cache spill behaviors not evaluated here.
5. **ANN-Only Scope:** Retrieval quality focused on vector similarity Recall@10. Downstream generative LLM answer fidelity (RAG answer quality) was not evaluated.
6. **No Direct Hardware Claims:** The observed latency reduction under `halfvec` operations was associated with reduced physical storage footprint and higher tuple density per page; however, isolating specific hardware or architectural mechanisms (such as memory-bus bandwidth or SIMD vectorization) was outside the empirical scope of this study. Cost profiles represent query planner cost adjustments, not hardware simulations.

---

## 11. Evidence Provenance & Cryptographic Hashes

All primary raw data are preserved in immutable ZIP archives and excluded from Git tracking:

| Evidence Archive | Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :--- |
| `Phase2_Full_Benchmark_Evidence_20261031.zip` | 18,526,371 | `b56bfd0fd702b5faecec379e290d9412f206bf0c867fefa8e8a09a62088722b0` |
| `Phase3A_Tie_Sensitivity_Evidence_20261031.zip` | 130,465 | `de19a122ea2d918cd5820d6c450cec873c33ecdab8a1f297cdcd9082de222d52` |
| `Phase3A_Evidence_Closure_Addendum_20261031.zip` | 37,500 | `3a33532c35045cb9a15eb7738e2bd8851c37304380b01cdf2b4cc7d6b671f7eb` |
| `Phase3_Analysis_Package_20261031.zip` | External | Offline statistical analysis package (reproduced by `scripts/analysis/analyze_phase3_sensitivity.py`) |

*Primary Benchmark Execution Manifest:* SHA-256 `3bdd37d228f2d045d082b3b8df4c1419c881c8d6c9fc9c8be68c85b4db614bee`.
