# Precision-Aware Query Planning for Vector Retrieval in PostgreSQL
## Comprehensive Empirical Research Results & Statistical Synthesis

**Author:** Yash Pradhan  
**Document Type:** Version-Controlled Empirical Research Synthesis  
**Target Venue:** IEEE Transactions on Knowledge and Data Engineering (TKDE) / IEEE International Conference on Data Engineering (ICDE)  
**Repository Branch:** `research/planner-sensitivity`  
**Base Benchmark Commit:** `3f2beee66cecd67aa8fc4cec69c364b92c07b884`  
**Final Audit Commit:** `8dfdce0a00338d0c6e0c4f70ea3e83d40e9b93ee`  

---

## 1. Research Questions & Hypotheses

This research investigates how vector precision quantization and relational physical storage policies alter PostgreSQL query planning and execution for Approximate Nearest Neighbor (ANN) vector retrieval:

- **RQ1 (Planner Indifference Frontier):** Under what combinations of planner cost settings ($random\_page\_cost \in \{1.0, 4.0, 8.0\}$ and $cpu\_operator\_cost \in \{0.0025, 0.010, 0.050\}$) does PostgreSQL elect a brute-force sequential scan with top-$N$ sort over an HNSW index for vector retrieval ($K=10$)?
- **RQ2 (Physical Storage Policy & Quantization Impact):** How does physical tuple layout (`FP32-EXTENDED` vs. `FP32-PLAIN` vs. `FP16-EXTENDED`) alter heap and TOAST page counts, shifting the optimizer's startup cost calculations and path selections across dataset scales ($N \in \{1{,}001, 5{,}000, 10{,}000\}$)?
- **RQ3 (Latency, Buffer Footprint, and Recall Accuracy):** What are the empirical execution time penalties ($P_{50}, P_{95}$), shared buffer page hits, and Recall@10 consequences of planner-selected paths compared to forced HNSW index scans?

### Hypotheses
1. **$H_1$ (Storage-Induced Plan Divergence):** Out-of-line TOASTing of `vector(768)` under `STORAGE EXTENDED` artificially lowers main heap pages ($relpages$), causing the `hnswcostestimate` startup-cost heuristic to discount index startup cost, while inline storage (`STORAGE PLAIN` or `halfvec(768)`) inflates heap page counts.
2. **$H_2$ (Scale-Dependent Plan Convergence):** At small corpus scales ($N=1{,}001$), planner cost profiles heavily influence path choice (HNSW vs. Seq Scan). At larger corpus scales ($N \ge 5{,}000$), the asymptotic cost disparity between $O(N)$ sequential distance evaluations and $O(\log N)$ HNSW graph traversals dominates, causing the planner to select HNSW index scans 100% of the time across all cost configurations.
3. **$H_3$ (Storage and Retrieval Parity of FP16):** Half-precision 16-bit quantization (`halfvec(768)`) yields $\approx 55\%$ total storage savings while maintaining $>0.98$ conventional Recall@10 and $>0.99$ tie-aware Recall@10 across all scales.

---

## 2. Experimental Design Matrix

The completed experimental evaluation spans a fully factorial experimental design across corpus scales, data representations, planner cost profiles, and execution modes:

```
3 Scales x 3 Representations x 9 Planner Cost Profiles x 3 Execution Modes x 20 Fixed Queries
```

### 2.1 The Three Vector Representations
1. **`FP32 Extended` (`vector(768)`, `STORAGE EXTENDED`):** Default PostgreSQL representation. Coordinates stored as 32-bit single-precision floats (3,076 bytes per tuple). Tuples exceed the 2 KiB page threshold and are moved out-of-line into TOAST relations.
2. **`FP32 Plain` (`vector(768)`, `STORAGE PLAIN`):** Inline storage policy enforced via `ALTER TABLE ... SET STORAGE PLAIN`. Forces 3,076-byte vectors into main heap blocks (fitting exactly 2 tuples per 8 KiB page), avoiding TOAST overhead while expanding main heap pages.
3. **`FP16 Extended` (`halfvec(768)`, `STORAGE EXTENDED`):** IEEE 754 16-bit half-precision floating-point coordinates (1,544 bytes per tuple). Sits naturally below the 2 KiB page threshold, remaining 100% inline within main heap blocks.

### 2.2 The Nine Planner Cost Profiles

Cost parameters are dimensionless relative units normalized to $seq\_page\_cost = 1.0$:

| Profile ID | $random\_page\_cost$ | $cpu\_operator\_cost$ | $seq\_page\_cost$ | $cpu\_tuple\_cost$ | Description |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | 1.0 | 0.0025 | 1.0 | 0.01 | Low I/O cost, pgvector default operator cost |
| **P2** | 1.0 | 0.0100 | 1.0 | 0.01 | Low I/O cost, elevated CPU operator cost |
| **P3** | 1.0 | 0.0500 | 1.0 | 0.01 | Low I/O cost, high CPU operator cost |
| **P4** | 4.0 | 0.0025 | 1.0 | 0.01 | **Canonical Profile** (PostgreSQL 18 default settings) |
| **P5** | 4.0 | 0.0100 | 1.0 | 0.01 | Standard I/O cost, elevated CPU operator cost |
| **P6** | 4.0 | 0.0500 | 1.0 | 0.01 | Standard I/O cost, high CPU operator cost |
| **P7** | 8.0 | 0.0025 | 1.0 | 0.01 | High I/O cost, pgvector default operator cost |
| **P8** | 8.0 | 0.0100 | 1.0 | 0.01 | High I/O cost, elevated CPU operator cost |
| **P9** | 8.0 | 0.0500 | 1.0 | 0.01 | High I/O cost, high CPU operator cost |

- **Invariants:** `hnsw.ef_search = 40`, `max_parallel_workers_per_gather = 0`, `LIMIT = 10`, `HNSW parameters: m=16, ef_construction=64`.
- **Query Set:** 20 pre-generated MS MARCO semantic queries with precomputed 768-dimensional embeddings.
- **Passes:** 2 unmeasured warm-up passes, 5 measured passes per query.

### 2.3 Workload Execution Accounting
- **Timed Executions:** 9,900 queries (8,100 planner-selected + 900 forced exact + 900 forced HNSW).
- **Execution-Free Plan Captures:** 4,860 JSON plan captures via `EXPLAIN (SETTINGS, FORMAT JSON)`.
- **Diagnostic Executions:** 540 detailed runs via `EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON)` under Profile P4.

---

## 3. Environment & Provenance Information

### 3.1 Host & Database Environment
- **Operating System:** Windows 11 Enterprise (x86_64, 64-bit).
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
1. **TOAST Inversion:** Under `STORAGE EXTENDED`, `fp32_extended` delegates $99.8\%$ of table data to TOAST relations (e.g. 41.43 MB TOAST vs 524 KB heap at 10k). In contrast, `fp32_plain` forces all 40.96 MB into heap blocks.
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

**Key Finding:** Sequential scans were chosen **exclusively** at the 1,001-row scale. At scales 5,000 and 10,000, the PostgreSQL query planner selected HNSW Index Scan 100% of the time across all 9 profiles and all representations.

---

## 6. Execution Latency Results

Descriptive statistics are computed over the 100 timed executions per condition (5 measured passes $\times$ 20 queries) under canonical Profile P4 ($random\_page\_cost = 4.0, cpu\_operator\_cost = 0.0025$):

### 6.1 Profile P4 Latency Summary

| Scale | Representation | Mode | $N$ | P50 (ms) | P95 (ms) | Mean (ms) | Std Dev (ms) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1,001** | `fp16_extended` | Exact (Seq Scan) | 100 | 1.699 | 2.790 | 1.845 | 0.428 |
| **1,001** | `fp16_extended` | Forced HNSW | 100 | 1.591 | 2.635 | 1.736 | 0.417 |
| **1,001** | `fp16_extended` | Planner Selected | 100 | 1.674 | 3.045 | 1.857 | 0.511 |
| **1,001** | `fp32_extended` | Exact (Seq Scan) | 100 | 5.127 | 10.495 | 6.171 | 6.542 |
| **1,001** | `fp32_extended` | Forced HNSW | 100 | 1.779 | 2.971 | 1.998 | 1.128 |
| **1,001** | `fp32_extended` | Planner Selected | 100 | 1.701 | 2.630 | 1.825 | 0.372 |
| **1,001** | `fp32_plain` | Exact (Seq Scan) | 100 | 1.899 | 3.429 | 2.131 | 0.603 |
| **1,001** | `fp32_plain` | Forced HNSW | 100 | 1.670 | 3.144 | 1.917 | 0.673 |
| **1,001** | `fp32_plain` | Planner Selected | 100 | 1.870 | 3.413 | 2.103 | 0.582 |
| **5,000** | `fp16_extended` | Exact (Seq Scan) | 100 | 4.504 | 9.206 | 4.883 | 2.010 |
| **5,000** | `fp16_extended` | Forced HNSW | 100 | 1.971 | 3.852 | 2.241 | 0.799 |
| **5,000** | `fp16_extended` | Planner Selected | 100 | 1.824 | 3.732 | 2.164 | 0.905 |
| **5,000** | `fp32_extended` | Exact (Seq Scan) | 100 | 29.416 | 53.224 | 36.027 | 13.062 |
| **5,000** | `fp32_extended` | Forced HNSW | 100 | 1.866 | 3.499 | 2.223 | 1.134 |
| **5,000** | `fp32_extended` | Planner Selected | 100 | 1.853 | 3.358 | 2.080 | 0.585 |
| **5,000** | `fp32_plain` | Exact (Seq Scan) | 100 | 6.182 | 10.486 | 6.968 | 2.738 |
| **5,000** | `fp32_plain` | Forced HNSW | 100 | 1.867 | 3.597 | 2.128 | 0.657 |
| **5,000** | `fp32_plain` | Planner Selected | 100 | 1.870 | 3.246 | 2.127 | 0.636 |
| **10,000** | `fp16_extended` | Exact (Seq Scan) | 100 | 12.770 | 28.180 | 14.252 | 6.939 |
| **10,000** | `fp16_extended` | Forced HNSW | 100 | 2.479 | 4.503 | 2.776 | 1.048 |
| **10,000** | `fp16_extended` | Planner Selected | 100 | 2.450 | 6.147 | 2.906 | 1.449 |
| **10,000** | `fp32_extended` | Exact (Seq Scan) | 100 | 91.954 | 126.625 | 93.638 | 23.979 |
| **10,000** | `fp32_extended` | Forced HNSW | 100 | 2.716 | 8.524 | 3.847 | 4.106 |
| **10,000** | `fp32_extended` | Planner Selected | 100 | 2.811 | 5.606 | 3.150 | 1.493 |
| **10,000** | `fp32_plain` | Exact (Seq Scan) | 100 | 53.556 | 70.400 | 52.800 | 10.872 |
| **10,000** | `fp32_plain` | Forced HNSW | 100 | 2.881 | 7.058 | 3.550 | 1.880 |
| **10,000** | `fp32_plain` | Planner Selected | 100 | 2.785 | 7.169 | 3.433 | 1.909 |

### 6.2 Latency Insights
1. **TOAST Overhead in Sequential Scan:** In exact sequential scan, `fp32_extended` incurs severe latency penalties due to out-of-line TOAST pointer dereferencing (91.95 ms P50 at 10k vs 53.56 ms P50 for `fp32_plain` and 12.77 ms P50 for `fp16_extended`).
2. **HNSW Convergence:** Under HNSW index scans, median latency remains between 1.6 ms and 2.9 ms across all representations and scales.
3. **FP16 Advantage at Scale:** At 10,000 rows, `fp16_extended` achieves the lowest P50 latency (2.45 ms planner-selected, 2.48 ms forced HNSW).

---

## 7. Retrieval Accuracy & Boundary-Tie Sensitivity

Retrieval accuracy was audited using the complete set of 1,980 query-condition units against the ground-truth top-10 FP32 reference set:

### 7.1 Sensitivity Across Numerical Tolerances ($\tau$)

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

### 7.2 The Six Boundary-Tie Queries
Exact sequential scan analysis identified six specific queries exhibiting boundary distance ties across the 20-query evaluation suite:
1. **Scale 1,001, Query 5:** Boundary distance $0.473267$; 4 chunks tied for 1 boundary slot ($|B|=9, |T|=4$).
2. **Scale 1,001, Query 13:** Boundary distance $0.451638$; 4 chunks tied for 3 boundary slots ($|B|=7, |T|=4$).
3. **Scale 1,001, Query 16:** Boundary distance $0.491378$; 4 chunks tied for 3 boundary slots ($|B|=7, |T|=4$).
4. **Scale 5,000, Query 3:** Boundary distance $0.406328$; 12 chunks tied for 8 boundary slots ($|B|=2, |T|=12$).
5. **Scale 5,000, Query 9:** Boundary distance $0.443279$; 12 chunks tied for 10 boundary slots ($|B|=0, |T|=12$).
6. **Scale 5,000, Query 17:** Boundary distance $0.452952$; 2 chunks tied for 1 boundary slot ($|B|=9, |T|=2$) at $\tau = 10^{-5}$.
*(At Scale 10,000, Query 9 exhibits 28 tied chunks for 8 boundary slots).*

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

### 9.1 Friedman Test Across 9 Cost Profiles (P1 – P9)
Tests whether planner cost profiles produce statistically significant latency differences under planner-selected execution:

| Scale | Representation | Friedman $Q$ Stat | Degrees of Freedom ($df$) | $p$-value | Significance ($\alpha = 0.05$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1,001** | `fp32_extended` | 50.547 | 8 | **$3.21 \times 10^{-8}$** | **Statistically Significant** |
| **1,001** | `fp32_plain` | 25.267 | 8 | **$1.40 \times 10^{-3}$** | **Statistically Significant** |
| **1,001** | `fp16_extended` | 15.547 | 8 | **$4.93 \times 10^{-2}$** | **Statistically Significant** |
| **5,000** | `fp32_extended` | 8.160 | 8 | 0.418 | Not Significant |
| **5,000** | `fp32_plain` | 6.000 | 8 | 0.647 | Not Significant |
| **5,000** | `fp16_extended` | 9.467 | 8 | 0.304 | Not Significant |
| **10,000** | `fp32_extended` | 5.653 | 8 | 0.686 | Not Significant |
| **10,000** | `fp32_plain` | 6.720 | 8 | 0.567 | Not Significant |
| **10,000** | `fp16_extended` | 10.507 | 8 | 0.231 | Not Significant |

**Inference:** At $N=1{,}001$, planner cost parameters induce significant plan shifts between Seq Scan and HNSW Index Scan ($p < 0.05$). At $N \ge 5{,}000$, cost profile changes produce no plan alterations (HNSW index scan is chosen 100% of the time), resulting in no statistically significant latency variance across profiles ($p > 0.20$).

### 9.2 Paired Wilcoxon Signed-Rank Tests (Profile P4, Forced HNSW)
Comparing FP16 vs. FP32 representations across the 20 queries with Holm-Bonferroni correction ($\alpha = 0.05$):

| Scale | Comparison | $W$ Stat | Raw $p$-value | Holm-Adjusted $p$ | Cliff's $\delta$ | Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10,000** | FP16 Ext vs. FP32 Plain | 28.0 | $4.29 \times 10^{-3}$ | **$2.57 \times 10^{-2}$** | $-0.47$ (Medium) | **Statistically Significant** |
| **10,000** | FP16 Ext vs. FP32 Ext | 41.0 | $1.78 \times 10^{-2}$ | $8.88 \times 10^{-2}$ | $-0.33$ (Small) | Not Significant |
| **1,001** | FP16 Ext vs. FP32 Plain | 43.0 | $2.17 \times 10^{-2}$ | $8.67 \times 10^{-2}$ | $-0.39$ (Medium) | Not Significant |
| **1,001** | FP16 Ext vs. FP32 Ext | 55.0 | $6.46 \times 10^{-2}$ | $1.94 \times 10^{-1}$ | $-0.39$ (Medium) | Not Significant |
| **5,000** | FP16 Ext vs. FP32 Plain | 91.0 | $6.14 \times 10^{-1}$ | 1.000 | $+0.13$ (Negligible) | Not Significant |
| **5,000** | FP16 Ext vs. FP32 Ext | 98.0 | $8.08 \times 10^{-1}$ | $8.08 \times 10^{-1}$ | $+0.08$ (Negligible) | Not Significant |

---

## 10. Research Limitations & Validity Threats

1. **Query Sample Size ($N=20$):** Evaluation utilized 20 representative MS MARCO semantic queries. While multi-pass repeated timing yielded 100 observations per condition, semantic variance across broader query workloads requires further evaluation.
2. **Warm-Cache Operating Regime:** Benchmark executions evaluated warmed buffer-cache behavior to measure algorithmic and planner efficiency. Cold-cache I/O from spinning disks or cold cloud volumes was not benchmarked.
3. **Single-Host Environment:** Experiments ran on a dedicated Windows workstation under PostgreSQL 18.4. Distributed PostgreSQL architectures (e.g. Citus) were not evaluated.
4. **ANN-Only Scope:** Retrieval quality focused on vector similarity Recall@10. Downstream generative LLM answer fidelity (RAG answer quality) was not evaluated.
5. **No Universal Hardware Claims:** Cost profiles represent query planner cost adjustments, not physical hardware emulation. No claims are made regarding physical SSD or NVMe speedup parity.

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
