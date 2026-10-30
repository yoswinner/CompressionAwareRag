# Precision-Aware Query Planning for Vector Retrieval in PostgreSQL

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/PostgreSQL-18.4-336791.svg)](https://www.postgresql.org/)
[![Extension](https://img.shields.io/badge/pgvector-0.8.4-green.svg)](https://github.com/pgvector/pgvector)
[![Embedding](https://img.shields.io/badge/Ollama-nomic--embed--text-orange.svg)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/Tests-107%20passed-brightgreen.svg)](tests/)

An independent experimental research project investigating how **coordinate-level numerical quantization** (`halfvec` / IEEE 754 Float16) and **physical storage layout** (`STORAGE EXTENDED` vs. `STORAGE PLAIN`) alter PostgreSQL query planner cost estimates, access-path selection (HNSW index scan vs. sequential scan), execution latency, shared buffer access dynamics, and nearest-neighbor Recall@10 across varying corpus sizes and planner-cost configurations.

---

## Table of Contents
1. [Research Overview](#1-research-overview)
2. [Problem Statement and Motivation](#2-problem-statement-and-motivation)
3. [Research Questions](#3-research-questions)
4. [Main Contributions](#4-main-contributions)
5. [Experimental Design](#5-experimental-design)
6. [Three Vector Representations](#6-three-vector-representations)
7. [Three Corpus Sizes](#7-three-corpus-sizes)
8. [Nine Planner-Cost Profiles](#8-nine-planner-cost-profiles)
9. [Benchmark Methodology](#9-benchmark-methodology)
10. [Final Empirical Findings](#10-final-empirical-findings)
11. [Statistical Analysis Methodology](#11-statistical-analysis-methodology)
12. [Repository Structure](#12-repository-structure)
13. [Environment and Dependencies](#13-environment-and-dependencies)
14. [Installation and Configuration](#14-installation-and-configuration)
15. [Database Preparation and Safety](#15-database-preparation-and-safety)
16. [Benchmark Reproduction Commands](#16-benchmark-reproduction-commands)
17. [Test Execution](#17-test-execution)
18. [Documentation Index](#18-documentation-index)
19. [Limitations and Threats to Validity](#19-limitations-and-threats-to-validity)
20. [Research References](#20-research-references)

---

## 1. Research Overview

Modern Relational Database Management Systems (RDBMS) increasingly serve as vector databases for Retrieval-Augmented Generation (RAG) by integrating vector index extensions such as `pgvector`. PostgreSQL employs a cost-based query optimizer that models page fetch I/O ($random\_page\_cost$, $seq\_page\_cost$) and CPU computation ($cpu\_operator\_cost$, $cpu\_tuple\_cost$) to choose between index scans and sequential table scans.

However, standard embedding vectors (e.g., 768-dimensional single-precision floats) exceed PostgreSQL's page tuple threshold (~2 KiB), causing them to be evicted from main heap blocks into secondary TOAST relations. This eviction drastically lowers main heap page counts ($relpages$), which interacts with the heuristic cost estimation formulas inside `pgvector`'s `hnswcostestimate` function.

This research project conducts an exhaustive factorial benchmark (9,900 timed queries, 4,860 execution-free plan captures, and 540 buffer diagnostics) to quantify how vector precision, relational storage policy, and planner cost parameters interact to govern query plan selection and physical retrieval performance.

---

## 2. Problem Statement and Motivation

In PostgreSQL with `pgvector`, vector tables encounter two architectural phenomena:

1. **The TOAST Storage Inversion:** Under default `STORAGE EXTENDED`, a `vector(768)` column requires 3,076 bytes per tuple (exceeding `TOAST_TUPLE_THRESHOLD` on 8 KiB pages). Vectors are evicted out-of-line into `pg_toast` tables, resulting in a primary heap that occupies very few pages ($relpages \approx 64$ for 5,000 rows) while the TOAST relation occupies over 20 MB.
2. **Planner Heuristic Distortion:** In `pgvector` (`src/hnsw.c`), `hnswcostestimate` computes index startup costs based on an estimated number of graph entry pages ($startupPages$). If the main table heap page count ($relpages$) is smaller than $startupPages$—which occurs precisely when vectors are TOASTed out-of-line—the optimizer artificially clamps startup cost to the heap page count. In contrast, when vectors are stored inline (via `STORAGE PLAIN` or 16-bit `halfvec(768)`), $relpages$ remains high, preventing this discount from firing.

Consequently, the query planner may arrive at divergent execution plans (sequential scan vs. HNSW index scan) solely due to physical column storage layout and cost parameter settings.

---

## 3. Research Questions

- **RQ1 (Planner Indifference Frontier):** Under what combinations of $random\_page\_cost \in \{1.0, 4.0, 8.0\}$ and $cpu\_operator\_cost \in \{0.0025, 0.010, 0.050\}$ does PostgreSQL select a sequential scan over an available HNSW index scan for vector $K$-NN retrieval ($K=10$)?
- **RQ2 (Physical Storage & Representation Impact):** How does physical vector layout (`FP32-EXTENDED` vs. `FP32-PLAIN` vs. `FP16-EXTENDED`) alter heap and TOAST page counts, shifting the optimizer's path selection across corpus scales ($N \in \{1{,}001, 5{,}000, 10{,}000\}$)?
- **RQ3 (Latency, Buffer Footprint, and Accuracy):** What are the empirical latency trade-offs ($P_{50}, P_{95}$), shared buffer page hits, and Recall@10 accuracy consequences of planner-selected paths compared to forced HNSW index execution?

---

## 4. Main Contributions

1. **Quantification of Planner Decision Boundaries:** Proves that cost-profile-driven plan divergence occurs exclusively at small corpus sizes ($N=1{,}001$, where 220 sequential scans were selected across 1,620 plans). At larger corpus sizes ($N \ge 5{,}000$), the asymptotic cost disparity ensures HNSW index scans are selected 100% of the time across all cost configurations.
2. **Empirical Evaluation of Storage Policy:** Demonstrates that storing uncompressed FP32 vectors inline via `STORAGE PLAIN` eliminates out-of-line TOAST pointer dereferencing overhead during sequential scans (reducing 10k Seq Scan $P_{50}$ from 91.95 ms to 53.56 ms), while expanding main heap pages by $78\times$.
3. **Comprehensive Evaluation of Float16 Quantization:** Shows that 16-bit half-precision quantization (`halfvec(768)`) delivers a consistent $\approx 55.3\%$ total table footprint reduction and a $49.98\%$ HNSW index graph reduction while maintaining $>0.98$ conventional Recall@10 and $>0.99$ tie-aware Recall@10.
4. **Boundary-Tie Tolerance Sensitivity Formulation:** Formulates a disjoint-set boundary tie metric ($\text{Recall@10}_{\text{tie\_aware}}$) and proves stability across three orders of numerical tolerance ($\tau \in \{10^{-7}, 10^{-6}, 10^{-5}\}$).
5. **Open, Reproducible Experimental Artifacts:** Delivers a fully verified 107-test test suite and automated analysis scripts that reproduce all descriptive and inferential statistics with zero database writes.

---

## 5. Experimental Design

The benchmark implements a complete multi-factor matrix:

$$\text{3 Scales} \times \text{3 Representations} \times \text{9 Cost Profiles} \times \text{3 Modes} \times \text{20 Queries}$$

- **Corpus Scales:** 1,001, 5,000, and 10,000 chunks sampled from canonical MS MARCO text passages.
- **Fixed Query Set:** 20 pre-generated evaluation queries with 768-dimensional embeddings generated via `nomic-embed-text`.
- **Pass Structure:** 2 unmeasured warm-up passes followed by 5 measured passes per query.
- **Randomization:** Trial presentation order within each pass is randomized using a fixed seed (`42`).

---

## 6. Three Vector Representations

| Representation | PostgreSQL Type | Storage Policy | Datum Size | Inlined in Heap? | Index Operator Class |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **FP32 Extended** | `vector(768)` | `EXTENDED` | 3,076 B | No (TOASTed) | `vector_cosine_ops` |
| **FP32 Plain** | `vector(768)` | `PLAIN` | 3,076 B | Yes (2 tuples/page) | `vector_cosine_ops` |
| **FP16 Extended** | `halfvec(768)` | `EXTENDED` | 1,544 B | Yes (4 tuples/page) | `halfvec_cosine_ops` |

---

## 7. Three Corpus Sizes

1. **1,001 Chunks:** Evaluates the small-scale planner sensitivity frontier where index startup cost and sequential page cost are comparable.
2. **5,000 Chunks:** Intermediate scale where index selectivity overtakes sequential page read costs.
3. **10,000 Chunks:** Production-scale dataset evaluating memory bus saturation, buffer page cache turnover, and asymptotic retrieval behavior.

---

## 8. Nine Planner-Cost Profiles

Planner costs are relative dimensionless units normalized to $seq\_page\_cost = 1.0$:

| Profile ID | $random\_page\_cost$ | $cpu\_operator\_cost$ | $seq\_page\_cost$ | $cpu\_tuple\_cost$ | Configuration Focus |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | 1.0 | 0.0025 | 1.0 | 0.01 | Low random I/O cost, default CPU operator cost |
| **P2** | 1.0 | 0.0100 | 1.0 | 0.01 | Low random I/O cost, elevated CPU operator cost |
| **P3** | 1.0 | 0.0500 | 1.0 | 0.01 | Low random I/O cost, high CPU operator cost |
| **P4** | 4.0 | 0.0025 | 1.0 | 0.01 | **Canonical Profile** (PostgreSQL 18 default settings) |
| **P5** | 4.0 | 0.0100 | 1.0 | 0.01 | Standard random I/O cost, elevated CPU operator cost |
| **P6** | 4.0 | 0.0500 | 1.0 | 0.01 | Standard random I/O cost, high CPU operator cost |
| **P7** | 8.0 | 0.0025 | 1.0 | 0.01 | High random I/O cost, default CPU operator cost |
| **P8** | 8.0 | 0.0100 | 1.0 | 0.01 | High random I/O cost, elevated CPU operator cost |
| **P9** | 8.0 | 0.0500 | 1.0 | 0.01 | High random I/O cost, high CPU operator cost |

*Fixed parameters across all profiles:* `hnsw.ef_search = 40`, `LIMIT = 10`, `max_parallel_workers_per_gather = 0`.

---

## 9. Benchmark Methodology

Execution is cleanly decoupled into three phases:

1. **Phase A (Execution-Free Plan Capture):** Query plans are captured across all 9 profiles via `EXPLAIN (SETTINGS, FORMAT JSON)` inside read-only transactions, evaluating optimizer cost models without executing queries (4,860 captures).
2. **Phase B (Wall-Clock Latency Measurement):** Parameterized `SELECT` queries executed with client-side `time.perf_counter()` timing. All connections set `prepare_threshold=None` and `prepare=False`, forcing PostgreSQL to re-plan every query and preventing stale prepared statement caching (9,900 timed queries).
3. **Phase C (Diagnostic Buffer Inspection):** Detailed execution plans captured via `EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON)` under Profile P4. Buffer usage is extracted from the root plan node to prevent recursive over-counting (540 diagnostic runs).

---

## 10. Final Empirical Findings

### 10.1 Physical Storage Reductions
- `fp16_extended` reduces total inclusive table footprint by **55.21%** (1,001 scale), **55.35%** (5,000 scale), and **55.37%** (10,000 scale) compared to `fp32_extended`.
- HNSW index graph size is halved: at 10,000 scale, `fp16_extended` requires **20.40 MB** versus **40.78 MB** for `fp32_extended` (**49.98% reduction**).
- `fp32_plain` reduces total size by $\approx 1.2\%\text{--}1.7\%$ by eliminating TOAST chunk header overhead.

### 10.2 Planner Plan Selection Breakdown
- Total planner-selected plans: **1,620**
  - **HNSW Index Scan:** 1,400 plans (**86.4%**)
  - **Sequential Scan (Seq Scan):** 220 plans (**13.6%**)
- **Scale-Dependent Finding:** All 220 sequential scans occurred at the **1,001-row scale**. At scales 5,000 and 10,000, the planner selected HNSW Index Scan **100% of the time (540/540 plans at each scale)** across all 9 cost profiles and all representations.

### 10.3 Canonical Profile P4 Latency Summary
Measured over 100 executions per condition (5 passes $\times$ 20 queries):

| Scale | Representation | Mode | P50 Latency | P95 Latency | Mean $\pm$ Std |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1,001** | `fp16_extended` | Exact (Seq Scan) | 1.699 ms | 2.790 ms | 1.845 $\pm$ 0.428 ms |
| **1,001** | `fp16_extended` | Forced HNSW | 1.591 ms | 2.635 ms | 1.736 $\pm$ 0.417 ms |
| **1,001** | `fp16_extended` | Planner Selected | 1.674 ms | 3.045 ms | 1.857 $\pm$ 0.511 ms |
| **1,001** | `fp32_extended` | Exact (Seq Scan) | 5.127 ms | 10.495 ms | 6.171 $\pm$ 6.542 ms |
| **1,001** | `fp32_extended` | Forced HNSW | 1.779 ms | 2.971 ms | 1.998 $\pm$ 1.128 ms |
| **1,001** | `fp32_extended` | Planner Selected | 1.701 ms | 2.630 ms | 1.825 $\pm$ 0.372 ms |
| **5,000** | `fp16_extended` | Forced HNSW | 1.971 ms | 3.852 ms | 2.241 $\pm$ 0.799 ms |
| **5,000** | `fp32_extended` | Forced HNSW | 1.866 ms | 3.499 ms | 2.223 $\pm$ 1.134 ms |
| **5,000** | `fp32_plain` | Forced HNSW | 1.867 ms | 3.597 ms | 2.128 $\pm$ 0.657 ms |
| **10,000** | `fp16_extended` | Forced HNSW | **2.479 ms** | 4.503 ms | 2.776 $\pm$ 1.048 ms |
| **10,000** | `fp32_extended` | Forced HNSW | 2.716 ms | 8.524 ms | 3.847 $\pm$ 4.106 ms |
| **10,000** | `fp32_plain` | Forced HNSW | 2.881 ms | 7.058 ms | 3.550 $\pm$ 1.880 ms |
| **10,000** | `fp16_extended` | Exact (Seq Scan) | 12.770 ms | 28.180 ms | 14.252 $\pm$ 6.939 ms |
| **10,000** | `fp32_plain` | Exact (Seq Scan) | 53.556 ms | 70.400 ms | 52.800 $\pm$ 10.872 ms |
| **10,000** | `fp32_extended` | Exact (Seq Scan) | 91.954 ms | 126.625 ms | 93.638 $\pm$ 23.979 ms |

### 10.4 Retrieval Quality & Boundary-Tie Sensitivity
- **Conventional Recall@10:** 0.9914 (1,001 scale), 0.9605 (5,000 scale), 0.9394 (10,000 scale).
- **Tie-Aware Recall@10:** 0.9971 (1,001 scale), 0.9847 (5,000 scale), 0.9758 (10,000 scale).
- **Tolerance Robustness:** Recall values are numerically invariant across three orders of magnitude ($\tau \in \{10^{-7}, 10^{-6}, 10^{-5}\}$).

### 10.5 Corrected Root Buffer Consumption
In exact sequential scan at 10,000 scale, `fp32_extended` touches **30,064 shared buffer pages** (due to TOAST dereferencing), whereas `fp16_extended` touches only **1,906.5 buffer pages**. Under HNSW index scan, all representations touch only **640 to 745 pages**.

---

## 11. Statistical Analysis Methodology

Inferential tests are conducted on the 20 query-level experimental units ($N=20$) with repeated passes averaged per query:

1. **Friedman Test Across 9 Profiles:**
   - At Scale 1,001: Statistically significant latency differences across profiles ($p = 3.21 \times 10^{-8}$ for `fp32_extended`, $p = 1.40 \times 10^{-3}$ for `fp32_plain`, $p = 4.93 \times 10^{-2}$ for `fp16_extended`) because profiles shift the plan choice between Seq Scan and HNSW Index Scan.
   - At Scales 5,000 and 10,000: No statistically significant differences across profiles ($p > 0.20$) because the planner selects HNSW Index Scan 100% of the time.
2. **Paired Wilcoxon Signed-Rank Tests (Profile P4, Forced HNSW):**
   - At Scale 10,000: `fp16_extended` vs. `fp32_plain` achieves statistically significant latency reduction ($W=28.0$, Holm-adjusted $p = 0.0257$, Cliff's $\delta = -0.47$).
   - `fp16_extended` vs. `fp32_extended` shows directional speedup with small-to-medium effect size (Cliff's $\delta = -0.33$, Holm-adjusted $p = 0.0888$).

---

## 12. Repository Structure

```
CompressionAwareRag/
├── README.md                                  # Research project overview and landing page
├── pytest.ini                                 # Pytest configuration
├── requirements.txt                           # Python project dependencies
├── .env.example                               # Environment configuration template
├── .gitignore                                 # Git exclusion rules (results & archives ignored)
├── data/raw/                                  # Canonical raw text datasets and query metadata
├── docs/                                      # Authoritative research documentation
│   ├── planner_sensitivity_protocol.md        # Formal experimental & reproducibility protocol
│   ├── research_results.md                    # Consolidated empirical results & statistical synthesis
│   └── reproducibility.md                     # Step-by-step replication and verification guide
├── scripts/                                   # Database provisioning and verification scripts
│   ├── provision_planner_sensitivity.py       # Atomic table and index provisioning
│   ├── verify_table_readiness.py              # Strictly read-only schema readiness audit
│   └── analysis/                              # Dedicated research analysis and reproduction scripts
│       ├── analyze_phase3_sensitivity.py      # Comprehensive statistical synthesis & tests
│       ├── run_phase3a_tie_sensitivity.py     # Boundary-tie tolerance sensitivity analysis
│       └── run_phase3a_addendum_closure.py    # Evidence chain closure and GUC equivalence audit
├── src/                                       # Core experimental framework and query execution engine
│   ├── planner_sensitivity_experiment.py      # Multi-profile 9,900-query benchmark runner
│   ├── benchmark.py                           # Baseline benchmarking utilities
│   ├── config.py                              # Configuration management
│   ├── database.py                            # PostgreSQL connection factory
│   ├── embeddings.py                          # Ollama vector generation client
│   └── representations.py                     # Vector formatting and quantization helpers
└── tests/                                     # Complete automated test suite (107 tests passing)
    ├── test_basic.py                          # Basic connection and schema tests
    ├── test_hnsw_experiment.py                # HNSW parameter and execution tests
    ├── test_ingestion_cache.py                # Ingestion caching tests
    ├── test_planner_sensitivity_experiment.py # Planner sensitivity harness tests
    ├── test_representations.py                # Vector representation tests
    └── test_scale_experiment.py               # Multi-scale integrity tests
```

---

## 13. Environment and Dependencies

- **PostgreSQL:** 18.4 (x86_64)
- **pgvector:** 0.8.4
- **Python:** 3.13.14
- **Driver:** `psycopg[binary] >= 3.1.0`
- **Embedding Client:** Ollama 0.5+ (`nomic-embed-text:latest`)

Install requirements:
```bash
pip install -r requirements.txt
```

---

## 14. Installation and Configuration

1. Clone repository:
   ```bash
   git clone -b research/planner-sensitivity https://github.com/yoswinner/CompressionAwareRag.git
   cd CompressionAwareRag
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy environment configuration:
   ```bash
   cp .env.example .env
   # Edit .env with local PostgreSQL credentials
   ```

---

## 15. Database Preparation and Safety

The experiment enforces strict catalog isolation:
- Experimental tables reside exclusively in schema `planner_sensitivity` inside `compression_aware_rag_exp` ($N=1{,}001$) and `compression_aware_rag_scale` ($N=5{,}000; 10{,}000$).
- Baseline catalog `compression_aware_rag` is protected with fail-closed assertions.
- All benchmark connections enforce `prepare_threshold=None` and execute with explicit `BEGIN READ ONLY` and `ROLLBACK` blocks.

Provision tables:
```bash
python scripts/provision_planner_sensitivity.py --scale 1001 --scale 5000 --scale 10000 --apply
```

Audit readiness (read-only):
```bash
python scripts/verify_table_readiness.py --scale all
```

---

## 16. Benchmark Reproduction Commands

Execute the full 9,900-query benchmark:
```bash
python src/planner_sensitivity_experiment.py --scale all --representation all --canonical-profile P4 --passes 5 --warmup 2
```

Run independent statistical analysis:
```bash
python scripts/analysis/analyze_phase3_sensitivity.py
```

Run boundary-tie sensitivity analysis:
```bash
python scripts/analysis/run_phase3a_tie_sensitivity.py
```

Run evidence closure audit:
```bash
python scripts/analysis/run_phase3a_addendum_closure.py
```

---

## 17. Test Execution

Run the complete 107-test test suite:
```bash
pytest -v
```
All 107 tests pass in approximately 13 seconds.

---

## 18. Documentation Index

- **[Experimental Protocol](docs/planner_sensitivity_protocol.md):** Complete formal protocol, parameter matrix, mathematical recall formulas, and safety guarantees.
- **[Empirical Results](docs/research_results.md):** Detailed numerical tables, storage metrics, plan selections, latency percentiles, and non-parametric statistical tests.
- **[Reproducibility Guide](docs/reproducibility.md):** Step-by-step replication instructions and cryptographic evidence verification.

---

## 19. Limitations and Threats to Validity

1. **Query Sample Size ($N=20$):** Evaluation utilized 20 representative MS MARCO semantic queries with 5 repeated timing passes (100 observations per condition). Broader query suites may exhibit different distribution profiles.
2. **Warm-Cache Operating Regime:** Benchmark evaluated in-memory buffer cache execution to assess algorithmic and planner behavior. Cold disk I/O was not evaluated.
3. **Discrete Corpus Scales:** Findings reflect the evaluated discrete corpus scales ($N=1{,}001, 5{,}000, 10{,}000$). Asymptotic crossover boundaries between 1,001 and 5,000 rows were not continuously interpolated.
4. **No Direct SIMD Claims:** Speedup in `halfvec` operations is attributed to reduced memory bus traffic and higher page density; no claims of hardware-specific SIMD execution are asserted without low-level performance counter profiling.

---

## 20. Research References

1. Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 42.4 (2018): 824-836.
2. pgvector contributors. "pgvector: Open-source vector similarity search for PostgreSQL." https://github.com/pgvector/pgvector (2024).
3. PostgreSQL Global Development Group. "PostgreSQL 18 Documentation: Query Planning and Cost Estimation." https://www.postgresql.org/docs/18/
4. Lewis, P., et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks." *Advances in Neural Information Processing Systems* 33 (2020): 9459-9474.
5. Conover, W. J. *Practical Nonparametric Statistics*. 3rd ed. John Wiley & Sons, 1999.
