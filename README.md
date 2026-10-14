# Compression-Aware Vector Database Indexing for RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/PostgreSQL-18.0%20(tested)-336791.svg)](https://www.postgresql.org/)
[![Extension](https://img.shields.io/badge/pgvector-0.5.0%20(tested)-green.svg)](https://github.com/pgvector/pgvector)
[![Embeddings](https://img.shields.io/badge/Ollama-nomic--embed--text-orange.svg)](https://ollama.com/)

An experimental investigation into **coordinate-level numerical quantization** (`halfvec` / IEEE 754 Float16) and **Hierarchical Navigable Small World (HNSW)** vector graph indexing in **PostgreSQL / pgvector**, evaluating physical storage layout, query buffer access dynamics, exact and approximate retrieval latency, and nearest-neighbor recall trade-offs for Retrieval-Augmented Generation (RAG) workloads.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [The DBMS Pathology: The TOAST Storage Inversion](#the-dbms-pathology-the-toast-storage-inversion)
- [Research Objectives & Scope](#research-objectives--scope)
- [What This Repository Implements & Evaluates](#what-this-repository-implements--evaluates)
- [Technical Architecture & Data Layout](#technical-architecture--data-layout)
- [Experimental Methodology & Benchmark Design](#experimental-methodology--benchmark-design)
- [Master Empirical Results Summary](#master-empirical-results-summary)
- [Important Trade-offs & Interpretative Boundaries](#important-trade-offs--interpretative-boundaries)
- [Repository Structure](#repository-structure)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [Configuration & Dataset Preparation](#configuration--dataset-preparation)
- [Running the Code & Test Suites](#running-the-code--test-suites)
- [Documentation Index & Benchmark Reports](#documentation-index--benchmark-reports)
- [Limitations & Future Research Roadmap](#limitations--future-research-roadmap)
- [Acknowledgements & References](#acknowledgements--references)

---

## Executive Overview

Retrieval-Augmented Generation (RAG) pipelines increasingly deploy relational databases with vector extensions—chiefly PostgreSQL with `pgvector`—to co-locate enterprise metadata with high-dimensional dense embeddings. Standard embedding models (such as `nomic-embed-text` with 768 dimensions) output 32-bit single-precision floating-point arrays (`vector` in pgvector).

In relational storage engines like PostgreSQL, storing 768-dimensional FP32 vectors introduces a severe physical storage pathology: **out-of-line TOAST (The Oversized-Attribute Storage Technique) eviction**. Because each FP32 datum exceeds PostgreSQL's ~2 KiB page-tuple threshold, vectors are evicted from the primary heap into secondary TOAST tables (`pg_toast_<oid>`), creating a **relational storage inversion** where secondary tables dwarf the primary heap by over 70x.

This project implements and experimentally benchmarks coordinate-level 16-bit half-precision quantization (`halfvec`), halving coordinate datum size (from 3,076 bytes to 1,544 bytes) to bring vectors below the TOAST threshold. Across evaluated scales from 1,001 to 10,000 chunks, inlining FP16 vectors into heap pages:
1. **Eliminates 100% of the out-of-line TOAST data payload in the evaluated schema** (0 bytes data payload vs. 40,960,000 bytes / 39.06 MiB for FP32 at 10k, retaining an empty 8 KiB TOAST index shell).
2. **Reduces total inclusive relation footprint by 55.4%** and HNSW index footprint by **49.98%**.
3. **Reports 93.35% lower Query 0 (Q0) shared-buffer access counts**, consistent with observed **6.22x–7.24x mean (7.12x–8.03x median) exact search speedups**.
4. **Preserves 96.0% Recall@10** at $ef_{search} = 40$ relative to exhaustive uncompressed FP32 ground truth, maintaining $\ge 95\%$ pairwise top-$K$ candidate-set agreement between FP32 and FP16 HNSW.

---

## The DBMS Pathology: The TOAST Storage Inversion

PostgreSQL divides physical table storage into fixed-size **8 KiB page blocks** (`BLCKSZ = 8192`). Tuples cannot span physical pages. To prevent large attributes from monopolizing block space:
- `TOAST_TUPLE_THRESHOLD` triggers out-of-line processing when tuple size exceeds approximately 2 KiB.
- `TOAST_TUPLE_TARGET` defines the target tuple length after compression or out-of-line eviction attempts (~2 KiB).

```
   PostgreSQL 8 KiB Page Block
  ┌───────────────────────────────────────────────────────────────┐
  │ Page Header & Item Pointers (ItemIdData)                      │
  ├───────────────────────────────────────────────────────────────┤
  │ [FP32 Tuple]                                                  │
  │   - id, chunk_id, metadata (Inline in Heap)                   │
  │   - embedding (3,076 B) > ~2 KiB Threshold ──► EVICTED TOAST! │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
           Secondary TOAST Relation Fork (pg_toast_<oid>)
          ┌───────────────────────────────────────────────┐
          │ Chunk 0 (2 KiB slice)                         │
          │ Chunk 1 (1 KiB slice)                         │
          │ (39.06 MiB across 5,000 TOAST pages at 10k!)  │
          └───────────────────────────────────────────────┘

  vs.

   PostgreSQL 8 KiB Page Block (FP16 halfvec)
  ┌───────────────────────────────────────────────────────────────┐
  │ Page Header & Item Pointers (ItemIdData)                      │
  ├───────────────────────────────────────────────────────────────┤
  │ [FP16 Tuple]                                                  │
  │   - id, chunk_id, metadata (Inline in Heap)                   │
  │   - embedding (1,544 B) < ~2 KiB Threshold ──► INLINED HEAP!  │
  ├───────────────────────────────────────────────────────────────┤
  │ [FP16 Tuple 2] (Inlined)                                      │
  ├───────────────────────────────────────────────────────────────┤
  │ [FP16 Tuple 3] (Inlined)                                      │
  └───────────────────────────────────────────────────────────────┘
     Zero Out-of-Line TOAST Data Payload (0 Bytes across tested scales)
```

### The Measured Storage Layout at 10,000 Chunks
In the evaluated dedicated vector-only schema at 10,000 chunks:
- The **FP32 primary heap** occupies **512 KiB** (64 heap pages).
- The **FP32 secondary TOAST data relation** stored **40,960,000 bytes** (39.06 MiB across 5,000 TOAST pages) of out-of-line data payload, accompanied by a 456 KiB TOAST index.
- The **FP16 representation** had zero out-of-line TOAST data payload in the evaluated schema, inlining tuples into primary heap blocks (2,000 heap pages / 15.63 MiB heap) and retaining only an empty 8 KiB TOAST index shell.
- In single-query execution plan diagnostics (`EXPLAIN (ANALYZE, BUFFERS)`), the exact-search diagnostic reported high shared-buffer access counts for FP32 (30,064 shared-buffer hits at 10k vs. 2,000 for FP16), alongside substantially lower observed exact-search latency for FP16.
- *Methodological qualification:* Shared-buffer hits represent runtime buffer accesses—which may include repeated page touches during tuple and datum assembly—and do not by themselves establish physical disk reads or buffer thrashing. Storage layout conclusions reflect the tested dedicated schema and configuration.

---

## Research Objectives & Scope

### Primary Objectives
1. **Quantify Storage Compaction:** Measure physical on-disk storage across coordinate datums, table heaps, secondary TOAST tables, TOAST indexes, auxiliary forks (`_fsm`, `_vm`), and HNSW graph indexes from 1,001 up to 10,000 chunks.
2. **Benchmark Retrieval Latency:** Compare client-observed query latencies for exhaustive exact sequential scans and approximate HNSW index traversals under controlled database conditions.
3. **Evaluate Retrieval Fidelity:** Measure Nearest-Neighbor Overlap (Recall@K relative to uncompressed FP32 exact ground truth) and pairwise neighborhood consistency between FP32 and FP16 HNSW graphs.
4. **Map Query Optimizer Transitions:** Track PostgreSQL planner cost modeling and plan selection across corpus scales under default cost parameters.
5. **Enforce Database Isolation & Safety:** Guarantee zero mutation of baseline production tables through dedicated isolated databases.

### Scope Boundary
This repository evaluates **vector storage and retrieval performance trade-offs** inside the database engine. It does **not** evaluate downstream LLM generation quality (e.g., ROUGE, BLEU, or semantic hallucination rates) or represent a production end-to-end RAG application.

---

## What This Repository Implements & Evaluates

- **Representation Engine:** PostgreSQL tables utilizing `vector(768)` (FP32) and `halfvec(768)` (FP16 coordinate compression) via `pgvector`.
- **Dedicated Vector-Only Architecture:** Compares a monolithic schema (mixing heavy raw text with vectors) against isolated, dedicated vector-only representation tables (`id`, `chunk_id`, `embedding`) with foreign-key alignment.
- **Retrieval Modes:**
  1. *Exact Search:* Exhaustive cosine distance (`<=>` operator) with `ORDER BY distance LIMIT K`.
  2. *Forced HNSW Search:* Approximate nearest neighbor search forcing index scan (`SET enable_seqscan = off; SET hnsw.ef_search = 40;`).
  3. *Planner-Selected Search:* Unconstrained optimizer path evaluating PostgreSQL cost-based plan crossover.
- **Experimental Isolation:** Multi-database isolation preventing cross-stage cache contamination:
  - `compression_aware_rag` (Baseline, 180 chunks)
  - `compression_aware_rag_exp` (Stages 1–3, 1,001 chunks)
  - `compression_aware_rag_scale` (Stage 4, 5,000 & 10,000 chunks)
  - `compression_aware_rag_test` (Isolated pytest fixture database)

---

## Technical Architecture & Data Layout

### Schema Architecture
The project establishes a normalized, vector-only representation design:

```sql
-- Dedicated FP32 Representation Schema
CREATE TABLE chunks_embedding_fp32 (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL
);

-- Dedicated FP16 Compressed Representation Schema
CREATE TABLE chunks_embedding_halfvec (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding halfvec(768) NOT NULL
);
```

### Storage Breakdown Mechanics
Each relation fork is tracked through PostgreSQL system catalogs (`pg_relation_size`, `pg_total_relation_size`, `pg_statio_user_tables`):
- **Datum Size:** FP32 datum measures 3,076 bytes (`pg_column_size`), while FP16 datum measures 1,544 bytes (49.80% reduction).
- **Heap Packing:** Inlined FP16 tuples pack ~5 tuples per 8 KiB page, with observed relation counts of 201 heap blocks at 1,001 chunks, 1,000 at 5,000 chunks, and 2,000 at 10,000 chunks.
- **TOAST Elimination:** The TOAST data relation for `chunks_embedding_halfvec` measures **0 bytes** across all scales. Only an 8 KiB empty TOAST index shell remains.

---

## Experimental Methodology & Benchmark Design

### Validated Experimental Parameters

| Parameter | Configuration | Verification / Source |
|---|---|---|
| **Host Environment** | Windows 11, PostgreSQL 18.0, pgvector 0.5.0 | Catalog queries (`PostgreSQL 18.0`) |
| **Embedding Model** | `nomic-embed-text` (768 dimensions), local Ollama instance | `src/config.py`, verified embeddings |
| **Evaluation Dataset** | MS MARCO passage ranking subset (BeIR corpus) | SHA-256 verified in `data/raw/` |
| **Chunking Parameters** | Character length: 300, Overlap: 50 | Verified chunking pipeline (`src/preprocessing.py`) |
| **Corpus Scales Tested** | 180 chunks (Baseline), 1,001 chunks (Stages 1–3), 5,000 & 10,000 chunks (Stage 4) | Dedicated databases across stages |
| **Evaluation Queries** | 20 fixed domain-diverse queries pre-embedded outside timing harness | `benchmark_scale_..._v2.json` |
| **HNSW Index Parameters** | Distance: Cosine (`vector_cosine_ops`), $m=16$, $ef_{construction}=64$, $ef_{search}=40$ | DDL index definitions |
| **Measurement Trials** | 100 timed trials per configuration (20 queries × 5 measurement passes) | Stage 4 v2 timing harness |
| **Timing Boundaries** | Strictly bounded around `cur.execute()` and `cur.fetchall()` via `time.perf_counter()` | GUC statements outside timer |
| **Trial Order** | Alternating paired order (`FP32`, `FP16` vs. `FP16`, `FP32`) | Mitigates systemic caching bias |

---

## Master Empirical Results Summary

All metrics are extracted from verified catalog queries and benchmark execution JSON artifacts.

### 1. Physical Storage Footprint Compaction

| Metric Category | 1,001 Chunks (Stage 2/3) | 5,000 Chunks (Stage 4) | 10,000 Chunks (Stage 4) | Key Architectural Observation |
|---|---|---|---|---|
| **FP32 Table Base Storage** | 4.13 MiB (4,333,568 B) | 20.20 MiB (21,184,512 B) | 40.31 MiB (42,270,720 B) | Dominated by secondary TOAST data relation |
| **FP16 Table Base Storage** | 1.65 MiB (1,728,512 B) | 7.98 MiB (8,364,032 B) | 15.90 MiB (16,670,720 B) | Inlined in heap; 0 bytes TOAST data payload |
| **Base Storage Reduction** | **60.11%** | **60.52%** | **60.56%** | Consistent ~60.5% base table compaction |
| **FP32 HNSW Index Size** | 3.90 MiB (4,087,808 B) | 19.48 MiB (20,422,656 B) | 38.89 MiB (40,779,776 B) | Graph index on 32-bit floats |
| **FP16 HNSW Index Size** | 1.95 MiB (2,048,000 B) | 9.74 MiB (10,215,424 B) | 19.45 MiB (20,398,080 B) | Graph index on 16-bit floats |
| **HNSW Index Reduction** | **49.90%** | **49.98%** | **49.98%** | Exactly matches coordinate halving ratio |
| **FP32 Total Inclusive Size** | *Unrecorded post-index total* | 39.68 MiB (41,607,168 B) | 79.20 MiB (83,050,496 B) | `pg_total_relation_size` inclusive of all indexes |
| **FP16 Total Inclusive Size** | *Unrecorded post-index total* | 17.72 MiB (18,579,456 B) | 35.35 MiB (37,068,800 B) | `pg_total_relation_size` inclusive of all indexes |
| **Inclusive Compaction** | N/A | **55.35%** | **55.37%** | Exceeds 50% coordinate ratio via elimination of out-of-line TOAST data payload |

### 2. Retrieval Latency & Shared-Buffer Accesses

Measurements reflect client-observed latency across 100 timed trials (Stage 4 v2 timing harness) and single-query Query 0 (Q0) plan diagnostics:

| Corpus Scale | Search Strategy | FP32 Mean (Median) Latency | FP16 Mean (Median) Latency | Latency Speedup | Q0 Shared-Buffer Hits | Buffer Hit Reduction |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **5,000 Chunks** | Exact Search (Seq Scan) | 42.285 ms (44.694 ms) | 6.802 ms (6.277 ms) | **6.22x mean (7.12x median)** | 15,032 vs. 1,000 | **93.35% fewer buffer hits** |
| **5,000 Chunks** | Forced HNSW ($ef=40$) | 4.129 ms (3.371 ms) | 3.372 ms (2.923 ms) | 1.22x mean (1.15x median) | 830 vs. 753 | 9.28% lower |
| **10,000 Chunks** | Exact Search (Seq Scan) | 89.004 ms (92.664 ms) | 12.301 ms (11.541 ms) | **7.24x mean (8.03x median)** | 30,064 vs. 2,000 | **93.35% fewer buffer hits** |
| **10,000 Chunks** | Forced HNSW ($ef=40$) | 3.674 ms (3.415 ms) | 3.856 ms (3.820 ms) | 0.95x mean (0.89x median) | 930 vs. 847 | 8.92% lower |

> [!NOTE]
> **Key Finding on Latency:** FP16 delivers dramatic **6.2x–8.0x speedups in exact sequential search** (83.91% lower mean and 85.96% lower median latency at 5k; 86.18% lower mean and 87.55% lower median latency at 10k), where the full vector payload is scanned. In approximate HNSW search, traversal latencies remain similar across both representations (mean latencies span approximately 3.37–4.13 ms; medians span 2.92–3.82 ms across 5,000 and 10,000 chunks). A plausible explanation is that graph navigation overhead may dominate over coordinate arithmetic once index blocks reside in memory; however, this hypothesis was not directly verified through profiling.

### 3. Retrieval Fidelity & Relative Index Agreement

Evaluated at $ef_{search} = 40$ across 20 fixed evaluation queries against uncompressed FP32 exact ground truth:

| Corpus Scale | Index Representation | Recall@3 | Recall@5 | Recall@10 | Pairwise Index Agreement@10 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **5,000 Chunks** | FP32 HNSW (`vector`) | 0.9167 (91.67%) | 0.9100 (91.00%) | 0.9550 (95.50%) | — |
| **5,000 Chunks** | FP16 HNSW (`halfvec`) | 0.9167 (91.67%) | 0.9100 (91.00%) | **0.9600 (96.00%)** | **0.9850 (98.50%)** |
| **10,000 Chunks** | FP32 HNSW (`vector`) | 0.9667 (96.67%) | 0.9500 (95.00%) | 0.9450 (94.50%) | — |
| **10,000 Chunks** | FP16 HNSW (`halfvec`) | 0.9500 (95.00%) | 0.9400 (94.00%) | **0.9600 (96.00%)** | **0.9500 (95.00%)** |

*Pairwise Index Agreement is defined as the mean top-$K$ set intersection overlap fraction $\frac{|S_{\text{FP32}} \cap S_{\text{FP16}}|}{K}$ averaged across 20 evaluation queries (see [docs/submission_assets/README.md](docs/submission_assets/README.md) for full formal documentation).*

### 4. Single-Run HNSW Index Construction Efficiency

Single-run wall-clock build times recorded on identical hardware ($m=16$, $ef_{construction}=64$):
- **1,001 Chunks:** FP32 0.233 s vs. FP16 0.161 s (~30.9% faster; 31.19% recorded).
- **5,000 Chunks:** FP32 1.462 s vs. FP16 1.264 s (~13.5% faster; 13.52% recorded).
- **10,000 Chunks:** FP32 3.762 s vs. FP16 2.641 s (~29.8% faster; 29.79% recorded).

---

## Important Trade-offs & Interpretative Boundaries

When interpreting these empirical findings, preserve the following methodological boundaries:

1. **Storage Scope Separation:** Pre-HNSW base relation total, HNSW index-only footprint, and inclusive post-index relation total must not be conflated.
2. **Q0 Diagnostics vs. 100-Trial Statistics:** Single-query Q0 buffer hit reductions (93.35%) are **consistent with** repeated-trial exact search speedups. However, because Q0 diagnostics represent single-query samples while latency statistics are derived from 100 timed trials, the Q0 diagnostic does not establish the causal mechanism of every measured latency improvement across repeated trials.
3. **Recall@K vs. Pairwise Agreement:** Recall@K measures overlap against uncompressed FP32 exact ground truth; pairwise agreement measures consistency between FP32 and FP16 HNSW candidate sets. Pairwise agreement is the symmetric set intersection fraction, **not** Jaccard index.
4. **Discrete Scaling Observations:** Observations for exact search latency and forced HNSW traversal are limited to the two tested Stage 4 v2 scales (5,000 and 10,000 chunks). Linear or logarithmic scaling laws must not be extrapolated from two data points.
5. **Cross-Stage Latency Isolation:** Stage 3 1,001-chunk latency is strictly excluded from Stage 4 scaling comparisons because Stage 3 timed a full transaction block (`BEGIN; SET LOCAL; ...; COMMIT;`), whereas Stage 4 v2 timed strictly around `cur.execute()` and `cur.fetchall()`.
6. **The 4-Byte FP32 Datum Discrepancy:** Catalog queries consistently record **3,076 bytes** for `pg_column_size(vector(768))`, whereas the documented formula $4d + 8$ yields **3,080 bytes**. This is preserved as an unresolved empirical measurement without assigning an unverified internal cause.
7. **10k FP32 Recall Non-Monotonicity:** In the 10,000-chunk FP32 HNSW benchmark, recorded Recall values are 0.9667 at K=3, 0.9500 at K=5, and 0.9450 at K=10. This is preserved as recorded from benchmark artifacts.

---

## Repository Structure

```text
CompressionAwareRag/
├── main.py                               # Baseline reproduction entry point (smoke test & benchmark)
├── requirements.txt                      # Core dependencies (psycopg, pgvector, pytest, etc.)
├── pytest.ini                            # Pytest configuration (pythonpath = .)
├── .env.example                          # Environment configuration template
├── data/
│   └── raw/
│       ├── msmarco_10k_canonical.meta.json# Tracked dataset provenance & SHA-256 verification manifest
│       ├── msmarco_subset.tsv            # Tracked baseline initial sample (100 docs / 180 chunks)
│       ├── msmarco_10k_canonical.tsv     # (Excluded via .gitignore; local/cached 10,000-chunk corpus)
│       └── msmarco_1k_chunks.tsv         # (Excluded via .gitignore; local/cached 1,001-chunk corpus)
├── docs/
│   ├── final_results_synthesis.md        # Comprehensive cross-stage research evidence & synthesis
│   ├── final_submission_handoff.md       # Final submission handoff package & Review 3 blueprint
│   ├── stage1_dataset_environment.md     # Stage 1: Dataset provenance & environment setup
│   ├── stage2_fp16_evaluation.md         # Stage 2: FP16 inlining & TOAST eviction analysis
│   ├── stage3_hnsw_benchmark.md          # Stage 3: HNSW index construction & 1k recall
│   ├── stage4_scale_benchmark.md         # Stage 4: Corpus scaling (5k & 10k) & v2 timing
│   ├── baseline_benchmark.md             # Baseline initial benchmark documentation
│   └── submission_assets/                # Verified CSV datasets & data dictionary
│       ├── README.md                     # Data dictionary & provenance guide for CSV assets
│       ├── chart1_storage.csv            # Physical storage footprint scaling (1k, 5k, 10k)
│       ├── chart2_q0_diagnostics.csv     # Q0 execution plan diagnostics (shared-buffer hits)
│       ├── chart3_hnsw_recall.csv        # Representation-specific ANN Recall@K vs. exact ref
│       ├── chart3_pairwise_agreement.csv # Pairwise FP32 vs. FP16 neighborhood agreement
│       └── chart4_latency.csv            # 100-trial client latency distributions
├── src/
│   ├── benchmark.py                      # Baseline benchmark runner
│   ├── config.py                         # Environment loading & connection string generator
│   ├── database.py                       # PostgreSQL schema initialization & pool helpers
│   ├── embeddings.py                     # Ollama embedding client
│   ├── hnsw_experiment.py                # Stage 3 HNSW indexing & recall routines
│   ├── ingestion.py                      # Data ingestion & batch embedding pipeline
│   ├── preprocessing.py                  # Character-based text chunking & tokenization
│   ├── representations.py                # Dedicated FP32 & FP16 schema management
│   ├── scale_experiment.py               # Stage 4 corpus scaling & v2 benchmark harness
│   └── search.py                         # Query execution routines (exact & HNSW)
└── tests/
    ├── conftest.py                       # Pytest fixtures & isolated test database setup
    ├── test_basic.py                     # Baseline pipeline tests
    ├── test_hnsw_experiment.py           # Stage 3 HNSW index test suite
    ├── test_ingestion_cache.py           # Ingestion cache verification tests
    ├── test_representations.py           # Stage 2 FP16 representation tests
    └── test_scale_experiment.py          # Stage 4 scaling harness test suite
```

---

## Prerequisites & Environment Setup

### System Prerequisites & Tested Environment
- **Tested Operating System:** Windows 11
- **Python:** Version 3.10 or higher (evaluated with Python 3.13)
- **PostgreSQL:** Evaluated on PostgreSQL 18.0 (requires development headers / extension loading)
- **pgvector:** Evaluated with pgvector 0.5.0 (`CREATE EXTENSION vector;`)
- **Ollama:** Running locally with the `nomic-embed-text` model pulled

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yoswinner/CompressionAwareRag.git
   cd CompressionAwareRag
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running and download the embedding model:**
   ```bash
   ollama pull nomic-embed-text
   ```

---

## Configuration & Dataset Preparation

1. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:
   ```bash
   cp .env.example .env
   ```
   Required variables in `.env`:
   - `DB_HOST`: Hostname (default: `localhost`)
   - `DB_PORT`: PostgreSQL port (default: `5432`)
   - `DB_USER`: Database user (default: `postgres`)
   - `DB_PASSWORD`: User password
   - `DB_NAME`: Default database (default: `postgres`)
   - `OLLAMA_API_URL`: Ollama endpoint (default: `http://127.0.0.1:11434/api/embeddings`)
   - `OLLAMA_MODEL`: Embedding model (default: `nomic-embed-text`)

2. **Dataset Preparation & Provenance:**
   The repository tracks cryptographic provenance and baseline sample data in `data/raw/`:
   - `data/raw/msmarco_10k_canonical.meta.json`: Tracked cryptographic metadata and provenance manifest.
   - `data/raw/msmarco_subset.tsv`: Tracked baseline initial sample (100 documents / 180 chunks).

   The larger evaluation corpus files (`msmarco_1k_chunks.tsv` and `msmarco_10k_canonical.tsv`) are excluded from version control via `.gitignore` to avoid bloating the repository. To prepare these datasets using verified existing routines:
   - **1,001-Chunk Corpus (Stages 1–3):** See the verified preparation procedure in [Stage 1 Dataset & Environment Setup Report](docs/stage1_dataset_environment.md):
     ```bash
     python -c "from src.ingestion import download_and_cache_dataset; download_and_cache_dataset(target_chunks=1000)"
     ```
   - **10,000-Chunk Corpus (Stage 4):** Prepare the canonical cache using the existing routine in `src/scale_experiment.py`:
     ```bash
     python -c "from src.scale_experiment import download_canonical_cache; download_canonical_cache(target_chunks=10000)"
     ```

---

## Running the Code & Test Suites

### 1. Running the Baseline Pipeline
Execute the baseline demonstration pipeline:
```bash
python main.py
```
This entry point executes a smoke test and benchmark exclusively on the baseline database (`compression_aware_rag` with 100 documents / 180 chunks), validating schema initialization and basic exact vs. HNSW search. It does **not** execute the multi-stage scale experiments (Stages 1–4 v2 across 1,001 to 10,000 chunks); those scaling experiments are executed via dedicated test suites and benchmark modules as documented in [Stage 4 Benchmark Report](docs/stage4_scale_benchmark.md).

### 2. Running Automated Tests
Run the complete test suite across all stages:
```bash
pytest tests/
```
To run specific modular test suites:
```bash
# Test Stage 2 FP16 representations and inlining:
pytest tests/test_representations.py

# Test Stage 3 HNSW indexing and recall calculation:
pytest tests/test_hnsw_experiment.py

# Test Stage 4 scaling harness and v2 timing logic:
pytest tests/test_scale_experiment.py
```

---

## Documentation Index & Benchmark Reports

Comprehensive experimental records are available in `docs/`:

- **[Final Submission Handoff & Research Dossier](docs/final_submission_handoff.md):** Unified briefing dossier, 12-slide presentation outline, and academic report blueprint.
- **[Comprehensive Cross-Stage Synthesis](docs/final_results_synthesis.md):** Detailed cross-stage analysis, DBMS storage mechanics, optimizer crossover mapping, and viva voce defense guide.
- **[Submission Assets Data Dictionary](docs/submission_assets/README.md):** Complete data dictionary and provenance guide for all five chart CSV datasets.
- **[Stage 4: Corpus Scaling Benchmark Report](docs/stage4_scale_benchmark.md):** Scale benchmark at 5,000 and 10,000 chunks under the corrected v2 timing harness.
- **[Stage 3: HNSW Indexing Benchmark Report](docs/stage3_hnsw_benchmark.md):** HNSW graph construction and recall evaluation at 1,001 chunks.
- **[Stage 2: FP16 Representation Evaluation Report](docs/stage2_fp16_evaluation.md):** In-depth analysis of TOAST eviction and coordinate quantization.
- **[Stage 1: Dataset & Environment Setup Report](docs/stage1_dataset_environment.md):** Initial MS MARCO data pipeline and database configuration.
- **[Baseline Benchmark Report](docs/baseline_benchmark.md):** Initial 180-chunk benchmark and methodology audit.

---

## Limitations & Future Research Roadmap

- **Corpus Scale Boundary:** Benchmarking in this project was rigorously validated up to 10,000 chunks (7.68 million dimensions). Future work should evaluate 100k to 1M+ vectors under constrained memory to observe disk-paging transitions.
- **Single-Run Build Durations:** HNSW index construction times represent single-run wall-clock measurements; multi-trial timing and Write-Ahead Log (WAL) volume measurement were outside the scope of this phase.
- **Cache Sizing & Memory Pressure:** Evaluated under warm single-host shared buffers; future studies should stress PostgreSQL `shared_buffers` limits to quantify cache-miss degradation.
- **Compound Quantization:** Evaluating coordinate quantization combined with Product Quantization (PQ) or Scalar Quantization (SQ) for billion-scale deployments.
- **Downstream Generation Quality:** Evaluating whether half-precision coordinate truncation alters downstream LLM generative response quality (e.g., via LLM-as-a-judge or RAGAS evaluation).

---

## Acknowledgements & References

- **pgvector:** Vector similarity search for PostgreSQL ([github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)).
- **PostgreSQL Global Development Group:** Documentation on TOAST storage architecture and table physical structure ([postgresql.org/docs](https://www.postgresql.org/docs/current/storage-toast.html)).
- **MS MARCO Dataset:** Microsoft Machine Reading Comprehension dataset via the BeIR benchmark suite.
- **Ollama & Nomic AI:** `nomic-embed-text` high-dimensional embedding model.
