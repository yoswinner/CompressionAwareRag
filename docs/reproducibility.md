# Reproducibility & Replication Guide

**Project:** Precision-Aware Query Planning for Vector Retrieval in PostgreSQL  
**Author:** Yash Pradhan  
**Target Venue:** IEEE Transactions on Knowledge and Data Engineering (TKDE) / IEEE ICDE  
**Repository Branch:** `research/planner-sensitivity`  

---

## 1. Prerequisites & Environment Setup

### 1.1 Core Software Requirements
- **PostgreSQL:** Version 18.x (tested on PostgreSQL 18.4, 64-bit).
- **pgvector:** Version 0.8.0+ (tested on `pgvector` v0.8.4 with `halfvec` support).
- **Python:** Python 3.11+ (tested on Python 3.13.14).
- **Ollama:** Version 0.5+ serving `nomic-embed-text:latest` (768 dimensions).

### 1.2 Python Virtual Environment Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` dependencies:
- `psycopg[binary]>=3.1.0`
- `pgvector`
- `requests`
- `python-dotenv`
- `tqdm`
- `pytest`

### 1.3 Database Setup & Schema Isolation
The experiment strictly isolates experimental tables inside a dedicated schema `planner_sensitivity` within two designated catalogs:
- `compression_aware_rag_exp` ($N=1{,}001$)
- `compression_aware_rag_scale` ($N=5{,}000$ and $N=10{,}000$)
- `compression_aware_rag_test` (used exclusively for pilot smoke-testing and pytest unit tests)

The production baseline database `compression_aware_rag` is strictly protected and disallowed from modification or benchmarking.

Configure connection credentials via `.env` (derived from `.env.example`):
```ini
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 2. Table Provisioning & Readiness Verification

### 2.1 Atomic Fail-Closed Provisioning
Provisioning builds the three representations (`fp32_extended`, `fp32_plain`, `fp16_extended`) across scales in atomic transactions:
```bash
# Verify source and provision all scales
python scripts/provision_planner_sensitivity.py --scale 1001 --scale 5000 --scale 10000 --apply
```
The script enforces:
- Valid `PRIMARY KEY` on `chunk_id`.
- `COUNT(DISTINCT chunk_id) == COUNT(*)`.
- Bidirectional 6-way chunk ID parity across representations.
- HNSW parameters: $m=16, ef\_construction=64$, cosine distance operator.
- Explicit in-transaction `ANALYZE`.

### 2.2 Independent Readiness Check
Verify table readiness without writing to the database:
```bash
python scripts/verify_table_readiness.py --scale all
```
Expected output:
```
[Scale 1001]  Database: compression_aware_rag_exp   => READY
[Scale 5000]  Database: compression_aware_rag_scale => READY
[Scale 10000] Database: compression_aware_rag_scale => READY
[Scale test]  Database: compression_aware_rag_test  => READY
[VERIFICATION PASSED] All experimental tables, storage policies, types, opclasses, and HNSW indexes are verified ready.
```

---

## 3. Unit & Integration Test Suite

Run the complete 107-test test suite:
```bash
pytest -v
```
Expected output:
```
============================ 107 passed in ~13s =============================
```

---

## 4. Benchmark Execution

### 4.1 Live Acceptance Pilot (Smoke Test)
Run a rapid pilot test on the test database ($N=180$, 2 queries, 2 warm-ups, 2 passes):
```bash
python src/planner_sensitivity_experiment.py --pilot
```

### 4.2 Full Research Benchmark Execution
Execute the full 9,900-query benchmark across all 9 profiles:
```bash
python src/planner_sensitivity_experiment.py --scale all --representation all --canonical-profile P4 --passes 5 --warmup 2
```
This produces:
- `results/planner_sensitivity/full_run_<timestamp>/latency_results.csv` (9,900 timed queries)
- `results/planner_sensitivity/full_run_<timestamp>/plan_capture.jsonl` (4,860 plan captures)
- `results/planner_sensitivity/full_run_<timestamp>/diagnostic_results.jsonl` (540 diagnostic runs)
- `results/planner_sensitivity/full_run_<timestamp>/summary_results.csv`
- `results/planner_sensitivity/full_run_<timestamp>/experiment_manifest.json`
- `results/planner_sensitivity/full_run_<timestamp>/environment_snapshot.json`

---

## 5. Research Analysis Scripts

All analysis scripts reside in `scripts/analysis/` and require zero database access (they consume the benchmark evidence directly):

### 5.1 Comprehensive Statistical Synthesis
```bash
python scripts/analysis/analyze_phase3_sensitivity.py
```
Outputs:
- Physical storage metrics and reductions table.
- Planner access-path selection distribution table.
- Canonical Profile P4 latency percentiles (P50, P95, Mean, Std).
- Boundary-tie tolerance sensitivity table ($\tau \in \{1\text{e-}7, 1\text{e-}6, 1\text{e-}5\}$).
- Corrected root-plan buffer consumption table.
- Friedman tests across profiles and paired Wilcoxon signed-rank tests with Holm-Bonferroni correction.

### 5.2 Boundary-Tie Sensitivity Audit
```bash
python scripts/analysis/run_phase3a_tie_sensitivity.py
```
Re-evaluates boundary-tie prevalence and calculates tie-aware Recall@10 across all 1,980 conditions.

### 5.3 Evidence Chain Closure & Equivalence Audit
```bash
python scripts/analysis/run_phase3a_addendum_closure.py
```
Exports explicit FP32 reference top-10 IDs, independently recalculates Recall@10, and audits forced-HNSW `enable_bitmapscan` equivalence.

---

## 6. Evidence Handling & Cryptographic Verification

The primary benchmark artifacts and audits are packaged in immutable ZIP archives in the repository root:
- `Phase2_Full_Benchmark_Evidence_20261031.zip` (SHA-256: `b56bfd0fd702b5faecec379e290d9412f206bf0c867fefa8e8a09a62088722b0`)
- `Phase3A_Tie_Sensitivity_Evidence_20261031.zip` (SHA-256: `de19a122ea2d918cd5820d6c450cec873c33ecdab8a1f297cdcd9082de222d52`)
- `Phase3A_Evidence_Closure_Addendum_20261031.zip` (SHA-256: `3a33532c35045cb9a15eb7738e2bd8851c37304380b01cdf2b4cc7d6b671f7eb`)

To verify archive integrity:
```python
import zipfile, hashlib

for archive in [
    "Phase2_Full_Benchmark_Evidence_20261031.zip",
    "Phase3A_Tie_Sensitivity_Evidence_20261031.zip",
    "Phase3A_Evidence_Closure_Addendum_20261031.zip"
]:
    with zipfile.ZipFile(archive, "r") as zf:
        bad = zf.testzip()
        assert bad is None, f"CRC corruption in {bad}"
    sha = hashlib.sha256(open(archive, "rb").read()).hexdigest()
    print(f"[OK] {archive}: Valid CRC, SHA-256: {sha}")
```
