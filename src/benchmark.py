import time
import statistics
import re
from datetime import datetime, timezone
from src.search import (
    exact_search,
    create_hnsw_index,
    hnsw_search,
    verify_hnsw_index,
    get_search_plan,
    verify_execution_plan
)
from src.database import get_connection
from src.embeddings import get_embeddings_batch

def calculate_recall(exact_results: list[dict], hnsw_results: list[dict]) -> float:
    """
    Calculate Recall@K between ground-truth exact results and ANN HNSW results.
    Recall@K = |exact_ids ∩ hnsw_ids| / |exact_ids|
    """
    exact_ids = {r["chunk_id"] for r in exact_results}
    hnsw_ids = {r["chunk_id"] for r in hnsw_results}
    if not exact_ids:
        return 0.0
    intersection = exact_ids.intersection(hnsw_ids)
    return len(intersection) / len(exact_ids)

def extract_plan_times(plan_text: str):
    """Extract server-side planning and execution times in ms from EXPLAIN ANALYZE."""
    plan_time_match = re.search(r"Planning Time:\s*([\d\.]+)\s*ms", plan_text)
    exec_time_match = re.search(r"Execution Time:\s*([\d\.]+)\s*ms", plan_text)
    planning_ms = float(plan_time_match.group(1)) if plan_time_match else None
    exec_ms = float(exec_time_match.group(1)) if exec_time_match else None
    return planning_ms, exec_ms

def run_benchmark(
    queries: list[str],
    top_k: int = 5,
    m: int = 16,
    ef_construction: int = 64,
    ef_search: int = 40,
    warmup_runs: int = 2,
    num_runs: int = 3,
    conn=None,
    rebuild_index: bool = False
) -> dict:
    """
    Controlled Benchmark for Exact vs HNSW Retrieval.
    - Pre-generates all embeddings once beforehand.
    - Reuses a single persistent database connection to eliminate connection overhead.
    - Verifies actual PostgreSQL execution plans for both Exact (Seq Scan) and HNSW (Index Scan).
    - Aborts if HNSW index scan cannot be verified.
    - Measures client-observed retrieval query latency with high-resolution timer.
    - Records individual query latencies and computes average and median.
    - Computes ground-truth Recall@K on actual chunk IDs.
    - Records server-side EXPLAIN ANALYZE metrics separately.
    - Restores all session planner settings cleanly.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM document_chunks;")
            num_vectors = cur.fetchone()[0]
            cur.execute("SELECT current_database();")
            active_db = cur.fetchone()[0]

        print(f"\n=======================================================")
        print(f"BENCHMARK RUN: {timestamp}")
        print(f"Database: {active_db} | Total Vectors: {num_vectors}")
        print(f"Configuration: Top-K={top_k}, m={m}, ef_construction={ef_construction}, ef_search={ef_search}")
        print(f"Warmup Runs: {warmup_runs} | Measurement Runs: {num_runs} | Queries: {len(queries)}")
        print(f"=======================================================")

        # 1. Pre-generate Query Embeddings (EXCLUDED from retrieval latency)
        print("\n--- Step 1: Pre-generating Query Embeddings ---")
        start_emb_time = time.perf_counter()
        query_embeddings = get_embeddings_batch(queries)
        emb_time = time.perf_counter() - start_emb_time
        print(f"Generated {len(queries)} embeddings in {emb_time:.3f}s (EXCLUDED from retrieval latency)")

        if not query_embeddings:
            raise ValueError("No query embeddings were generated. Benchmark aborted.")

        sample_emb = query_embeddings[0]

        # 2. Verify / Build HNSW Index
        print("\n--- Step 2: HNSW Index Verification ---")
        index_exists = verify_hnsw_index(conn=conn)
        index_build_time = None
        if not index_exists or rebuild_index:
            print(f"Building HNSW index (m={m}, ef_construction={ef_construction})...")
            start_idx = time.perf_counter()
            create_hnsw_index(m=m, ef_construction=ef_construction, conn=conn)
            index_build_time = time.perf_counter() - start_idx
            print(f"HNSW index built in {index_build_time:.3f}s (EXCLUDED from retrieval latency)")
        else:
            print(f"HNSW index already exists and verified intact.")

        if not verify_hnsw_index(conn=conn):
            raise RuntimeError("HNSW index verification failed! Index does not exist on document_chunks.")

        # 3. Exact Search Plan Verification
        print("\n--- Step 3: Verifying Exact Search Execution Plan ---")
        exact_plan = get_search_plan(sample_emb, top_k=top_k, force_hnsw=False, conn=conn, analyze=False)
        exact_is_seqscan = verify_execution_plan(exact_plan, "Seq Scan")
        exact_uses_hnsw = verify_execution_plan(exact_plan, "document_chunks_embedding_hnsw_idx")

        print("Natural query plan for exact search:")
        for line in exact_plan.splitlines():
            print(f"  {line}")

        if not exact_is_seqscan or exact_uses_hnsw:
            raise RuntimeError(
                f"Exact baseline verification failed! Expected sequential scan without HNSW index. Plan:\n{exact_plan}"
            )
        print("Verified: Exact search executes exhaustive sequential scan with distance sort.")

        # Server-side execution profile for exact search
        exact_analyze = get_search_plan(sample_emb, top_k=top_k, force_hnsw=False, conn=conn, analyze=True)
        exact_plan_ms, exact_exec_ms = extract_plan_times(exact_analyze)

        # 4. Exact Search Warmup and Timing
        print(f"\n--- Step 4: Measuring Exact Search Latency ({warmup_runs} warmups, {num_runs} runs x {len(queries)} queries) ---")
        for _ in range(warmup_runs):
            for q_emb in query_embeddings:
                exact_search(q_emb, top_k=top_k, conn=conn)

        exact_latencies = []
        exact_last_results = []
        for run_idx in range(num_runs):
            for q_idx, q_emb in enumerate(query_embeddings):
                t0 = time.perf_counter()
                res = exact_search(q_emb, top_k=top_k, conn=conn)
                t1 = time.perf_counter()
                exact_latencies.append(t1 - t0)
                if run_idx == num_runs - 1:
                    exact_last_results.append(res)

        avg_exact_latency = sum(exact_latencies) / len(exact_latencies)
        median_exact_latency = statistics.median(exact_latencies)

        # 5. HNSW Search Plan Verification
        print("\n--- Step 5: Verifying HNSW Search Execution Plan ---")
        natural_hnsw_plan = get_search_plan(sample_emb, top_k=top_k, force_hnsw=False, ef_search=ef_search, conn=conn)
        forced_hnsw_plan = get_search_plan(sample_emb, top_k=top_k, force_hnsw=True, ef_search=ef_search, conn=conn)

        print("Natural planner decision (at current scale):")
        for line in natural_hnsw_plan.splitlines():
            print(f"  {line}")

        print("Diagnostic forced HNSW plan (session enable_seqscan = off):")
        for line in forced_hnsw_plan.splitlines():
            print(f"  {line}")

        hnsw_verified = verify_execution_plan(forced_hnsw_plan, "Index Scan using document_chunks_embedding_hnsw_idx")
        if not hnsw_verified:
            raise RuntimeError(
                f"HNSW Plan Verification ABORTED: Planner did not use 'document_chunks_embedding_hnsw_idx' "
                f"even with enable_seqscan=off. Plan:\n{forced_hnsw_plan}"
            )
        print("Verified: Forced diagnostic HNSW plan genuinely uses document_chunks_embedding_hnsw_idx.")

        # Server-side execution profile for HNSW search
        hnsw_analyze = get_search_plan(sample_emb, top_k=top_k, force_hnsw=True, ef_search=ef_search, conn=conn, analyze=True)
        hnsw_plan_ms, hnsw_exec_ms = extract_plan_times(hnsw_analyze)

        # 6. HNSW Search Warmup and Timing (with persistent session settings)
        print(f"\n--- Step 6: Measuring HNSW Search Latency ({warmup_runs} warmups, {num_runs} runs x {len(queries)} queries) ---")
        hnsw_latencies = []
        hnsw_last_results = []
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off;")
            cur.execute(f"SET hnsw.ef_search = {int(ef_search)};")
        try:
            for _ in range(warmup_runs):
                for q_emb in query_embeddings:
                    hnsw_search(q_emb, top_k=top_k, ef_search=None, conn=conn)

            for run_idx in range(num_runs):
                for q_idx, q_emb in enumerate(query_embeddings):
                    t0 = time.perf_counter()
                    res = hnsw_search(q_emb, top_k=top_k, ef_search=None, conn=conn)
                    t1 = time.perf_counter()
                    hnsw_latencies.append(t1 - t0)
                    if run_idx == num_runs - 1:
                        hnsw_last_results.append(res)
        finally:
            with conn.cursor() as cur:
                cur.execute("RESET enable_seqscan;")
                cur.execute("RESET hnsw.ef_search;")

        avg_hnsw_latency = sum(hnsw_latencies) / len(hnsw_latencies)
        median_hnsw_latency = statistics.median(hnsw_latencies)

        # 7. Ground-Truth Recall@K Calculation
        print("\n--- Step 7: Calculating Recall@K ---")
        recalls = []
        for i, (exact_res, hnsw_res) in enumerate(zip(exact_last_results, hnsw_last_results)):
            rec = calculate_recall(exact_res, hnsw_res)
            recalls.append(rec)
            exact_ids = [r["chunk_id"] for r in exact_res]
            hnsw_ids = [r["chunk_id"] for r in hnsw_res]
            print(f"  Query {i+1}: Recall@{top_k} = {rec:.4f} | Exact IDs: {exact_ids} | HNSW IDs: {hnsw_ids}")

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

        # Summary Display
        print("\n=======================================================")
        print("BENCHMARK SUMMARY RESULTS")
        print("=======================================================")
        print(f"Timestamp:                 {timestamp}")
        print(f"Database:                  {active_db}")
        print(f"Total Vectors:             {num_vectors}")
        print(f"Top-K:                     {top_k}")
        print(f"Queries:                   {len(queries)}")
        print(f"Measurements Per Method:   {len(exact_latencies)} ({num_runs} runs x {len(queries)} queries)")
        print(f"Index Parameters:          m={m}, ef_construction={ef_construction}, ef_search={ef_search}")
        print(f"Recall@{top_k}:                 {avg_recall:.4f}")
        print(f"Exact Retrieval Latency:   Avg = {avg_exact_latency*1000:.3f} ms | Median = {median_exact_latency*1000:.3f} ms")
        print(f"HNSW Retrieval Latency:    Avg = {avg_hnsw_latency*1000:.3f} ms | Median = {median_hnsw_latency*1000:.3f} ms")
        if exact_exec_ms is not None and hnsw_exec_ms is not None:
            print(f"Server-Side Execution:     Exact = {exact_exec_ms:.3f} ms | HNSW = {hnsw_exec_ms:.3f} ms (EXPLAIN ANALYZE)")
        print(f"Metric Description:        client-observed retrieval query latency with a persistent connection")
        print("=======================================================\n")

        return {
            "timestamp": timestamp,
            "database": active_db,
            "num_vectors": num_vectors,
            "top_k": top_k,
            "num_queries": len(queries),
            "warmup_runs": warmup_runs,
            "num_runs": num_runs,
            "total_measurements": len(exact_latencies),
            "index_config": {
                "m": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search
            },
            "index_build_time_seconds": index_build_time,
            "exact_plan": exact_plan,
            "natural_hnsw_plan": natural_hnsw_plan,
            "forced_hnsw_plan": forced_hnsw_plan,
            "exact_latencies_seconds": exact_latencies,
            "exact_latency_avg_ms": avg_exact_latency * 1000,
            "exact_latency_median_ms": median_exact_latency * 1000,
            "hnsw_latencies_seconds": hnsw_latencies,
            "hnsw_latency_avg_ms": avg_hnsw_latency * 1000,
            "hnsw_latency_median_ms": median_hnsw_latency * 1000,
            "recall": avg_recall,
            "per_query_recalls": recalls,
            "server_side_metrics": {
                "exact_planning_ms": exact_plan_ms,
                "exact_execution_ms": exact_exec_ms,
                "hnsw_planning_ms": hnsw_plan_ms,
                "hnsw_execution_ms": hnsw_exec_ms
            },
            "metric_description": "client-observed retrieval query latency with a persistent connection"
        }

    finally:
        if should_close:
            conn.close()
