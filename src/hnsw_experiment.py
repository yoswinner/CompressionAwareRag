import time
import re
import statistics
import psycopg
from pgvector.psycopg import register_vector
from src.config import get_db_connection_string
from src.embeddings import get_embedding
from src.representations import (
    EVALUATION_QUERIES,
    exact_search_fp32
)

EXP_DB_NAME = "compression_aware_rag_exp"
TEST_DB_NAME = "compression_aware_rag_test"
ALLOWED_DATABASES = {EXP_DB_NAME, TEST_DB_NAME}

def assert_allowed_db(conn, allowed_dbs: set[str] = ALLOWED_DATABASES):
    """
    Fail-closed safety guard:
    Verifies that the connection is strictly targeting an allowed database.
    Prevents any mutation against production database 'compression_aware_rag'.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        active_db = cur.fetchone()[0]
        if active_db not in allowed_dbs:
            raise RuntimeError(
                f"CRITICAL SAFETY GUARD VIOLATION: Operation aborted! "
                f"Active database is '{active_db}', but allowed databases are {allowed_dbs}. "
                f"Refusing to execute HNSW operation on unauthorized database."
            )
        return active_db

def get_server_settings(conn=None, target_db: str = EXP_DB_NAME) -> dict:
    """
    Retrieves key PostgreSQL server memory and buffer configuration settings.
    Records maintenance_work_mem as the configured graph-build memory budget,
    not measured peak process RAM.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        with conn.cursor() as cur:
            cur.execute("SHOW maintenance_work_mem;")
            mwm = cur.fetchone()[0]
            cur.execute("SHOW shared_buffers;")
            sb = cur.fetchone()[0]
            cur.execute("SHOW work_mem;")
            wm = cur.fetchone()[0]
            return {
                "maintenance_work_mem": mwm,
                "shared_buffers": sb,
                "work_mem": wm
            }
    finally:
        if should_close:
            conn.close()

def build_hnsw_indexes(
    conn=None,
    target_db: str = EXP_DB_NAME,
    m: int = 16,
    ef_construction: int = 64
) -> dict:
    """
    Builds the HNSW indexes on chunks_embedding_fp32 and chunks_embedding_halfvec.
    Measures wall-clock build time using time.perf_counter() strictly around CREATE INDEX.
    If an index already exists, reports it and does not claim a new build duration.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        settings = get_server_settings(conn=conn, target_db=target_db)
        results = {
            "target_db": target_db,
            "server_settings": settings,
            "indexes": {}
        }

        with conn.cursor() as cur:
            # Check existing indexes
            cur.execute("""
                SELECT indexname
                FROM pg_indexes
                WHERE tablename IN ('chunks_embedding_fp32', 'chunks_embedding_halfvec');
            """)
            existing_indexes = {r[0] for r in cur.fetchall()}

            # 1. FP32 HNSW Index
            fp32_idx = "chunks_embedding_fp32_hnsw_idx"
            if fp32_idx in existing_indexes:
                cur.execute(f"SELECT pg_relation_size('{fp32_idx}'::regclass);")
                fp32_size = cur.fetchone()[0]
                results["indexes"]["fp32"] = {
                    "index_name": fp32_idx,
                    "table_name": "chunks_embedding_fp32",
                    "operator_class": "vector_cosine_ops",
                    "m": m,
                    "ef_construction": ef_construction,
                    "already_existed": True,
                    "build_time_seconds": None,
                    "index_size_bytes": fp32_size
                }
                print(f"Index '{fp32_idx}' already exists. Skipping build, size: {fp32_size} bytes.")
            else:
                print(f"Building FP32 HNSW index '{fp32_idx}' (m={m}, ef_construction={ef_construction})...")
                t0 = time.perf_counter()
                cur.execute(f"""
                    CREATE INDEX {fp32_idx}
                    ON chunks_embedding_fp32
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction});
                """)
                t1 = time.perf_counter()
                build_time_fp32 = t1 - t0

                cur.execute(f"SELECT pg_relation_size('{fp32_idx}'::regclass);")
                fp32_size = cur.fetchone()[0]
                results["indexes"]["fp32"] = {
                    "index_name": fp32_idx,
                    "table_name": "chunks_embedding_fp32",
                    "operator_class": "vector_cosine_ops",
                    "m": m,
                    "ef_construction": ef_construction,
                    "already_existed": False,
                    "build_time_seconds": build_time_fp32,
                    "index_size_bytes": fp32_size
                }
                print(f"Built '{fp32_idx}' in {build_time_fp32:.4f}s, size: {fp32_size} bytes.")

            # 2. FP16 (halfvec) HNSW Index
            half_idx = "chunks_embedding_halfvec_hnsw_idx"
            if half_idx in existing_indexes:
                cur.execute(f"SELECT pg_relation_size('{half_idx}'::regclass);")
                half_size = cur.fetchone()[0]
                results["indexes"]["halfvec"] = {
                    "index_name": half_idx,
                    "table_name": "chunks_embedding_halfvec",
                    "operator_class": "halfvec_cosine_ops",
                    "m": m,
                    "ef_construction": ef_construction,
                    "already_existed": True,
                    "build_time_seconds": None,
                    "index_size_bytes": half_size
                }
                print(f"Index '{half_idx}' already exists. Skipping build, size: {half_size} bytes.")
            else:
                print(f"Building FP16 HNSW index '{half_idx}' (m={m}, ef_construction={ef_construction})...")
                t2 = time.perf_counter()
                cur.execute(f"""
                    CREATE INDEX {half_idx}
                    ON chunks_embedding_halfvec
                    USING hnsw (embedding halfvec_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction});
                """)
                t3 = time.perf_counter()
                build_time_half = t3 - t2

                cur.execute(f"SELECT pg_relation_size('{half_idx}'::regclass);")
                half_size = cur.fetchone()[0]
                results["indexes"]["halfvec"] = {
                    "index_name": half_idx,
                    "table_name": "chunks_embedding_halfvec",
                    "operator_class": "halfvec_cosine_ops",
                    "m": m,
                    "ef_construction": ef_construction,
                    "already_existed": False,
                    "build_time_seconds": build_time_half,
                    "index_size_bytes": half_size
                }
                print(f"Built '{half_idx}' in {build_time_half:.4f}s, size: {half_size} bytes.")

        return results
    finally:
        if should_close:
            conn.close()

def get_hnsw_index_sizes(conn=None, target_db: str = EXP_DB_NAME) -> dict:
    """
    Returns exact relation size on disk (pg_relation_size) for both HNSW indexes.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    relname,
                    pg_relation_size(oid) AS size_bytes
                FROM pg_class
                WHERE relname IN ('chunks_embedding_fp32_hnsw_idx', 'chunks_embedding_halfvec_hnsw_idx');
            """)
            sizes = {r[0]: r[1] for r in cur.fetchall()}
            fp32_bytes = sizes.get("chunks_embedding_fp32_hnsw_idx", 0)
            half_bytes = sizes.get("chunks_embedding_halfvec_hnsw_idx", 0)

            return {
                "fp32_index_bytes": fp32_bytes,
                "halfvec_index_bytes": half_bytes,
                "size_reduction_bytes": fp32_bytes - half_bytes,
                "size_reduction_pct": ((fp32_bytes - half_bytes) / fp32_bytes * 100.0) if fp32_bytes > 0 else 0.0,
                "size_reduction_ratio": (fp32_bytes / half_bytes) if half_bytes > 0 else 0.0
            }
    finally:
        if should_close:
            conn.close()

def parse_execution_plan(plan_lines: list[str]) -> dict:
    """
    Parses EXPLAIN (ANALYZE, BUFFERS) lines to extract:
    - scan_type (e.g. 'Index Scan', 'Seq Scan')
    - index_name_used (e.g. 'chunks_embedding_fp32_hnsw_idx')
    - execution_time_ms
    - planning_time_ms
    - shared_hit_buffers
    - shared_read_buffers
    """
    plan_text = "\n".join(plan_lines)
    scan_type = "Unknown"
    index_name = None

    if "Index Scan using" in plan_text:
        scan_type = "Index Scan"
        m = re.search(r"Index Scan using (\S+)", plan_text)
        if m:
            index_name = m.group(1)
    elif "Seq Scan on" in plan_text:
        scan_type = "Seq Scan"

    # Execution time
    exec_time = None
    m_exec = re.search(r"Execution Time:\s+([\d\.]+)\s+ms", plan_text)
    if m_exec:
        exec_time = float(m_exec.group(1))

    # Planning time
    plan_time = None
    m_plan = re.search(r"Planning Time:\s+([\d\.]+)\s+ms", plan_text)
    if m_plan:
        plan_time = float(m_plan.group(1))

    # Shared buffers
    shared_hit = 0
    shared_read = 0
    m_hit = re.search(r"shared hit=(\d+)", plan_text)
    if m_hit:
        shared_hit = int(m_hit.group(1))
    m_read = re.search(r"shared read=(\d+)", plan_text)
    if m_read:
        shared_read = int(m_read.group(1))

    return {
        "scan_type": scan_type,
        "index_name_used": index_name,
        "execution_time_ms": exec_time,
        "planning_time_ms": plan_time,
        "shared_hit_buffers": shared_hit,
        "shared_read_buffers": shared_read,
        "raw_plan": plan_lines
    }

def get_query_plan(
    query_embedding: list[float],
    representation: str = "fp32",
    enable_seqscan: bool = False,
    ef_search: int = 40,
    top_k: int = 10,
    conn=None,
    target_db: str = EXP_DB_NAME
) -> dict:
    """
    Captures EXPLAIN (ANALYZE, BUFFERS) for FP32 or FP16 representation query.
    Binds the query vector parameter twice (in SELECT and ORDER BY).
    If enable_seqscan is False, programmatically asserts that the expected HNSW index was used.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        table_name = "chunks_embedding_fp32" if representation == "fp32" else "chunks_embedding_halfvec"
        cast_type = "vector(768)" if representation == "fp32" else "halfvec(768)"
        expected_index = f"{table_name}_hnsw_idx"

        with conn.cursor() as cur:
            cur.execute(f"SET enable_seqscan = {'on' if enable_seqscan else 'off'};")
            cur.execute(f"SET hnsw.ef_search = {ef_search};")

            query_sql = f"""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, (embedding <=> %s::{cast_type}) AS distance
                FROM {table_name}
                ORDER BY embedding <=> %s::{cast_type} ASC
                LIMIT %s;
            """
            cur.execute(query_sql, (query_embedding, query_embedding, top_k))
            raw_lines = [r[0] for r in cur.fetchall()]

        parsed = parse_execution_plan(raw_lines)

        # Verification assertion for forced-index execution
        if not enable_seqscan:
            if parsed["scan_type"] != "Index Scan" or parsed["index_name_used"] != expected_index:
                raise RuntimeError(
                    f"FORCED-INDEX VERIFICATION FAILURE for '{representation}': Expected Index Scan using "
                    f"'{expected_index}', but planner chose scan_type='{parsed['scan_type']}', "
                    f"index_name='{parsed['index_name_used']}'."
                )

        return parsed
    finally:
        if should_close:
            conn.close()

def search_hnsw(
    query_embedding: list[float],
    representation: str = "fp32",
    enable_seqscan: bool = False,
    ef_search: int = 40,
    top_k: int = 10,
    conn=None,
    target_db: str = EXP_DB_NAME
) -> list[dict]:
    """
    Executes nearest-neighbor search for FP32 or FP16 representation.
    Binds the query vector parameter twice: in SELECT distance and in ORDER BY distance operator.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        table_name = "chunks_embedding_fp32" if representation == "fp32" else "chunks_embedding_halfvec"
        cast_type = "vector(768)" if representation == "fp32" else "halfvec(768)"

        with conn.cursor() as cur:
            cur.execute(f"SET enable_seqscan = {'on' if enable_seqscan else 'off'};")
            cur.execute(f"SET hnsw.ef_search = {ef_search};")

            query_sql = f"""
                SELECT chunk_id, (embedding <=> %s::{cast_type}) AS distance
                FROM {table_name}
                ORDER BY embedding <=> %s::{cast_type} ASC
                LIMIT %s;
            """
            cur.execute(query_sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()
            return [{"chunk_id": r[0], "distance": float(r[1])} for r in rows]
    finally:
        if should_close:
            conn.close()

def calculate_overlap_recall(reference_results: list[dict], candidate_results: list[dict], top_k: int) -> float:
    """
    Calculates overlap fraction (Recall@K) of candidate top-K against reference top-K:
    Recall@K = |Top-K_ref ∩ Top-K_cand| / K
    """
    if top_k <= 0:
        return 0.0
    ref_set = {r["chunk_id"] for r in reference_results[:top_k]}
    cand_set = {r["chunk_id"] for r in candidate_results[:top_k]}
    return len(ref_set.intersection(cand_set)) / float(top_k)

def run_hnsw_benchmark(
    queries: list[str] | None = None,
    top_k_values: list[int] = [3, 5, 10],
    ef_search: int = 40,
    warmup_runs: int = 2,
    measurement_runs: int = 5,
    target_db: str = EXP_DB_NAME
) -> dict:
    """
    Executes the comprehensive Stage 3 HNSW benchmark:
    - Pre-generates query embeddings once via Ollama (excluded from retrieval timing)
    - Reuses persistent database connection
    - Performs 2 warm-up cycles across all queries
    - Obtains FP32 exact-search reference results for all queries
    - Runs 5 measurement passes (100 trials per representation and mode)
    - Alternates paired FP32/FP16 execution order between passes to eliminate bias
    - Programmatically verifies forced-index plans use the intended HNSW index
    - Strictly separates Forced-HNSW results from Planner-Selected results
    - Calculates Recall@3, Recall@5, and Recall@10 against FP32 exact ground truth
    """
    if queries is None:
        queries = EVALUATION_QUERIES

    conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
    register_vector(conn)

    try:
        assert_allowed_db(conn)
        settings = get_server_settings(conn=conn, target_db=target_db)

        # 1. Pre-generate all query embeddings once
        print(f"Pre-generating embeddings for {len(queries)} evaluation queries via Ollama...")
        t_emb_0 = time.perf_counter()
        precomputed_embeddings = [get_embedding(q) for q in queries]
        emb_gen_time = time.perf_counter() - t_emb_0
        print(f"Generated {len(precomputed_embeddings)} query embeddings in {emb_gen_time:.2f}s.")

        max_k = max(top_k_values)

        # 2. Compute FP32 Exact ground truth reference rankings
        print("Computing FP32 exact-search ground truth reference rankings...")
        exact_references = {}
        for q_idx, emb in enumerate(precomputed_embeddings):
            exact_references[q_idx] = {}
            for k in top_k_values:
                exact_references[q_idx][k] = exact_search_fp32(
                    emb, top_k=k, conn=conn, target_db=target_db
                )

        # 3. Warm-up passes
        print(f"Executing {warmup_runs} warm-up cycles for all modes...")
        for _ in range(warmup_runs):
            for emb in precomputed_embeddings:
                # Warm up forced HNSW
                search_hnsw(emb, representation="fp32", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
                search_hnsw(emb, representation="halfvec", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
                # Warm up planner-selected
                search_hnsw(emb, representation="fp32", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
                search_hnsw(emb, representation="halfvec", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)

        # 4. Timed Measurement Trials with Alternating Paired Order
        print(f"Executing {measurement_runs} measurement passes (100 trials per representation and mode)...")
        forced_fp32_latencies = []
        forced_halfvec_latencies = []
        planner_fp32_latencies = []
        planner_halfvec_latencies = []

        # Recall tracking per K for forced HNSW against FP32 exact ground truth
        forced_fp32_recall = {k: [] for k in top_k_values}
        forced_halfvec_recall = {k: [] for k in top_k_values}
        relative_index_agreement = {k: [] for k in top_k_values}

        # Evaluate forced Recall on pass 0
        for q_idx, emb in enumerate(precomputed_embeddings):
            for k in top_k_values:
                cand_fp32 = search_hnsw(emb, representation="fp32", enable_seqscan=False, ef_search=ef_search, top_k=k, conn=conn, target_db=target_db)
                cand_half = search_hnsw(emb, representation="halfvec", enable_seqscan=False, ef_search=ef_search, top_k=k, conn=conn, target_db=target_db)
                ref_exact = exact_references[q_idx][k]

                rec_fp32 = calculate_overlap_recall(ref_exact, cand_fp32, top_k=k)
                rec_half = calculate_overlap_recall(ref_exact, cand_half, top_k=k)
                rec_rel = calculate_overlap_recall(cand_fp32, cand_half, top_k=k)

                forced_fp32_recall[k].append(rec_fp32)
                forced_halfvec_recall[k].append(rec_half)
                relative_index_agreement[k].append(rec_rel)

        # Multi-run timed trials
        for run_id in range(measurement_runs):
            # Alternating order: Run 0, 2, 4 (odd passes): FP32 then Halfvec
            #                    Run 1, 3 (even passes): Halfvec then FP32
            order = ["fp32", "halfvec"] if (run_id % 2 == 0) else ["halfvec", "fp32"]

            for emb in precomputed_embeddings:
                # A. Forced-Index Mode (enable_seqscan = off)
                for rep in order:
                    t0 = time.perf_counter()
                    search_hnsw(emb, representation=rep, enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
                    t1 = time.perf_counter()
                    lat_ms = (t1 - t0) * 1000.0
                    if rep == "fp32":
                        forced_fp32_latencies.append(lat_ms)
                    else:
                        forced_halfvec_latencies.append(lat_ms)

                # B. Planner-Selected Mode (enable_seqscan = on)
                for rep in order:
                    t2 = time.perf_counter()
                    search_hnsw(emb, representation=rep, enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
                    t3 = time.perf_counter()
                    lat_ms = (t3 - t2) * 1000.0
                    if rep == "fp32":
                        planner_fp32_latencies.append(lat_ms)
                    else:
                        planner_halfvec_latencies.append(lat_ms)

        # 5. Diagnostic Plan Capture & Programmatic Verification (Query 0)
        sample_emb = precomputed_embeddings[0]
        plan_forced_fp32 = get_query_plan(sample_emb, representation="fp32", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
        plan_forced_half = get_query_plan(sample_emb, representation="halfvec", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
        plan_planner_fp32 = get_query_plan(sample_emb, representation="fp32", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
        plan_planner_half = get_query_plan(sample_emb, representation="halfvec", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)

        # 6. Physical Index Sizes
        index_sizes = get_hnsw_index_sizes(conn=conn, target_db=target_db)

        # 7. Aggregate Statistics
        def calc_stats(lat_list):
            return {
                "mean": statistics.mean(lat_list),
                "median": statistics.median(lat_list),
                "min": min(lat_list),
                "max": max(lat_list),
                "stdev": statistics.stdev(lat_list) if len(lat_list) > 1 else 0.0
            }

        avg_forced_fp32_recall = {f"Recall@{k}": statistics.mean(forced_fp32_recall[k]) for k in top_k_values}
        avg_forced_half_recall = {f"Recall@{k}": statistics.mean(forced_halfvec_recall[k]) for k in top_k_values}
        avg_relative_agreement = {f"Agreement@{k}": statistics.mean(relative_index_agreement[k]) for k in top_k_values}

        return {
            "target_db": target_db,
            "server_settings": settings,
            "query_count": len(queries),
            "measurement_runs_per_query": measurement_runs,
            "total_trials_per_mode": len(forced_fp32_latencies),
            "ef_search": ef_search,
            "top_k_values": top_k_values,
            "index_sizes": index_sizes,
            "forced_hnsw": {
                "description": "Forced HNSW index scan (enable_seqscan = off, ef_search = 40)",
                "fp32_latency_ms": calc_stats(forced_fp32_latencies),
                "halfvec_latency_ms": calc_stats(forced_halfvec_latencies),
                "fp32_recall_vs_exact": avg_forced_fp32_recall,
                "halfvec_recall_vs_exact": avg_forced_half_recall,
                "relative_index_agreement": avg_relative_agreement,
                "sample_plans": {
                    "fp32": plan_forced_fp32,
                    "halfvec": plan_forced_half
                }
            },
            "planner_selected": {
                "description": "Planner-selected execution under default settings (enable_seqscan = on)",
                "fp32_latency_ms": calc_stats(planner_fp32_latencies),
                "halfvec_latency_ms": calc_stats(planner_halfvec_latencies),
                "sample_plans": {
                    "fp32": plan_planner_fp32,
                    "halfvec": plan_planner_half
                }
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    finally:
        conn.close()
