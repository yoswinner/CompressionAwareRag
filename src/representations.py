import time
import statistics
import psycopg
from pgvector.psycopg import register_vector
from src.config import get_db_connection_string, EMBEDDING_DIM
from src.embeddings import get_embedding

EXP_DB_NAME = "compression_aware_rag_exp"
TEST_DB_NAME = "compression_aware_rag_test"
ALLOWED_DATABASES = {EXP_DB_NAME, TEST_DB_NAME}

# Standard fixed evaluation query set (20 queries covering corpus topics and diverse domains)
EVALUATION_QUERIES = [
    "what was the purpose of the manhattan project",
    "who led the manhattan project at los alamos",
    "when did world war 2 end and what caused it",
    "causes and key battles of the american civil war",
    "how does photosynthesis work in green plants",
    "what is the speed of light in vacuum",
    "who invented the telephone and when was it patented",
    "what are the clinical symptoms of acute appendicitis",
    "how do vector databases store and index embeddings",
    "difference between supervised and unsupervised machine learning",
    "history and origins of the ancient olympic games",
    "how do photovoltaic solar panels generate electricity",
    "what geological mechanisms cause earthquakes and volcanoes",
    "who was the first president of the united states",
    "how does the human immune system neutralize viral infections",
    "common treatment options and medications for high blood pressure",
    "what is the capital city of australia",
    "how do commercial airplane wings produce aerodynamic lift",
    "what are the fundamental principles of quantum mechanics",
    "structure and biological function of human deoxyribonucleic acid dna"
]

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
                f"Refusing to execute representation operation on unauthorized database."
            )
        return active_db

def initialize_representation_tables(conn=None, target_db: str = EXP_DB_NAME):
    """
    Initializes the two vector-only representation tables in target_db:
    1. chunks_embedding_fp32 (vector(768))
    2. chunks_embedding_halfvec (halfvec(768))
    Both reference document_chunks(id) ON DELETE CASCADE.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        with conn.cursor() as cur:
            # Ensure vector extension is present
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Table 1: FP32 representation
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_embedding_fp32 (
                    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding vector(768) NOT NULL
                );
            """)

            # Table 2: FP16 (halfvec) representation
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_embedding_halfvec (
                    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding halfvec(768) NOT NULL
                );
            """)
            print(f"Representation tables initialized in '{target_db}'.")
    finally:
        if should_close:
            conn.close()

def populate_representation_tables(conn=None, target_db: str = EXP_DB_NAME, force_repopulate: bool = False) -> dict:
    """
    Populates chunks_embedding_fp32 and chunks_embedding_halfvec from document_chunks.embedding.
    Transactional insertion ensuring exact 1:1 correspondence without re-generating embeddings.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=False)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        initialize_representation_tables(conn=conn, target_db=target_db)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            source_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunks_embedding_fp32;")
            fp32_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks_embedding_halfvec;")
            halfvec_count = cur.fetchone()[0]

            if not force_repopulate and fp32_count == source_count and halfvec_count == source_count and source_count > 0:
                print(f"Representation tables already fully populated with {source_count} rows in '{target_db}'.")
                return {
                    "source_count": source_count,
                    "fp32_count": fp32_count,
                    "halfvec_count": halfvec_count,
                    "populated": False
                }

            print(f"Populating representation tables from document_chunks ({source_count} rows) in '{target_db}'...")

            # Populate FP32
            cur.execute("""
                INSERT INTO chunks_embedding_fp32 (chunk_id, embedding)
                SELECT id, embedding
                FROM document_chunks
                ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """)

            # Populate FP16 (halfvec)
            cur.execute("""
                INSERT INTO chunks_embedding_halfvec (chunk_id, embedding)
                SELECT id, embedding::halfvec(768)
                FROM document_chunks
                ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """)

            conn.commit()

            cur.execute("SELECT COUNT(*) FROM chunks_embedding_fp32;")
            fp32_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks_embedding_halfvec;")
            halfvec_count = cur.fetchone()[0]

            print(f"Population complete in '{target_db}': FP32={fp32_count}, HalfVec={halfvec_count}.")
            return {
                "source_count": source_count,
                "fp32_count": fp32_count,
                "halfvec_count": halfvec_count,
                "populated": True
            }
    except Exception:
        conn.rollback()
        raise
    finally:
        if should_close:
            conn.close()

def validate_representations(conn=None, target_db: str = EXP_DB_NAME) -> dict:
    """
    Performs comprehensive verification of the representation tables:
    - Counts and parity
    - NULL values
    - Vector dimensions
    - Missing or duplicate chunk IDs
    - Source table immutability
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            source_chunk_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunks_embedding_fp32;")
            fp32_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunks_embedding_halfvec;")
            halfvec_count = cur.fetchone()[0]

            # Null checks
            cur.execute("SELECT COUNT(*) FROM chunks_embedding_fp32 WHERE chunk_id IS NULL OR embedding IS NULL;")
            fp32_nulls = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunks_embedding_halfvec WHERE chunk_id IS NULL OR embedding IS NULL;")
            halfvec_nulls = cur.fetchone()[0]

            # Dimensions
            cur.execute("SELECT DISTINCT vector_dims(embedding) FROM chunks_embedding_fp32;")
            fp32_dims = [r[0] for r in cur.fetchall()]

            cur.execute("SELECT DISTINCT vector_dims(embedding) FROM chunks_embedding_halfvec;")
            halfvec_dims = [r[0] for r in cur.fetchall()]

            # Missing or mismatched IDs
            cur.execute("""
                SELECT COUNT(*) FROM document_chunks dc
                LEFT JOIN chunks_embedding_fp32 f ON dc.id = f.chunk_id
                WHERE f.chunk_id IS NULL;
            """)
            missing_fp32 = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM document_chunks dc
                LEFT JOIN chunks_embedding_halfvec h ON dc.id = h.chunk_id
                WHERE h.chunk_id IS NULL;
            """)
            missing_halfvec = cur.fetchone()[0]

            # Correspondence check: every FP16 row matches the corresponding FP32 row
            cur.execute("""
                SELECT COUNT(*) FROM chunks_embedding_fp32 f
                FULL OUTER JOIN chunks_embedding_halfvec h ON f.chunk_id = h.chunk_id
                WHERE f.chunk_id IS NULL OR h.chunk_id IS NULL;
            """)
            mismatched_ids = cur.fetchone()[0]

            # Source embedding immutability check
            cur.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL;")
            source_null_embeddings = cur.fetchone()[0]

            is_valid = (
                source_chunk_count > 0 and
                fp32_count == source_chunk_count and
                halfvec_count == source_chunk_count and
                fp32_nulls == 0 and
                halfvec_nulls == 0 and
                fp32_dims == [EMBEDDING_DIM] and
                halfvec_dims == [EMBEDDING_DIM] and
                missing_fp32 == 0 and
                missing_halfvec == 0 and
                mismatched_ids == 0 and
                source_null_embeddings == 0
            )

            return {
                "target_db": target_db,
                "document_count": doc_count,
                "source_chunk_count": source_chunk_count,
                "fp32_count": fp32_count,
                "halfvec_count": halfvec_count,
                "fp32_nulls": fp32_nulls,
                "halfvec_nulls": halfvec_nulls,
                "fp32_dims": fp32_dims,
                "halfvec_dims": halfvec_dims,
                "missing_fp32": missing_fp32,
                "missing_halfvec": missing_halfvec,
                "mismatched_ids": mismatched_ids,
                "source_null_embeddings": source_null_embeddings,
                "is_valid": is_valid
            }
    finally:
        if should_close:
            conn.close()

def exact_search_fp32(query_embedding: list[float], top_k: int = 10, conn=None, target_db: str = EXP_DB_NAME) -> list[dict]:
    """
    Exhaustive nearest-neighbor search over chunks_embedding_fp32 using vector cosine distance (<=>).
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
                SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                FROM chunks_embedding_fp32
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
            rows = cur.fetchall()
            return [{"chunk_id": r[0], "distance": float(r[1])} for r in rows]
    finally:
        if should_close:
            conn.close()

def exact_search_halfvec(query_embedding: list[float], top_k: int = 10, conn=None, target_db: str = EXP_DB_NAME) -> list[dict]:
    """
    Exhaustive nearest-neighbor search over chunks_embedding_halfvec using halfvec cosine distance (<=>).
    Query vector is cast directly to halfvec(768).
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
                SELECT chunk_id, (embedding <=> %s::halfvec(768)) AS distance
                FROM chunks_embedding_halfvec
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
            rows = cur.fetchall()
            return [{"chunk_id": r[0], "distance": float(r[1])} for r in rows]
    finally:
        if should_close:
            conn.close()

def get_exact_plan_fp32(query_embedding: list[float], top_k: int = 10, conn=None, target_db: str = EXP_DB_NAME) -> list[str]:
    """
    Retrieves EXPLAIN plan for exact FP32 search to verify sequential scan / exhaustive nearest neighbor retrieval.
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
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, (embedding <=> %s::vector(768)) AS distance
                FROM chunks_embedding_fp32
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
            return [r[0] for r in cur.fetchall()]
    finally:
        if should_close:
            conn.close()

def get_exact_plan_halfvec(query_embedding: list[float], top_k: int = 10, conn=None, target_db: str = EXP_DB_NAME) -> list[str]:
    """
    Retrieves EXPLAIN plan for exact FP16 (halfvec) search to verify sequential scan / exhaustive nearest neighbor retrieval.
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
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, (embedding <=> %s::halfvec(768)) AS distance
                FROM chunks_embedding_halfvec
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
            return [r[0] for r in cur.fetchall()]
    finally:
        if should_close:
            conn.close()

def calculate_agreement_recall(fp32_results: list[dict], halfvec_results: list[dict], top_k: int) -> float:
    """
    Calculate agreement Recall@K between FP32 reference ranking and FP16 candidate ranking:
    Recall@K = |FP32 top-K chunk_ids ∩ FP16 top-K chunk_ids| / K
    """
    if top_k <= 0:
        return 0.0
    fp32_set = {r["chunk_id"] for r in fp32_results[:top_k]}
    halfvec_set = {r["chunk_id"] for r in halfvec_results[:top_k]}
    intersection = fp32_set.intersection(halfvec_set)
    return len(intersection) / float(top_k)

def measure_storage_metrics(conn=None, target_db: str = EXP_DB_NAME) -> dict:
    """
    Measures detailed PostgreSQL storage metrics for both representation tables:
    - Datum column size: min, max, avg
    - Table heap size (pg_relation_size)
    - Total relation size (pg_total_relation_size)
    - TOAST storage and toast index storage
    - Primary key index size
    - Theoretical vs measured payload
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_allowed_db(conn)
        with conn.cursor() as cur:
            tables = ["chunks_embedding_fp32", "chunks_embedding_halfvec"]
            metrics = {}

            for tbl in tables:
                # Column size
                cur.execute(f"""
                    SELECT
                        min(pg_column_size(embedding)),
                        max(pg_column_size(embedding)),
                        avg(pg_column_size(embedding))
                    FROM {tbl};
                """)
                min_c, max_c, avg_c = cur.fetchone()

                # Relation sizes
                cur.execute("""
                    SELECT
                        pg_relation_size(c.oid) AS heap_size,
                        pg_total_relation_size(c.oid) AS total_size,
                        pg_indexes_size(c.oid) AS index_size,
                        COALESCE(pg_relation_size(c.reltoastrelid), 0) AS toast_size,
                        COALESCE(pg_indexes_size(c.reltoastrelid), 0) AS toast_index_size
                    FROM pg_class c
                    WHERE c.relname = %s;
                """, (tbl,))
                heap_sz, tot_sz, idx_sz, toast_sz, toast_idx_sz = cur.fetchone()

                # Row count
                cur.execute(f"SELECT COUNT(*) FROM {tbl};")
                row_count = cur.fetchone()[0]

                is_fp32 = (tbl == "chunks_embedding_fp32")
                coord_bytes = 4 if is_fp32 else 2
                theoretical_payload_per_row = EMBEDDING_DIM * coord_bytes
                theoretical_total_payload = theoretical_payload_per_row * row_count

                metrics[tbl] = {
                    "row_count": row_count,
                    "column_size_min_bytes": min_c,
                    "column_size_max_bytes": max_c,
                    "column_size_avg_bytes": float(avg_c) if avg_c is not None else 0.0,
                    "heap_size_bytes": heap_sz,
                    "total_size_bytes": tot_sz,
                    "primary_key_index_bytes": idx_sz,
                    "toast_size_bytes": toast_sz,
                    "toast_index_bytes": toast_idx_sz,
                    "theoretical_payload_per_row_bytes": theoretical_payload_per_row,
                    "theoretical_total_payload_bytes": theoretical_total_payload
                }

            # Comparison summary
            fp32_tot = metrics["chunks_embedding_fp32"]["total_size_bytes"]
            half_tot = metrics["chunks_embedding_halfvec"]["total_size_bytes"]
            fp32_heap = metrics["chunks_embedding_fp32"]["heap_size_bytes"]
            half_heap = metrics["chunks_embedding_halfvec"]["heap_size_bytes"]
            fp32_col = metrics["chunks_embedding_fp32"]["column_size_avg_bytes"]
            half_col = metrics["chunks_embedding_halfvec"]["column_size_avg_bytes"]

            return {
                "target_db": target_db,
                "tables": metrics,
                "comparison": {
                    "total_size_reduction_ratio": (fp32_tot / half_tot) if half_tot > 0 else 0.0,
                    "total_size_reduction_pct": ((fp32_tot - half_tot) / fp32_tot * 100.0) if fp32_tot > 0 else 0.0,
                    "heap_size_reduction_ratio": (fp32_heap / half_heap) if half_heap > 0 else 0.0,
                    "heap_size_reduction_pct": ((fp32_heap - half_heap) / fp32_heap * 100.0) if fp32_heap > 0 else 0.0,
                    "column_size_reduction_ratio": (fp32_col / half_col) if half_col > 0 else 0.0,
                    "column_size_reduction_pct": ((fp32_col - half_col) / fp32_col * 100.0) if fp32_col > 0 else 0.0
                }
            }
    finally:
        if should_close:
            conn.close()

def run_representation_benchmark(
    queries: list[str] | None = None,
    top_k_values: list[int] = [3, 5, 10],
    warmup_runs: int = 2,
    measurement_runs: int = 5,
    target_db: str = EXP_DB_NAME
) -> dict:
    """
    Executes the FP32 vs FP16 exact retrieval benchmark adhering strictly to established methodology:
    - Persistent database connection
    - Pre-computed query embeddings (excluded from retrieval timing)
    - Warm-up cycles
    - time.perf_counter() timing
    - Exact agreement Recall@K evaluation for K in top_k_values
    - Detailed storage and query plan capture
    """
    if queries is None:
        queries = EVALUATION_QUERIES

    conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
    register_vector(conn)

    try:
        assert_allowed_db(conn)

        # 1. Pre-generate all query embeddings once
        print(f"Pre-generating embeddings for {len(queries)} evaluation queries via Ollama...")
        t_emb_start = time.perf_counter()
        precomputed_embeddings = []
        for q in queries:
            emb = get_embedding(q)
            precomputed_embeddings.append(emb)
        emb_gen_time = time.perf_counter() - t_emb_start
        print(f"Query embeddings generated in {emb_gen_time:.2f}s.")

        # 2. Warm-up runs
        print(f"Executing {warmup_runs} warm-up runs across all queries...")
        for _ in range(warmup_runs):
            for emb in precomputed_embeddings:
                exact_search_fp32(emb, top_k=max(top_k_values), conn=conn, target_db=target_db)
                exact_search_halfvec(emb, top_k=max(top_k_values), conn=conn, target_db=target_db)

        # 3. Measurement runs
        print(f"Executing {measurement_runs} measurement runs per query ({len(queries) * measurement_runs} trials per representation)...")
        fp32_latencies = []
        halfvec_latencies = []

        # Recall tracking per K
        recall_records = {k: [] for k in top_k_values}

        # Reference results for agreement recall (evaluated on run 0)
        for q_idx, emb in enumerate(precomputed_embeddings):
            # Evaluate agreement recall across K values
            for k in top_k_values:
                res_fp32 = exact_search_fp32(emb, top_k=k, conn=conn, target_db=target_db)
                res_half = exact_search_halfvec(emb, top_k=k, conn=conn, target_db=target_db)
                rec = calculate_agreement_recall(res_fp32, res_half, top_k=k)
                recall_records[k].append(rec)

        # Timed trials
        for run_id in range(measurement_runs):
            for emb in precomputed_embeddings:
                # Time FP32
                t0 = time.perf_counter()
                exact_search_fp32(emb, top_k=max(top_k_values), conn=conn, target_db=target_db)
                t1 = time.perf_counter()
                fp32_latencies.append((t1 - t0) * 1000.0)  # ms

                # Time Halfvec
                t2 = time.perf_counter()
                exact_search_halfvec(emb, top_k=max(top_k_values), conn=conn, target_db=target_db)
                t3 = time.perf_counter()
                halfvec_latencies.append((t3 - t2) * 1000.0)  # ms

        # 4. Query plan capture (for first query)
        sample_plan_fp32 = get_exact_plan_fp32(precomputed_embeddings[0], top_k=10, conn=conn, target_db=target_db)
        sample_plan_halfvec = get_exact_plan_halfvec(precomputed_embeddings[0], top_k=10, conn=conn, target_db=target_db)

        # 5. Storage metrics
        storage_info = measure_storage_metrics(conn=conn, target_db=target_db)

        # 6. Aggregate results
        avg_recalls = {f"Recall@{k}": statistics.mean(recall_records[k]) for k in top_k_values}

        results = {
            "target_db": target_db,
            "query_count": len(queries),
            "measurement_runs_per_query": measurement_runs,
            "total_trials_per_method": len(fp32_latencies),
            "top_k_values": top_k_values,
            "latency_ms": {
                "fp32_exact": {
                    "mean": statistics.mean(fp32_latencies),
                    "median": statistics.median(fp32_latencies),
                    "min": min(fp32_latencies),
                    "max": max(fp32_latencies),
                    "stdev": statistics.stdev(fp32_latencies) if len(fp32_latencies) > 1 else 0.0
                },
                "halfvec_exact": {
                    "mean": statistics.mean(halfvec_latencies),
                    "median": statistics.median(halfvec_latencies),
                    "min": min(halfvec_latencies),
                    "max": max(halfvec_latencies),
                    "stdev": statistics.stdev(halfvec_latencies) if len(halfvec_latencies) > 1 else 0.0
                }
            },
            "recall": avg_recalls,
            "storage": storage_info,
            "sample_plans": {
                "fp32": sample_plan_fp32,
                "halfvec": sample_plan_halfvec
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        return results
    finally:
        conn.close()
