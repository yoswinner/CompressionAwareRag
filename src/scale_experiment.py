import os
import csv
import json
import time
import re
import hashlib
import statistics
import psycopg
from pgvector.psycopg import register_vector
from datasets import load_dataset
from src.config import (
    get_db_connection_string,
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    EMBEDDING_DIM
)
from src.preprocessing import chunk_text
from src.embeddings import get_embedding, get_embeddings_batch
from src.representations import EVALUATION_QUERIES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CANONICAL_CACHE_PATH = os.path.join(DATA_DIR, "raw", "msmarco_10k_canonical.tsv")
CANONICAL_MANIFEST_PATH = os.path.join(DATA_DIR, "raw", "msmarco_10k_canonical.meta.json")

SCALE_DB_NAME = "compression_aware_rag_scale"
EXP_DB_NAME = "compression_aware_rag_exp"
TEST_DB_NAME = "compression_aware_rag_test"
PROD_DB_NAME = "compression_aware_rag"

# ==============================================================================
# 1. EXCLUSIVE SAFETY GUARDS & TRANSACTION PROTOCOLS
# ==============================================================================

def assert_scale_mutation_db(conn):
    """
    Strict fail-closed safety guard for Stage 4 mutations.
    Permits mutations EXCLUSIVELY in 'compression_aware_rag_scale'.
    Rejects test DB, exp DB, production DB, and chatbot DB unconditionally.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        active_db = cur.fetchone()[0]
        if active_db != SCALE_DB_NAME:
            raise RuntimeError(
                f"CRITICAL SAFETY VIOLATION: Mutation attempted on unauthorized database '{active_db}'. "
                f"Stage 4 mutations must strictly target '{SCALE_DB_NAME}'. "
                f"Operation aborted to preserve environment isolation."
            )
        return active_db

def get_exp_read_only_connection():
    """
    Establishes a connection to 'compression_aware_rag_exp' and immediately enters
    read-only transaction mode BEFORE any query that begins the transaction.
    Then verifies current_database() within that verified read-only transaction.
    """
    conn = psycopg.connect(
        get_db_connection_string(EXP_DB_NAME),
        autocommit=False
    )
    conn.read_only = True
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        active_db = cur.fetchone()[0]
        if active_db != EXP_DB_NAME:
            conn.close()
            raise RuntimeError(
                f"CRITICAL SAFETY VIOLATION: Read-only reuse attempted on unauthorized database '{active_db}'. "
                f"Expected '{EXP_DB_NAME}'. Operation aborted."
            )
    return conn

# ==============================================================================
# 2. ADMINISTRATIVE BOOTSTRAP & SCHEMA INITIALIZATION
# ==============================================================================

def create_scaling_database_if_missing(target_db: str = SCALE_DB_NAME) -> bool:
    """
    Creates the dedicated scaling database using an administrative connection to 'postgres'
    with autocommit=True (since CREATE DATABASE cannot run in a transaction block).
    Never touches production or unrelated databases.
    """
    admin_conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    with psycopg.connect(admin_conn_str, autocommit=True) as admin_conn:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target_db,))
            if not cur.fetchone():
                print(f"Creating dedicated scaling database '{target_db}'...")
                cur.execute(f"CREATE DATABASE {target_db};")
                return True
            else:
                print(f"Scaling database '{target_db}' already exists.")
                return False

def initialize_scaling_schema(conn=None, target_db: str = SCALE_DB_NAME):
    """
    Initializes pgvector, base tables (documents, document_chunks), and scale-tagged
    representation tables (5k and 10k) in target_db.
    Enforces assert_scale_mutation_db if targeting SCALE_DB_NAME.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        should_close = True

    try:
        if target_db == SCALE_DB_NAME:
            assert_scale_mutation_db(conn)

        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        register_vector(conn)

        with conn.cursor() as cur:
            # Base tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) UNIQUE NOT NULL,
                    content TEXT NOT NULL
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(768),
                    UNIQUE(document_id, chunk_index)
                );
            """)

            # Scale 5,000 Representation Tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_5k_fp32 (
                    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding vector(768) NOT NULL
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_5k_halfvec (
                    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding halfvec(768) NOT NULL
                );
            """)

            # Scale 10,000 Representation Tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_10k_fp32 (
                    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding vector(768) NOT NULL
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_10k_halfvec (
                    chunk_id INT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding halfvec(768) NOT NULL
                );
            """)
            print(f"Scaling schema initialized in '{target_db}'.")
    finally:
        if should_close:
            conn.close()

# ==============================================================================
# 3. CANONICAL CACHE DOWNLOAD & KEY-SET VALIDATION
# ==============================================================================

def download_canonical_cache(
    target_chunks: int = 10000,
    chunk_size: int = 300,
    overlap: int = 50,
    output_path: str = CANONICAL_CACHE_PATH,
    manifest_path: str = CANONICAL_MANIFEST_PATH,
    force_download: bool = False
) -> tuple[int, int]:
    """
    Streams documents from Hugging Face 'BeIR/msmarco' (corpus split) exactly once
    until target_chunks is satisfied. Writes atomically to output_path and generates a manifest.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not force_download and os.path.exists(output_path) and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if meta.get("complete", False) and meta.get("actual_chunks", 0) >= target_chunks:
                print(f"Valid canonical cache exists: {output_path} ({meta['actual_documents']} docs, {meta['actual_chunks']} chunks).")
                return meta["actual_documents"], meta["actual_chunks"]
        except Exception as e:
            print(f"Warning: Manifest read failed ({e}). Re-streaming...")

    print(f"Streaming canonical MS MARCO dataset (target: ~{target_chunks} chunks)...")
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)

    tmp_path = output_path + ".tmp"
    doc_count = 0
    total_chunks = 0
    hasher = hashlib.sha256()

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for item in ds:
            doc_id = str(item.get("_id", "")).strip()
            text = item.get("text", "").strip()
            if not doc_id or not text:
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                continue

            line = f"{doc_id}\t{text}\n"
            hasher.update(line.encode("utf-8"))
            writer.writerow([doc_id, text])
            doc_count += 1
            total_chunks += len(chunks)

            if total_chunks >= target_chunks:
                break

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)

    manifest_data = {
        "source": "BeIR/msmarco",
        "config": "corpus",
        "split": "corpus",
        "target_chunks": target_chunks,
        "actual_documents": doc_count,
        "actual_chunks": total_chunks,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "sha256_tsv": hasher.hexdigest(),
        "complete": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    print(f"Canonical cache created: {doc_count} docs, {total_chunks} chunks -> {output_path}")
    return doc_count, total_chunks

def validate_canonical_cache_against_baseline(
    canonical_tsv_path: str = CANONICAL_CACHE_PATH,
    chunk_size: int = 300,
    overlap: int = 50
) -> dict:
    """
    Validates the first 1,001 chunks of the canonical cache against compression_aware_rag_exp:
    - Uses a strictly read-only connection entering read-only transaction mode before any statement.
    - Builds baseline_map[(doc_id, chunk_index)] = sha256(chunk_text).
    - Verifies matching keys, matching hashes, and complete key-set equality.
    - Raises RuntimeError immediately if any mismatch is found.
    """
    if not os.path.exists(canonical_tsv_path):
        raise FileNotFoundError(f"Canonical cache not found: {canonical_tsv_path}")

    # 1. Fetch baseline chunks in read-only mode
    conn_ro = get_exp_read_only_connection()
    try:
        with conn_ro.cursor() as cur:
            cur.execute("""
                SELECT d.doc_id, dc.chunk_index, dc.chunk_text
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id;
            """)
            baseline_rows = cur.fetchall()
    finally:
        conn_ro.close()

    if len(baseline_rows) != 1001:
        raise RuntimeError(
            f"BASELINE INTEGRITY FAILURE: Expected 1,001 chunks in '{EXP_DB_NAME}', "
            f"found {len(baseline_rows)}."
        )

    baseline_map = {
        (str(r[0]).strip(), int(r[1])): hashlib.sha256(r[2].encode("utf-8")).hexdigest()
        for r in baseline_rows
    }

    # 2. Stream and chunk canonical TSV
    visited_keys = set()
    matched_count = 0
    chunk_counter = 0

    with open(canonical_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue
            doc_id = row[0].strip()
            text = row[1].strip()

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for c_idx, c_txt in enumerate(chunks):
                if chunk_counter < 1001:
                    key = (doc_id, c_idx)
                    h = hashlib.sha256(c_txt.encode("utf-8")).hexdigest()

                    if key not in baseline_map:
                        raise RuntimeError(
                            f"CANONICAL VALIDATION FAILURE at chunk {chunk_counter}: "
                            f"Key {key} not found in baseline map!"
                        )
                    if baseline_map[key] != h:
                        raise RuntimeError(
                            f"CANONICAL VALIDATION FAILURE at chunk {chunk_counter}: "
                            f"SHA-256 mismatch for key {key}!"
                        )

                    visited_keys.add(key)
                    matched_count += 1

                chunk_counter += 1
                if chunk_counter >= 1001:
                    break
            if chunk_counter >= 1001:
                break

    # 3. Assert complete key-set equality
    expected_keys = set(baseline_map.keys())
    if visited_keys != expected_keys:
        missing = expected_keys - visited_keys
        extra = visited_keys - expected_keys
        raise RuntimeError(
            f"KEY-SET EQUALITY FAILURE: Missing={len(missing)}, Extra={len(extra)}."
        )

    print(f"Cryptographic Validation Succeeded: 1,001 chunks verified (Matched={matched_count}, Key-set equality=True).")
    return {
        "verified": True,
        "matched_chunks": matched_count,
        "total_baseline_chunks": len(baseline_map)
    }

# ==============================================================================
# 4. INGESTION & EMBEDDING WITH ZERO OVERSHOOT
# ==============================================================================

def ingest_scaling_corpus(
    conn=None,
    target_db: str = SCALE_DB_NAME,
    canonical_tsv_path: str = CANONICAL_CACHE_PATH,
    chunk_size: int = 300,
    overlap: int = 50,
    target_chunks: int = 10000,
    batch_size: int = 25
) -> dict:
    """
    Ingests base documents and document_chunks into target_db up to EXACTLY target_chunks.
    - Reuses the first 1,001 validated embeddings from compression_aware_rag_exp via read-only transaction.
    - Generates embeddings for chunks 1,001..target_chunks-1 via Ollama nomic-embed-text.
    - Slices final document chunks to guarantee zero chunk overshoot.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_scale_mutation_db(conn)
        initialize_scaling_schema(conn=conn, target_db=target_db)

        # Check existing chunk count
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            existing_chunks = cur.fetchone()[0]

        if existing_chunks >= target_chunks:
            print(f"Target chunks already satisfied in '{target_db}' ({existing_chunks} chunks). Skipping ingestion.")
            return {"target_db": target_db, "chunks": existing_chunks, "already_populated": True}

        # 1. Fetch reusable 1,001 embeddings from compression_aware_rag_exp
        print("Fetching reusable baseline embeddings from 'compression_aware_rag_exp' (read-only)...")
        conn_ro = get_exp_read_only_connection()
        try:
            with conn_ro.cursor() as cur:
                cur.execute("""
                    SELECT d.doc_id, dc.chunk_index, dc.embedding
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id;
                """)
                reusable_embs = {(str(r[0]).strip(), int(r[1])): r[2] for r in cur.fetchall()}
        finally:
            conn_ro.close()

        print(f"Loaded {len(reusable_embs)} reusable embeddings.")

        # 2. Ingest documents and chunks sequentially
        inserted_chunks = existing_chunks
        t0 = time.perf_counter()

        with open(canonical_tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    continue
                doc_id = row[0].strip()
                content = row[1].strip()

                if inserted_chunks >= target_chunks:
                    break

                chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    continue

                # Strict cutoff: slice final chunks to prevent overshoot
                if inserted_chunks + len(chunks) > target_chunks:
                    chunks = chunks[:target_chunks - inserted_chunks]

                with conn.cursor() as cur:
                    # Insert document
                    cur.execute(
                        "INSERT INTO documents (doc_id, content) VALUES (%s, %s) "
                        "ON CONFLICT (doc_id) DO NOTHING RETURNING id;",
                        (doc_id, content)
                    )
                    res = cur.fetchone()
                    if res is not None:
                        db_doc_id = res[0]
                    else:
                        cur.execute("SELECT id FROM documents WHERE doc_id = %s;", (doc_id,))
                        db_doc_id = cur.fetchone()[0]

                    # Prepare embeddings (reuse baseline or call Ollama)
                    embs_to_insert = []
                    chunks_needing_embed = []
                    indices_needing_embed = []

                    for idx, c_txt in enumerate(chunks):
                        key = (doc_id, idx)
                        if key in reusable_embs:
                            embs_to_insert.append((idx, c_txt, reusable_embs[key]))
                        else:
                            chunks_needing_embed.append(c_txt)
                            indices_needing_embed.append(idx)

                    # Generate new embeddings in batches
                    if chunks_needing_embed:
                        new_embs = get_embeddings_batch(chunks_needing_embed)
                        for idx, c_txt, emb in zip(indices_needing_embed, chunks_needing_embed, new_embs):
                            embs_to_insert.append((idx, c_txt, emb))

                    # Insert chunks
                    for idx, c_txt, emb in embs_to_insert:
                        cur.execute(
                            "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) "
                            "VALUES (%s, %s, %s, %s) "
                            "ON CONFLICT (document_id, chunk_index) DO NOTHING RETURNING id;",
                            (db_doc_id, idx, c_txt, emb)
                        )
                        if cur.fetchone() is not None:
                            inserted_chunks += 1

                if inserted_chunks % 500 == 0:
                    print(f"Ingestion progress: {inserted_chunks}/{target_chunks} chunks ({inserted_chunks/target_chunks*100:.1f}%)...")

        elapsed = time.perf_counter() - t0
        print(f"Ingestion complete: {inserted_chunks} chunks in {elapsed:.2f}s.")
        return {"target_db": target_db, "chunks": inserted_chunks, "elapsed_seconds": elapsed}
    finally:
        if should_close:
            conn.close()

def populate_scale_representation_tables(
    scale: int,
    conn=None,
    target_db: str = SCALE_DB_NAME
) -> dict:
    """
    Populates chunks_{scale//1000}k_fp32 and chunks_{scale//1000}k_halfvec with EXACTLY scale rows.
    Takes the first scale chunks from document_chunks ordered by id.
    Casts FP32 to FP16 in database (embedding::halfvec(768)).
    """
    if scale not in (5000, 10000):
        raise ValueError(f"Unsupported scale: {scale}. Must be 5000 or 10000.")

    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_scale_mutation_db(conn)
        tag = f"{scale // 1000}k"
        fp32_table = f"chunks_{tag}_fp32"
        half_table = f"chunks_{tag}_halfvec"

        with conn.cursor() as cur:
            # Check source count
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            source_count = cur.fetchone()[0]
            if source_count < scale:
                raise RuntimeError(
                    f"INSUFFICIENT SOURCE CHUNKS: Table 'document_chunks' has {source_count} rows, "
                    f"but scale {scale} requires at least {scale} rows."
                )

            # Check if already populated
            cur.execute(f"SELECT COUNT(*) FROM {fp32_table};")
            fp32_count = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {half_table};")
            half_count = cur.fetchone()[0]

            if fp32_count == scale and half_count == scale:
                print(f"Scale tables for {tag} already populated with exactly {scale} rows.")
                return {"scale": scale, "fp32_count": fp32_count, "halfvec_count": half_count, "already_populated": True}

            print(f"Populating representation tables for {tag} ({scale} rows)...")

            # Transactional population
            cur.execute(f"TRUNCATE TABLE {fp32_table} CASCADE;")
            cur.execute(f"TRUNCATE TABLE {half_table} CASCADE;")

            cur.execute(f"""
                INSERT INTO {fp32_table} (chunk_id, embedding)
                SELECT id, embedding
                FROM document_chunks
                ORDER BY id ASC
                LIMIT %s;
            """, (scale,))

            cur.execute(f"""
                INSERT INTO {half_table} (chunk_id, embedding)
                SELECT id, embedding::halfvec(768)
                FROM document_chunks
                ORDER BY id ASC
                LIMIT %s;
            """, (scale,))

            # Verification
            cur.execute(f"SELECT COUNT(*) FROM {fp32_table};")
            final_fp32 = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {half_table};")
            final_half = cur.fetchone()[0]

            # Null check
            cur.execute(f"SELECT COUNT(*) FROM {fp32_table} WHERE embedding IS NULL;")
            nulls_fp32 = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {half_table} WHERE embedding IS NULL;")
            nulls_half = cur.fetchone()[0]

            # Dims check
            cur.execute(f"SELECT DISTINCT vector_dims(embedding) FROM {fp32_table};")
            dims_fp32 = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT DISTINCT vector_dims(embedding) FROM {half_table};")
            dims_half = [r[0] for r in cur.fetchall()]

            if final_fp32 != scale or final_half != scale or nulls_fp32 > 0 or nulls_half > 0:
                raise RuntimeError(
                    f"POPULATION INTEGRITY FAILURE: FP32={final_fp32}, Half={final_half}, "
                    f"NullsFP32={nulls_fp32}, NullsHalf={nulls_half}."
                )

            print(f"Scale {tag} successfully populated: {final_fp32} FP32 rows, {final_half} HalfVec rows (0 nulls, dims: {dims_fp32}).")
            return {
                "scale": scale,
                "fp32_count": final_fp32,
                "halfvec_count": final_half,
                "dims": dims_fp32,
                "nulls": 0,
                "already_populated": False
            }
    finally:
        if should_close:
            conn.close()

# ==============================================================================
# 5. HNSW INDEX CONSTRUCTION & DISAGGREGATED STORAGE METRICS
# ==============================================================================

def build_scale_hnsw_indexes(
    scale: int,
    conn=None,
    target_db: str = SCALE_DB_NAME,
    m: int = 16,
    ef_construction: int = 64
) -> dict:
    """
    Idempotently builds HNSW indexes for the target scale (5k or 10k).
    Measures wall-clock build duration strictly around CREATE INDEX.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        assert_scale_mutation_db(conn)
        tag = f"{scale // 1000}k"
        fp32_tbl = f"chunks_{tag}_fp32"
        half_tbl = f"chunks_{tag}_halfvec"
        fp32_idx = f"chunks_{tag}_fp32_hnsw_idx"
        half_idx = f"chunks_{tag}_halfvec_hnsw_idx"

        results = {"scale": scale, "indexes": {}}

        with conn.cursor() as cur:
            # Check existing indexes
            cur.execute(f"""
                SELECT indexname
                FROM pg_indexes
                WHERE tablename IN ('{fp32_tbl}', '{half_tbl}');
            """)
            existing = {r[0] for r in cur.fetchall()}

            # 1. FP32 HNSW
            if fp32_idx in existing:
                cur.execute(f"SELECT pg_relation_size('{fp32_idx}'::regclass);")
                size = cur.fetchone()[0]
                results["indexes"]["fp32"] = {
                    "index_name": fp32_idx,
                    "already_existed": True,
                    "build_time_seconds": None,
                    "size_bytes": size
                }
                print(f"Index '{fp32_idx}' already exists. Size: {size} bytes.")
            else:
                print(f"Building FP32 HNSW index '{fp32_idx}' (m={m}, ef_construction={ef_construction})...")
                t0 = time.perf_counter()
                cur.execute(f"""
                    CREATE INDEX {fp32_idx}
                    ON {fp32_tbl}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction});
                """)
                b_time = time.perf_counter() - t0
                cur.execute(f"SELECT pg_relation_size('{fp32_idx}'::regclass);")
                size = cur.fetchone()[0]
                results["indexes"]["fp32"] = {
                    "index_name": fp32_idx,
                    "already_existed": False,
                    "build_time_seconds": b_time,
                    "size_bytes": size
                }
                print(f"Built '{fp32_idx}' in {b_time:.4f}s, size: {size} bytes.")

            # 2. FP16 (halfvec) HNSW
            if half_idx in existing:
                cur.execute(f"SELECT pg_relation_size('{half_idx}'::regclass);")
                size = cur.fetchone()[0]
                results["indexes"]["halfvec"] = {
                    "index_name": half_idx,
                    "already_existed": True,
                    "build_time_seconds": None,
                    "size_bytes": size
                }
                print(f"Index '{half_idx}' already exists. Size: {size} bytes.")
            else:
                print(f"Building FP16 HNSW index '{half_idx}' (m={m}, ef_construction={ef_construction})...")
                t1 = time.perf_counter()
                cur.execute(f"""
                    CREATE INDEX {half_idx}
                    ON {half_tbl}
                    USING hnsw (embedding halfvec_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction});
                """)
                b_time = time.perf_counter() - t1
                cur.execute(f"SELECT pg_relation_size('{half_idx}'::regclass);")
                size = cur.fetchone()[0]
                results["indexes"]["halfvec"] = {
                    "index_name": half_idx,
                    "already_existed": False,
                    "build_time_seconds": b_time,
                    "size_bytes": size
                }
                print(f"Built '{half_idx}' in {b_time:.4f}s, size: {size} bytes.")

        return results
    finally:
        if should_close:
            conn.close()

def get_disaggregated_storage_metrics(
    scale: int,
    conn=None,
    target_db: str = SCALE_DB_NAME
) -> dict:
    """
    Measures disaggregated storage footprints for both representation tables:
    Heap, TOAST, PK Index, HNSW Index, and Inclusive Total (pg_total_relation_size).
    Avoids double-counting by distinctly labeling components.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        if target_db == SCALE_DB_NAME:
            assert_scale_mutation_db(conn)

        tag = f"{scale // 1000}k"
        tables = [f"chunks_{tag}_fp32", f"chunks_{tag}_halfvec"]
        metrics = {}

        with conn.cursor() as cur:
            for tbl in tables:
                # Column size
                cur.execute(f"""
                    SELECT min(pg_column_size(embedding)),
                           max(pg_column_size(embedding)),
                           avg(pg_column_size(embedding))
                    FROM {tbl};
                """)
                min_c, max_c, avg_c = cur.fetchone()

                # Relation components
                cur.execute("""
                    SELECT
                        pg_relation_size(c.oid) AS heap_bytes,
                        COALESCE(pg_relation_size(c.reltoastrelid), 0) AS toast_table_bytes,
                        COALESCE(pg_indexes_size(c.reltoastrelid), 0) AS toast_index_bytes,
                        pg_total_relation_size(c.oid) AS total_inclusive_bytes
                    FROM pg_class c
                    WHERE c.relname = %s;
                """, (tbl,))
                heap_b, toast_b, toast_idx_b, total_inc_b = cur.fetchone()

                # Primary key index
                cur.execute(f"SELECT pg_relation_size('{tbl}_pkey'::regclass);")
                pk_b = cur.fetchone()[0]

                # HNSW index (if exists)
                hnsw_idx_name = f"{tbl}_hnsw_idx"
                cur.execute(f"""
                    SELECT pg_relation_size(oid)
                    FROM pg_class
                    WHERE relname = '{hnsw_idx_name}';
                """)
                res_hnsw = cur.fetchone()
                hnsw_b = res_hnsw[0] if res_hnsw else 0

                is_fp32 = "fp32" in tbl
                metrics["fp32" if is_fp32 else "halfvec"] = {
                    "table_name": tbl,
                    "row_count": scale,
                    "column_size_min_bytes": min_c,
                    "column_size_max_bytes": max_c,
                    "column_size_avg_bytes": float(avg_c) if avg_c is not None else 0.0,
                    "heap_bytes": heap_b,
                    "toast_table_bytes": toast_b,
                    "toast_index_bytes": toast_idx_b,
                    "toast_total_bytes": toast_b + toast_idx_b,
                    "pk_index_bytes": pk_b,
                    "hnsw_index_bytes": hnsw_b,
                    "total_inclusive_bytes": total_inc_b
                }

        fp32_m = metrics["fp32"]
        half_m = metrics["halfvec"]

        return {
            "scale": scale,
            "metrics": metrics,
            "comparisons": {
                "hnsw_index_reduction_pct": ((fp32_m["hnsw_index_bytes"] - half_m["hnsw_index_bytes"]) / fp32_m["hnsw_index_bytes"] * 100.0) if fp32_m["hnsw_index_bytes"] > 0 else 0.0,
                "hnsw_index_reduction_ratio": (fp32_m["hnsw_index_bytes"] / half_m["hnsw_index_bytes"]) if half_m["hnsw_index_bytes"] > 0 else 0.0,
                "total_inclusive_reduction_pct": ((fp32_m["total_inclusive_bytes"] - half_m["total_inclusive_bytes"]) / fp32_m["total_inclusive_bytes"] * 100.0) if fp32_m["total_inclusive_bytes"] > 0 else 0.0,
                "total_inclusive_reduction_ratio": (fp32_m["total_inclusive_bytes"] / half_m["total_inclusive_bytes"]) if half_m["total_inclusive_bytes"] > 0 else 0.0,
                "column_size_reduction_pct": ((fp32_m["column_size_avg_bytes"] - half_m["column_size_avg_bytes"]) / fp32_m["column_size_avg_bytes"] * 100.0) if fp32_m["column_size_avg_bytes"] > 0 else 0.0
            }
        }
    finally:
        if should_close:
            conn.close()

# ==============================================================================
# 6. BENCHMARK EXECUTION (SIX CONFIGURATIONS)
# ==============================================================================

def parse_plan_details(raw_lines: list[str]) -> dict:
    """Parses EXPLAIN (ANALYZE, BUFFERS) text for execution mode diagnostics."""
    plan_text = "\n".join(raw_lines)
    scan_type = "Unknown"
    index_name = None

    if "Index Scan using" in plan_text:
        scan_type = "Index Scan"
        m = re.search(r"Index Scan using (\S+)", plan_text)
        if m:
            index_name = m.group(1)
    elif "Seq Scan on" in plan_text:
        scan_type = "Seq Scan"
    elif "Bitmap Heap Scan on" in plan_text:
        scan_type = "Bitmap Scan"

    exec_time = None
    m_exec = re.search(r"Execution Time:\s+([\d\.]+)\s+ms", plan_text)
    if m_exec:
        exec_time = float(m_exec.group(1))

    plan_time = None
    m_plan = re.search(r"Planning Time:\s+([\d\.]+)\s+ms", plan_text)
    if m_plan:
        plan_time = float(m_plan.group(1))

    shared_hit = 0
    shared_read = 0
    m_hit = re.search(r"shared hit=(\d+)", plan_text)
    if m_hit:
        shared_hit = int(m_hit.group(1))
    m_read = re.search(r"(?:shared\s+)?read=(\d+)", plan_text)
    if m_read:
        shared_read = int(m_read.group(1))

    return {
        "scan_type": scan_type,
        "index_name_used": index_name,
        "execution_time_ms": exec_time,
        "planning_time_ms": plan_time,
        "shared_hit_buffers": shared_hit,
        "shared_read_buffers": shared_read,
        "raw_plan": raw_lines
    }

def exact_search_scale(
    query_embedding: list[float],
    scale: int,
    representation: str = "fp32",
    top_k: int = 10,
    conn=None,
    target_db: str = SCALE_DB_NAME,
    apply_guc: bool = True
) -> list[dict]:
    """
    Executes exhaustive nearest-neighbor search for scale tables.
    If apply_guc is True, configures session planner:
    SET enable_indexscan = off; SET enable_bitmapscan = off; SET enable_seqscan = on;
    Orders by the computed 'distance' alias so the query vector is bound only once.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        tag = f"{scale // 1000}k"
        tbl = f"chunks_{tag}_{representation}"
        cast_type = "vector(768)" if representation == "fp32" else "halfvec(768)"

        with conn.cursor() as cur:
            if apply_guc:
                cur.execute("SET enable_indexscan = off;")
                cur.execute("SET enable_bitmapscan = off;")
                cur.execute("SET enable_seqscan = on;")

            cur.execute(f"""
                SELECT chunk_id, (embedding <=> %s::{cast_type}) AS distance
                FROM {tbl}
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
            rows = cur.fetchall()
            return [{"chunk_id": r[0], "distance": float(r[1])} for r in rows]
    finally:
        if should_close:
            conn.close()

def get_exact_plan_scale(
    query_embedding: list[float],
    scale: int,
    representation: str = "fp32",
    top_k: int = 10,
    conn=None,
    target_db: str = SCALE_DB_NAME
) -> dict:
    """
    Captures EXPLAIN (ANALYZE, BUFFERS) for exact search.
    Programmatically asserts that the scan is a Seq Scan and touches NO index.
    Orders by distance alias to match exact_search_scale.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        tag = f"{scale // 1000}k"
        tbl = f"chunks_{tag}_{representation}"
        cast_type = "vector(768)" if representation == "fp32" else "halfvec(768)"

        with conn.cursor() as cur:
            cur.execute("SET enable_indexscan = off;")
            cur.execute("SET enable_bitmapscan = off;")
            cur.execute("SET enable_seqscan = on;")

            cur.execute(f"""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, (embedding <=> %s::{cast_type}) AS distance
                FROM {tbl}
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
            raw_lines = [r[0] for r in cur.fetchall()]

        parsed = parse_plan_details(raw_lines)
        if parsed["scan_type"] != "Seq Scan" or parsed["index_name_used"] is not None:
            raise RuntimeError(
                f"EXHAUSTIVE EXACT SCAN VIOLATION on '{tbl}': Expected Seq Scan with 0 indexes, "
                f"but planner chose scan_type='{parsed['scan_type']}', index_name='{parsed['index_name_used']}'."
            )
        return parsed
    finally:
        if should_close:
            conn.close()

def search_scale_hnsw(
    query_embedding: list[float],
    scale: int,
    representation: str = "fp32",
    enable_seqscan: bool = False,
    ef_search: int = 40,
    top_k: int = 10,
    conn=None,
    target_db: str = SCALE_DB_NAME,
    apply_guc: bool = True
) -> list[dict]:
    """
    Executes HNSW retrieval query on scale tables.
    If apply_guc is True, sets indexscan, bitmapscan, seqscan, and hnsw.ef_search.
    Retains direct distance expression in ORDER BY so pgvector can match the index operator class.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        tag = f"{scale // 1000}k"
        tbl = f"chunks_{tag}_{representation}"
        cast_type = "vector(768)" if representation == "fp32" else "halfvec(768)"

        with conn.cursor() as cur:
            if apply_guc:
                cur.execute("SET enable_indexscan = on;")
                cur.execute("SET enable_bitmapscan = on;")
                cur.execute(f"SET enable_seqscan = {'on' if enable_seqscan else 'off'};")
                cur.execute(f"SET hnsw.ef_search = {ef_search};")

            cur.execute(f"""
                SELECT chunk_id, (embedding <=> %s::{cast_type}) AS distance
                FROM {tbl}
                ORDER BY embedding <=> %s::{cast_type} ASC
                LIMIT %s;
            """, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()
            return [{"chunk_id": r[0], "distance": float(r[1])} for r in rows]
    finally:
        if should_close:
            conn.close()

def get_hnsw_plan_scale(
    query_embedding: list[float],
    scale: int,
    representation: str = "fp32",
    enable_seqscan: bool = False,
    ef_search: int = 40,
    top_k: int = 10,
    conn=None,
    target_db: str = SCALE_DB_NAME
) -> dict:
    """
    Captures EXPLAIN (ANALYZE, BUFFERS) for HNSW retrieval.
    If enable_seqscan is False, programmatically asserts Index Scan using the expected HNSW index.
    """
    should_close = False
    if conn is None:
        conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
        register_vector(conn)
        should_close = True

    try:
        tag = f"{scale // 1000}k"
        tbl = f"chunks_{tag}_{representation}"
        cast_type = "vector(768)" if representation == "fp32" else "halfvec(768)"
        expected_index = f"{tbl}_hnsw_idx"

        with conn.cursor() as cur:
            cur.execute("SET enable_indexscan = on;")
            cur.execute("SET enable_bitmapscan = on;")
            cur.execute(f"SET enable_seqscan = {'on' if enable_seqscan else 'off'};")
            cur.execute(f"SET hnsw.ef_search = {ef_search};")

            cur.execute(f"""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, (embedding <=> %s::{cast_type}) AS distance
                FROM {tbl}
                ORDER BY embedding <=> %s::{cast_type} ASC
                LIMIT %s;
            """, (query_embedding, query_embedding, top_k))
            raw_lines = [r[0] for r in cur.fetchall()]

        parsed = parse_plan_details(raw_lines)
        if not enable_seqscan:
            if parsed["scan_type"] != "Index Scan" or parsed["index_name_used"] != expected_index:
                raise RuntimeError(
                    f"FORCED-INDEX VERIFICATION FAILURE on '{tbl}': Expected Index Scan using "
                    f"'{expected_index}', but planner chose scan_type='{parsed['scan_type']}', "
                    f"index_name='{parsed['index_name_used']}'."
                )
        return parsed
    finally:
        if should_close:
            conn.close()

def calculate_overlap_recall(ref_results: list[dict], cand_results: list[dict], top_k: int) -> float:
    """Recall@K = |Top-K_ref ∩ Top-K_cand| / K"""
    if top_k <= 0:
        return 0.0
    ref_set = {r["chunk_id"] for r in ref_results[:top_k]}
    cand_set = {r["chunk_id"] for r in cand_results[:top_k]}
    return len(ref_set.intersection(cand_set)) / float(top_k)

def compute_paired_stats(fp32_list: list[float], halfvec_list: list[float]) -> dict:
    """Calculates paired latency difference (FP32 - Halfvec) and standard stats."""
    deltas = [f - h for f, h in zip(fp32_list, halfvec_list)]
    return {
        "mean_diff_ms": statistics.mean(deltas),
        "median_diff_ms": statistics.median(deltas),
        "min_diff_ms": min(deltas),
        "max_diff_ms": max(deltas),
        "fp32_faster_count": sum(1 for d in deltas if d < 0),
        "halfvec_faster_count": sum(1 for d in deltas if d > 0)
    }

def summarize_latencies(lat_list: list[float]) -> dict:
    """Computes mean, median, IQR, min, max, stdev."""
    if not lat_list:
        return {}
    sorted_l = sorted(lat_list)
    q1 = statistics.quantiles(sorted_l, n=4)[0] if len(sorted_l) >= 4 else min(sorted_l)
    q3 = statistics.quantiles(sorted_l, n=4)[2] if len(sorted_l) >= 4 else max(sorted_l)
    return {
        "mean": statistics.mean(lat_list),
        "median": statistics.median(lat_list),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(lat_list),
        "max": max(lat_list),
        "stdev": statistics.stdev(lat_list) if len(lat_list) > 1 else 0.0
    }

def run_scale_benchmark(
    scale: int,
    queries: list[str] | None = None,
    top_k_values: list[int] = [3, 5, 10],
    ef_search: int = 40,
    warmup_runs: int = 2,
    measurement_runs: int = 5,
    target_db: str = SCALE_DB_NAME
) -> dict:
    """
    Executes the full 6-configuration benchmark for the specified scale:
    1. FP32 Exact Search (forced seqscan)
    2. FP16 Exact Search (forced seqscan)
    3. FP32 Forced HNSW
    4. FP16 Forced HNSW
    5. FP32 Planner-Selected
    6. FP16 Planner-Selected
    Precomputes query embeddings, computes scale-specific FP32 exact ground truth,
    runs 100 trials per config with alternating order, records paired per-query stats,
    and programmatically verifies plans.
    """
    if queries is None:
        queries = EVALUATION_QUERIES

    conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
    register_vector(conn)

    try:
        if target_db == SCALE_DB_NAME:
            assert_scale_mutation_db(conn)

        print(f"\n=======================================================")
        print(f"RUNNING STAGE 4 BENCHMARK FOR SCALE {scale} CHUNKS ({target_db})")
        print(f"=======================================================")

        # 1. Precompute query embeddings once
        print(f"Pre-generating embeddings for {len(queries)} evaluation queries via Ollama...")
        t0 = time.perf_counter()
        precomputed_embeddings = [get_embedding(q) for q in queries]
        print(f"Precomputed {len(precomputed_embeddings)} vectors in {time.perf_counter() - t0:.2f}s.")

        max_k = max(top_k_values)

        # 2. Compute scale-specific FP32 exact ground truth
        print(f"Computing independent FP32 exact ground truth for scale {scale}...")
        with conn.cursor() as cur:
            cur.execute("SET enable_indexscan = off;")
            cur.execute("SET enable_bitmapscan = off;")
            cur.execute("SET enable_seqscan = on;")
        exact_ground_truths = {}
        for q_idx, emb in enumerate(precomputed_embeddings):
            exact_ground_truths[q_idx] = {}
            for k in top_k_values:
                exact_ground_truths[q_idx][k] = exact_search_scale(
                    emb, scale=scale, representation="fp32", top_k=k, conn=conn, target_db=target_db, apply_guc=False
                )

        # 3. Warm-up passes
        print(f"Executing {warmup_runs} warm-up passes across all 6 configurations...")
        for _ in range(warmup_runs):
            # Warm-up Exact
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = off;")
                cur.execute("SET enable_bitmapscan = off;")
                cur.execute("SET enable_seqscan = on;")
            for emb in precomputed_embeddings:
                exact_search_scale(emb, scale=scale, representation="fp32", top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)
                exact_search_scale(emb, scale=scale, representation="halfvec", top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)

            # Warm-up Forced HNSW
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = on;")
                cur.execute("SET enable_bitmapscan = on;")
                cur.execute("SET enable_seqscan = off;")
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
            for emb in precomputed_embeddings:
                search_scale_hnsw(emb, scale=scale, representation="fp32", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)
                search_scale_hnsw(emb, scale=scale, representation="halfvec", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)

            # Warm-up Planner Auto
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = on;")
                cur.execute("SET enable_bitmapscan = on;")
                cur.execute("SET enable_seqscan = on;")
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
            for emb in precomputed_embeddings:
                search_scale_hnsw(emb, scale=scale, representation="fp32", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)
                search_scale_hnsw(emb, scale=scale, representation="halfvec", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)

        # 4. Recall Evaluation against scale FP32 ground truth
        print("Evaluating ANN Recall@K against scale-specific FP32 ground truth...")
        forced_fp32_recall = {k: [] for k in top_k_values}
        forced_half_recall = {k: [] for k in top_k_values}
        exact_half_recall = {k: [] for k in top_k_values}
        relative_agreement = {k: [] for k in top_k_values}

        for q_idx, emb in enumerate(precomputed_embeddings):
            for k in top_k_values:
                ref_exact = exact_ground_truths[q_idx][k]
                cand_exact_half = exact_search_scale(emb, scale=scale, representation="halfvec", top_k=k, conn=conn, target_db=target_db, apply_guc=True)
                cand_forced_fp32 = search_scale_hnsw(emb, scale=scale, representation="fp32", enable_seqscan=False, ef_search=ef_search, top_k=k, conn=conn, target_db=target_db, apply_guc=True)
                cand_forced_half = search_scale_hnsw(emb, scale=scale, representation="halfvec", enable_seqscan=False, ef_search=ef_search, top_k=k, conn=conn, target_db=target_db, apply_guc=True)

                exact_half_recall[k].append(calculate_overlap_recall(ref_exact, cand_exact_half, top_k=k))
                forced_fp32_recall[k].append(calculate_overlap_recall(ref_exact, cand_forced_fp32, top_k=k))
                forced_half_recall[k].append(calculate_overlap_recall(ref_exact, cand_forced_half, top_k=k))
                relative_agreement[k].append(calculate_overlap_recall(cand_forced_fp32, cand_forced_half, top_k=k))

        # 5. Timed Measurement Trials (100 trials per configuration)
        print(f"Executing {measurement_runs} measurement passes (100 trials per configuration)...")
        latencies = {
            "fp32_exact": [],
            "halfvec_exact": [],
            "fp32_forced_hnsw": [],
            "halfvec_forced_hnsw": [],
            "fp32_planner_auto": [],
            "halfvec_planner_auto": []
        }

        for run_id in range(measurement_runs):
            order = ["fp32", "halfvec"] if (run_id % 2 == 0) else ["halfvec", "fp32"]

            # Mode 1: Exact Search (Exhaustive)
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = off;")
                cur.execute("SET enable_bitmapscan = off;")
                cur.execute("SET enable_seqscan = on;")
            for emb in precomputed_embeddings:
                for rep in order:
                    t_s = time.perf_counter()
                    exact_search_scale(emb, scale=scale, representation=rep, top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)
                    latencies[f"{rep}_exact"].append((time.perf_counter() - t_s) * 1000.0)

            # Mode 2: Forced HNSW
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = on;")
                cur.execute("SET enable_bitmapscan = on;")
                cur.execute("SET enable_seqscan = off;")
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
            for emb in precomputed_embeddings:
                for rep in order:
                    t_s = time.perf_counter()
                    search_scale_hnsw(emb, scale=scale, representation=rep, enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)
                    latencies[f"{rep}_forced_hnsw"].append((time.perf_counter() - t_s) * 1000.0)

            # Mode 3: Planner-Selected
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = on;")
                cur.execute("SET enable_bitmapscan = on;")
                cur.execute("SET enable_seqscan = on;")
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
            for emb in precomputed_embeddings:
                for rep in order:
                    t_s = time.perf_counter()
                    search_scale_hnsw(emb, scale=scale, representation=rep, enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db, apply_guc=False)
                    latencies[f"{rep}_planner_auto"].append((time.perf_counter() - t_s) * 1000.0)

        # Restore GUC defaults
        with conn.cursor() as cur:
            cur.execute("RESET enable_indexscan;")
            cur.execute("RESET enable_bitmapscan;")
            cur.execute("RESET enable_seqscan;")
            cur.execute("RESET hnsw.ef_search;")

        # 6. Diagnostic Execution Plans & Verification (Sample Query 0)
        sample_emb = precomputed_embeddings[0]
        plans = {
            "fp32_exact": get_exact_plan_scale(sample_emb, scale=scale, representation="fp32", top_k=max_k, conn=conn, target_db=target_db),
            "halfvec_exact": get_exact_plan_scale(sample_emb, scale=scale, representation="halfvec", top_k=max_k, conn=conn, target_db=target_db),
            "fp32_forced_hnsw": get_hnsw_plan_scale(sample_emb, scale=scale, representation="fp32", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db),
            "halfvec_forced_hnsw": get_hnsw_plan_scale(sample_emb, scale=scale, representation="halfvec", enable_seqscan=False, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db),
            "fp32_planner_auto": get_hnsw_plan_scale(sample_emb, scale=scale, representation="fp32", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db),
            "halfvec_planner_auto": get_hnsw_plan_scale(sample_emb, scale=scale, representation="halfvec", enable_seqscan=True, ef_search=ef_search, top_k=max_k, conn=conn, target_db=target_db)
        }

        # 7. Disaggregated Storage Metrics
        storage = get_disaggregated_storage_metrics(scale=scale, conn=conn, target_db=target_db)

        # 8. Package Results
        return {
            "scale": scale,
            "target_db": target_db,
            "query_count": len(queries),
            "trials_per_config": len(latencies["fp32_exact"]),
            "storage": storage,
            "exact_search": {
                "fp32_latency_ms": summarize_latencies(latencies["fp32_exact"]),
                "halfvec_latency_ms": summarize_latencies(latencies["halfvec_exact"]),
                "paired_diff": compute_paired_stats(latencies["fp32_exact"], latencies["halfvec_exact"]),
                "halfvec_exact_recall": {f"Recall@{k}": statistics.mean(exact_half_recall[k]) for k in top_k_values},
                "sample_plans": {"fp32": plans["fp32_exact"], "halfvec": plans["halfvec_exact"]}
            },
            "forced_hnsw": {
                "fp32_latency_ms": summarize_latencies(latencies["fp32_forced_hnsw"]),
                "halfvec_latency_ms": summarize_latencies(latencies["halfvec_forced_hnsw"]),
                "paired_diff": compute_paired_stats(latencies["fp32_forced_hnsw"], latencies["halfvec_forced_hnsw"]),
                "fp32_recall_vs_exact": {f"Recall@{k}": statistics.mean(forced_fp32_recall[k]) for k in top_k_values},
                "halfvec_recall_vs_exact": {f"Recall@{k}": statistics.mean(forced_half_recall[k]) for k in top_k_values},
                "relative_agreement": {f"Agreement@{k}": statistics.mean(relative_agreement[k]) for k in top_k_values},
                "sample_plans": {"fp32": plans["fp32_forced_hnsw"], "halfvec": plans["halfvec_forced_hnsw"]}
            },
            "planner_selected": {
                "fp32_latency_ms": summarize_latencies(latencies["fp32_planner_auto"]),
                "halfvec_latency_ms": summarize_latencies(latencies["halfvec_planner_auto"]),
                "sample_plans": {"fp32": plans["fp32_planner_auto"], "halfvec": plans["halfvec_planner_auto"]}
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    finally:
        conn.close()
