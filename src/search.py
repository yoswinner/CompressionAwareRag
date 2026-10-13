from src.database import get_connection

def exact_search(query_emb: list[float], top_k: int = 5, conn=None):
    """
    Perform an exhaustive cosine-similarity search.
    Supports caller-supplied connection for connection reuse and low latency.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.document_id, c.chunk_text, c.embedding <=> %s::vector AS distance
                FROM document_chunks c
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_emb, top_k))
            results = cur.fetchall()
            return [{"chunk_id": r[0], "document_id": r[1], "chunk_text": r[2], "distance": r[3]} for r in results]
    finally:
        if should_close:
            conn.close()

def create_hnsw_index(m: int = 16, ef_construction: int = 64, conn=None):
    """
    Create or rebuild the HNSW index on document_chunks.
    Supports caller-supplied connection.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx;")
            query = f"""
                CREATE INDEX document_chunks_embedding_hnsw_idx 
                ON document_chunks USING hnsw (embedding vector_cosine_ops) 
                WITH (m = {int(m)}, ef_construction = {int(ef_construction)});
            """
            cur.execute(query)
            print(f"HNSW Index created with m={m}, ef_construction={ef_construction}.")
    finally:
        if should_close:
            conn.close()

def verify_hnsw_index(conn=None) -> bool:
    """
    Verify whether the HNSW index document_chunks_embedding_hnsw_idx exists.
    Supports caller-supplied connection.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'document_chunks' "
                "AND indexname = 'document_chunks_embedding_hnsw_idx';"
            )
            return cur.fetchone() is not None
    finally:
        if should_close:
            conn.close()

def hnsw_search(query_emb: list[float], top_k: int = 5, ef_search: int | None = 40, conn=None):
    """
    Perform ANN search using HNSW index.
    Supports caller-supplied connection to eliminate connection setup overhead.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            if ef_search is not None:
                cur.execute(f"SET hnsw.ef_search = {int(ef_search)};")
            cur.execute("""
                SELECT c.id, c.document_id, c.chunk_text, c.embedding <=> %s::vector AS distance
                FROM document_chunks c
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_emb, top_k))
            results = cur.fetchall()
            return [{"chunk_id": r[0], "document_id": r[1], "chunk_text": r[2], "distance": r[3]} for r in results]
    finally:
        if should_close:
            conn.close()

def get_search_plan(
    query_emb: list[float],
    top_k: int = 5,
    force_hnsw: bool = False,
    ef_search: int | None = 40,
    conn=None,
    analyze: bool = False
) -> str:
    """
    Execute EXPLAIN (or EXPLAIN ANALYZE) on the vector search query.
    If force_hnsw is True, temporarily disables sequential scan on the session
    to verify index scan behavior, then restores the planner setting in finally.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            if force_hnsw:
                cur.execute("SET enable_seqscan = off;")
                if ef_search is not None:
                    cur.execute(f"SET hnsw.ef_search = {int(ef_search)};")
            try:
                explain_sql = "EXPLAIN (ANALYZE, COSTS, BUFFERS, FORMAT TEXT)" if analyze else "EXPLAIN (COSTS, BUFFERS, FORMAT TEXT)"
                cur.execute(f"""
                    {explain_sql}
                    SELECT c.id, c.document_id, c.chunk_text, c.embedding <=> %s::vector AS distance
                    FROM document_chunks c
                    ORDER BY distance ASC
                    LIMIT %s;
                """, (query_emb, top_k))
                rows = cur.fetchall()
                return "\n".join(r[0] for r in rows)
            finally:
                if force_hnsw:
                    cur.execute("RESET enable_seqscan;")
                    if ef_search is not None:
                        cur.execute("RESET hnsw.ef_search;")
    finally:
        if should_close:
            conn.close()

def verify_execution_plan(plan_text: str, expected_pattern: str) -> bool:
    """Verify that expected_pattern appears in the EXPLAIN plan text."""
    return expected_pattern.lower() in plan_text.lower()
