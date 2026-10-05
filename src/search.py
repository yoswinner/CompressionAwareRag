from src.database import get_connection

def exact_search(query_emb: list[float], top_k: int = 5):
    conn = get_connection()
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
        conn.close()

def create_hnsw_index(m: int = 16, ef_construction: int = 64):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx;")
            query = f"""
                CREATE INDEX document_chunks_embedding_hnsw_idx 
                ON document_chunks USING hnsw (embedding vector_cosine_ops) 
                WITH (m = {m}, ef_construction = {ef_construction});
            """
            cur.execute(query)
            print(f"HNSW Index created with m={m}, ef_construction={ef_construction}.")
    finally:
        conn.close()

def verify_hnsw_index():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'document_chunks' AND indexname = 'document_chunks_embedding_hnsw_idx';")
            return cur.fetchone() is not None
    finally:
        conn.close()

def hnsw_search(query_emb: list[float], top_k: int = 5, ef_search: int = 40):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {ef_search};")
            cur.execute("""
                SELECT c.id, c.document_id, c.chunk_text, c.embedding <=> %s::vector AS distance
                FROM document_chunks c
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_emb, top_k))
            results = cur.fetchall()
            return [{"chunk_id": r[0], "document_id": r[1], "chunk_text": r[2], "distance": r[3]} for r in results]
    finally:
        conn.close()
