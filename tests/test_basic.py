import os
import pytest
from src.database import get_connection, initialize_schema, clear_database
from src.embeddings import get_embedding
from src.preprocessing import chunk_text
from src.search import exact_search, create_hnsw_index, hnsw_search, verify_hnsw_index
from src.benchmark import calculate_recall

def test_db_connection():
    conn = get_connection()
    assert conn is not None
    conn.close()

def test_embedding_dimension():
    emb = get_embedding("test")
    assert len(emb) == 768

def test_chunking():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 0
    assert len(chunks[0]) == 300

def test_insertion_and_search():
    initialize_schema()
    clear_database()
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            doc_id = "test_doc"
            cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) RETURNING id;", (doc_id, "test content"))
            db_doc_id = cur.fetchone()[0]
            
            emb = get_embedding("test content")
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                (db_doc_id, 0, "test content", emb)
            )
    finally:
        conn.close()

    query_emb = get_embedding("test content")
    res = exact_search(query_emb, top_k=1)
    assert len(res) == 1
    assert res[0]["document_id"] > 0
    
    create_hnsw_index(m=4, ef_construction=10)
    assert verify_hnsw_index() is True
    
    res_hnsw = hnsw_search(query_emb, top_k=1, ef_search=10)
    assert len(res_hnsw) == 1

def test_recall_calculation():
    exact = [{"chunk_id": 1}, {"chunk_id": 2}]
    hnsw = [{"chunk_id": 1}, {"chunk_id": 3}]
    recall = calculate_recall(exact, hnsw)
    assert recall == 0.5
