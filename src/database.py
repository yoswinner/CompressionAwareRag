import psycopg
from pgvector.psycopg import register_vector
from src.config import get_db_connection_string

def get_connection():
    conn = psycopg.connect(get_db_connection_string(), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn

def initialize_schema():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
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
            print("Database schema initialized successfully.")
    finally:
        conn.close()

def clear_database():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents CASCADE;")
            print("Database cleared.")
    finally:
        conn.close()
