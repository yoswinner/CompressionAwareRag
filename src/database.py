import psycopg
from pgvector.psycopg import register_vector
from src.config import get_db_connection_string

TEST_DB_NAME = "compression_aware_rag_test"

def get_connection(db_name: str | None = None):
    """
    Open and return a PostgreSQL connection with pgvector registered.
    Does not execute CREATE EXTENSION on every connection.
    """
    conn = psycopg.connect(get_db_connection_string(db_name), autocommit=True)
    register_vector(conn)
    return conn

def assert_test_database(conn=None):
    """
    Fail-closed safety guard:
    Verifies that the target database is strictly 'compression_aware_rag_test'.
    Raises RuntimeError immediately if connected to any other database.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database();")
            active_db = cur.fetchone()[0]
            if active_db != TEST_DB_NAME:
                raise RuntimeError(
                    f"CRITICAL SAFETY GUARD VIOLATION: Operation aborted! "
                    f"Active database is '{active_db}', but required test database is '{TEST_DB_NAME}'. "
                    f"Refusing to execute mutating action on non-test database."
                )
    finally:
        if should_close:
            conn.close()

def initialize_schema(conn=None):
    """
    Initializes the required database schema (vector extension, documents, document_chunks).
    Supports a caller-supplied connection.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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
        if should_close:
            conn.close()

def clear_database(conn=None, allow_non_test: bool = False):
    """
    Truncates document tables with a fail-closed safety check.
    Prevents accidental data deletion on non-test databases unless explicitly overridden.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database();")
            active_db = cur.fetchone()[0]
            if active_db != TEST_DB_NAME and not allow_non_test:
                raise RuntimeError(
                    f"CRITICAL SAFETY GUARD VIOLATION: clear_database() aborted! "
                    f"Target database is '{active_db}', which is not '{TEST_DB_NAME}'. "
                    f"Destructive operations against non-test databases are blocked."
                )
            cur.execute("TRUNCATE TABLE documents CASCADE;")
            print(f"Database '{active_db}' cleared.")
    finally:
        if should_close:
            conn.close()
