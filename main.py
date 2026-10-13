import psycopg
from src.database import get_connection, initialize_schema
from src.ingestion import ingest_data
from src.benchmark import run_benchmark

if __name__ == "__main__":
    DATASET_SIZE = 100 
    
    # Verify existing data before doing anything
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents;")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM document_chunks;")
            chunk_count = cur.fetchone()[0]
            cur.execute("SELECT current_database();")
            active_db = cur.fetchone()[0]
    finally:
        conn.close()

    print(f"Connected to database: {active_db}")
    print(f"Existing dataset: {doc_count} documents, {chunk_count} chunks.")

    if doc_count == 0:
        print("Database is empty. Initializing schema and ingesting dataset...")
        initialize_schema()
        ingest_data(num_samples=DATASET_SIZE, chunk_size=300, overlap=50)
    else:
        print("Existing dataset detected. Preserving data without re-clearing or re-ingesting.")
    
    queries = [
        "What is the capital of France?",
        "How do vector databases work?",
        "What is Retrieval-Augmented Generation?",
        "What are the consequences of the Manhattan Project?",
        "How to run large language models locally?"
    ]
    
    print("\nRunning Stabilized Baseline Benchmark (Exact vs HNSW)...")
    results = run_benchmark(
        queries=queries,
        top_k=3,
        m=16,
        ef_construction=64,
        ef_search=40,
        warmup_runs=2,
        num_runs=5,
        rebuild_index=False
    )
