from src.database import initialize_schema, clear_database
from src.ingestion import ingest_data
from src.benchmark import run_benchmark

if __name__ == "__main__":
    DATASET_SIZE = 100 
    
    print("Initializing Database...")
    initialize_schema()
    clear_database()
    
    print(f"Ingesting {DATASET_SIZE} Data Samples...")
    ingest_data(num_samples=DATASET_SIZE, chunk_size=300, overlap=50)
    
    queries = [
        "What is the capital of France?",
        "How do vector databases work?",
        "What is Retrieval-Augmented Generation?",
        "What are the consequences of the Manhattan Project?",
        "How to run large language models locally?"
    ]
    
    print("Running Benchmarks...")
    run_benchmark(queries, top_k=3, m=16, ef_construction=64, ef_search=40, warmup_runs=2, num_runs=5)
