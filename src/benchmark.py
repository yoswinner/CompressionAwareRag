import time
import statistics
from src.search import exact_search, create_hnsw_index, hnsw_search, verify_hnsw_index
from src.database import get_connection
from src.embeddings import get_embeddings_batch

def calculate_recall(exact_results, hnsw_results):
    exact_ids = {r["chunk_id"] for r in exact_results}
    hnsw_ids = {r["chunk_id"] for r in hnsw_results}
    if not exact_ids:
        return 0.0
    intersection = exact_ids.intersection(hnsw_ids)
    return len(intersection) / len(exact_ids)

def run_benchmark(queries: list[str], top_k: int = 5, m: int = 16, ef_construction: int = 64, ef_search: int = 40, warmup_runs: int = 2, num_runs: int = 3):
    print("\n--- Pre-generating Query Embeddings ---")
    start_emb_time = time.time()
    query_embeddings = get_embeddings_batch(queries)
    emb_time = time.time() - start_emb_time
    print(f"Generated {len(queries)} embeddings in {emb_time:.2f}s (EXCLUDED from retrieval latency)")
    
    print("\n--- Running Benchmark ---")
    
    # 1. Exact Search
    for _ in range(warmup_runs):
        for q_emb in query_embeddings:
            exact_search(q_emb, top_k)
            
    exact_latencies = []
    exact_all_results = []
    for _ in range(num_runs):
        run_latencies = []
        for q_emb in query_embeddings:
            start_time = time.time()
            res = exact_search(q_emb, top_k)
            run_latencies.append(time.time() - start_time)
            if _ == num_runs - 1:
                exact_all_results.append(res)
        exact_latencies.append(sum(run_latencies) / len(run_latencies))
    
    avg_exact_latency = sum(exact_latencies) / len(exact_latencies)
    median_exact_latency = statistics.median(exact_latencies)

    # 2. Create HNSW Index
    print("\nCreating HNSW Index...")
    create_hnsw_index(m, ef_construction)
    
    if not verify_hnsw_index():
        print("ERROR: HNSW Index verification failed!")
        return
    print("HNSW Index verified.")

    # 3. HNSW Search
    for _ in range(warmup_runs):
        for q_emb in query_embeddings:
            hnsw_search(q_emb, top_k, ef_search)
            
    hnsw_latencies = []
    hnsw_all_results = []
    for _ in range(num_runs):
        run_latencies = []
        for q_emb in query_embeddings:
            start_time = time.time()
            res = hnsw_search(q_emb, top_k, ef_search)
            run_latencies.append(time.time() - start_time)
            if _ == num_runs - 1:
                hnsw_all_results.append(res)
        hnsw_latencies.append(sum(run_latencies) / len(run_latencies))
        
    avg_hnsw_latency = sum(hnsw_latencies) / len(hnsw_latencies)
    median_hnsw_latency = statistics.median(hnsw_latencies)

    # 4. Calculate Recall@K
    recalls = []
    for exact_res, hnsw_res in zip(exact_all_results, hnsw_all_results):
        recalls.append(calculate_recall(exact_res, hnsw_res))
        
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_chunks;")
        num_vectors = cur.fetchone()[0]
    conn.close()
    
    print("\n--- Benchmark Results ---")
    print(f"Number of vectors: {num_vectors}")
    print(f"K = {top_k}")
    print(f"Index Config: m={m}, ef_construction={ef_construction}, ef_search={ef_search}")
    print(f"Recall@{top_k}: {avg_recall:.4f}")
    print(f"Exact Avg Latency: {avg_exact_latency:.6f}s | Median: {median_exact_latency:.6f}s")
    print(f"HNSW Avg Latency:  {avg_hnsw_latency:.6f}s | Median: {median_hnsw_latency:.6f}s")
    print("Note: Latencies measure strictly database execution time.")
    
    return {
        "num_vectors": num_vectors,
        "exact_latency": avg_exact_latency,
        "hnsw_latency": avg_hnsw_latency,
        "recall": avg_recall
    }
