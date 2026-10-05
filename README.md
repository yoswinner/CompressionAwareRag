# Compression-Aware Vector Database Indexing for RAG

## Objective
Implement and experimentally evaluate vector database indexing and vector compression techniques for memory-efficient and low-latency Retrieval-Augmented Generation (RAG) systems.

## Architecture
- **Language**: Python 3
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: Local Ollama model (`nomic-embed-text`, 768 dimensions)
- **Dataset**: MS MARCO passage ranking subset (configurable sizes via BeIR corpus)

## Setup Instructions
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure PostgreSQL is running and pgvector is installed.
3. Copy `.env.example` to `.env` and fill in your DB credentials.
4. Ensure Ollama is running locally with `nomic-embed-text`:
   ```bash
   ollama pull nomic-embed-text
   ```

## Database Initialization & Data Ingestion
The dataset size is fully configurable in `main.py` (e.g., 100, 1000, 10000 vectors). The initial development smoke test was validated with 7 vectors, but the actual benchmarking allows testing meaningful scale.
To configure the dataset size, update `DATASET_SIZE` in `main.py` and run it.

## Benchmarking Improvements
The baseline benchmarking is scientifically structured:
1. **Embedding Generation is Excluded**: Query embeddings are generated *before* timing starts. The reported latency is strictly the database execution time.
2. **Warm-up & Repetitions**: The benchmark executes configurable warm-up runs to prime the cache, and calculates the median/average over multiple repetitions.
3. **Exact vs HNSW**: Uses identical pre-generated query embeddings to evaluate HNSW Recall@K and latency tradeoffs accurately.
4. **Index Verification**: Automatically verifies the presence of the HNSW index before proceeding.

Run the pipeline:
```bash
python main.py
```

## Running Tests
Run basic functional tests for configurable dataset ingestion, embedding generation, exact/HNSW search isolation, recall verification, and more:
```bash
pytest tests/
```
