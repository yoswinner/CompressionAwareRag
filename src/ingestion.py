import os
import csv
import json
import time
from datasets import load_dataset
from src.database import get_connection, initialize_schema
from src.preprocessing import chunk_text
from src.embeddings import get_embeddings_batch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CACHE_PATH = os.path.join(DATA_DIR, "raw", "msmarco_1k_chunks.tsv")
DEFAULT_MANIFEST_PATH = os.path.join(DATA_DIR, "raw", "msmarco_1k_chunks.meta.json")
LEGACY_DATA_PATH = os.path.join(DATA_DIR, "raw", "msmarco_subset.tsv")

def download_and_cache_dataset(
    target_chunks: int = 1000,
    target_docs: int | None = None,
    chunk_size: int = 300,
    overlap: int = 50,
    output_path: str = DEFAULT_CACHE_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    force_download: bool = False
) -> tuple[int, int]:
    """
    Stream documents from Hugging Face BeIR/msmarco using the datasets library,
    computing chunk counts in real time until target_chunks (or target_docs) is satisfied.
    Writes atomically to output_path and generates a metadata manifest.
    Avoids silent partial downloads or duplicate re-downloads.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if a valid, complete cache already exists
    if not force_download and os.path.exists(output_path) and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if meta.get("complete", False):
                doc_count = meta.get("actual_documents", 0)
                chunk_count = meta.get("actual_chunks", 0)
                if (target_chunks and chunk_count >= target_chunks) or (target_docs and doc_count >= target_docs):
                    print(f"Verified local cache exists: {output_path} ({doc_count} docs, {chunk_count} chunks). Skipping download.")
                    return doc_count, chunk_count
        except Exception as e:
            print(f"Warning: Manifest verification failed ({e}). Re-caching...")

    print(f"Streaming dataset from Hugging Face 'BeIR/msmarco' (target: ~{target_chunks} chunks)...")
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)

    tmp_path = output_path + ".tmp"
    doc_count = 0
    total_chunks = 0

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

            writer.writerow([doc_id, text])
            doc_count += 1
            total_chunks += len(chunks)

            if target_chunks and total_chunks >= target_chunks:
                break
            if target_docs and doc_count >= target_docs:
                break

    # Atomically replace destination file upon successful completion
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)

    manifest_data = {
        "source": "BeIR/msmarco",
        "config": "corpus",
        "split": "corpus",
        "target_chunks": target_chunks,
        "target_docs": target_docs,
        "actual_documents": doc_count,
        "actual_chunks": total_chunks,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "complete": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    print(f"Caching complete: {doc_count} documents yielding {total_chunks} chunks saved to {output_path}.")
    return doc_count, total_chunks

def load_cached_dataset(tsv_path: str = DEFAULT_CACHE_PATH, max_docs: int | None = None) -> list[tuple[str, str]]:
    """
    Read cached documents from a TSV file.
    Returns a list of (doc_id, text) tuples.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Cached dataset file not found: {tsv_path}")

    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                records.append((row[0].strip(), row[1].strip()))
                if max_docs and len(records) >= max_docs:
                    break
    return records

def ingest_dataset_to_db(
    target_db: str = "compression_aware_rag_exp",
    tsv_path: str = DEFAULT_CACHE_PATH,
    chunk_size: int = 300,
    overlap: int = 50,
    max_chunks: int | None = None,
    batch_size: int = 25
) -> dict:
    """
    Ingest cached documents and chunks with embeddings into the specified target database.
    Contains a fail-closed guard preventing accidental modification of the production database.
    """
    if target_db == "compression_aware_rag":
        raise RuntimeError(
            "CRITICAL SAFETY GUARD VIOLATION: Ingestion into production database 'compression_aware_rag' is blocked! "
            "Use target_db='compression_aware_rag_exp' or 'compression_aware_rag_test'."
        )

    docs = load_cached_dataset(tsv_path)
    print(f"Loaded {len(docs)} documents from cache {tsv_path}.")

    conn = get_connection(target_db)
    initialize_schema(conn=conn)

    inserted_docs = 0
    inserted_chunks = 0
    t0 = time.perf_counter()

    try:
        with conn.cursor() as cur:
            for doc_id, content in docs:
                if max_chunks and inserted_chunks >= max_chunks:
                    break

                chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    continue

                # If adding this doc exceeds max_chunks, optionally truncate chunks or break
                if max_chunks and (inserted_chunks + len(chunks) > max_chunks):
                    chunks = chunks[:max_chunks - inserted_chunks]

                # Insert parent document
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

                # Pre-generate embeddings for chunks
                embeddings = get_embeddings_batch(chunks)

                for idx, (chunk_txt, emb) in enumerate(zip(chunks, embeddings)):
                    cur.execute(
                        "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (document_id, chunk_index) DO NOTHING RETURNING id;",
                        (db_doc_id, idx, chunk_txt, emb)
                    )
                    if cur.fetchone() is not None:
                        inserted_chunks += 1

                inserted_docs += 1
                if inserted_docs % 50 == 0:
                    print(f"Progress: {inserted_docs} docs, {inserted_chunks} chunks inserted into {target_db}...")

        elapsed = time.perf_counter() - t0
        print(f"Ingestion to {target_db} complete in {elapsed:.2f}s: {inserted_docs} documents, {inserted_chunks} chunks.")
        return {
            "target_db": target_db,
            "documents_ingested": inserted_docs,
            "chunks_inserted": inserted_chunks,
            "elapsed_seconds": elapsed
        }
    finally:
        conn.close()

def ingest_data(num_samples: int = 100, chunk_size: int = 300, overlap: int = 50):
    """
    Backward-compatible convenience wrapper for legacy scripts.
    """
    return ingest_dataset_to_db(
        target_db="compression_aware_rag_exp",
        tsv_path=DEFAULT_CACHE_PATH,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunks=None
    )
