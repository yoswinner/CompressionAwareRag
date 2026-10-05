import os
import csv
import requests
import time
from src.database import get_connection
from src.preprocessing import chunk_text
from src.embeddings import get_embeddings_batch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "msmarco_subset.tsv")

def download_sample_data(num_samples: int):
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    if os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)
        if existing_count >= num_samples:
            print(f"Data file already has {existing_count} samples (>= {num_samples} requested). Skipping download.")
            return

    print(f"Downloading {num_samples} samples from BeIR/msmarco using Hugging Face Datasets API...")
    
    with open(RAW_DATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        offset = 0
        while offset < num_samples:
            length = min(100, num_samples - offset)
            url = f"https://datasets-server.huggingface.co/rows?dataset=BeIR%2Fmsmarco&config=corpus&split=corpus&offset={offset}&length={length}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                for row_data in data.get("rows", []):
                    r = row_data["row"]
                    writer.writerow([r["_id"], r["text"]])
                offset += length
                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching data at offset {offset}: {e}")
                break
    print("Sample data downloaded.")

def ingest_data(num_samples: int = 100, chunk_size: int = 300, overlap: int = 50):
    download_sample_data(num_samples)
    conn = get_connection()
    count = 0
    try:
        with conn.cursor() as cur:
            with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 2:
                        continue
                    if count >= num_samples:
                        break
                    doc_id, content = row[0], row[1]
                    cur.execute("INSERT INTO documents (doc_id, content) VALUES (%s, %s) ON CONFLICT (doc_id) DO NOTHING RETURNING id;", (doc_id, content))
                    result = cur.fetchone()
                    if result is None:
                        continue
                    db_doc_id = result[0]
                    chunks = chunk_text(content, chunk_size, overlap)
                    if not chunks:
                        continue
                    embeddings = get_embeddings_batch(chunks)
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        cur.execute(
                            "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                            (db_doc_id, idx, chunk, emb)
                        )
                    count += 1
                    if count % 50 == 0:
                        print(f"Ingested {count}/{num_samples} documents...")
            print(f"Ingestion complete. Total {count} documents ingested.")
    finally:
        conn.close()
