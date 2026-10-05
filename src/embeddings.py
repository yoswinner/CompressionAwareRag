import urllib.request
import json
from src.config import OLLAMA_API_URL, OLLAMA_MODEL, EMBEDDING_DIM

def get_embedding(text: str) -> list[float]:
    data = json.dumps({"model": OLLAMA_MODEL, "prompt": text}).encode('utf-8')
    req = urllib.request.Request(OLLAMA_API_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            embedding = result.get('embedding', [])
            if len(embedding) != EMBEDDING_DIM:
                raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}, got {len(embedding)}")
            return embedding
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        raise

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        embeddings.append(get_embedding(text))
    return embeddings
