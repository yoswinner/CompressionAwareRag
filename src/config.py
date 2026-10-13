import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "postgres")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
EMBEDDING_DIM = 768

def get_db_name() -> str:
    """Return the active database name from environment or loaded configuration."""
    return os.getenv("DB_NAME", DB_NAME)

def get_db_connection_string(db_name: str | None = None) -> str:
    """Return PostgreSQL connection string for the specified or configured database."""
    target_db = db_name if db_name is not None else get_db_name()
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{target_db}"
