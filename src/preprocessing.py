def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Deterministic simple character-based chunking with overlap.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks
