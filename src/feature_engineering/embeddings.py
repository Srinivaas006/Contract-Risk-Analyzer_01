"""
Document Embedding Generator — Week 3: Day 1-3
Generates dense vector representations of legal contract text using sentence-transformers.
These embeddings are stored in the Vector Database to enable semantic search
across the entire contract repository.

Model used: all-MiniLM-L6-v2 (90MB, fast, high quality, runs on CPU)
"""
import numpy as np
from typing import List, Union
import os


# ─────────────────────────────────────────────────────────────────
# Model Management (lazy-loaded on first use)
# ─────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2


def get_model():
    """Lazy-load the embedding model (downloads ~90MB on first call)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"📦 Loading embedding model: {MODEL_NAME}")
            print("   (First run downloads ~90MB — subsequent runs are instant)")
            _model = SentenceTransformer(MODEL_NAME)
            print("✅ Embedding model ready!")
        except ImportError:
            print("⚠️  sentence-transformers not installed. Falling back to TF-IDF embeddings.")
            print("   Install with: pip install sentence-transformers")
            _model = "tfidf"  # Flag to use fallback
    return _model


def _tfidf_fallback_embedding(text: str) -> np.ndarray:
    """
    Simple TF-IDF hash-based fallback when sentence-transformers is unavailable.
    Not semantically rich, but keeps the system functional.
    """
    import hashlib
    # Create a pseudo-embedding from character n-gram hashes
    words = text.lower().split()[:100]
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for i, word in enumerate(words):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vec[h] += 1.0 / (i + 1)  # TF weighting
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a single normalized embedding vector for a text string.

    Args:
        text: The contract text or query string to embed.

    Returns:
        numpy array of shape (384,) — L2 normalized.
    """
    model = get_model()

    if model == "tfidf":
        return _tfidf_fallback_embedding(text)

    embedding = model.encode(
        text,
        normalize_embeddings=True,   # L2 normalization → cosine similarity = dot product
        show_progress_bar=False
    )
    return embedding.astype(np.float32)


def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a batch of texts efficiently.

    Args:
        texts:      List of contract text strings.
        batch_size: Number of texts to process in each GPU/CPU batch.

    Returns:
        numpy array of shape (len(texts), 384) — each row is L2 normalized.
    """
    model = get_model()

    if model == "tfidf":
        return np.stack([_tfidf_fallback_embedding(t) for t in texts])

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings.astype(np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    # Since vectors are already L2-normalized, dot product = cosine similarity
    return float(np.dot(vec_a, vec_b))


def embed_contract_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[dict]:
    """
    Split a long contract into overlapping chunks and embed each one.
    Useful for very long contracts that exceed model token limits.

    Args:
        text:       Full contract text.
        chunk_size: Characters per chunk (not tokens).
        overlap:    Overlap between chunks for context continuity.

    Returns:
        List of dicts: {"chunk_id": int, "text": str, "embedding": np.ndarray}
    """
    chunks = []
    words = text.split()
    word_chunk_size = chunk_size // 6  # ~6 chars/word average

    for i in range(0, len(words), word_chunk_size - overlap // 6):
        chunk_words = words[i:i + word_chunk_size]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text) < 20:  # Skip very short chunks
            break
        embedding = generate_embedding(chunk_text)
        chunks.append({
            "chunk_id": len(chunks),
            "text": chunk_text,
            "embedding": embedding
        })

    return chunks


if __name__ == "__main__":
    print("Testing Embedding Generator...")

    texts = [
        "This agreement shall automatically renew for successive one-year periods.",
        "Payment is due within 30 days of invoice date.",
        "All intellectual property created shall be work for hire.",
    ]

    print(f"\nGenerating embeddings for {len(texts)} clauses...")
    embeddings = generate_embeddings(texts)
    print(f"✅ Embedding matrix shape: {embeddings.shape}")

    # Test similarity
    sim = cosine_similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between clause 0 and clause 1: {sim:.4f}")
    sim2 = cosine_similarity(embeddings[0], embeddings[0])
    print(f"Self-similarity (should be 1.0):           {sim2:.4f}")
