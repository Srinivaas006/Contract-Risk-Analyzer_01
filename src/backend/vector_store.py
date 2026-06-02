"""
Local Vector Store — Week 3: Day 1-3
Replaces external Pinecone/Milvus dependency with a fully LOCAL vector database
using NumPy for similarity search. No API keys or external services needed.

Stores contract embeddings as .npy files and metadata as JSON.
Supports cosine similarity search across all indexed contracts.
"""
import numpy as np
import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class ContractVectorStore:
    """
    Local vector database for contract embeddings.

    On production, replace this with Pinecone or Milvus for scale.
    For local dev / demo, this works entirely offline with zero dependencies
    beyond NumPy (already installed).

    Storage layout:
        data/vector_store/
        ├── index.json        ← Contract metadata (id, filename, risk_score, timestamp)
        └── vectors.npy       ← Embedding matrix (N x embedding_dim)
    """

    def __init__(self, store_path: str = "data/vector_store"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self.index_file = os.path.join(store_path, "index.json")
        self.vectors_file = os.path.join(store_path, "vectors.npy")

        # In-memory state
        self.metadata: List[Dict] = []
        self.vectors: Optional[np.ndarray] = None

        self._load()

    # ────────────────────────────────────────────────────────────
    # Persistence
    # ────────────────────────────────────────────────────────────

    def _load(self):
        """Load index and vectors from disk."""
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                self.metadata = json.load(f)

        if os.path.exists(self.vectors_file) and self.metadata:
            self.vectors = np.load(self.vectors_file)
            # Validate dimensions match
            if self.vectors.shape[0] != len(self.metadata):
                print("⚠️  Vector store dimension mismatch — resetting store.")
                self.metadata = []
                self.vectors = None

    def _save(self):
        """Persist index and vectors to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        if self.vectors is not None:
            np.save(self.vectors_file, self.vectors)

    # ────────────────────────────────────────────────────────────
    # Write Operations
    # ────────────────────────────────────────────────────────────

    def add_contract(self, contract_id: str, embedding: np.ndarray, metadata: dict):
        """
        Add a new contract embedding to the store.

        Args:
            contract_id: Unique identifier for the contract.
            embedding:   Numpy array (embedding_dim,) — should be L2 normalized.
            metadata:    Dict of contract metadata (filename, risk_score, etc.)
        """
        # Check if contract already exists — update if so
        for i, m in enumerate(self.metadata):
            if m.get("contract_id") == contract_id:
                self.metadata[i] = {"contract_id": contract_id, "indexed_at": datetime.now().isoformat(), **metadata}
                self.vectors[i] = embedding.astype(np.float32)
                self._save()
                return

        # New contract
        new_vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if self.vectors is None:
            self.vectors = new_vec
        else:
            self.vectors = np.vstack([self.vectors, new_vec])

        self.metadata.append({
            "contract_id": contract_id,
            "indexed_at": datetime.now().isoformat(),
            **metadata
        })
        self._save()

    def delete_contract(self, contract_id: str) -> bool:
        """Remove a contract from the vector store."""
        for i, m in enumerate(self.metadata):
            if m.get("contract_id") == contract_id:
                self.metadata.pop(i)
                if self.vectors is not None:
                    self.vectors = np.delete(self.vectors, i, axis=0)
                    if self.vectors.shape[0] == 0:
                        self.vectors = None
                self._save()
                return True
        return False

    # ────────────────────────────────────────────────────────────
    # Read / Search Operations
    # ────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Semantic similarity search across all indexed contracts.

        Uses cosine similarity (dot product of L2-normalized vectors).
        Returns top_k most similar contracts with their similarity scores.

        Args:
            query_embedding: Numpy array (embedding_dim,) for the search query.
            top_k:           Number of results to return.

        Returns:
            List of metadata dicts, each with an added "similarity_score" field.
        """
        if self.vectors is None or len(self.metadata) == 0:
            return []

        # L2-normalize query vector
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        # L2-normalize stored vectors (row-wise)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        v_norm = self.vectors / (norms + 1e-10)

        # Cosine similarity = dot product of normalized vectors
        similarities = (v_norm @ q_norm).flatten()

        # Get top-k indices sorted by similarity
        k = min(top_k, len(self.metadata))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            entry = self.metadata[idx].copy()
            entry["similarity_score"] = round(float(similarities[idx]), 4)
            results.append(entry)

        return results

    def get_all(self) -> List[Dict]:
        """Return all indexed contract metadata."""
        return list(self.metadata)

    def get_contract(self, contract_id: str) -> Optional[Dict]:
        """Get metadata for a specific contract by ID."""
        for m in self.metadata:
            if m.get("contract_id") == contract_id:
                return m
        return None

    def __len__(self) -> int:
        return len(self.metadata)

    def __repr__(self) -> str:
        n = len(self.metadata)
        dim = self.vectors.shape[1] if self.vectors is not None else 0
        return f"ContractVectorStore(contracts={n}, embedding_dim={dim}, path='{self.store_path}')"


if __name__ == "__main__":
    import tempfile

    print("Testing ContractVectorStore...")

    with tempfile.TemporaryDirectory() as tmp:
        store = ContractVectorStore(store_path=tmp)

        # Add mock contracts
        for i in range(3):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            store.add_contract(
                contract_id=f"contract_{i}",
                embedding=vec,
                metadata={"filename": f"contract_{i}.pdf", "risk_score": 30 + i * 15}
            )

        print(f"Stored: {len(store)} contracts")

        # Search
        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, top_k=2)
        print(f"Search results: {[r['contract_id'] for r in results]}")
        print(f"Similarity scores: {[r['similarity_score'] for r in results]}")
        print("✅ ContractVectorStore working correctly!")
