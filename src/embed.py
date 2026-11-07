"""Step 3: embed the chunks and retrieve by cosine similarity.

No vector database. The corpus is a few hundred chunks, so the whole index is
one float32 matrix in a .npy file and retrieval is a single matrix-vector
product in numpy. A database here would add an install, a service and a schema
to a problem that fits in a few kilobytes of RAM.

Vectors are L2-normalised at write time, so cosine similarity is a dot product.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ROOT, embed_model
from .process import load_chunks

VECTORS = ROOT / "data" / "processed" / "vectors.npy"
VECTOR_META = ROOT / "data" / "processed" / "vectors_meta.json"
BATCH = 64
TOP_K = 5


def _client():
    from openai import OpenAI

    return OpenAI()


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    model = model or embed_model()
    client = _client()
    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        response = client.embeddings.create(model=model, input=batch)
        out.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return _normalise(np.array(out, dtype=np.float32))


def build() -> dict:
    chunks = load_chunks()
    model = embed_model()
    matrix = embed_texts([c["text"] for c in chunks], model=model)
    np.save(VECTORS, matrix)
    VECTOR_META.write_text(json.dumps({
        "embed_model": model,
        "chunk_count": len(chunks),
        "dimensions": int(matrix.shape[1]),
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }, indent=2) + "\n", encoding="utf-8")
    return {"chunks": len(chunks), "dimensions": int(matrix.shape[1]), "model": model}


class Index:
    """The retrieval index: vectors plus the chunk records they came from."""

    def __init__(self) -> None:
        if not VECTORS.exists():
            raise FileNotFoundError("data/processed/vectors.npy missing — run `python -m src.embed`")
        self.matrix = np.load(VECTORS)
        self.meta = json.loads(VECTOR_META.read_text(encoding="utf-8"))
        self.chunks = load_chunks()
        if self.meta["chunk_ids"] != [c["chunk_id"] for c in self.chunks]:
            raise RuntimeError(
                "vectors.npy is stale: its chunk ids do not match chunks.json. "
                "Re-run `python -m src.process` then `python -m src.embed`."
            )

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Top-k chunks by cosine similarity, each with its score."""
        query = embed_texts([question], model=self.meta["embed_model"])[0]
        scores = self.matrix @ query
        order = np.argsort(-scores)[:top_k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order]


def main() -> int:
    stats = build()
    print(f"embedded {stats['chunks']} chunks with {stats['model']} "
          f"({stats['dimensions']} dimensions)")
    print("vectors -> data/processed/vectors.npy (gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
