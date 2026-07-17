"""Vector storage: a FAISS index + a row-aligned JSONL record store.

FAISS row i  <=>  records[i]. The FAISS index holds only vectors (fast ANN);
the JSONL sidecar holds image_path + caption + attributes for display and
metadata-aware reranking. Keeping them separate keeps the vector layer dumb and
swappable, which is what the assignment asks for.

For 1k images we use IndexFlatIP (exact, cosine over normalized vectors). See
`build_index` for the drop-in swap to IVF/HNSW when the corpus grows to millions.
"""

from __future__ import annotations

import json

import faiss
import numpy as np

from fashion_retrieval.configs.config import Config, CONFIG


def build_index(dim: int) -> faiss.Index:
    """Exact cosine index. Swap the body for IVF/HNSW at scale:

        # ~1M+ vectors, approximate but sub-linear:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist=4096,
                                   faiss.METRIC_INNER_PRODUCT)
        index.train(sample_vectors)   # IVF needs a training pass
    """
    return faiss.IndexFlatIP(dim)


class VectorStore:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.index: faiss.Index | None = None
        self.records: list[dict] = []

    #write path (indexer)
    def create(self, embeddings: np.ndarray, records: list[dict]) -> "VectorStore":
        assert len(embeddings) == len(records), "vectors and records must be row-aligned"
        self.index = build_index(embeddings.shape[1])
        self.index.add(embeddings)
        self.records = records
        return self

    def save(self, index_path=None, records_path=None) -> None:
        """Write the index (and the records sidecar unless records_path is None).
        Paths default to config. The two indexes share one sidecar, so only the
        first one needs to write it."""
        self.config.ensure_dirs()
        faiss.write_index(self.index, str(index_path or self.config.text_index_path))
        if records_path is not None:
            with open(records_path, "w") as f:
                for rec in self.records:
                    f.write(json.dumps(rec) + "\n")

    #read path (retriever)
    @classmethod
    def load(cls, config: Config = CONFIG, index_path=None, records_path=None) -> "VectorStore":
        store = cls(config)
        store.index = faiss.read_index(str(index_path or config.text_index_path))
        with open(records_path or config.records_path) as f:
            store.records = [json.loads(line) for line in f if line.strip()]
        assert store.index.ntotal == len(store.records), "index/records out of sync"
        return store

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        """Return up to k (record, cosine_score) pairs, best first."""
        q = query_vec.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(self.records[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def __len__(self) -> int:
        return len(self.records)
