"""Late fusion over the two indexes: fused = a*text_sim + (1-a)*image_sim.

One text-query vector scores against both the caption index and the image index
(shared SigLIP space). We pull every row's similarity from each, combine, and hand
the top candidates to the reranker.

Heads up: text-text cosines run higher than text-image ones, so a=0.5 leans toward
text unless you normalize per query first. See the retriever docstring.
"""

from __future__ import annotations

import numpy as np


def full_scores(index, query_vec: np.ndarray, n: int) -> np.ndarray:
    """Similarity of the query against every row of index, in row order."""
    q = query_vec.reshape(1, -1).astype("float32")
    scores, ids = index.search(q, n)
    out = np.zeros(n, dtype="float32")
    out[ids[0]] = scores[0]          # search returns ids sorted by score, so scatter back
    return out


def fuse(text_index, image_index, query_vec: np.ndarray, n: int,
         alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fused, text_sim, image_sim), each length n in row order."""
    text_sim = full_scores(text_index, query_vec, n)
    image_sim = full_scores(image_index, query_vec, n)
    fused = alpha * text_sim + (1.0 - alpha) * image_sim
    return fused, text_sim, image_sim


def fused_candidates(text_index, image_index, query_vec: np.ndarray, records: list[dict],
                     alpha: float, top_candidates: int,
                     ) -> tuple[list[tuple[dict, float]], np.ndarray, np.ndarray]:
    """Top-`top_candidates` (record, fused_score) pairs, plus the full text_sim and
    image_sim arrays for reporting."""
    n = len(records)
    fused, text_sim, image_sim = fuse(text_index, image_index, query_vec, n, alpha)
    top = np.argsort(-fused)[:top_candidates]
    candidates = [(records[row], float(fused[row])) for row in top]
    return candidates, text_sim, image_sim
