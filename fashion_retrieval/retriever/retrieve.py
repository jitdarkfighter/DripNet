"""Query -> top-k images. No VLM or LLM online, just encode -> fuse -> rerank.

    from fashion_retrieval.retriever.retrieve import FusionRetriever
    r = FusionRetriever()
    for hit in r.search("a red tie and a white shirt in a formal setting"):
        print(hit["score"], hit["image_path"])

    python -m fashion_retrieval.retriever.retrieve "blue shirt on a park bench" -k 5 -a 0.5

a is the fusion weight: 1 = caption-only, 0 = image-only. Since text-text cosines
outweigh text-image ones, a=0.5 still leans text unless you normalize per query.
"""

from __future__ import annotations

import argparse
import os

import faiss

from fashion_retrieval.configs.config import Config, CONFIG, ROOT
from fashion_retrieval.indexer.embed import SiglipEncoder
from fashion_retrieval.indexer.storage import VectorStore
from fashion_retrieval.retriever.fusion import fused_candidates
from fashion_retrieval.retriever.rerank import rerank


class FusionRetriever:
    def __init__(self, config: Config = CONFIG, encoder: SiglipEncoder | None = None):
        self.config = config
        # the text index loads the shared records; the image index is just vectors
        self.store = VectorStore.load(config, index_path=config.text_index_path,
                                      records_path=config.records_path)
        self.records = self.store.records
        # records.jsonl stores project-relative image paths; anchor them to THIS
        # checkout so they open regardless of cwd (os.path.join leaves any already
        # absolute path untouched, so an older index keeps working too).
        for rec in self.records:
            rec["image_path"] = os.path.join(str(ROOT), rec["image_path"])
        self.image_index = faiss.read_index(str(config.image_index_path))
        assert self.image_index.ntotal == len(self.records), "image index / records out of sync"
        self.encoder = encoder or SiglipEncoder(config)

    def search(self, query: str, k: int | None = None, a: float | None = None) -> list[dict]:
        k = k or self.config.top_k
        a = self.config.fusion_alpha if a is None else a
        q_vec = self.encoder.embed_query(query)

        candidates, text_sim, image_sim = fused_candidates(
            self.store.index, self.image_index, q_vec, self.records,
            alpha=a, top_candidates=self.config.top_candidates,
        )
        ranked = rerank(query, candidates, self.config)

        id2row = {r["id"]: i for i, r in enumerate(self.records)}
        results = []
        for record, score, debug in ranked[:k]:
            row = id2row[record["id"]]
            results.append({
                "id": record["id"],
                "image_path": record["image_path"],
                "caption": record.get("caption", ""),
                "attributes": record.get("attributes", {}),
                "score": round(score, 4), # fused score + rerank boost
                "fused": round(debug["cosine"], 4), # fused score before the boost
                "text_sim": round(float(text_sim[row]), 4),
                "image_sim": round(float(image_sim[row]), 4),
                "matched": debug["matched"],
            })
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the fashion index.")
    parser.add_argument("query", help="natural-language search query")
    parser.add_argument("-k", type=int, default=None, help="number of results")
    parser.add_argument("-a", type=float, default=None, help="fusion weight (text vs image)")
    args = parser.parse_args()

    retriever = FusionRetriever()
    hits = retriever.search(args.query, k=args.k, a=args.a)
    print(f'\nQuery: "{args.query}"\n')
    for rank, hit in enumerate(hits, 1):
        print(f"{rank:2d}. score={hit['score']:.3f} "
              f"(fused={hit['fused']:.3f} = text {hit['text_sim']:.2f} / img {hit['image_sim']:.2f})  "
              f"{hit['image_path']}")
        if hit["matched"]:
            print(f"     matched attrs: {', '.join(hit['matched'])}")


if __name__ == "__main__":
    main()
