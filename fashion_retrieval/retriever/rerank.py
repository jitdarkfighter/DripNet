"""Metadata-aware reranking.

The FAISS stage ranks by caption<->query cosine similarity. Reranking adds a
light, transparent boost when a candidate's STRUCTURED attributes (from the VLM
JSON) explicitly match terms in the query. This is what lifts compositional and
attribute-heavy queries ("red tie AND white shirt in a formal setting") above a
pure-embedding baseline, without putting an LLM in the online loop.

Score = cosine + rerank_weight * attribute_match_fraction
"""

from __future__ import annotations

import re

from fashion_retrieval.configs.config import Config, CONFIG

_WORD = re.compile(r"[a-z]+")
# Words too generic to signal an attribute match.
_STOP = {
    "a", "an", "the", "in", "on", "of", "with", "and", "at", "to", "for", "is",
    "person", "someone", "wearing", "wears", "outfit", "look", "style", "who",
}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _attribute_text(record: dict, fields: tuple[str, ...]) -> set[str]:
    """Flatten a record's attribute values into a token set."""
    attrs = record.get("attributes", {}) or {}
    parts: list[str] = []
    for f in fields:
        v = attrs.get(f)
        if isinstance(v, list):
            parts.extend(str(x) for x in v if x)
        elif v:
            parts.append(str(v))
    return _tokens(" ".join(parts))


def rerank(
    query: str,
    candidates: list[tuple[dict, float]],
    config: Config = CONFIG,
) -> list[tuple[dict, float, dict]]:
    """Re-score candidates using attribute overlap with the query.

    Returns (record, final_score, debug) sorted best-first, where debug exposes
    the cosine score and matched terms for explainability.
    """
    q_tokens = _tokens(query)
    reranked = []
    for record, cosine in candidates:
        if config.rerank_enabled and q_tokens:
            attr_tokens = _attribute_text(record, config.rerank_fields)
            matched = q_tokens & attr_tokens
            match_frac = len(matched) / len(q_tokens)
            boost = config.rerank_weight * match_frac
        else:
            matched, boost = set(), 0.0
        final = cosine + boost
        reranked.append(
            (record, final, {"cosine": cosine, "boost": boost, "matched": sorted(matched)})
        )
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked
