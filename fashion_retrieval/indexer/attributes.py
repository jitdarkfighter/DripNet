from __future__ import annotations


# remove duplicates
def _phrase(*parts) -> str:
    """Join non-empty parts, skipping words already seen. The VLM tends to repeat the
    color inside the item, e.g. color='yellow' + item='yellow cardigan' should just be 'yellow cardigan'."""
    words, seen = [], set()
    for p in parts:
        if not p:
            continue
        for w in str(p).strip().split():
            if w and w.lower() not in seen:
                seen.add(w.lower())
                words.append(w)
    return " ".join(words)

# flatten the features, so that the sentence can be meaningful
def _feat(rec: dict, slot: str) -> str:
    """<slot>_features list -> phrase: ['polka dot', 'lace trim'] should be 'polka dot lace trim'."""
    return " ".join(str(x) for x in (rec.get(f"{slot}_features") or []) if x)


# normalize the accessories
def _accessories(rec: dict) -> list[str]:
    """[{'item':'bracelet','color':'silver'}] -> ['silver bracelet']."""
    out = []
    for a in rec.get("accessories") or []:
        if isinstance(a, dict):
            p = _phrase(a.get("color"), a.get("item"))
            if p:
                out.append(p)
        elif a:
            out.append(str(a))
    return out


# for the re-ranker, we want a dict of {field: 'space-joined string'} for the reranker to tokenize
def flat_attrs(rec: dict) -> dict[str, str]:
    """{field: 'space-joined string'} dict for the reranker to tokenize."""
    return {
        "top": _phrase(rec.get("top_color"), _feat(rec, "top"), rec.get("top")),
        "bottom": _phrase(rec.get("bottom_color"), _feat(rec, "bottom"), rec.get("bottom")),
        "dress": _phrase(rec.get("dress_color"), _feat(rec, "dress"), rec.get("dress")),
        "outerwear": _phrase(rec.get("outerwear_color"), _feat(rec, "outerwear"), rec.get("outerwear")),
        "shoes": _phrase(rec.get("shoes_color"), rec.get("shoes")),
        "accessories": ", ".join(_accessories(rec)),
        "action": rec.get("action") or "",
        "style": rec.get("style") or "",
        "environment": rec.get("environment") or "",
    }


# Prompt template to pass to siglip text encoder
def attrs_to_text(rec: dict) -> str:
    """One sentence for the text encoder, empty fields dropped."""
    a = flat_attrs(rec)
    garments = [a[k] for k in ("top", "bottom", "dress", "outerwear", "shoes") if a[k]]
    bits = []
    if garments:
        bits.append("A person wearing " + ", ".join(garments))
    if a["accessories"]:
        bits.append("with " + a["accessories"])
    if a["action"]:
        bits.append(a["action"])
    if a["style"]:
        bits.append(a["style"] + " style")
    if a["environment"]:
        bits.append("in " + a["environment"])
    return (", ".join(bits) + ".") if bits else "An unspecified fashion image."
