"""Offline indexing: images -> captions + embeddings -> two FAISS indexes.

Two stages you can run separately:
  caption_and_embed_all : write per-image JSON + .npy (needs the GPU, resumable)
  build_indexes         : read those into text.faiss + image.faiss + records.jsonl (no GPU)

    python -m fashion_retrieval.indexer.index                 # both stages
    python -m fashion_retrieval.indexer.index --caption-only  # just JSON + .npy
    python -m fashion_retrieval.indexer.index --build-only    # just the indexes
    python -m fashion_retrieval.indexer.index --limit 20      # first 20 images
"""

from __future__ import annotations

# Just fallbacks for the VLM's JSON schema, which is the source of truth.

import argparse
import glob
import json
import os

import numpy as np
from tqdm import tqdm

from fashion_retrieval.configs.config import Config, CONFIG, ROOT
from fashion_retrieval.indexer.attributes import attrs_to_text, flat_attrs
from fashion_retrieval.indexer.embed import SiglipEncoder
from fashion_retrieval.indexer.storage import VectorStore


def caption_and_embed_all(config: Config = CONFIG, limit: int | None = None) -> int:
    """Caption + embed every image in images_dir, skipping any that already have
    both a JSON and a .npy. Returns how many were processed this run."""
    from fashion_retrieval.indexer.caption import Qwen3VLCaptioner

    config.ensure_dirs()
    names = sorted(os.listdir(config.images_dir))
    if limit:
        names = names[:limit]

    todo = []
    for name in names:
        stem = os.path.splitext(name)[0]
        js = os.path.join(config.metadata_dir, f"{stem}.json")
        npy = os.path.join(config.image_emb_dir, f"{stem}.npy")
        if not (os.path.exists(js) and os.path.exists(npy)):
            todo.append((stem, os.path.join(str(config.images_dir), name), js, npy))
    if not todo:
        print("nothing to caption — all images already have JSON + .npy")
        return 0

    print(f"captioning + embedding {len(todo)} image(s) ...")
    captioner = Qwen3VLCaptioner.load(config)
    encoder = SiglipEncoder(config)
    for stem, img_path, js, npy in tqdm(todo, desc="caption+embed"):
        metadata = captioner.caption(img_path)
        with open(js, "w") as f:
            json.dump(metadata, f, indent=2)
        np.save(npy, encoder.embed_image(img_path))
    return len(todo)


def _resolve_image(stem: str, config: Config) -> str:
    """Find an image by stem, checking the dataset dir before curated drops.

    Returns a path *relative to the project root* so records.jsonl stays portable
    across machines; consumers join it back onto ROOT when they load the records.
    """
    for cand in (
        os.path.join(str(config.images_dir), f"{stem}.jpg"),
        os.path.join(str(config.curated_dir), f"{stem}.jpg"),
        os.path.join(str(config.curated_dir), f"{stem}.jpeg"),
        os.path.join(str(config.curated_dir), f"{stem}.png"),
    ):
        if os.path.exists(cand):
            return os.path.relpath(cand, ROOT)
    return os.path.relpath(os.path.join(str(config.images_dir), f"{stem}.jpg"), ROOT)


def load_records(config: Config = CONFIG) -> list[dict]:
    """Read the metadata JSONs into {id, image_path, caption, attributes} records."""
    records = []
    for fp in sorted(glob.glob(f"{config.metadata_dir}/*.json")):
        with open(fp) as f:
            rec = json.load(f)
        if rec.get("_parse_error") or rec.get("_exception"):
            continue
        stem = os.path.splitext(os.path.basename(fp))[0]
        records.append({
            "id": stem,
            "image_path": _resolve_image(stem, config),
            "caption": attrs_to_text(rec),
            "attributes": flat_attrs(rec),
        })
    return records


def build_indexes(config: Config = CONFIG) -> None:
    """Build the text + image indexes from the JSON/.npy on disk. The image side
    just loads the stored vectors; only the text side needs SigLIP."""
    config.ensure_dirs()
    records = load_records(config)
    if not records:
        raise SystemExit("no usable metadata records; run --caption-only first")

    # image vectors are stacked in records order, so every record needs its .npy
    missing = [r["id"] for r in records
               if not os.path.exists(os.path.join(config.image_emb_dir, f"{r['id']}.npy"))]
    if missing:
        raise SystemExit(f"{len(missing)} record(s) missing an image .npy, e.g. {missing[:3]}; "
                         f"run --caption-only to backfill")

    print(f"[1/2] Text index: embedding {len(records)} captions with {config.siglip_model} ...")
    encoder = SiglipEncoder(config)
    text_vecs = encoder.embed_documents([r["caption"] for r in records])
    VectorStore(config).create(text_vecs, records).save(
        index_path=config.text_index_path, records_path=config.records_path,
    )

    print(f"[2/2] Image index: stacking {len(records)} stored SigLIP vectors ...")
    img_vecs = np.stack([
        np.load(os.path.join(config.image_emb_dir, f"{r['id']}.npy")) for r in records
    ]).astype("float32")
    VectorStore(config).create(img_vecs, records).save(
        index_path=config.image_index_path, records_path=None,   # sidecar already written above
    )
    print(f"Done. text={config.text_index_path}  image={config.image_index_path}  "
          f"records={config.records_path}")


def run(config: Config = CONFIG, caption_only: bool = False,
        build_only: bool = False, limit: int | None = None) -> None:
    if not build_only:
        caption_and_embed_all(config, limit=limit)
    if not caption_only:
        build_indexes(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fashion retrieval indexes.")
    parser.add_argument("--caption-only", action="store_true",
                        help="only write per-image JSON + .npy (skip FAISS build)")
    parser.add_argument("--build-only", action="store_true",
                        help="only build the FAISS indexes from existing JSON + .npy")
    parser.add_argument("--limit", type=int, default=None,
                        help="caption only the first N images (for quick tests)")
    args = parser.parse_args()
    run(caption_only=args.caption_only, build_only=args.build_only, limit=args.limit)


if __name__ == "__main__":
    main()
