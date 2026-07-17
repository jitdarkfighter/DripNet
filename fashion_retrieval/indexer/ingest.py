from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from fashion_retrieval.configs.config import Config, CONFIG
from fashion_retrieval.indexer.embed import SiglipEncoder
from fashion_retrieval.utils.io import IMG_EXTS, append_meta, next_index, save_rgb_jpg

# the pipeline basically (check architecture)
class Ingestor:
    """Keeps the captioner + encoder loaded so repeated ingests don't reload them."""

    def __init__(self, config: Config = CONFIG, captioner=None, encoder: SiglipEncoder | None = None):
        from fashion_retrieval.indexer.caption import Qwen3VLCaptioner

        self.config = config
        config.ensure_dirs()
        self.captioner = captioner or Qwen3VLCaptioner.load(config)
        self.encoder = encoder or SiglipEncoder(config)

    def ingest_image(self, src_path: str, source: str = "online") -> tuple[str, str, dict]:
        """Caption + embed one image and write it out. Returns (stem, dst_path, metadata)."""
        cfg = self.config
        stem = f"image_{next_index(cfg.images_dir):05d}"
        dst = os.path.join(str(cfg.images_dir), f"{stem}.jpg")

        w, h = save_rgb_jpg(Image.open(src_path), dst)

        metadata = self.captioner.caption(dst)
        with open(os.path.join(str(cfg.metadata_dir), f"{stem}.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        np.save(os.path.join(str(cfg.image_emb_dir), f"{stem}.npy"), self.encoder.embed_image(dst))

        append_meta({
            "image_path": dst, "width": w, "height": h,
            "source": source, "source_file": os.path.basename(src_path),
            "objects": {"category_names": []},
        }, cfg.dataset_metadata)

        return stem, dst, metadata

    def ingest_folder(self, folder: str, move_done: bool = True) -> list[tuple[str, str, dict]]:
        """Ingest every image in folder, moving done files to _ingested/ by default."""
        srcs = sorted(p for p in (os.path.join(folder, n) for n in os.listdir(folder))
                      if os.path.splitext(p)[1].lower() in IMG_EXTS)
        results = []
        done_dir = os.path.join(folder, "_ingested")
        if move_done and srcs:
            os.makedirs(done_dir, exist_ok=True)
        for src in srcs:
            results.append(self.ingest_image(src))
            if move_done:
                os.replace(src, os.path.join(done_dir, os.path.basename(src)))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest image(s) into the fashion DBs.")
    parser.add_argument("images", nargs="*", help="image path(s) to ingest")
    parser.add_argument("--folder", help="ingest every image in this folder")
    args = parser.parse_args()

    ing = Ingestor()
    if args.folder:
        for stem, path, _ in ing.ingest_folder(args.folder):
            print(f"ingested {path}  ({stem})")
    for src in args.images:
        stem, path, _ = ing.ingest_image(src)
        print(f"ingested {src} -> {path}  ({stem})")
    print("Re-run `python -m fashion_retrieval.indexer.index --build-only` to make them searchable.")


if __name__ == "__main__":
    main()
