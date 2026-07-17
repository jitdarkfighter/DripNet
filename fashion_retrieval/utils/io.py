"""Shared I/O helpers: image save, numbering, metadata append, JSON parsing.

Used by the indexer and the ingest pipeline so they don't each carry their own copy.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def next_index(img_dir: str | Path) -> int:
    """Next free integer in the image_NNNNN.jpg sequence under img_dir."""
    idxs = [
        int(m.group(1))
        for p in glob.glob(f"{img_dir}/image_*.jpg")
        if (m := re.search(r"image_(\d+)\.jpg$", p))
    ]
    return (max(idxs) + 1) if idxs else 0


def save_rgb_jpg(img: Image.Image, path: str | Path, quality: int = 95) -> tuple[int, int]:
    """Save `img` as an RGB JPEG at `path`; return its (width, height)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(str(path), quality=quality)
    return img.size


def load_meta(meta_path: str | Path) -> list[dict]:
    """Load the dataset metadata list, or [] if it does not exist yet."""
    return json.load(open(meta_path)) if os.path.exists(meta_path) else []


def save_meta(meta: list[dict], meta_path: str | Path) -> None:
    """Persist the dataset metadata list (default=str tolerates stray types)."""
    with open(meta_path, "w") as f:
        json.dump(meta, f, default=str)


def append_meta(record: dict, meta_path: str | Path) -> None:
    """Append one record to the dataset metadata list on disk."""
    meta = load_meta(meta_path)
    meta.append(record)
    save_meta(meta, meta_path)


def parse_json_or_raw(text: str) -> dict:
    """Parse the VLM response as JSON. If it wrapped the object in prose, retry on
    the outermost {...}. On failure return a _parse_error record instead of raising,
    so one bad response doesn't drop the image."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"_raw": text, "_parse_error": True}
