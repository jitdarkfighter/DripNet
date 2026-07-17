

import os
import json
import random
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

SEED = 50
N_SAMPLES = 1000       
SHUFFLE_BUFFER = 3000   
OUT_DIR = "data/sampled_images"
META_PATH = "data/sampled_dataset_metadata.json"

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

dataset = load_dataset(
    "detection-datasets/fashionpedia",
    split="train",
    streaming=True,
)

_category_feature = getattr(dataset.features["objects"]["category"], "feature", None)
CATEGORY_NAMES = getattr(_category_feature, "names", None)
if CATEGORY_NAMES:
    print(f"{len(CATEGORY_NAMES)} garment categories, e.g.: {CATEGORY_NAMES[:8]}")

sampled_stream = dataset.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER).take(N_SAMPLES)

metadata = []
for i, sample in enumerate(tqdm(sampled_stream, total=N_SAMPLES, desc="saving")):
    image = sample["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    fname = f"image_{i:05d}.jpg"
    fpath = os.path.join(OUT_DIR, fname)
    image.save(fpath, quality=95)

    record = {k: v for k, v in sample.items() if k != "image"}
    record["image_path"] = fpath
    cats = record.get("objects", {}).get("category", [])
    if CATEGORY_NAMES:
        record["objects"]["category_names"] = [CATEGORY_NAMES[c] for c in cats]
    metadata.append(record)

with open(META_PATH, "w") as f:
    json.dump(metadata, f, default=str)

print(f"\nSaved {len(metadata)} images to {OUT_DIR}/")
print(f"Wrote metadata to {META_PATH}")

cat_counter = Counter()
for rec in metadata:
    names = rec.get("objects", {}).get("category_names", [])
    cat_counter.update(set(names))

print(f"\nTotal images stored: {len(metadata)}  (criteria: 500-1,000)")
if CATEGORY_NAMES:
    print(f"Distinct garment types present: {len(cat_counter)} / {len(CATEGORY_NAMES)}")
    print("Top 20 garment categories by image count:")
    for name, cnt in cat_counter.most_common(20):
        print(f"  {name:20s} {cnt:4d} images ({100 * cnt / len(metadata):.1f}%)")
