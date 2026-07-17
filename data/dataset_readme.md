# Dataset — Sampled Fashionpedia Subset

1,000 images randomly sampled (reservoir, `SEED=50`) from `detection-datasets/fashionpedia`
(HuggingFace `train`, ~45k). Used for the multimodal fashion retrieval system.

Sampled dataset, meta-data generated using Qwen, are all available in `bigly_files.zip`
[Bigly_files.zip Drive link](https://drive.google.com/file/d/1NkqAM56hqsUvmvfEH0wbcgBtIM30LRCd/view?usp=sharing)


Each image has its own JSON; `attributes.jsonl` is a small dump (first 20 records) for a
quick look at the metadata.

## Criteria (assignment §2)

| Criterion | Requirement | Result |
|---|---|---|
| Count | 500–1,000 | 1,000 ✅ |
| Clothing types | formal / casual / outerwear | 46/46 categories, all 3 buckets ✅ |
| Color theory | wide garment palette | 12/12 families, neutral-skewed ⚠️ |
| Environment | office / street / park / home | no labels; street/park present, office & home absent ⚠️ |

Fashionopedia simply lacks these images. Not an issue with the Image search system

## Files

| Path | Description |
|---|---|
| `sampled_images/image_*.jpg` | 1,000 sampled JPEGs |
| `sampled_dataset_metadata.json` | Per-image `image_id`, `width`, `height`, `objects` (bbox/category), `image_path` |
| `attributes.jsonl` | First 20 attribute records (quick metadata preview) |
| `dataset_analysis.json` | Machine-readable summary |
| `analysis_montage.jpg` | 5×5 sample grid |
| `analysis_color_palette.png` | Garment color-family distribution |
| `analyze_dataset.py` | Regenerates the above |
