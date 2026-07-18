from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    ### VLM captioner (indexing only, never online) 
    vlm_model: str = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
    vlm_max_new_tokens: int = 320
    vlm_min_pixels: int = 256 * 28 * 28
    vlm_max_pixels: int = 512 * 28 * 28 #Doesnt fit on my GPU otherwise, but works decent

    ### data 
    dataset_metadata: Path = ROOT / "data" / "sampled_dataset_metadata.json"
    images_dir: Path = ROOT / "data" / "sampled_images"
    curated_dir: Path = ROOT / "data" / "curated"      
    incoming_dir: Path = ROOT / "data" / "incoming"   

    # per-image artifacts (captioner writes, indexer reads) 
    # one file per image, named by stem: image_00042.json / image_00042.npy
    metadata_dir: Path = ROOT / "data" / "qwen3_vl_metadata"
    image_emb_dir: Path = ROOT / "data" / "siglip_image_emb"

    ### FAISS artifacts 
    index_dir: Path = ROOT / "data" / "index"
    text_index_path: Path = ROOT / "data" / "index" / "text.faiss"
    image_index_path: Path = ROOT / "data" / "index" / "image.faiss"
    records_path: Path = ROOT / "data" / "index" / "records.jsonl"

    ### SigLIP encoder (one model for text + image, offline and online) 
    # vectors. Kept on CPU to stay off the VLM's GPU.
    siglip_model: str = "google/siglip2-base-patch16-224"
    siglip_max_length: int = 64        
    embed_device: str = "cpu"
    embed_batch_size: int = 64

    ### retrieval 
    top_candidates: int = 100         
    top_k: int = 10                    
    fusion_alpha: float = 0.5         

    ### reranking (attribute boost over the fused score) 
    rerank_enabled: bool = True
    rerank_weight: float = 0.15

    ### attribute fields (from flat_attrs) that the rerank matches against
    rerank_fields: tuple[str, ...] = (
        "top", "bottom", "dress", "outerwear", "shoes",
        "accessories", "action", "style", "environment",
    )

    def ensure_dirs(self) -> None:
        for d in (self.index_dir, self.metadata_dir, self.image_emb_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
