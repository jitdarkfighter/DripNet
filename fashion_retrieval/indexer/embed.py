from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from fashion_retrieval.configs.config import Config, CONFIG


class SiglipEncoder:
    """Text embedding (embed_documents / embed_query) and image embedding
    (embed_image / embed_images) from a single SigLIP model."""

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.device = config.embed_device
        self.batch_size = config.embed_batch_size
        self.max_length = config.siglip_max_length
        self.model = SiglipModel.from_pretrained(config.siglip_model).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(config.siglip_model)

    @property
    def dim(self) -> int:
        return self.model.config.text_config.hidden_size

    # Text Encoder 
    @torch.inference_mode() #since i have less memory, otherwise it will OOM
    def _encode_text(self, texts: list[str]) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            # SigLIP's text encoder is a fixed 64-token context, so pad/truncate.
            inputs = self.processor(
                text=texts[i : i + self.batch_size], return_tensors="pt",
                padding="max_length", truncation=True, max_length=self.max_length,
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            chunks.append(feats.cpu())
        return torch.cat(chunks).numpy().astype("float32")

    def embed_documents(self, captions: list[str]) -> np.ndarray:
        return self._encode_text(list(captions))

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode_text([query])[0]

    # image Encoder
    @torch.inference_mode()
    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        chunks = []
        for i in range(0, len(image_paths), self.batch_size):
            imgs = [Image.open(p).convert("RGB") for p in image_paths[i : i + self.batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            chunks.append(feats.cpu())
        return torch.cat(chunks).numpy().astype("float32")

    def embed_image(self, image_path: str) -> np.ndarray:
        return self.embed_images([image_path])[0]
