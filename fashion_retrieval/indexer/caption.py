from __future__ import annotations

from PIL import Image

from fashion_retrieval.configs.config import Config, CONFIG
from fashion_retrieval.utils.io import parse_json_or_raw

# Structured-extraction prompt; the schema here is what attributes.py expects.
PROMPT = """
You are an expert fashion vision-language model that extracts structured metadata for a multimodal fashion retrieval system.

Your goal is to produce consistent, retrieval-friendly metadata describing the PRIMARY person in the image.

GENERAL RULES

- Return ONLY valid JSON matching the schema exactly, with no markdown or extra text.
- Base every attribute on visible evidence only.
- Describe a garment when you can identify it with reasonable confidence, even if partially occluded.
- Use null for any attribute you cannot determine, and [] for any empty list.
- Use concise, consistent terminology across images.

SCHEMA COMPLETENESS

Always include EVERY key from the schema, in the exact order shown, in every response.
Fill unknown scalar fields with null and unknown list fields with []. Keep the top/bottom keys present even when the outfit is a one-piece (set them to null / []).

INSPECTION ORDER

Inspect the primary person from head to toe: head/hair accessories, neck, upper body, outerwear, lower body, footwear, accessories (bags, watches, jewelry, belts, hats, glasses), pose/action, then scene/background.

ONE-PIECE RULE

For a one-piece outfit (dress, gown, jumpsuit, romper, bodysuit worn as the primary garment):
- Fill dress, dress_color, dress_features.
- Set top, top_color, bottom, bottom_color to null and top_features, bottom_features to [].

For separates, fill top and bottom, and set dress = null, dress_color = null, dress_features = [].

GARMENT RULES

The garment fields are: top, bottom, dress, outerwear, shoes.

- Write each garment as a garment NOUN, optionally with a fit or style word
  (e.g. "oversized hoodie", "cropped denim jacket", "sleeveless tank top", "wide-leg trousers",
   "straight-leg jeans", "pleated skirt", "bomber jacket", "combat boots", "strapless cocktail dress").
- Put the garment's color in its matching *_color field, and keep the garment field itself color-free.
    write:  "dress": "cocktail dress", "dress_color": "black"   (not "dress": "black dress")
- Put patterns and fits in *_features, and always keep a noun in the garment field.
    write:  "dress": "polka dot dress", "dress_features": ["polka dot"]   (not "dress": "polka dot")
- When you can only see a sleeve, fit, or pattern, combine it with the best-guess noun
  (e.g. "long-sleeved top", "sleeveless dress"). If the type is unclear, use a generic noun:
  top, bottom, dress, jacket, or shoes.

COLORS

Store each color only in its *_color field, as a plain dominant color name
(e.g. black, white, navy, olive, beige). Choose the closest concrete color; use null if truly uncertain.

FEATURES

Use *_features for visual characteristics such as: striped, floral, plaid, polka dot, ruched,
pleated, lace, embroidered, ruffled, cropped, oversized, fitted, distressed, sheer, quilted, ribbed.
Keep colors and the garment noun out of the features list.

ACCESSORIES

List every clearly visible accessory as an object with "item" and "color". Use [] when there are none.

  [{"item": "watch", "color": "gold"}, {"item": "crossbody bag", "color": "brown"}]

STYLE

Give the overall fashion style as a short phrase
(e.g. casual streetwear, business formal, elegant evening, minimalist casual, athleisure, bohemian, smart casual).

ENVIRONMENT

Give the scene as ONE short canonical phrase, using the shortest form
(e.g. city street, park, fashion runway, studio, beach, shopping mall, office, living room, red carpet).

ACTION

Give the person's primary action as a single verb, preferring:
standing, walking, posing, sitting, running, jumping, talking.
Use a short verb phrase for anything else, or null when no person/action is visible.

OUTPUT

Return ONLY JSON with ALL of these keys, in this exact order (fill unknowns with null or []):

{
  "top": null,
  "top_color": null,
  "top_features": [],
  "bottom": null,
  "bottom_color": null,
  "bottom_features": [],
  "dress": null,
  "dress_color": null,
  "dress_features": [],
  "outerwear": null,
  "outerwear_color": null,
  "outerwear_features": [],
  "shoes": null,
  "shoes_color": null,
  "accessories": [],
  "action": null,
  "style": null,
  "environment": null
}
"""


class Qwen3VLCaptioner:
    """Qwen3-VL-4B (4-bit). Load once, caption many. Shared by the batch indexer
    and the ingest pipeline."""

    def __init__(self, model=None, processor=None, prompt: str = PROMPT, max_new_tokens: int = 320):
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    @classmethod
    def load(cls, config: Config = CONFIG) -> "Qwen3VLCaptioner":
        """Load the model + processor with the configured pixel bounds."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            config.vlm_model,
            min_pixels=config.vlm_min_pixels,
            max_pixels=config.vlm_max_pixels,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_model, dtype="auto", device_map="auto",
        ).eval()
        
        return cls(model=model, processor=processor, max_new_tokens=config.vlm_max_new_tokens)

    def caption(self, image_path: str, system_prompt: str | None = None) -> dict:
        """Caption one image, returning the parsed metadata dict (or a
        _parse_error record if the JSON is malformed)."""
        import torch

        image = Image.open(image_path).convert("RGB")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt},
        ]})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        response = self.processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        return parse_json_or_raw(response)
