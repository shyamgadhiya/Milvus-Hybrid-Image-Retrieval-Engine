import torch
from PIL import Image
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel
from config import Config

class BaseEmbeddingModel:
    """Base interface for embedding models."""
    def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
        raise NotImplementedError
        
    def encode_text(self, text: Union[str, List[str]]) -> List[float]:
        raise NotImplementedError


class HFCLIPModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = Config.MODEL_NAME, device: str = Config.DEVICE):
        self.device = device
        print(f"[*] Loading model '{model_name}' on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def _extract_tensor(self, output) -> torch.Tensor:
        """Safely extracts pooled representation across transformers versions."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        return output[0]

    def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            raw_feats = self.model.get_image_features(**inputs)
            feats = self._extract_tensor(raw_feats)
            # L2 normalization for true cosine distance
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

    def encode_text(self, text: str) -> List[float]:
        # Wrap query for better semantic zero-shot alignment
        formatted_prompt = f"a photo of {text.strip()}"
        inputs = self.processor(text=[formatted_prompt], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            raw_feats = self.model.get_text_features(**inputs)
            feats = self._extract_tensor(raw_feats)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy()[0].tolist()