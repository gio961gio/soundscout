from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/clap-htsat-unfused"


class ClapEmbedder:
    """Thin wrapper around the public CLAP model: audio and text -> shared embedding space."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _as_embedding(output: object, modality: str) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        attr = "audio_embeds" if modality == "audio" else "text_embeds"
        value = getattr(output, attr, None)
        if value is None:
            value = getattr(output, "pooler_output", None)
        if value is None:
            raise RuntimeError(f"Il modello CLAP non ha restituito un embedding {modality} utilizzabile.")
        return value

    def embed_audio(self, segments: list[np.ndarray], sample_rate: int = 48_000) -> np.ndarray:
        try:
            inputs = self.processor(audio=segments, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        except TypeError:
            inputs = self.processor(audios=segments, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            output = self.model.get_audio_features(**inputs)
            features = self._as_embedding(output, "audio")
            features = F.normalize(features.float(), p=2, dim=-1)
            mean_embedding = features.mean(dim=0, keepdim=True)
            mean_embedding = F.normalize(mean_embedding, p=2, dim=-1)
        return mean_embedding[0].detach().cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
            features = self._as_embedding(output, "text")
            features = F.normalize(features.float(), p=2, dim=-1)
        return features[0].detach().cpu().numpy().astype(np.float32)
