import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from torchcrf import CRF
from .transcription_model import BaseTranscriptionModel


class CRFTranscriptionModel(nn.Module):
    def __init__(
        self,
        base_model: BaseTranscriptionModel,
        hidden_size: int,
        num_quality_classes: int,
        num_bass_classes: int,
        num_key_classes: int,
        pitch_chroma_dim: int = 12,
    ) -> None:
        super().__init__()
        self.base_model = base_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()  # ensure it stays in eval mode

        self.num_root_quality_classes = (num_quality_classes - 1) * 12 + 1

        # pitch_chroma_logits (12次元) を入力に使う
        bass_input_dim = hidden_size + pitch_chroma_dim
        self.bass_proj = nn.Linear(bass_input_dim, num_bass_classes)
        # chroma -> root_chord
        self.root_chord_proj = nn.Sequential(
            nn.Linear(pitch_chroma_dim, self.num_root_quality_classes),
            nn.LayerNorm(self.num_root_quality_classes),
        )
        # key -> key
        self.key_proj = nn.Linear(num_key_classes, num_key_classes)

        # CRF Layers (using pytorch-crf, configured for batch_first)
        self.crf_bass = CRF(num_tags=num_bass_classes, batch_first=True)
        self.crf_root_chord = CRF(num_tags=self.num_root_quality_classes, batch_first=True)
        self.crf_key = CRF(num_tags=num_key_classes, batch_first=True)

    def train(self, mode: bool = True):
        # Override train to keep base_model in eval mode
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        waveform: torch.Tensor,
        root_chord_target: Optional[torch.Tensor] = None,
        bass_target: Optional[torch.Tensor] = None,
        key_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            base_outputs = self.base_model(waveform)

        chroma_logits = base_outputs["pitch_chroma_logits"]  # (B, T, 12)
        key_logits = base_outputs["initial_key_logits"]  # (B, T, num_key_classes)
        features = base_outputs["initial_features"]  # (B, T, hidden_size)

        bass_input = torch.cat([features, chroma_logits], dim=-1)
        bass_emissions = self.bass_proj(bass_input)
        root_chord_emissions = self.root_chord_proj(chroma_logits)
        key_emissions = self.key_proj(key_logits)

        outputs = {
            "bass_emissions": bass_emissions,
            "root_chord_emissions": root_chord_emissions,
            "key_emissions": key_emissions,
            "base_outputs": base_outputs,
        }

        # If targets are provided (training mode), compute CRF NLL
        if root_chord_target is not None and bass_target is not None and key_target is not None:
            # torchcrf returns positive log likelihood. We negate it to get NLL loss.
            # TorchCRF will throw an index out of bounds error if any element (even if masked out) is out of bonds
            def get_clean_target(target):
                t = target.clone()
                t[~mask] = 0
                return t

            outputs["crf_loss_bass"] = -self.crf_bass(
                bass_emissions, get_clean_target(bass_target), mask=mask, reduction="mean"
            )
            outputs["crf_loss_root_chord"] = -self.crf_root_chord(
                root_chord_emissions, get_clean_target(root_chord_target), mask=mask, reduction="mean"
            )
            outputs["crf_loss_key"] = -self.crf_key(
                key_emissions, get_clean_target(key_target), mask=mask, reduction="mean"
            )

        return outputs

    def decode(self, waveform: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Inference method to get Viterbi decoded paths.
        """
        self.eval()
        outputs = self.forward(waveform, root_chord_target=None, bass_target=None, key_target=None, mask=mask)

        return {
            "bass_predictions": self.crf_bass.decode(outputs["bass_emissions"], mask=mask),
            "root_chord_predictions": self.crf_root_chord.decode(outputs["root_chord_emissions"], mask=mask),
            "key_predictions": self.crf_key.decode(outputs["key_emissions"], mask=mask),
            "base_outputs": outputs["base_outputs"],
        }
