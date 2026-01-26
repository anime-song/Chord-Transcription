import torch
from typing import Dict


class LabelShifter:
    def __init__(self):
        pass

    def __call__(self, batch: Dict[str, torch.Tensor], shift_pitch: int) -> Dict[str, torch.Tensor]:
        if shift_pitch == 0:
            return batch

        n_semitones = shift_pitch
        with torch.no_grad():
            # ラベルの移調 (root, bass, key)
            for key in ["root_index", "bass_index", "key_index"]:
                if key in batch:
                    labels = batch[key]
                    valid_mask = labels > 0
                    labels[valid_mask] = (labels[valid_mask] - 1 + n_semitones) % 12 + 1
                    batch[key] = labels

            # Chord25dベクトルの移調
            if "chord25" in batch:
                pitch_vec = batch["chord25"][:, :12]
                bass_vec = batch["chord25"][:, 12:]
                shifted_pitch_vec = torch.roll(pitch_vec, shifts=n_semitones, dims=-1)
                shifted_bass_vec_notes = torch.roll(bass_vec[:, 1:], shifts=n_semitones, dims=-1)
                shifted_bass_vec = torch.cat([bass_vec[:, :1], shifted_bass_vec_notes], dim=-1)
                batch["chord25"] = torch.cat([shifted_pitch_vec, shifted_bass_vec], dim=-1)

        return batch
