import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union, Tuple

try:
    from torchcrf import CRF
except ImportError:
    print("Warning: torchcrf not found. Please install pytorch-crf.")
    CRF = None

from .transcription_model import BaseTranscriptionModel
from .segment_model import SegmentTranscriptionModel


class TranscriptionCRFModel(nn.Module):
    def __init__(self, base_model: BaseTranscriptionModel | SegmentTranscriptionModel):
        super().__init__()
        self.base_model = base_model

        # Base modelのパラメータを凍結
        for param in self.base_model.parameters():
            param.requires_grad = False

        # クラス数の取得
        self.num_root_tags = base_model.num_root_quality_classes
        self.num_bass_tags = base_model.num_bass_classes
        self.num_key_tags = base_model.key_head.out_features

        if CRF is not None:
            self.crf_root = CRF(self.num_root_tags, batch_first=True)
            self.crf_bass = CRF(self.num_bass_tags, batch_first=True)
            self.crf_key = CRF(self.num_key_tags, batch_first=True)
        else:
            raise ImportError("pytorch-crf is required for TranscriptionCRFModel")

        # New Projection Heads
        self.root_head = nn.Linear(self.num_root_tags, self.num_root_tags)
        self.bass_head = nn.Linear(self.num_bass_tags, self.num_bass_tags)
        self.key_head = nn.Linear(self.num_key_tags, self.num_key_tags)

    def _get_emissions(self, waveform: torch.Tensor):
        # Base modelの推論 (勾配計算なし)
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(waveform)
            # 各タスクのLogitsを取得
            root_logits = outputs.get("initial_root_chord_logits")
            bass_logits = outputs.get("initial_bass_logits")
            key_logits = outputs.get("initial_key_logits")  # [B, T, C]

            if root_logits is None:
                raise KeyError("base model output missing 'initial_root_chord_logits'")
            if bass_logits is None:
                raise KeyError("base model output missing 'initial_bass_logits'")
            if key_logits is None:
                raise KeyError("base model output missing 'initial_key_logits'")

        emissions_root = self.root_head(root_logits)
        emissions_bass = self.bass_head(bass_logits)
        emissions_key = self.key_head(key_logits)
        return emissions_root, emissions_bass, emissions_key

    def forward(
        self,
        waveform: torch.Tensor,
        root_labels: Optional[torch.Tensor] = None,
        bass_labels: Optional[torch.Tensor] = None,
        key_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, List[List[int]]]]:
        """
        Args:
            waveform: (B, C, T) 音声波形
            root_labels: (B, T)  Root学習用の正解タグ
            bass_labels: (B, T)  Bass学習用の正解タグ
            key_labels: (B, T)   Key学習用の正解タグ
            mask: (B, T)    パディング用のマスク (有効な部分は1, パディングは0)

        Returns:
            ラベルが1つでも指定された場合 (学習時):
                loss (scalar Tensor): 指定されたタスクの負の対数尤度の合計
            ラベルが全てNoneの場合 (推論時):
                predictions (Dict[str, List[List[int]]]): 各タスクのViterbiデコード結果
                    keys: 'root', 'bass', 'key'
        """
        emissions_root, emissions_bass, emissions_key = self._get_emissions(waveform)

        if mask is not None:
            mask = mask.bool()

            # pytorch-crf は mask[:, 0] が全てTrueであることを要求します。
            # もしFalseが含まれている場合(開始位置がパディングされている場合など)、
            # 強制的にTrueにし、ラベルをモデルの予測値で補完します。
            if not mask[:, 0].all():
                # Invalidなバッチインデックスを取得
                invalid_indices = torch.where(~mask[:, 0])[0]

                # Maskを修正
                mask[invalid_indices, 0] = True

                # ラベルを補完 (学習時)
                if root_labels is not None:
                    # Argmaxで予測値を「正解」として扱う
                    preds = emissions_root.argmax(dim=-1)
                    root_labels[invalid_indices, 0] = preds[invalid_indices, 0]

                if bass_labels is not None:
                    preds = emissions_bass.argmax(dim=-1)
                    bass_labels[invalid_indices, 0] = preds[invalid_indices, 0]

                if key_labels is not None:
                    preds = emissions_key.argmax(dim=-1)
                    key_labels[invalid_indices, 0] = preds[invalid_indices, 0]

        # 学習モード (ラベルがある場合)
        if root_labels is not None or bass_labels is not None or key_labels is not None:
            loss = torch.tensor(0.0, device=waveform.device)

            if root_labels is not None:
                safe_root = root_labels.masked_fill(root_labels < 0, 0)
                nll_root = -self.crf_root(emissions_root, safe_root, mask=mask, reduction="mean")
                loss += nll_root

            if bass_labels is not None:
                safe_bass = bass_labels.masked_fill(bass_labels < 0, 0)
                nll_bass = -self.crf_bass(emissions_bass, safe_bass, mask=mask, reduction="mean")
                loss += nll_bass

            if key_labels is not None:
                safe_key = key_labels.masked_fill(key_labels < 0, 0)
                nll_key = -self.crf_key(emissions_key, safe_key, mask=mask, reduction="mean")
                loss += nll_key

            return loss

        else:
            # 推論モード (デコード)
            preds = {}
            preds["root"] = self.crf_root.decode(emissions_root, mask=mask)
            preds["bass"] = self.crf_bass.decode(emissions_bass, mask=mask)
            preds["key"] = self.crf_key.decode(emissions_key, mask=mask)
            return preds

    def get_loss_and_preds(
        self,
        waveform: torch.Tensor,
        root_labels: Optional[torch.Tensor] = None,
        bass_labels: Optional[torch.Tensor] = None,
        key_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[List[int]]]]:
        """
        検証用: LossとPredictionsの両方を返します。
        """
        emissions_root, emissions_bass, emissions_key = self._get_emissions(waveform)

        if mask is not None:
            mask = mask.bool()
            # mask fix logic (duplicated for now or shared if I refactor more, sticking to inline for clarity/safety of replace)
            if not mask[:, 0].all():
                invalid_indices = torch.where(~mask[:, 0])[0]
                mask[invalid_indices, 0] = True
                if root_labels is not None:
                    preds = emissions_root.argmax(dim=-1)
                    root_labels[invalid_indices, 0] = preds[invalid_indices, 0]
                if bass_labels is not None:
                    preds = emissions_bass.argmax(dim=-1)
                    bass_labels[invalid_indices, 0] = preds[invalid_indices, 0]
                if key_labels is not None:
                    preds = emissions_key.argmax(dim=-1)
                    key_labels[invalid_indices, 0] = preds[invalid_indices, 0]

        loss = torch.tensor(0.0, device=waveform.device)
        if root_labels is not None:
            safe_root = root_labels.masked_fill(root_labels < 0, 0)
            loss += -self.crf_root(emissions_root, safe_root, mask=mask, reduction="mean")
        if bass_labels is not None:
            safe_bass = bass_labels.masked_fill(bass_labels < 0, 0)
            loss += -self.crf_bass(emissions_bass, safe_bass, mask=mask, reduction="mean")
        if key_labels is not None:
            safe_key = key_labels.masked_fill(key_labels < 0, 0)
            loss += -self.crf_key(emissions_key, safe_key, mask=mask, reduction="mean")

        preds = {}
        preds["root"] = self.crf_root.decode(emissions_root, mask=mask)
        preds["bass"] = self.crf_bass.decode(emissions_bass, mask=mask)
        preds["key"] = self.crf_key.decode(emissions_key, mask=mask)

        return loss, preds

    def train(self, mode: bool = True):
        """
        trainモードの上書き。
        base_modelは常にevalモードを維持するようにします。
        """
        super().train(mode)
        self.base_model.eval()
        return self
