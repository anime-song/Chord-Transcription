import torch
import torch.nn as nn
import random
from typing import Dict, Tuple, List


class SpecAugment(nn.Module):
    """
    スペクトログラムにSpecAugmentを適用するモジュール。
    時間マスキングと周波数マスキングを行います。
    学習時(`model.train()`)にのみ適用されます。
    """

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        p: float = 1.0,
    ):
        """
        Args:
            freq_mask_param (int): 周波数マスクの最大幅 (F)
            time_mask_param (int): 時間マスクの最大幅 (T)
            num_freq_masks (int): 適用する周波数マスクの数
            num_time_masks (int): 適用する時間マスクの数
            p (float): Augmentationを適用する確率
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def forward(self, spec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            spec (torch.Tensor): 入力スペクトログラム (B, C, F, T)
        Returns:
            tuple: (Augmentationが適用されたスペクトログラム, マスク情報)
        """
        device = spec.device
        batch_size, _, num_mels, num_frames = spec.shape
        freq_mask = torch.zeros(batch_size, num_mels, device=device, dtype=torch.bool)
        time_mask = torch.zeros(batch_size, num_frames, device=device, dtype=torch.bool)

        # self.trainingはnn.Moduleが持つフラグで、model.train() / model.eval()で切り替わる
        if not self.training or random.random() > self.p:
            return spec, {"freq_mask": freq_mask, "time_mask": time_mask}

        # 元のスペクトログラムをコピーして変更
        aug_spec = spec.clone()

        for i in range(batch_size):
            # 周波数マスキング
            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                if f == 0:
                    continue
                f0 = random.randint(0, num_mels - f)
                aug_spec[i, :, f0 : f0 + f, :] = 0
                freq_mask[i, f0 : f0 + f] = True

            # 時間マスキング
            for _ in range(self.num_time_masks):
                t = random.randint(0, self.time_mask_param)
                if t == 0:
                    continue
                t0 = random.randint(0, num_frames - t)
                aug_spec[i, :, :, t0 : t0 + t] = 0
                time_mask[i, t0 : t0 + t] = True

        return aug_spec, {"freq_mask": freq_mask, "time_mask": time_mask}


class MiniBatchMixtureMasking(nn.Module):
    """
    Mini-batch based Mixture Masking (MM).
    入力 or 隠れ表現に対して、時間/周波数の連続区間を
    同一バッチ内の他サンプルとの平均 (x + y) / 2 で置換します。
    学習時のみ適用されます。
    """

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        p: float = 1.0,
        fallback_when_batch1: str = "zero",  # "skip" or "zero"
    ):
        """
        Args:
            freq_mask_param (int): 周波数マスクの最大幅 (F)
            time_mask_param (int): 時間マスクの最大幅 (T)
            num_freq_masks (int): 適用する周波数マスクの数
            num_time_masks (int): 適用する時間マスクの数
            p (float): Augmentationを適用する確率
            fallback_when_batch1 (str): バッチサイズが1のときの挙動
                - "skip": 何もしない
                - "zero": ゼロ詰め(ZM)にフォールバック
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p
        assert fallback_when_batch1 in ("skip", "zero")
        self.fallback_when_batch1 = fallback_when_batch1

    def forward(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): (B, C, F, T)
        Returns:
            tuple: (Augmented tensor, info dict)
                info["freq_mask"]: (B, F) bool
                info["time_mask"]: (B, T) bool
                info["partner_idx"]: (B,) long (適用時のみ有効, それ以外は-1)
        """
        device = x.device
        B, C, F, T = x.shape

        freq_mask = torch.zeros(B, F, device=device, dtype=torch.bool)
        time_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
        partner_idx = torch.full((B,), -1, device=device, dtype=torch.long)

        if (not self.training) or (random.random() > self.p):
            return x, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}

        # バッチサイズ1のときの扱い
        if B < 2:
            if self.fallback_when_batch1 == "skip":
                return x, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}
            # "zero" フォールバック（ZM）
            aug = x.clone()
            for i in range(B):
                for _ in range(self.num_freq_masks):
                    f = random.randint(0, self.freq_mask_param)
                    if f > 0:
                        f0 = random.randint(0, F - f)
                        aug[i, :, f0 : f0 + f, :] = 0
                        freq_mask[i, f0 : f0 + f] = True
                for _ in range(self.num_time_masks):
                    t = random.randint(0, self.time_mask_param)
                    if t > 0:
                        t0 = random.randint(0, T - t)
                        aug[i, :, :, t0 : t0 + t] = 0
                        time_mask[i, t0 : t0 + t] = True
            return aug, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}

        # 通常: MM を適用
        aug = x.clone()
        # partner をサンプルごとに一人だけ選ぶ（マスク数に関わらず固定）
        if group_ids is not None:
            group_ids = group_ids.to(device)
            # group_id -> [indices] を作る
            groups = {}
            for i in range(B):
                g = int(group_ids[i].item())
                groups.setdefault(g, []).append(i)

            # 各グループ内でペアを選ぶ（同ステム同士のみ）
            for g, idxs in groups.items():
                n = len(idxs)
                if n == 1:
                    # グループに1件しかないときはスキップ（必要ならゼロ詰めにするなど拡張可）
                    continue
                for i in idxs:
                    if n == 2:
                        j = idxs[1] if i == idxs[0] else idxs[0]
                    else:
                        # 自分以外からランダムに選ぶ
                        while True:
                            j = random.choice(idxs)
                            if j != i:
                                break
                    partner_idx[i] = j
        else:
            # バッチ全体から相手を選ぶ
            for i in range(B):
                j = random.randrange(B - 1)
                if j >= i:
                    j += 1
                partner_idx[i] = j

        # 元の x を参照して置換（連続マスクの順序で結果が歪まないように）
        for i in range(B):
            y = x[int(partner_idx[i].item())]  # (C, F, T)
            # 周波数方向マスク
            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                if f == 0:
                    continue
                f0 = random.randint(0, F - f)
                # (C, f, T) を平均に置換
                aug[i, :, f0 : f0 + f, :] = 0.5 * (x[i, :, f0 : f0 + f, :] + y[:, f0 : f0 + f, :])
                freq_mask[i, f0 : f0 + f] = True

            # 時間方向マスク
            for _ in range(self.num_time_masks):
                t = random.randint(0, self.time_mask_param)
                if t == 0:
                    continue
                t0 = random.randint(0, T - t)
                # (C, F, t) を平均に置換
                aug[i, :, :, t0 : t0 + t] = 0.5 * (x[i, :, :, t0 : t0 + t] + y[:, :, t0 : t0 + t])
                time_mask[i, t0 : t0 + t] = True

        return aug, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}
