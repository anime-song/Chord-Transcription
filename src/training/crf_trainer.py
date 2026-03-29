import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .augmentations import apply_batch_time_stretch, apply_batch_waveform_augmentation
from ..data.audio_augmentation import StereoWaveformAugmentation
from ..chord_transcription.models.factory import save_model_build_config_sidecar


class CRFTrainer:
    """
    CRF 学習専用の実行ループ。
    optimizer / writer / checkpoint metadata は train script 側で組み立て、
    ここでは CRF の学習進行だけを扱う。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        writer: Any,
        scheduler: Any = None,
        checkpoint_extras: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.config = config
        self.writer = writer
        # checkpoint に追加で埋めたい metadata は外側から受け取る。
        self.checkpoint_extras = dict(checkpoint_extras or {})

        self.train_cfg = config.get("crf_training", {})
        if not self.train_cfg:
            raise KeyError("crf_training section is missing from config.")
        self.ignore_index = int(config.get("base_model_training", {}).get("loss", {}).get("ce_ignore_index", -100))

        self.global_step = 0
        self.start_epoch = 1
        self.validate_every = self.train_cfg.get("validate_every_n_epochs", 1)

        self.accum_steps = max(1, int(self.train_cfg.get("grad_accum_steps", 1)))

        # Disable automatic mixed precision for CRF logsumexp numerical stability if requested
        self.amp_enabled = self.train_cfg.get("amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.device_waveform_augmentation: Optional[StereoWaveformAugmentation] = None

        loader_cfg = self.config.get("data_loader", {})
        waveform_aug_cfg = loader_cfg.get("waveform_augmentation", {}) or {}
        waveform_aug_backend = str(waveform_aug_cfg.get("backend", "cpu")).lower()
        if bool(waveform_aug_cfg.get("enabled", False)) and waveform_aug_backend == "cuda":
            if self.device.type != "cuda":
                raise ValueError("waveform_augmentation.backend='cuda' requires experiment.device='cuda'")
            self.device_waveform_augmentation = StereoWaveformAugmentation(
                sample_rate=int(loader_cfg.get("sample_rate", 22050)),
                config=waveform_aug_cfg,
            )
            print(f"{self.device_waveform_augmentation.summary()} [cuda mix]")

    def train(self):
        epochs = self.train_cfg["epochs"]
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} (CRF) ---")

            # 1. CRF 学習
            train_log = self._train_epoch(epoch)
            torch.cuda.empty_cache()

            # 2. 必要なタイミングだけ検証
            if (epoch % self.validate_every == 0) or (epoch == epochs):
                valid_log = self._validate_epoch(epoch)
            else:
                valid_log = {}

            # 3. エポック終端で scheduler 更新
            if self.scheduler:
                self.scheduler.step()

            # 4. ログ出力
            log_str = f"[Epoch {epoch}/{epochs}] "
            for k, v in train_log.items():
                log_str += f"{k}: {v:.4f} | "
                self.writer.add_scalar(f"Train/{k}", v, epoch)

            if valid_log:
                for k, v in valid_log.items():
                    log_str += f"val_{k}: {v:.4f} | "
                    self.writer.add_scalar(f"Valid/{k}", v, epoch)
            print(log_str.strip(" | "))

            # 5. CRF 固有の健全性指標を記録
            self._log_model_health(epoch)

            # 6. checkpoint 保存
            if epoch % self.train_cfg.get("model_save_every_n_epochs", 1) == 0:
                self._save_checkpoint(epoch)

    @torch.no_grad()
    def _log_model_health(self, epoch: int) -> None:
        """
        Projection層のノルムとCRF遷移行列の健全性指標をTensorBoardに記録する。

        指標と目標値:
          weight_norm_root_chord_proj
            初期値 ~89 (問題あり) → 目標: 20以下
            50以上で警戒、80以上は weight_decay をさらに強化すること

          weight_norm_bass_proj / weight_norm_key_proj
            初期値 ~3 (正常) → 横ばいを維持すること
            10を超えたら警戒

          trans_dominance_{root_chord|bass|key}
            全タグ中「自己遷移スコアが最大」なタグの割合
            初期値 root_chord ~78%、bass/key 100%
            → 目標: root_chord も 95% 以上
            80% 未満が続く場合は CRF が平滑化できていない

          trans_margin_{root_chord|bass|key}
            対角成分(自己遷移)の平均 - 非対角最大値の平均
            正の値ほど平滑化が強く効く
            → 目標: 正の値 (> 0) を維持
            負になると CRF が他クラスへ積極的に遷移する状態
        """
        health_log = {}

        # Projection 層のノルム (Sequential でも Linear でも対応)
        for attr in ["root_chord_proj", "bass_proj", "key_proj"]:
            layer = getattr(self.model, attr, None)
            if layer is None:
                continue
            # Sequential の場合は最初の Linear を探す
            if isinstance(layer, torch.nn.Linear):
                linear = layer
            else:
                linear = next((m for m in layer.modules() if isinstance(m, torch.nn.Linear)), None)
            if linear is None:
                continue
            norm = linear.weight.data.norm().item()
            health_log[f"Health/weight_norm_{attr}"] = norm

        # CRF 遷移行列の健全性
        for crf_name in ["crf_root_chord", "crf_bass", "crf_key"]:
            crf = getattr(self.model, crf_name, None)
            if crf is None:
                continue
            trans = crf.transitions.detach().cpu()  # (C, C)
            diag = trans.diag()
            # 非対角成分の各行の最大値
            off_diag = trans.clone()
            off_diag.fill_diagonal_(-1e9)
            off_diag_max = off_diag.max(dim=1).values
            # 自己遷移が優勢なタグの割合
            dominance_ratio = (diag > off_diag_max).float().mean().item()
            # 遷移マージン: 対角平均 - 非対角最大平均
            margin = (diag - off_diag_max).mean().item()

            short = crf_name.replace("crf_", "")
            health_log[f"Health/trans_dominance_{short}"] = dominance_ratio
            health_log[f"Health/trans_margin_{short}"] = margin

        for k, v in health_log.items():
            self.writer.add_scalar(k, v, epoch)

        # 標準出力への簡易表示
        rc_norm = health_log.get("Health/weight_norm_root_chord_proj", float("nan"))
        rc_margin = health_log.get("Health/trans_margin_root_chord", float("nan"))
        bass_norm = health_log.get("Health/weight_norm_bass_proj", float("nan"))
        rc_dom = health_log.get("Health/trans_dominance_root_chord", float("nan"))
        print(
            f"  [Health] root_chord_proj norm={rc_norm:.2f} "
            f"trans_margin={rc_margin:.4f} trans_dominance={rc_dom:.2%} "
            f"| bass_proj norm={bass_norm:.2f}"
        )

    def _get_crf_mask(
        self, root_target: torch.Tensor, bass_target: torch.Tensor, key_target: torch.Tensor
    ) -> torch.Tensor:
        """Create a shared valid-token mask across all CRF targets."""
        return (
            (root_target != self.ignore_index) & (bass_target != self.ignore_index) & (key_target != self.ignore_index)
        ).bool()

    @staticmethod
    def _filter_batch_by_keep_mask(batch: Dict[str, torch.Tensor], keep_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Keep only samples where keep_mask is True, for all tensors with batch dimension."""
        filtered_batch: Dict[str, torch.Tensor] = {}
        batch_size = int(keep_mask.shape[0])

        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == batch_size:
                filtered_batch[key] = value[keep_mask]
            else:
                filtered_batch[key] = value

        return filtered_batch

    @staticmethod
    def _valid_crf_sample_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Return per-sample validity for torchcrf:
        - first timestep must be valid
        - mask must be right-padded (no True after first False)
        - at least one valid timestep exists
        """
        starts_valid = mask[:, 0]
        has_any_valid = mask.any(dim=1)
        saw_false = (~mask).cumsum(dim=1) > 0
        has_true_after_false = (mask & saw_false).any(dim=1)
        return starts_valid & has_any_valid & (~has_true_after_false)

    def _prepare_crf_batch(self, batch: Dict[str, torch.Tensor]):
        """torchcrf の制約を満たさないサンプルを落として mask を作る。"""
        mask = self._get_crf_mask(batch["root_chord_index"], batch["bass_index"], batch["key_index"])
        keep_mask = self._valid_crf_sample_mask(mask)
        dropped = int((~keep_mask).sum().item())

        if dropped == 0:
            return batch, mask, dropped

        batch = self._filter_batch_by_keep_mask(batch, keep_mask)
        mask = mask[keep_mask]
        return batch, mask, dropped

    def _optimizer_step(self):
        # accumulation の区切りごとにまとめて optimizer を進める。
        grad_clip_norm = self.train_cfg.get("grad_clip_norm", None)
        if grad_clip_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_losses = {}
        micro_step = 0
        processed_batches = 0
        skipped_samples = 0
        seen_samples = 0

        if tqdm:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        else:
            progress_bar = self.train_loader

        self.optimizer.zero_grad(set_to_none=True)

        loss_weights = self.train_cfg.get("loss_weights", {"bass": 1.0, "root_chord": 1.0, "key": 1.0})

        for batch in progress_bar:
            for key in batch:
                batch[key] = batch[key].to(self.device)

            seen_samples += int(batch["audio"].shape[0])
            batch = apply_batch_waveform_augmentation(batch, self.device_waveform_augmentation)
            batch = apply_batch_time_stretch(batch)
            batch, mask, dropped = self._prepare_crf_batch(batch)
            skipped_samples += dropped

            if int(mask.shape[0]) == 0:
                continue

            processed_batches += 1

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(
                    waveform=batch["audio"],
                    root_chord_target=batch["root_chord_index"],
                    bass_target=batch["bass_index"],
                    key_target=batch["key_index"],
                    mask=mask,
                )

                loss_bass = outputs.get("crf_loss_bass", torch.tensor(0.0, device=self.device))
                loss_root = outputs.get("crf_loss_root_chord", torch.tensor(0.0, device=self.device))
                loss_key = outputs.get("crf_loss_key", torch.tensor(0.0, device=self.device))

                supervised_total_loss = (
                    loss_bass * loss_weights.get("bass", 1.0)
                    + loss_root * loss_weights.get("root_chord", 1.0)
                    + loss_key * loss_weights.get("key", 1.0)
                )

            supervised_loss_for_backward = supervised_total_loss / self.accum_steps
            if supervised_loss_for_backward.requires_grad:
                self.scaler.scale(supervised_loss_for_backward).backward()

            total_loss = supervised_total_loss.detach()

            running_losses.setdefault("crf_bass", 0.0)
            running_losses["crf_bass"] += loss_bass.item()
            running_losses.setdefault("crf_root", 0.0)
            running_losses["crf_root"] += loss_root.item()
            running_losses.setdefault("crf_key", 0.0)
            running_losses["crf_key"] += loss_key.item()
            running_losses.setdefault("total", 0.0)
            running_losses["total"] += total_loss.item()

            micro_step += 1
            if micro_step % self.accum_steps == 0:
                self._optimizer_step()

            if tqdm:
                progress_bar.set_postfix(loss=total_loss.item())

        if micro_step > 0 and micro_step % self.accum_steps != 0:
            self._optimizer_step()

        denom = max(1, processed_batches)
        epoch_losses = {k: v / denom for k, v in running_losses.items()}
        epoch_losses["skipped_samples_ratio"] = float(skipped_samples) / float(max(1, seen_samples))

        return epoch_losses

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_losses = {}
        running_metrics = {}
        processed_batches = 0
        skipped_samples = 0
        seen_samples = 0

        loss_weights = self.train_cfg.get("loss_weights", {"bass": 1.0, "root_chord": 1.0, "key": 1.0})

        for batch in self.valid_loader:
            for key in batch:
                batch[key] = batch[key].to(self.device)

            seen_samples += int(batch["audio"].shape[0])
            batch = apply_batch_time_stretch(batch)
            batch, mask, dropped = self._prepare_crf_batch(batch)
            skipped_samples += dropped

            if int(mask.shape[0]) == 0:
                continue

            processed_batches += 1

            # 1. Compute loss
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(
                    waveform=batch["audio"],
                    root_chord_target=batch["root_chord_index"],
                    bass_target=batch["bass_index"],
                    key_target=batch["key_index"],
                    mask=mask,
                )

                loss_bass = outputs.get("crf_loss_bass", torch.tensor(0.0, device=self.device))
                loss_root = outputs.get("crf_loss_root_chord", torch.tensor(0.0, device=self.device))
                loss_key = outputs.get("crf_loss_key", torch.tensor(0.0, device=self.device))

                total_loss = (
                    loss_bass * loss_weights.get("bass", 1.0)
                    + loss_root * loss_weights.get("root_chord", 1.0)
                    + loss_key * loss_weights.get("key", 1.0)
                )

            running_losses.setdefault("crf_bass", 0.0)
            running_losses["crf_bass"] += loss_bass.item()
            running_losses.setdefault("crf_root", 0.0)
            running_losses["crf_root"] += loss_root.item()
            running_losses.setdefault("crf_key", 0.0)
            running_losses["crf_key"] += loss_key.item()
            running_losses.setdefault("total", 0.0)
            running_losses["total"] += total_loss.item()

            # 2. Compute metrics via Viterbi decoding
            decoded = self.model.decode(batch["audio"], mask=mask)

            # Calculate accuracy from predicted sequences
            def _calc_acc(preds_list, targets, ignore_index: int):
                correct = 0
                total = 0
                for b, preds in enumerate(preds_list):
                    t = targets[b][: len(preds)]
                    valid = t != ignore_index

                    if valid.sum() > 0:
                        pt = torch.tensor(preds, device=self.device)
                        correct += (pt[valid] == t[valid]).sum().item()
                        total += valid.sum().item()
                return correct, total

            c_bass, t_bass = _calc_acc(decoded["bass_predictions"], batch["bass_index"], self.ignore_index)
            c_root, t_root = _calc_acc(decoded["root_chord_predictions"], batch["root_chord_index"], self.ignore_index)
            c_key, t_key = _calc_acc(decoded["key_predictions"], batch["key_index"], self.ignore_index)

            running_metrics.setdefault("c_bass", 0)
            running_metrics["c_bass"] += c_bass
            running_metrics.setdefault("t_bass", 0)
            running_metrics["t_bass"] += t_bass

            running_metrics.setdefault("c_root", 0)
            running_metrics["c_root"] += c_root
            running_metrics.setdefault("t_root", 0)
            running_metrics["t_root"] += t_root

            running_metrics.setdefault("c_key", 0)
            running_metrics["c_key"] += c_key
            running_metrics.setdefault("t_key", 0)
            running_metrics["t_key"] += t_key

        denom = max(1, processed_batches)
        epoch_results = {k: v / denom for k, v in running_losses.items()}

        # Aggregate metrics
        if running_metrics.get("t_bass", 0) > 0:
            epoch_results["acc/bass_viterbi"] = running_metrics["c_bass"] / running_metrics["t_bass"]
        if running_metrics.get("t_root", 0) > 0:
            epoch_results["acc/root_viterbi"] = running_metrics["c_root"] / running_metrics["t_root"]
        if running_metrics.get("t_key", 0) > 0:
            epoch_results["acc/key_viterbi"] = running_metrics["c_key"] / running_metrics["t_key"]
        epoch_results["skipped_samples_ratio"] = float(skipped_samples) / float(max(1, seen_samples))

        return epoch_results

    def _save_checkpoint(self, epoch: int):
        ckpt_dir = Path(self.config["data"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_path = ckpt_dir / f"crf_model_epoch_{epoch:03d}.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        # model config などの補助 metadata は外側から受け取ったものを足す。
        checkpoint.update(self.checkpoint_extras)
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.amp_enabled:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
        model_build_config = checkpoint.get("model_build_config")
        if model_build_config is not None:
            sidecar_path = save_model_build_config_sidecar(save_path, model_build_config)
            print(f"Model config saved to {sidecar_path}")

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)

        # CRF resume は optimizer / scheduler / scaler まで含めて復元する。
        model_state = checkpoint.get("model_state_dict")
        self.model.load_state_dict(model_state, strict=strict)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.amp_enabled and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed CRFTraining from epoch {self.start_epoch - 1}.")
