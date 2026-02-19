import json

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, List, Union, Optional, Tuple, Set
import time
from pathlib import Path
from copy import deepcopy

try:
    from tqdm.auto import tqdm
except ImportError:
    print("Warning: tqdm is not installed. Progress bar will not be shown. Please install with 'pip install tqdm'")
    tqdm = None

from .losses import compute_losses, compute_metrics
from .losses import BalancedSoftmaxLoss
from src.models.segment_model import resolve_segment_decode_params


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class Trainer:
    """
    モデルの学習プロセス全体（学習、検証、ロギング、保存）を管理するクラス。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Any = None,
        quality_class_counts: List[int] = None,
        train_config_key: str = "base_model_training",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.config = config
        if train_config_key not in config:
            raise KeyError(f"Training configuration '{train_config_key}' not found in config.")
        self.train_config_key = train_config_key
        self.train_cfg = config[train_config_key]
        if not isinstance(self.train_cfg, dict):
            raise TypeError(f"Config section '{train_config_key}' must be a mapping.")

        base_loss_cfg = config.get("base_model_training", {}).get("loss", {})
        self.loss_cfg = self.train_cfg.get("loss", base_loss_cfg) or {}
        self.ce_ignore_index = int(self.loss_cfg.get("ce_ignore_index", -100))
        self.segment_decode_cfg = resolve_segment_decode_params(config.get("segment_decode", {}))

        if quality_class_counts is None:
            raise ValueError("BalancedSoftmaxLossには quality_class_counts が必要です。")
        tau = self.loss_cfg.get("balanced_softmax_tau", 1.0)
        root_chord_class_counts = self._build_root_chord_class_counts(quality_class_counts)
        self.root_chord_loss_fn = BalancedSoftmaxLoss(class_counts=root_chord_class_counts, tau=tau).to(self.device)

        # TensorBoard用のWriterをセットアップ
        log_dir = Path(config["data"]["log_dir"]) / f'{config["experiment"]["name"]}_{time.strftime("%Y%m%d-%H%M%S")}'
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs will be saved to: {log_dir}")

        self.global_step = 0
        self.start_epoch = 1
        self.validate_every = self.train_cfg.get("validate_every_n_epochs", 1)
        print(f"Validation will run every {self.validate_every} epochs.")

        self.accum_steps = max(1, int(self.train_cfg.get("grad_accum_steps", 1)))

        self.amp_enabled = self.train_cfg.get("amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)

        self.ema = ModelEma(self.model, decay=0.99, device=self.device)

    def _filter_state_dict(
        self, incoming_state: Dict[str, Any], target_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Drop entries whose keys are missing or shapes mismatch to allow partial loading."""
        filtered = {}
        skipped = []
        for k, v in incoming_state.items():
            if (
                k in target_state
                and target_state[k].shape == v.shape
                and ("backbone" in k or "boundary_head" in k)
            ):
                filtered[k] = v
            else:
                skipped.append(k)
        return filtered, skipped

    def train(self):
        """設定されたエポック数だけ学習ループを実行します。"""
        epochs = self.train_cfg["epochs"]
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")

            # 1エポックの学習
            train_log = self._train_epoch(epoch)

            torch.cuda.empty_cache()
            # 1エポックの検証
            if (epoch % self.validate_every == 0) or (epoch == epochs):
                valid_log = self._validate_epoch(epoch)
            else:
                valid_log = {}

            # 学習率の更新（スケジューラがあれば）
            if self.scheduler:
                self.scheduler.step()

            # ログの表示とTensorBoardへの書き込み
            log_str = f"[Epoch {epoch}/{epochs}] "
            for k, v in train_log.items():
                log_str += f"{k}: {v:.4f} | "
                self.writer.add_scalar(f"Train/{k}", v, epoch)

            if valid_log:
                for k, v in valid_log.items():
                    log_str += f"val_{k}: {v:.4f} | "
                    self.writer.add_scalar(f"Valid/{k}", v, epoch)
            print(log_str.strip(" | "))

            # モデルのチェックポイントを保存
            if epoch % self.train_cfg.get("model_save_every_n_epochs", 1) == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポック分の学習処理。"""
        self.model.train()
        running_losses = {}
        usage_accum: Optional[torch.Tensor] = None
        usage_count = 0
        micro_step = 0

        # tqdmがインストールされていれば、DataLoaderをtqdmでラップ
        if tqdm:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        else:
            progress_bar = self.train_loader

        self.optimizer.zero_grad(set_to_none=True)

        total_batches = len(self.train_loader)

        for batch_index, batch in enumerate(progress_bar, start=1):
            # バッチをデバイスに移動
            for key in batch:
                batch[key] = batch[key].to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch["audio"])
                losses = compute_losses(
                    outputs,
                    batch,
                    self.loss_cfg,
                    root_chord_loss_fn=self.root_chord_loss_fn,
                    segment_decode_cfg=self.segment_decode_cfg,
                )
                total_loss = sum(losses.values())

                usage = outputs.get("repeat/sec_type_usage")
                if usage is not None:
                    usage = usage.detach().cpu()
                    if usage_accum is None:
                        usage_accum = usage.clone()
                    else:
                        usage_accum += usage
                    usage_count += 1

                loss_for_backward = total_loss / self.accum_steps
                self.scaler.scale(loss_for_backward).backward()

                for k, v in losses.items():
                    running_losses.setdefault(k, 0.0)
                    running_losses[k] += v.item()
                running_losses.setdefault("total", 0.0)
                running_losses["total"] += total_loss.item()

                micro_step += 1
                last_batch = batch_index == total_batches
                do_step = (micro_step % self.accum_steps == 0) or last_batch

                if do_step:
                    grad_clip_norm = self.train_cfg.get("grad_clip_norm", None)
                    if grad_clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.ema.update(self.model)
                    self.global_step += 1

                if tqdm:
                    progress_bar.set_postfix(loss=total_loss.item())

        # エポックの平均損失を計算
        denom = len(self.train_loader)
        epoch_losses = {k: v / denom for k, v in running_losses.items()}

        if usage_accum is not None and usage_count > 0:
            mean_usage = usage_accum / usage_count
            for idx, value in enumerate(mean_usage.tolist()):
                self.writer.add_scalar(f"Train/section_type_usage/{idx}", value, epoch)
        return epoch_losses

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポック分の検証処理。"""
        self.model.eval()
        running_losses = {}
        running_ema_losses = {}
        running_metrics = {}
        running_ema = {}
        usage_accum: Optional[torch.Tensor] = None
        usage_count = 0

        for batch in self.valid_loader:
            for key in batch:
                batch[key] = batch[key].to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch["audio"])
                ema_outputs = self.ema.module(batch["audio"])

                # ロス計算
                losses = compute_losses(
                    outputs, batch, self.loss_cfg,
                    root_chord_loss_fn=self.root_chord_loss_fn,
                    segment_decode_cfg=self.segment_decode_cfg,
                )
                for k, v in losses.items():
                    running_losses.setdefault(k, 0.0)
                    running_losses[k] += v.item()
                running_losses.setdefault("total", 0.0)
                running_losses["total"] += sum(v.item() for v in losses.values())

                ema_losses = compute_losses(
                    ema_outputs, batch, self.loss_cfg,
                    root_chord_loss_fn=self.root_chord_loss_fn,
                    segment_decode_cfg=self.segment_decode_cfg,
                )
                for k, v in ema_losses.items():
                    running_ema_losses.setdefault(f"ema_{k}", 0.0)
                    running_ema_losses[f"ema_{k}"] += v.item()
                running_ema_losses.setdefault("ema_total", 0.0)
                running_ema_losses["ema_total"] += sum(v.item() for v in ema_losses.values())

            usage = outputs.get("repeat/sec_type_usage")
            if usage is not None:
                usage = usage.detach().cpu()
                if usage_accum is None:
                    usage_accum = usage.clone()
                else:
                    usage_accum += usage
                usage_count += 1

            metrics = compute_metrics(outputs, batch, self.ce_ignore_index)
            for k, v in metrics.items():
                running_metrics.setdefault(k, 0.0)
                running_metrics[k] += v.item()

            metrics_ema = compute_metrics(ema_outputs, batch, self.ce_ignore_index)
            for k, v in metrics_ema.items():
                running_ema.setdefault(f"ema_{k}", 0.0)
                running_ema[f"ema_{k}"] += v.item()

        denom = len(self.valid_loader)
        # ロスとメトリクスをまとめる
        epoch_results = {k: v / denom for k, v in running_losses.items()}
        epoch_results.update({k: v / denom for k, v in running_ema_losses.items()})
        epoch_results.update({k: v / denom for k, v in running_metrics.items()})
        epoch_results.update({k: v / denom for k, v in running_ema.items()})

        if usage_accum is not None and usage_count > 0:
            mean_usage = usage_accum / usage_count
            for idx, value in enumerate(mean_usage.tolist()):
                self.writer.add_scalar(f"Valid/section_type_usage/{idx}", value, epoch)
        return epoch_results

    def _save_checkpoint(self, epoch: int):
        """モデルのチェックポイントを保存します。"""
        ckpt_dir = Path(self.config["data"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_path = ckpt_dir / f"model_epoch_{epoch:03d}.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.amp_enabled:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = True,
        resume_epoch: Optional[int] = None,
        load_backbone_only: bool = False,
    ):
        """保存済みチェックポイントから学習状態を復元します。

        resume_epoch を指定した場合はエポックとグローバルステップをリセットし、
        Optimizer/Scheduler/Scaler の状態は読み込まない（重みのみ読み込み）。
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if resume_epoch is not None:
            resume_epoch = int(resume_epoch)
            if resume_epoch < 1:
                raise ValueError("resume_epoch must be >= 1")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise KeyError("checkpoint does not contain 'model_state_dict'")
        if load_backbone_only:
            model_state, _ = self._filter_state_dict(model_state, self.model.state_dict())
        self.model.load_state_dict(model_state, strict=strict)

        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is not None:
            if load_backbone_only:
                ema_state, _ = self._filter_state_dict(ema_state, self.ema.module.state_dict())
            self.ema.module.load_state_dict(ema_state, strict=strict)

        load_optimizer_state = resume_epoch is None

        optim_state = checkpoint.get("optimizer_state_dict")
        try:
            if optim_state is not None and load_optimizer_state:
                self.optimizer.load_state_dict(optim_state)
        except Exception:
            print("[WARN] optimizerが正しく読み込めませんでした")

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint and load_optimizer_state:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.amp_enabled and "scaler_state_dict" in checkpoint and load_optimizer_state:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        last_epoch = int(checkpoint.get("epoch", 0))
        if resume_epoch is not None:
            self.global_step = 0
            self.start_epoch = resume_epoch
            self._align_scheduler_for_resume(resume_epoch)
            print(
                f"Loaded weights from epoch {last_epoch}; restarting from epoch {self.start_epoch} with global_step reset."
            )
        else:
            self.global_step = int(checkpoint.get("global_step", self.global_step))
            self.start_epoch = last_epoch + 1
            print(f"Resumed training from epoch {last_epoch}, global_step {self.global_step}.")

        if not load_optimizer_state:
            print("Skipped optimizer/scheduler/scaler states because resume_epoch was specified.")

    def _align_scheduler_for_resume(self, resume_epoch: int):
        """When resuming without scheduler state, advance scheduler to match the desired start epoch."""
        if self.scheduler is None:
            return

        target_last_epoch = max(resume_epoch - 2, -1)
        self.scheduler.last_epoch = target_last_epoch

        # Keep internal counters in sync to avoid warnings in PyTorch.
        if hasattr(self.scheduler, "_step_count"):
            self.scheduler._step_count = max(target_last_epoch + 1, 0)

        try:
            if target_last_epoch < 0:
                lrs = [group["lr"] for group in self.optimizer.param_groups]
            else:
                lrs = list(self.scheduler.get_lr())
            for lr, param_group in zip(lrs, self.optimizer.param_groups):
                param_group["lr"] = lr
            if hasattr(self.scheduler, "_last_lr"):
                self.scheduler._last_lr = lrs
        except Exception:
            # If the scheduler does not expose get_lr, fall back to leaving LRs as-is.
            pass

    def _build_root_chord_class_counts(self, quality_class_counts: List[int]) -> List[int]:
        """qualityの頻度からroot-chord用のクラス頻度を12倍して生成します"""
        quality_json_path = Path(self.config["data"]["quality_json_path"])
        with quality_json_path.open("r", encoding="utf-8") as f:
            quality_map = json.load(f)

        label_to_index = {v: int(k) for k, v in quality_map.items()}
        non_chord_index = label_to_index.get("N")
        if non_chord_index is None:
            raise ValueError("quality vocabulary must contain 'N' for non-chord handling")

        num_quality_classes = self.config["model"]["num_quality_classes"]
        num_root_classes = self.config["model"]["num_root_classes"]
        if len(quality_class_counts) != num_quality_classes:
            raise ValueError("quality_class_counts length does not match num_quality_classes")

        counts_without_non_chord = [count for idx, count in enumerate(quality_class_counts) if idx != non_chord_index]
        if len(counts_without_non_chord) != num_quality_classes - 1:
            raise ValueError("Failed to filter out non-chord count from quality_class_counts")

        root_chord_counts = counts_without_non_chord * (num_root_classes - 1)
        root_chord_counts.append(quality_class_counts[non_chord_index])
        return root_chord_counts
