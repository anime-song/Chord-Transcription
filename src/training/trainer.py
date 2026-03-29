import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Optional, Tuple
from pathlib import Path
from copy import deepcopy

try:
    from tqdm.auto import tqdm
except ImportError:
    print("Warning: tqdm is not installed. Progress bar will not be shown. Please install with 'pip install tqdm'")
    tqdm = None

from .losses import compute_losses, compute_metrics
from .augmentations import apply_batch_cutmix, apply_batch_time_stretch, apply_batch_waveform_augmentation
from ..data.audio_augmentation import StereoWaveformAugmentation
from ..chord_transcription.models.factory import save_model_build_config_sidecar
from ..chord_transcription.models.repeat_ssm import RepeatPairBuilderCPU


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
    base model 学習の実行ループを担当する。
    optimizer / writer / checkpoint metadata の組み立ては train script 側で行い、
    ここでは学習・検証・保存の進行だけを扱う。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        root_chord_loss_fn: torch.nn.Module,
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
        # checkpoint に追加で埋めたい metadata は外側から受け取る。
        self.checkpoint_extras = dict(checkpoint_extras or {})
        self.writer = writer
        self.train_cfg = config.get("base_model_training", {})
        if not isinstance(self.train_cfg, dict):
            raise TypeError("Config section 'base_model_training' must be a mapping.")
        if not self.train_cfg:
            raise KeyError("Training configuration 'base_model_training' not found in config.")

        self.loss_cfg = self.train_cfg.get("loss", {}) or {}
        self.ce_ignore_index = int(self.loss_cfg.get("ce_ignore_index", -100))
        self.root_chord_loss_fn = root_chord_loss_fn

        self.global_step = 0
        self.start_epoch = 1
        self.validate_every = self.train_cfg.get("validate_every_n_epochs", 1)
        print(f"Validation will run every {self.validate_every} epochs.")

        self.accum_steps = max(1, int(self.train_cfg.get("grad_accum_steps", 1)))

        self.amp_enabled = self.train_cfg.get("amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)

        self.ema = ModelEma(self.model, decay=0.99, device=self.device)
        self.device_waveform_augmentation: Optional[StereoWaveformAugmentation] = None
        self.repeat_pair_builder: Optional[RepeatPairBuilderCPU] = None
        self.repeat_enable_from_epoch = 1
        self._last_label_head_detach: Optional[bool] = None
        self.num_bass_classes = int(self.config.get("model", {}).get("num_bass_classes", 13))

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
        self._build_repeat_modules()

    def _build_repeat_modules(self) -> None:
        repeat_cfg = self.train_cfg.get("repeat_ssm", {}) or {}
        if not bool(repeat_cfg.get("enabled", False)):
            return
        self.repeat_enable_from_epoch = int(repeat_cfg.get("enable_from_epoch", 1))
        if self.repeat_enable_from_epoch < 1:
            raise ValueError("repeat_ssm.enable_from_epoch must be >= 1")

        self.repeat_pair_builder = RepeatPairBuilderCPU(
            window_size=int(repeat_cfg.get("window_size", 4)),
            max_span_ratio=float(repeat_cfg.get("max_span_ratio", 1.5)),
            ignore_index=int(repeat_cfg.get("ignore_index", -100)),
        )
        self.repeat_pair_builder.eval()
        if self.repeat_enable_from_epoch > 1:
            print(f"Repeat SSM will be enabled from epoch {self.repeat_enable_from_epoch}.")

    def _is_repeat_ssm_active(self, epoch: Optional[int]) -> bool:
        if self.repeat_pair_builder is None:
            return False
        if epoch is None:
            return True
        return int(epoch) >= self.repeat_enable_from_epoch

    def _iter_label_head_detach_targets(self):
        seen = set()
        for module in (self.model, self.ema.module):
            current = module
            while current is not None:
                module_id = id(current)
                if module_id in seen:
                    break
                seen.add(module_id)
                if hasattr(current, "set_label_head_detach"):
                    yield current
                    break
                current = getattr(current, "base_model", None)

    def _update_label_head_detach_state(self, epoch: Optional[int]) -> None:
        detach_enabled = not self._is_repeat_ssm_active(epoch)
        if self._last_label_head_detach is not None and self._last_label_head_detach == detach_enabled:
            return
        for module in self._iter_label_head_detach_targets():
            module.set_label_head_detach(detach_enabled)
        self._last_label_head_detach = detach_enabled

    def _combine_repeat_chord_labels(
        self,
        root_chord_labels: torch.Tensor,
        bass_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.repeat_pair_builder is None:
            raise RuntimeError("repeat_pair_builder is not initialized")
        if root_chord_labels.shape != bass_labels.shape:
            raise ValueError("repeat root_chord labels and bass labels must have the same shape")

        ignore_index = self.repeat_pair_builder.ignore_index
        combined = root_chord_labels.new_full(root_chord_labels.shape, ignore_index)
        valid_mask = (root_chord_labels != ignore_index) & (bass_labels != ignore_index)
        if not bool(valid_mask.any().item()):
            return combined

        combined[valid_mask] = root_chord_labels[valid_mask] * self.num_bass_classes + bass_labels[valid_mask]
        return combined

    @torch.no_grad()
    def _compute_repeat_ssm_output(self, batch: Dict[str, torch.Tensor], epoch: Optional[int] = None):
        if not self._is_repeat_ssm_active(epoch):
            return None
        if "root_chord_index" not in batch or "bass_index" not in batch:
            return None

        repeat_chord_labels = self._combine_repeat_chord_labels(batch["root_chord_index"], batch["bass_index"])
        return self.repeat_pair_builder(repeat_chord_labels)

    def _attach_repeat_ssm_output(
        self,
        outputs: Dict[str, Any],
        repeat_ssm_output,
    ) -> Dict[str, Any]:
        if repeat_ssm_output is not None:
            outputs["repeat_ssm_output"] = repeat_ssm_output
        return outputs

    def _filter_state_dict(
        self,
        incoming_state: Dict[str, Any],
        target_state: Dict[str, Any],
        key_prefixes: Tuple[str, ...],
    ) -> Tuple[Dict[str, Any], int]:
        """init-from 用に、指定 prefix と shape が合う重みだけを抜き出す。"""
        filtered = {}
        skipped_count = 0
        for k, v in incoming_state.items():
            key_allowed = any(k.startswith(prefix) for prefix in key_prefixes)
            if key_allowed and k in target_state and target_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped_count += 1
        return filtered, skipped_count

    def train(self):
        """設定されたエポック数だけ学習ループを実行します。"""
        epochs = self.train_cfg["epochs"]
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")

            # 1. 通常学習
            train_log = self._train_epoch(epoch)

            torch.cuda.empty_cache()
            # 2. 必要なタイミングだけ検証
            if (epoch % self.validate_every == 0) or (epoch == epochs):
                valid_log = self._validate_epoch(epoch)
            else:
                valid_log = {}

            # 3. エポック終端で scheduler を更新
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

            # 5. checkpoint 保存
            if epoch % self.train_cfg.get("model_save_every_n_epochs", 1) == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポック分の学習処理。"""
        self.model.train()
        self._update_label_head_detach_state(epoch)
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
        batch_cutmix_cfg = self.train_cfg.get("batch_cutmix", {})
        sample_rate = self.config["data_loader"].get("sample_rate", 22050)
        hop_length = self.config["data_loader"].get("hop_length", 512)

        for batch_index, batch in enumerate(progress_bar, start=1):
            # バッチをデバイスに移動
            for key in batch:
                batch[key] = batch[key].to(self.device)

            # 0. device 上で waveform augmentation を掛ける。
            batch = apply_batch_waveform_augmentation(batch, self.device_waveform_augmentation)

            # 1. バッチ単位のタイムストレッチ (GPU上で長さを揃える)
            batch = apply_batch_time_stretch(batch)

            # 2. バッチ単位の時間CutMix拡張 (長さが揃ったテンソルに対して適用)
            batch = apply_batch_cutmix(batch, batch_cutmix_cfg, sample_rate, hop_length)
            repeat_ssm_output = self._compute_repeat_ssm_output(batch, epoch=epoch)

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch["audio"])
                outputs = self._attach_repeat_ssm_output(outputs, repeat_ssm_output)
                losses = compute_losses(
                    outputs,
                    batch,
                    self.loss_cfg,
                    root_chord_loss_fn=self.root_chord_loss_fn,
                )
                if losses:
                    supervised_total_loss = sum(losses.values())
                else:
                    supervised_total_loss = batch["audio"].sum() * 0.0

                usage = outputs.get("repeat/sec_type_usage")
                if usage is not None:
                    usage = usage.detach().cpu()
                    if usage_accum is None:
                        usage_accum = usage.clone()
                    else:
                        usage_accum += usage
                    usage_count += 1

            # accumulation 対応のため、ここでは backward だけ先に進める。
            supervised_loss_for_backward = supervised_total_loss / self.accum_steps
            if supervised_loss_for_backward.requires_grad:
                self.scaler.scale(supervised_loss_for_backward).backward()

            total_loss = supervised_total_loss.detach()

            for k, v in losses.items():
                running_losses.setdefault(k, 0.0)
                running_losses[k] += v.item()

            running_losses.setdefault("total", 0.0)
            running_losses["total"] += total_loss.item()

            micro_step += 1
            last_batch = batch_index == total_batches
            do_step = (micro_step % self.accum_steps == 0) or last_batch

            if do_step:
                # accumulation の区切りか最後の batch で optimizer を進める。
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
        self._update_label_head_detach_state(epoch)
        running_losses = {}
        running_ema_losses = {}
        running_metrics = {}
        running_ema = {}
        usage_accum: Optional[torch.Tensor] = None
        usage_count = 0

        for batch in self.valid_loader:
            for key in batch:
                batch[key] = batch[key].to(self.device)

            # 1. バッチ単位のタイムストレッチ (GPU上で長さを揃える)
            batch = apply_batch_time_stretch(batch)
            repeat_ssm_output = self._compute_repeat_ssm_output(batch, epoch=epoch)

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch["audio"])
                ema_outputs = self.ema.module(batch["audio"])
                outputs = self._attach_repeat_ssm_output(outputs, repeat_ssm_output)
                ema_outputs = self._attach_repeat_ssm_output(ema_outputs, repeat_ssm_output)

                # ロス計算
                losses = compute_losses(
                    outputs,
                    batch,
                    self.loss_cfg,
                    root_chord_loss_fn=self.root_chord_loss_fn,
                )
                for k, v in losses.items():
                    running_losses.setdefault(k, 0.0)
                    running_losses[k] += v.item()
                running_losses.setdefault("total", 0.0)
                running_losses["total"] += sum(v.item() for v in losses.values())

                ema_losses = compute_losses(
                    ema_outputs,
                    batch,
                    self.loss_cfg,
                    root_chord_loss_fn=self.root_chord_loss_fn,
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
        # 学習ループ自身が持つ状態だけを基本として保存し、
        # model config などの補助 metadata は外側から受け取ったものを足す。
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
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

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = True,
        resume_epoch: Optional[int] = None,
        load_prefixes: Optional[Tuple[str, ...]] = None,
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
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)

        # init-from のときは prefix に合う重みだけを読み込む。
        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise KeyError("checkpoint does not contain 'model_state_dict'")
        if load_prefixes is not None:
            model_state, skipped_count = self._filter_state_dict(
                model_state,
                self.model.state_dict(),
                key_prefixes=load_prefixes,
            )
            print(
                f"Loaded {len(model_state)} model tensors matching prefixes {load_prefixes}; skipped {skipped_count}."
            )
        self.model.load_state_dict(model_state, strict=strict)

        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is not None:
            if load_prefixes is not None:
                ema_state, skipped_count = self._filter_state_dict(
                    ema_state,
                    self.ema.module.state_dict(),
                    key_prefixes=load_prefixes,
                )
                print(
                    f"Loaded {len(ema_state)} EMA tensors matching prefixes {load_prefixes}; skipped {skipped_count}."
                )
            self.ema.module.load_state_dict(ema_state, strict=strict)

        # resume_epoch を指定したケースは「重みだけ使って新しい学習を始める」扱い。
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
        """scheduler 状態を読まない再開時に、epoch だけ手動で整える。"""
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
