import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import random
import math


def apply_batch_cutmix(batch, config, sample_rate, hop_length):
    """
    バッチ内の波形とラベルを「コードチェンジタイミング」でクロスフェード合成する（Batch Temporal CutMix）。
    A（現在のバッチ要素）の一部を、B（次のバッチ要素、循環）の一部で置き換える。
    """
    if not config.get("enabled", False):
        return batch

    p = config.get("p", 0.5)
    fade_seconds = config.get("fade_seconds", 0.05)
    fade_samples = int(fade_seconds * sample_rate)

    # window for crossfade
    device = batch["audio"].device
    if fade_samples > 0:
        window_out = torch.linspace(1.0, 0.0, fade_samples, device=device)
        window_in = torch.linspace(0.0, 1.0, fade_samples, device=device)
    else:
        window_out, window_in = None, None

    B = batch["audio"].shape[0]
    repeat_loss_mask = torch.ones((B,), dtype=torch.bool, device=device)
    existing_repeat_loss_mask = batch.get("repeat_loss_mask")
    if existing_repeat_loss_mask is not None:
        repeat_loss_mask &= existing_repeat_loss_mask.bool()

    # 対象キーをリストアップ
    audio_key = "audio"
    frame_keys = [
        "root_index",
        "bass_index",
        "quality_index",
        "key_index",
        "chord25",
        "boundary",
        "beat",
        "downbeat",
        "key_boundary",
    ]

    # 既存のものを書き換えるためクローンはしない（inplaceでも可）または新しいディクショナリを作る
    # inplaceで更新するように設定
    mixed_audio = batch[audio_key].clone()

    mixed_labels = {}
    for k in frame_keys:
        if k in batch:
            mixed_labels[k] = batch[k].clone()

    for idx_A in range(B):
        if random.random() >= p:
            continue

        idx_B = (idx_A + 1) % B  # 相手となるバッチインデックス

        # Aの境界（コードチェンジ）フレームを探す
        boundary_A = batch["boundary"][idx_A].squeeze(-1)  # (T,)
        change_frames_A = torch.where(boundary_A == 1.0)[0]

        if len(change_frames_A) == 0:
            continue

        # ランダムにカット位置を選ぶ
        t_cut_frame = change_frames_A[random.randint(0, len(change_frames_A) - 1)].item()
        repeat_loss_mask[idx_A] = False

        # フレームインデックスをオーディオサンプルインデックスに変換
        # hop_length * frame_index
        t_cut_sample = t_cut_frame * hop_length

        max_samples = mixed_audio.shape[-1]
        t_cut_sample = min(t_cut_sample, max_samples)

        # カット位置が早すぎる/遅すぎる場合はスキップするか、フェード長を調整する
        half_fade = fade_samples // 2
        start_fade = t_cut_sample - half_fade
        end_fade = t_cut_sample + (fade_samples - half_fade)

        if start_fade < 0 or end_fade > max_samples:
            # 簡略化のため端すぎる場合は境界でのハードカットにするかスキップ
            start_fade = t_cut_sample
            end_fade = t_cut_sample
            actual_fade_samples = 0
        else:
            actual_fade_samples = fade_samples

        # オーディオのクロスフェード合成
        # 1. 0 ~ start_fade: A
        # 2. start_fade ~ end_fade: A * win_out + B * win_in
        # 3. end_fade ~ end: B

        if actual_fade_samples > 0:
            mixed_audio[idx_A, :, start_fade:end_fade] = (
                batch["audio"][idx_A, :, start_fade:end_fade] * window_out
                + batch["audio"][idx_B, :, start_fade:end_fade] * window_in
            )

        if end_fade < max_samples:
            mixed_audio[idx_A, :, end_fade:] = batch["audio"][idx_B, :, end_fade:]

        # ラベルの合成 (フレーム単位)
        # T_cut_frame 以降は完全に B のラベルを使用
        for k in frame_keys:
            if k in mixed_labels:
                T_max = mixed_labels[k].shape[1]
                if t_cut_frame < T_max:
                    mixed_labels[k][idx_A, t_cut_frame:] = batch[k][idx_B, t_cut_frame:]

        # T_cut_frame の位置に強引に転調フラグ(key_boundary = 1.0)を立てる
        if "key_boundary" in mixed_labels:
            T_max = mixed_labels["key_boundary"].shape[1]
            if t_cut_frame < T_max:
                mixed_labels["key_boundary"][idx_A, t_cut_frame] = 1.0

    # バッチを更新
    batch[audio_key] = mixed_audio
    batch["repeat_loss_mask"] = repeat_loss_mask
    for k in frame_keys:
        if k in mixed_labels:
            batch[k] = mixed_labels[k]

    return batch


@torch.no_grad()
def apply_batch_waveform_augmentation(batch: dict, augmentation) -> dict:
    """
    trainer 側で device に載った mix waveform に対して augmentation を掛ける。
    valid region のみを処理し、padding 領域は 0 のまま維持する。
    """
    if augmentation is None:
        return batch

    waveform = batch.get("audio")
    target_samples = batch.get("target_samples")
    if waveform is None or target_samples is None:
        return batch

    augmented = waveform.clone()
    batch_size = int(augmented.shape[0])
    total_samples = int(augmented.shape[-1])
    is_batched_samples = torch.is_tensor(target_samples) and target_samples.dim() > 0

    for b in range(batch_size):
        if is_batched_samples:
            valid_samples = int(target_samples[b].item())
        elif torch.is_tensor(target_samples):
            valid_samples = int(target_samples.item())
        else:
            valid_samples = int(target_samples)

        valid_samples = max(0, min(valid_samples, total_samples))
        if valid_samples == 0:
            augmented[b].zero_()
            continue

        augmented[b, :, :valid_samples] = augmentation(augmented[b, :, :valid_samples])
        if valid_samples < total_samples:
            augmented[b, :, valid_samples:] = 0

    batch["audio"] = augmented
    return batch


@torch.no_grad()
def time_stretch_waveform(x: torch.Tensor, rate: float, n_fft=256, hop_length=128, win_length=256) -> torch.Tensor:
    """
    x: (B, C, T) または (C, T) のCUDAテンソル想定
    rate: >1 で速く(短く), <1 で遅く(長く)
    """
    device, dtype = x.device, x.dtype
    # Batch次元の考慮: (C, T) -> (1, C, T)
    is_unbatched = x.dim() == 2
    if is_unbatched:
        x = x.unsqueeze(0)

    B, C, T = x.shape
    x_2d = x.reshape(B * C, T)

    window = torch.hann_window(win_length, device=device, dtype=dtype)

    # (B*C, F, Frames) complex
    spec = torch.stft(
        x_2d, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True
    )

    # docsの例の通り phase_advance を作る
    freq = spec.size(-2)
    phase_advance = torch.linspace(0, math.pi * hop_length, freq, device=device, dtype=dtype)[..., None]

    spec_st = AF.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)

    y_2d = torch.istft(spec_st, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    y = y_2d.reshape(B, C, -1)

    if is_unbatched:
        y = y.squeeze(0)
    return y


def apply_batch_time_stretch(batch: dict) -> dict:
    """
    バッチ内の各オーディオに対して個別の time_stretch_rate でタイムストレッチを適用し、
    目標の長さに揃えた新しいバッチを返します。
    """
    time_stretch_rate = batch.get("time_stretch_rate")
    target_samples = batch.get("target_samples")
    waveform = batch.get("audio")

    if time_stretch_rate is None or target_samples is None or waveform is None:
        return batch

    is_batched_rates = time_stretch_rate.dim() > 0
    is_batched_samples = target_samples.dim() > 0

    fixed_target_len = (
        int(target_samples[0] / time_stretch_rate[0]) if is_batched_rates else int(target_samples / time_stretch_rate)
    )

    def _stretch_batch_tensor(x: torch.Tensor) -> torch.Tensor:
        processed_waves = []
        batch_size = x.shape[0]
        for b in range(batch_size):
            rate = float(time_stretch_rate[b]) if is_batched_rates else float(time_stretch_rate)
            in_samples = int(target_samples[b]) if is_batched_samples else int(target_samples)

            wave_b = x[b, ..., :in_samples]
            original_shape = wave_b.shape[:-1]
            wave_b = wave_b.reshape(-1, wave_b.shape[-1])

            if abs(rate - 1.0) > 1e-5:
                wave_b = time_stretch_waveform(wave_b, rate=rate)

            wave_b = wave_b.reshape(*original_shape, -1)
            if wave_b.shape[-1] < fixed_target_len:
                wave_b = F.pad(wave_b, (0, fixed_target_len - wave_b.shape[-1]))
            elif wave_b.shape[-1] > fixed_target_len:
                wave_b = wave_b[..., :fixed_target_len]
            processed_waves.append(wave_b)
        return torch.stack(processed_waves, dim=0)

    batch["audio"] = _stretch_batch_tensor(waveform)
    return batch
