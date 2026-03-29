import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F


class StereoWaveformAugmentation:
    """
    2ch waveform に掛ける軽量なデータ拡張。
    stem 単位でも、mix 後の waveform に対しても使える。
    入力は常に (2, T) の float waveform を想定する。
    """

    def __init__(self, sample_rate: int, config: Dict[str, Any]):
        self.sample_rate = int(sample_rate)
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.eq_cfg = dict(self.config.get("eq_tilt", {}))
        self.saturation_cfg = dict(self.config.get("saturation", {}))
        self.noise_cfg = dict(self.config.get("colored_noise", {}))
        self.compression_cfg = dict(self.config.get("compression", {}))
        self.stereo_cfg = dict(self.config.get("stereo", {}))
        self._eps = 1e-8
        self._freq_cache_np: Dict[int, np.ndarray] = {}
        self._freq_cache_torch: Dict[tuple[int, str], torch.Tensor] = {}

    def summary(self) -> str:
        return (
            "Waveform augmentation enabled: "
            f"EQ(p={self.eq_cfg.get('p', 0.0)}), "
            f"Saturation(p={self.saturation_cfg.get('p', 0.0)}), "
            f"Noise(p={self.noise_cfg.get('p', 0.0)}), "
            f"Compression(p={self.compression_cfg.get('p', 0.0)}), "
            f"Mono(p={self.stereo_cfg.get('mono_p', 0.0)}), "
            f"Balance(p={self.stereo_cfg.get('balance_p', 0.0)})"
        )

    def __call__(self, audio):
        if torch.is_tensor(audio):
            return self._call_torch(audio)
        return self._call_numpy(audio)

    def _call_numpy(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.asarray(audio, dtype=np.float32)

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError(f"StereoWaveformAugmentation expects audio shape (2, T), got {waveform.shape}")

        waveform = np.array(waveform, copy=True)

        if random.random() < float(self.eq_cfg.get("p", 0.0)):
            waveform = self._apply_eq_tilt(waveform)
        if random.random() < float(self.saturation_cfg.get("p", 0.0)):
            waveform = self._apply_saturation(waveform)
        if random.random() < float(self.noise_cfg.get("p", 0.0)):
            waveform = self._apply_colored_noise(waveform)
        if random.random() < float(self.compression_cfg.get("p", 0.0)):
            waveform = self._apply_compression(waveform)

        mono_p = float(self.stereo_cfg.get("mono_p", 0.0))
        balance_p = float(self.stereo_cfg.get("balance_p", 0.0))
        if mono_p > 0.0 and random.random() < mono_p:
            mono = np.mean(waveform, axis=0, keepdims=True, dtype=np.float32)
            waveform = np.repeat(mono, 2, axis=0)
        elif balance_p > 0.0 and random.random() < balance_p:
            max_balance_db = float(self.stereo_cfg.get("max_balance_db", 3.0))
            balance_db = random.uniform(-max_balance_db, max_balance_db)
            left_gain = 10.0 ** (-balance_db / 20.0)
            right_gain = 10.0 ** (balance_db / 20.0)
            gain_norm = np.sqrt((left_gain * left_gain + right_gain * right_gain) / 2.0)
            waveform[0] *= left_gain / max(gain_norm, self._eps)
            waveform[1] *= right_gain / max(gain_norm, self._eps)

        return np.asarray(waveform, dtype=np.float32)

    def _call_torch(self, audio: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return audio.to(dtype=torch.float32)

        waveform = audio.to(dtype=torch.float32)
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError(f"StereoWaveformAugmentation expects audio shape (2, T), got {tuple(waveform.shape)}")

        waveform = waveform.clone()

        if random.random() < float(self.eq_cfg.get("p", 0.0)):
            waveform = self._apply_eq_tilt_torch(waveform)
        if random.random() < float(self.saturation_cfg.get("p", 0.0)):
            waveform = self._apply_saturation_torch(waveform)
        if random.random() < float(self.noise_cfg.get("p", 0.0)):
            waveform = self._apply_colored_noise_torch(waveform)
        if random.random() < float(self.compression_cfg.get("p", 0.0)):
            waveform = self._apply_compression_torch(waveform)

        mono_p = float(self.stereo_cfg.get("mono_p", 0.0))
        balance_p = float(self.stereo_cfg.get("balance_p", 0.0))
        if mono_p > 0.0 and random.random() < mono_p:
            mono = torch.mean(waveform, dim=0, keepdim=True)
            waveform = mono.repeat(2, 1)
        elif balance_p > 0.0 and random.random() < balance_p:
            max_balance_db = float(self.stereo_cfg.get("max_balance_db", 3.0))
            balance_db = random.uniform(-max_balance_db, max_balance_db)
            left_gain = 10.0 ** (-balance_db / 20.0)
            right_gain = 10.0 ** (balance_db / 20.0)
            gain_norm = np.sqrt((left_gain * left_gain + right_gain * right_gain) / 2.0)
            waveform[0] *= left_gain / max(gain_norm, self._eps)
            waveform[1] *= right_gain / max(gain_norm, self._eps)

        return waveform.to(dtype=torch.float32)

    def _get_rfft_freqs(self, num_samples: int) -> np.ndarray:
        freqs = self._freq_cache_np.get(num_samples)
        if freqs is None:
            freqs = np.fft.rfftfreq(num_samples, d=1.0 / float(self.sample_rate)).astype(np.float32)
            self._freq_cache_np[num_samples] = freqs
        return freqs

    def _get_rfft_freqs_torch(self, num_samples: int, device: torch.device) -> torch.Tensor:
        cache_key = (num_samples, str(device))
        freqs = self._freq_cache_torch.get(cache_key)
        if freqs is None:
            freqs = torch.fft.rfftfreq(
                num_samples,
                d=1.0 / float(self.sample_rate),
                device=device,
                dtype=torch.float32,
            )
            self._freq_cache_torch[cache_key] = freqs
        return freqs

    def _apply_eq_tilt(self, waveform: np.ndarray) -> np.ndarray:
        slope_db_per_octave = random.uniform(
            float(self.eq_cfg.get("min_db_per_octave", -3.0)),
            float(self.eq_cfg.get("max_db_per_octave", 3.0)),
        )
        pivot_hz = max(float(self.eq_cfg.get("pivot_hz", 700.0)), 20.0)
        freqs = self._get_rfft_freqs(waveform.shape[-1])

        octave_offset = np.log2(np.maximum(freqs, 1.0) / pivot_hz)
        gain_db = slope_db_per_octave * octave_offset
        gain = np.power(10.0, gain_db / 20.0).astype(np.float32, copy=False)
        gain[0] = 1.0

        spec = np.fft.rfft(waveform, axis=-1)
        spec *= gain[None, :]
        return np.fft.irfft(spec, n=waveform.shape[-1], axis=-1).astype(np.float32, copy=False)

    def _apply_eq_tilt_torch(self, waveform: torch.Tensor) -> torch.Tensor:
        slope_db_per_octave = random.uniform(
            float(self.eq_cfg.get("min_db_per_octave", -3.0)),
            float(self.eq_cfg.get("max_db_per_octave", 3.0)),
        )
        pivot_hz = max(float(self.eq_cfg.get("pivot_hz", 700.0)), 20.0)
        freqs = self._get_rfft_freqs_torch(waveform.shape[-1], waveform.device)

        octave_offset = torch.log2(torch.clamp(freqs, min=1.0) / pivot_hz)
        gain_db = slope_db_per_octave * octave_offset
        gain = torch.pow(10.0, gain_db / 20.0).to(dtype=torch.float32)
        gain[0] = 1.0

        spec = torch.fft.rfft(waveform, dim=-1)
        spec *= gain.unsqueeze(0)
        return torch.fft.irfft(spec, n=waveform.shape[-1], dim=-1).to(dtype=torch.float32)

    def _apply_saturation(self, waveform: np.ndarray) -> np.ndarray:
        drive_db = random.uniform(
            float(self.saturation_cfg.get("min_drive_db", 1.0)),
            float(self.saturation_cfg.get("max_drive_db", 6.0)),
        )
        wet = random.uniform(
            float(self.saturation_cfg.get("min_wet", 0.6)),
            float(self.saturation_cfg.get("max_wet", 1.0)),
        )
        drive = 10.0 ** (drive_db / 20.0)
        saturated = np.tanh(waveform * drive) / max(np.tanh(drive), self._eps)
        return ((1.0 - wet) * waveform + wet * saturated).astype(np.float32, copy=False)

    def _apply_saturation_torch(self, waveform: torch.Tensor) -> torch.Tensor:
        drive_db = random.uniform(
            float(self.saturation_cfg.get("min_drive_db", 1.0)),
            float(self.saturation_cfg.get("max_drive_db", 6.0)),
        )
        wet = random.uniform(
            float(self.saturation_cfg.get("min_wet", 0.6)),
            float(self.saturation_cfg.get("max_wet", 1.0)),
        )
        drive = 10.0 ** (drive_db / 20.0)
        saturated = torch.tanh(waveform * drive) / max(np.tanh(drive), self._eps)
        return ((1.0 - wet) * waveform + wet * saturated).to(dtype=torch.float32)

    def _apply_colored_noise(self, waveform: np.ndarray) -> np.ndarray:
        noise_types = self.noise_cfg.get("types", ["white", "pink", "brown"])
        noise_type = str(random.choice(noise_types))
        noise = np.random.randn(*waveform.shape).astype(np.float32)

        if noise_type != "white":
            freqs = self._get_rfft_freqs(waveform.shape[-1])
            spec = np.fft.rfft(noise, axis=-1)
            if noise_type == "pink":
                scale = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
            elif noise_type == "brown":
                scale = 1.0 / np.maximum(freqs, 1.0)
            else:
                scale = np.ones_like(freqs, dtype=np.float32)
            scale[0] = 0.0
            spec *= scale[None, :]
            noise = np.fft.irfft(spec, n=waveform.shape[-1], axis=-1).astype(np.float32, copy=False)

        signal_rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
        if signal_rms < self._eps or noise_rms < self._eps:
            return waveform

        snr_db = random.uniform(
            float(self.noise_cfg.get("min_snr_db", 20.0)),
            float(self.noise_cfg.get("max_snr_db", 35.0)),
        )
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise *= target_noise_rms / max(noise_rms, self._eps)
        return (waveform + noise).astype(np.float32, copy=False)

    def _apply_colored_noise_torch(self, waveform: torch.Tensor) -> torch.Tensor:
        noise_types = self.noise_cfg.get("types", ["white", "pink", "brown"])
        noise_type = str(random.choice(noise_types))
        noise = torch.randn_like(waveform, dtype=torch.float32)

        if noise_type != "white":
            freqs = self._get_rfft_freqs_torch(waveform.shape[-1], waveform.device)
            spec = torch.fft.rfft(noise, dim=-1)
            if noise_type == "pink":
                scale = torch.rsqrt(torch.clamp(freqs, min=1.0))
            elif noise_type == "brown":
                scale = torch.reciprocal(torch.clamp(freqs, min=1.0))
            else:
                scale = torch.ones_like(freqs, dtype=torch.float32)
            scale[0] = 0.0
            spec *= scale.unsqueeze(0)
            noise = torch.fft.irfft(spec, n=waveform.shape[-1], dim=-1).to(dtype=torch.float32)

        signal_rms = torch.sqrt(torch.mean(torch.square(waveform)) + self._eps)
        noise_rms = torch.sqrt(torch.mean(torch.square(noise)) + self._eps)
        if float(signal_rms.item()) < self._eps or float(noise_rms.item()) < self._eps:
            return waveform

        snr_db = random.uniform(
            float(self.noise_cfg.get("min_snr_db", 20.0)),
            float(self.noise_cfg.get("max_snr_db", 35.0)),
        )
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / torch.clamp(noise_rms, min=self._eps))
        return (waveform + noise).to(dtype=torch.float32)

    def _apply_compression(self, waveform: np.ndarray) -> np.ndarray:
        threshold_db = random.uniform(
            float(self.compression_cfg.get("min_threshold_db", -24.0)),
            float(self.compression_cfg.get("max_threshold_db", -12.0)),
        )
        ratio = random.uniform(
            float(self.compression_cfg.get("min_ratio", 1.25)),
            float(self.compression_cfg.get("max_ratio", 2.5)),
        )
        frame_ms = float(self.compression_cfg.get("frame_ms", 20.0))
        makeup_db = random.uniform(
            float(self.compression_cfg.get("min_makeup_db", 0.0)),
            float(self.compression_cfg.get("max_makeup_db", 1.5)),
        )

        frame_len = max(64, int(self.sample_rate * frame_ms / 1000.0))
        num_samples = waveform.shape[-1]
        padded_len = int(np.ceil(num_samples / frame_len) * frame_len)
        if padded_len != num_samples:
            padded = np.pad(waveform, ((0, 0), (0, padded_len - num_samples)))
        else:
            padded = waveform

        framed = padded.reshape(2, -1, frame_len)
        frame_rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=(0, 2)) + self._eps)
        frame_db = 20.0 * np.log10(frame_rms + self._eps)
        over_db = np.maximum(frame_db - threshold_db, 0.0)
        gain_db = -over_db * (1.0 - 1.0 / max(ratio, 1.0))

        if gain_db.size >= 3:
            gain_db = np.convolve(gain_db, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")

        gain_db = gain_db + makeup_db
        frame_gain = np.power(10.0, gain_db / 20.0).astype(np.float32, copy=False)
        sample_gain = np.repeat(frame_gain, frame_len)[:num_samples]
        return (waveform * sample_gain[None, :]).astype(np.float32, copy=False)

    def _apply_compression_torch(self, waveform: torch.Tensor) -> torch.Tensor:
        threshold_db = random.uniform(
            float(self.compression_cfg.get("min_threshold_db", -24.0)),
            float(self.compression_cfg.get("max_threshold_db", -12.0)),
        )
        ratio = random.uniform(
            float(self.compression_cfg.get("min_ratio", 1.25)),
            float(self.compression_cfg.get("max_ratio", 2.5)),
        )
        frame_ms = float(self.compression_cfg.get("frame_ms", 20.0))
        makeup_db = random.uniform(
            float(self.compression_cfg.get("min_makeup_db", 0.0)),
            float(self.compression_cfg.get("max_makeup_db", 1.5)),
        )

        frame_len = max(64, int(self.sample_rate * frame_ms / 1000.0))
        num_samples = waveform.shape[-1]
        padded_len = int(np.ceil(num_samples / frame_len) * frame_len)
        if padded_len != num_samples:
            padded = F.pad(waveform, (0, padded_len - num_samples))
        else:
            padded = waveform

        framed = padded.reshape(2, -1, frame_len)
        frame_rms = torch.sqrt(torch.mean(torch.square(framed), dim=(0, 2)) + self._eps)
        frame_db = 20.0 * torch.log10(frame_rms + self._eps)
        over_db = torch.clamp(frame_db - threshold_db, min=0.0)
        gain_db = -over_db * (1.0 - 1.0 / max(ratio, 1.0))

        if gain_db.numel() >= 3:
            kernel = torch.tensor([0.25, 0.5, 0.25], device=waveform.device, dtype=torch.float32).view(1, 1, 3)
            gain_db = F.conv1d(gain_db.view(1, 1, -1), kernel, padding=1).view(-1)

        gain_db = gain_db + makeup_db
        frame_gain = torch.pow(10.0, gain_db / 20.0).to(dtype=torch.float32)
        sample_gain = frame_gain.repeat_interleave(frame_len)[:num_samples]
        return (waveform * sample_gain.unsqueeze(0)).to(dtype=torch.float32)
