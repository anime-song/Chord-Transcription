from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Literal

import torch

from .chord_transcription.models.factory import (
    load_label_vocab_from_checkpoint,
    load_model_build_config_from_checkpoint,
)
from .chord_transcription.transcribers import TranscriptionPredictor


StateChoice = Literal["auto", "ema", "model"]


def _default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".pt"
    return input_path.with_name(f"{input_path.stem}_public{suffix}")


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _move_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _move_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(value) for value in obj)
    return obj


def _resolve_state_key(checkpoint: Dict[str, Any], preferred: StateChoice) -> str:
    if preferred == "auto":
        if checkpoint.get("ema_state_dict") is not None:
            return "ema_state_dict"
        if checkpoint.get("model_state_dict") is not None:
            return "model_state_dict"
    elif preferred == "ema":
        if checkpoint.get("ema_state_dict") is not None:
            return "ema_state_dict"
        raise KeyError("Checkpoint does not contain 'ema_state_dict'.")
    elif preferred == "model":
        if checkpoint.get("model_state_dict") is not None:
            return "model_state_dict"
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    else:
        raise ValueError(f"Unsupported state choice: {preferred}")

    raise KeyError("Checkpoint must contain 'ema_state_dict' or 'model_state_dict'.")


def _build_public_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    model_build_config: Dict[str, Any],
    label_vocab: Dict[str, Any],
    state_key: str,
    keep_epoch: bool,
) -> Dict[str, Any]:
    exported = {
        "model_build_config": model_build_config,
        "label_vocab": label_vocab,
        state_key: _move_to_cpu(checkpoint[state_key]),
    }
    if keep_epoch and "epoch" in checkpoint:
        exported["epoch"] = int(checkpoint["epoch"])
    return exported


def _verify_checkpoint(checkpoint_path: Path) -> None:
    predictor = TranscriptionPredictor.from_checkpoint(checkpoint_path, device="cpu")
    del predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an inference-only checkpoint with just weights and metadata."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the source training checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the exported checkpoint. Defaults to '<input>_public.pt'.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="auto",
        choices=["auto", "ema", "model"],
        help="Which weight set to keep. 'auto' prefers EMA when present.",
    )
    parser.add_argument(
        "--keep-epoch",
        action="store_true",
        help="Keep the source epoch field in the exported checkpoint.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load the exported checkpoint with the predictor after saving.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite it.")

    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    model_build_config = load_model_build_config_from_checkpoint(input_path, map_location="cpu")
    if model_build_config is None:
        raise KeyError("Checkpoint must contain 'model_build_config' or a valid sidecar config.")
    label_vocab = load_label_vocab_from_checkpoint(checkpoint)
    state_key = _resolve_state_key(checkpoint, args.state)

    exported = _build_public_checkpoint(
        checkpoint,
        model_build_config=model_build_config,
        label_vocab=label_vocab,
        state_key=state_key,
        keep_epoch=bool(args.keep_epoch),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)

    removed_keys = sorted(set(checkpoint.keys()) - set(exported.keys()))
    source_size = input_path.stat().st_size
    output_size = output_path.stat().st_size

    print(f"Saved public checkpoint to: {output_path}")
    print(f"Retained keys: {sorted(exported.keys())}")
    print(f"Removed keys: {removed_keys}")
    print(f"Selected weight set: {state_key}")
    print(f"Size: {_format_bytes(source_size)} -> {_format_bytes(output_size)}")

    if args.verify:
        print("Verifying exported checkpoint by constructing a predictor...")
        _verify_checkpoint(output_path)
        print("Verification passed.")


if __name__ == "__main__":
    main()
