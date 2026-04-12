from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


CHECKPOINT_EXTENSIONS: tuple[str, ...] = (".pt", ".pth", ".ckpt", ".bin")
DEFAULT_CHECKPOINT_FILENAMES: tuple[str, ...] = (
    "model.pt",
    "checkpoint.pt",
    "pytorch_model.bin",
)


def _is_checkpoint_file(path: Path) -> bool:
    return (
        path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS and not path.name.endswith(".model_config.json")
    )


def _extract_epoch_from_name(path: Path) -> int:
    matches = re.findall(r"(?:^|[_-])epoch[_-]?(\d+)(?:$|[_-])", path.stem.lower())
    if not matches:
        return -1
    return max(int(value) for value in matches)


def _checkpoint_preference_key(path: Path) -> tuple[int, int, int, str]:
    stem_lower = path.stem.lower()
    is_public = int(stem_lower.endswith("_public") or stem_lower == "public")
    has_epoch = int(_extract_epoch_from_name(path) >= 0)
    epoch = _extract_epoch_from_name(path)
    return (
        is_public,
        has_epoch,
        epoch,
        path.name.lower(),
    )


def _select_preferred_checkpoint(candidates: list[Path], directory: Path) -> Path:
    ranked = sorted(candidates, key=_checkpoint_preference_key, reverse=True)
    best = ranked[0]
    best_key = _checkpoint_preference_key(best)
    tied = [candidate for candidate in ranked if _checkpoint_preference_key(candidate) == best_key]
    if len(tied) == 1:
        return best

    candidate_list = ", ".join(path.relative_to(directory).as_posix() for path in tied[:8])
    raise FileNotFoundError(
        "Multiple checkpoint files matched the default selection rule. "
        f"Specify `filename=` explicitly. Tied candidates under {directory}: {candidate_list}"
    )


def _resolve_checkpoint_in_directory(
    directory: Path,
    *,
    filename: Optional[str] = None,
) -> Path:
    directory = directory.expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {directory}")

    if filename is not None:
        candidate = (directory / filename).expanduser().resolve()
        if not _is_checkpoint_file(candidate):
            raise FileNotFoundError(f"Checkpoint file not found: {candidate}")
        return candidate

    for default_name in DEFAULT_CHECKPOINT_FILENAMES:
        candidate = directory / default_name
        if _is_checkpoint_file(candidate):
            return candidate.resolve()

    candidates = sorted(path.resolve() for path in directory.rglob("*") if _is_checkpoint_file(path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under: {directory}")
    if len(candidates) == 1:
        return candidates[0]
    return _select_preferred_checkpoint(candidates, directory)


def _looks_like_hugging_face_repo_id(source: str) -> bool:
    if not source:
        return False
    if source.startswith((".", "/", "~")):
        return False
    if "\\" in source:
        return False
    if Path(source).suffix.lower() in CHECKPOINT_EXTENSIONS:
        return False
    return source.count("/") == 1


def _download_from_hugging_face(
    repo_id: str,
    *,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Loading from Hugging Face requires `huggingface-hub`. "
            "Install the base package dependencies or add `huggingface-hub`."
        ) from exc

    cache_dir_str = str(Path(cache_dir).expanduser()) if cache_dir is not None else None

    if filename is not None:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir_str,
            local_files_only=local_files_only,
            token=token,
        )
        return Path(downloaded_path).expanduser().resolve()

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir_str,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=[
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.bin",
            "*.model_config.json",
        ],
    )
    return _resolve_checkpoint_in_directory(Path(snapshot_path), filename=None)


def resolve_pretrained_checkpoint_path(
    pretrained_model_name_or_path: str | Path,
    *,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
) -> Path:
    """
    ローカル path / directory、または Hugging Face repo id から checkpoint path を解決する。
    """
    if isinstance(pretrained_model_name_or_path, Path):
        source = str(pretrained_model_name_or_path.expanduser())
    else:
        source = str(pretrained_model_name_or_path).strip()

    if not source:
        raise ValueError("pretrained_model_name_or_path must be a non-empty string or path.")

    local_candidate = Path(source).expanduser()
    if local_candidate.exists():
        if local_candidate.is_file():
            if not _is_checkpoint_file(local_candidate):
                raise FileNotFoundError(f"Checkpoint file not found: {local_candidate}")
            return local_candidate.resolve()
        return _resolve_checkpoint_in_directory(local_candidate, filename=filename)

    if _looks_like_hugging_face_repo_id(source):
        return _download_from_hugging_face(
            source,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )

    if filename is not None:
        missing_candidate = (local_candidate / filename).expanduser()
        raise FileNotFoundError(f"Checkpoint file not found: {missing_candidate}")
    raise FileNotFoundError(f"Checkpoint path not found: {local_candidate}")


__all__ = [
    "CHECKPOINT_EXTENSIONS",
    "DEFAULT_CHECKPOINT_FILENAMES",
    "resolve_pretrained_checkpoint_path",
]
