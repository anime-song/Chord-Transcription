import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .transcribers import PredictionDecoder, TranscriptionPredictor


def save_events_tsv(events: List[Dict[str, Any]], output_path: Path, headers: List[str]) -> None:
    """
    イベントリストを指定されたパスにTSV形式で保存する。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        label_key = headers[2]
        for event in events:
            row = [
                f"{event.get('start_time', 0.0):.6f}",
                f"{event.get('end_time', 0.0):.6f}",
                str(event.get(label_key, "N/A")),
            ]
            file.write("\t".join(row) + "\n")
    print(f"\n[OK] 予測結果を保存しました: {output_path}")


def save_beat_events_tsv(events: List[Dict[str, Any]], output_path: Path) -> None:
    """beat イベントを 秒<TAB>拍番号 形式で保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        for event in events:
            row = [
                f"{event.get('time_sec', 0.0):.6f}",
                str(event.get("beat", 0)),
            ]
            file.write("\t".join(row) + "\n")
    print(f"\n[OK] beat 予測結果を保存しました: {output_path}")


def print_events_preview(events: List[Dict[str, Any]], event_type: str) -> None:
    """イベントリストの最初の数件をプレビュー表示する。"""
    if not events:
        print(f"[INFO] 表示できる有効な {event_type} イベントがありません。")
        return

    max_show = min(20, len(events))
    label_key = event_type.lower()

    print(f"\n--- {event_type} 予測プレビュー (先頭{max_show}件) ---")
    for index in range(max_show):
        event = events[index]
        start = event.get("start_time", 0.0)
        end = event.get("end_time", 0.0)
        label = event.get(label_key, "N/A")
        print(f"{start:8.3f}s - {end:8.3f}s | {label}")
    print("------------------------------------------")


def print_beat_events_preview(events: List[Dict[str, Any]]) -> None:
    """beat イベントの最初の数件をプレビュー表示する。"""
    if not events:
        print("[INFO] 表示できる有効な Beat イベントがありません。")
        return

    max_show = min(20, len(events))
    print(f"\n--- Beat 予測プレビュー (先頭{max_show}件) ---")
    for index in range(max_show):
        event = events[index]
        print(f"{event.get('time_sec', 0.0):8.3f}s | {event.get('beat', 0)}")
    print("------------------------------------------")


def upscale_time_axis(matrix: np.ndarray, target_time_bins: int = 128) -> np.ndarray:
    """
    時間軸方向を最近傍（繰り返し）で拡大する。
    ぼやけを防ぐため補間は行わず、指定サイズ以上になるまで繰り返す。
    """
    if matrix.size == 0 or matrix.shape[0] >= target_time_bins:
        return matrix

    repeat = int(math.ceil(target_time_bins / matrix.shape[0]))
    upscaled = np.repeat(matrix, repeat, axis=0)
    return upscaled[:target_time_bins]


def save_pitch_chroma_logits_image(pitch_chroma_scores: Any, output_path: Path, target_time_bins: int = 128) -> None:
    """
    pitch_chroma_scores (T, 12) をそのまま PNG で保存する。時間軸を最近傍で target_time_bins まで拡大する。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib が無いため pitch chroma 画像を保存できません: {exc}")
        return

    if pitch_chroma_scores is None:
        print("[WARN] pitch_chroma_scores が見つからないため画像を保存できません。")
        return

    logits_np = np.asarray(pitch_chroma_scores, dtype=np.float32)
    if logits_np.ndim != 2:
        print(f"[WARN] pitch_chroma_scores の想定外の形状です: {logits_np.shape}")
        return

    upsampled = upscale_time_axis(logits_np, target_time_bins=target_time_bins)
    image_data = upsampled.T

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6.0, image_data.shape[1] / 20), 4))
    plt.imshow(image_data, aspect="auto", origin="lower", interpolation="nearest", cmap="gray", vmin=0)
    plt.xlabel("Time (upsampled)")
    plt.ylabel("Pitch class")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n[OK] pitch chroma イメージを保存しました: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="checkpoint から音声のコード進行とキーを推論するスクリプト")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="学習済み checkpoint のローカルパス、または Hugging Face repo id。",
    )
    parser.add_argument("--audio", type=str, required=True, help="入力音声ファイルへのパス")
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Hugging Face repo / checkpoint directory 内で使う checkpoint ファイル名。",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hugging Face 上の branch / tag / commit。",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face キャッシュディレクトリ。省略時は既定値を使う。",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="ネットワークアクセスを行わず、ローカルキャッシュだけを使う。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (例: 'cpu', 'cuda')",
    )
    parser.add_argument("--output_dir", type=str, default="predictions", help="出力ファイルを保存するディレクトリ")
    parser.add_argument(
        "--decode",
        type=str,
        default="auto",
        choices=["auto", "none", "argmax", "hmm", "crf", "crf_pool"],
        help="フレーム index の復号方法。checkpoint 種別に応じて auto が推奨。",
    )
    parser.add_argument(
        "--min_duration_chord",
        type=float,
        default=0.1,
        help="コードイベントの最小継続時間(秒)。これより短いイベントは前のイベントに結合されます。",
    )
    parser.add_argument(
        "--min_duration_key",
        type=float,
        default=2.0,
        help="キーイベントの最小継続時間(秒)。これより短いイベントは前のイベントに結合されます。",
    )
    parser.add_argument(
        "--no_romanize",
        action="store_true",
        help="chord-romanizer による臨時記号の補正を行わない。",
    )
    args = parser.parse_args()

    try:
        # checkpoint だけで predictor / decoder を組み立てる。
        predictor = TranscriptionPredictor.from_pretrained(
            args.checkpoint,
            device=args.device,
            filename=args.filename,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        decoder = PredictionDecoder.from_metadata(predictor.metadata)

        # まずはフレーム単位の model 出力を取る。
        prediction = predictor.predict_file(args.audio, decode_mode=args.decode)

        # 文字列ラベル化とイベント化は後段に分離する。
        frames = decoder.decode_frames(prediction)
        events = decoder.to_events(
            frames,
            min_duration_chord=args.min_duration_chord,
            min_duration_key=args.min_duration_key,
            romanize=not args.no_romanize,
        )

        if prediction.time_sec.size == 0:
            print("[ERROR] 予測を生成できませんでした。")
            return

        audio_path = Path(args.audio)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prediction.pitch_chroma_scores is not None:
            chroma_img_path = output_dir / f"{audio_path.stem}_pitch_chroma.png"
            save_pitch_chroma_logits_image(prediction.pitch_chroma_scores, chroma_img_path, target_time_bins=128)

        if events.chord_events:
            print_events_preview(events.chord_events, "Chord")
            chord_output_path = output_dir / f"{audio_path.stem}.chords.txt"
            save_events_tsv(events.chord_events, chord_output_path, ["start_time", "end_time", "chord"])

        if events.key_events:
            print_events_preview(events.key_events, "Key")
            key_output_path = output_dir / f"{audio_path.stem}.key.txt"
            save_events_tsv(events.key_events, key_output_path, ["start_time", "end_time", "key"])

        if events.beat_events:
            print_beat_events_preview(events.beat_events)
            beat_output_path = output_dir / f"{audio_path.stem}.beats.txt"
            save_beat_events_tsv(events.beat_events, beat_output_path)

    except (FileNotFoundError, KeyError, ImportError, AttributeError, ValueError) as exc:
        print(f"\n[エラー] 処理中にエラーが発生しました: {exc}")
        print("チェックポイントや推論 API の内容が正しいか確認してください。")
    except Exception as exc:
        print(f"\n[予期せぬエラー] 予期せぬエラーが発生しました: {exc}")


if __name__ == "__main__":
    main()
