import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .transcribers.base_transcriber import AudioTranscriber, load_config


def process_predictions_to_events(
    predictions: Dict[str, Any],
    min_duration_chord: float = 0.1,
    min_duration_key: float = 0.5,
    min_duration_tempo: float = 2.0,
    tempo_change_ratio: float = 0.05,
    tempo_change_bpm: float = 4.0,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    フレームごとの予測結果を、コード・キー・テンポのイベントに変換する。
    """
    # フレームごとの予測データを抽出
    frame_times = predictions.get("time_sec")
    root_chords = predictions.get("root_chord", [])
    basses = predictions.get("bass", [])
    keys = predictions.get("key", [])
    tempos = predictions.get("tempo_bpm", [])

    if frame_times is None:
        return [], [], []

    num_frames = len(frame_times)
    if num_frames == 0:
        return [], [], []

    # 最後のイベントの終了時刻を推定するために、最終フレームの時間に1フレーム分の長さを加える
    if num_frames > 1:
        frame_duration = frame_times[1] - frame_times[0]
    else:
        # 1フレームしかない場合は、仮の値（0.1秒など）を設定
        frame_duration = 0.1

    chord_events: List[Dict] = []
    key_events: List[Dict] = []
    tempo_events: List[Dict] = []

    # コードイベントの処理
    if root_chords and basses:
        limit = min(len(root_chords), len(basses), num_frames)
        if limit > 0:

            def format_chord(chord_label: str, bass_label: str) -> str:
                if chord_label == "N" or chord_label is None:
                    return "N"
                if bass_label in (None, "", "N"):
                    return chord_label

                def extract_root(label: str) -> str:
                    if not label:
                        return ""
                    if len(label) >= 2 and label[1] in {"#", "b"}:
                        return label[:2]
                    return label[:1]

                if extract_root(chord_label) == bass_label:
                    return chord_label
                return f"{chord_label}/{bass_label}"

            last_chord_label = format_chord(root_chords[0], basses[0])
            segment_start_time = frame_times[0]

            for i in range(1, limit):
                current_chord_label = format_chord(root_chords[i], basses[i])

                # コードが変化した場合、前のセグメントを記録
                if current_chord_label != last_chord_label:
                    chord_events.append(
                        {"start_time": segment_start_time, "end_time": frame_times[i], "chord": last_chord_label}
                    )
                    # 新しいセグメントを開始
                    segment_start_time = frame_times[i]
                    last_chord_label = current_chord_label

            # 最後のセグメントを追加
            end_time = frame_times[limit - 1] + frame_duration
            chord_events.append({"start_time": segment_start_time, "end_time": end_time, "chord": last_chord_label})

    # キーイベントの処理
    if keys:
        limit = min(len(keys), num_frames)
        if limit > 0:
            last_key_label = keys[0]
            segment_start_time = frame_times[0]

            for i in range(1, limit):
                current_key_label = keys[i]

                # キーが変化した場合、前のセグメントを記録
                if current_key_label != last_key_label:
                    key_events.append(
                        {"start_time": segment_start_time, "end_time": frame_times[i], "key": last_key_label}
                    )
                    # 新しいセグメントを開始
                    segment_start_time = frame_times[i]
                    last_key_label = current_key_label

            # 最後のセグメントを追加
            end_time = frame_times[limit - 1] + frame_duration
            key_events.append({"start_time": segment_start_time, "end_time": end_time, "key": last_key_label})

    # --- chord-romanizerによる臨時記号の修正 ---
    if chord_events and key_events:
        try:
            from chord_romanizer import Romanizer, ChordParser

            romanizer = Romanizer(simplify_accidentals=True)
            progression_sequence = []

            # IDによるマッピングを保持して、フィルタリング後の結果を元のイベントに戻せるようにする
            parsed_chord_map = {}  # {id(parsed_chord): index_in_chord_events}

            # 各コードセグメントに対して、最も重なりの大きいキーを割り当てる
            for i, chord_event in enumerate(chord_events):
                c_start = chord_event["start_time"]
                c_end = chord_event["end_time"]

                best_key = "C"  # Default
                max_overlap = -1.0

                for key_event in key_events:
                    k_start = key_event["start_time"]
                    k_end = key_event["end_time"]

                    overlap_start = max(c_start, k_start)
                    overlap_end = min(c_end, k_end)
                    overlap = max(0.0, overlap_end - overlap_start)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_key = key_event["key"]

                if best_key == "N":
                    best_key = "C"

                parsed = ChordParser.parse(chord_event["chord"])

                # Parse失敗時は "N" (No Chord) として扱うことでクラッシュを回避
                if parsed is None:
                    # Nをパースしてダミーオブジェクトを作成
                    parsed = ChordParser.parse("N")

                if parsed:
                    parsed_chord_map[id(parsed)] = i
                    progression_sequence.append((parsed, best_key))

            # Romanizer実行
            annotated = romanizer.annotate_progression(progression_sequence)

            # 結果を適用 (IDで照合)
            for result in annotated:
                if result and result.chord:  # result.chord is the Original ParsedChord object
                    idx = parsed_chord_map.get(id(result.chord))
                    if idx is not None and result.symbol_fixed:
                        chord_events[idx]["chord"] = result.symbol_fixed

        except ImportError as e:
            print(f"[WARN] chord-romanizer not found or failed to import: {e}")
        except Exception as e:
            print(f"[WARN] Failed to run chord-romanizer: {e}")

    def filter_events(events: List[Dict], min_dur: float, label_key: str) -> List[Dict]:
        if not events:
            return []
        filtered = []
        for event in events:
            duration = event["end_time"] - event["start_time"]
            if duration < min_dur and filtered:
                # 前のイベントを延長
                filtered[-1]["end_time"] = event["end_time"]
            else:
                # 前のイベントと同じラベルなら結合
                if filtered and filtered[-1][label_key] == event[label_key]:
                    filtered[-1]["end_time"] = event["end_time"]
                else:
                    filtered.append(event)
        return filtered

    def filter_tempo_events(events: List[Dict], min_dur: float, merge_bpm: float) -> List[Dict]:
        if not events:
            return []
        filtered: List[Dict] = []
        for event in events:
            duration = float(event["end_time"] - event["start_time"])
            if duration < min_dur and filtered:
                filtered[-1]["end_time"] = event["end_time"]
                continue

            if filtered:
                prev_tempo = float(filtered[-1]["tempo"])
                curr_tempo = float(event["tempo"])
                if abs(curr_tempo - prev_tempo) <= merge_bpm:
                    filtered[-1]["end_time"] = event["end_time"]
                    continue

            filtered.append(dict(event))
        return filtered

    # テンポイベントの処理
    if tempos:
        limit = min(len(tempos), num_frames)
        if limit > 0:
            segment_start_time = frame_times[0]
            segment_tempos: List[float] = [float(tempos[0])]

            for i in range(1, limit):
                current_tempo = float(tempos[i])
                if not np.isfinite(current_tempo):
                    continue

                representative = float(np.median(segment_tempos)) if segment_tempos else current_tempo
                threshold = max(float(tempo_change_bpm), abs(representative) * float(tempo_change_ratio))

                if abs(current_tempo - representative) >= threshold:
                    tempo_events.append(
                        {
                            "start_time": segment_start_time,
                            "end_time": frame_times[i],
                            "tempo": round(representative, 2),
                        }
                    )
                    segment_start_time = frame_times[i]
                    segment_tempos = [current_tempo]
                else:
                    segment_tempos.append(current_tempo)

            representative = float(np.median(segment_tempos)) if segment_tempos else float(tempos[limit - 1])
            end_time = frame_times[limit - 1] + frame_duration
            tempo_events.append(
                {"start_time": segment_start_time, "end_time": end_time, "tempo": round(representative, 2)}
            )

    # フィルタリング適用
    if min_duration_chord > 0:
        chord_events = filter_events(chord_events, min_duration_chord, "chord")

    if min_duration_key > 0:
        key_events = filter_events(key_events, min_duration_key, "key")

    if min_duration_tempo > 0:
        tempo_events = filter_tempo_events(tempo_events, min_duration_tempo, merge_bpm=max(1.0, tempo_change_bpm / 2))

    return chord_events, key_events, tempo_events


def save_events_tsv(events: List[Dict], output_path: Path, headers: List[str]) -> None:
    """
    イベントリストを指定されたパスにTSV形式で保存する。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        # 各イベントをタブ区切りで書き込み
        for event in events:
            # 3番目のヘッダー（'chord' or 'key'）を動的に取得
            label_key = headers[2]
            row = [
                f"{event.get('start_time', 0.0):.6f}",
                f"{event.get('end_time', 0.0):.6f}",
                str(event.get(label_key, "N/A")),
            ]
            f.write("\t".join(row) + "\n")
    print(f"\n[OK] 予測結果を保存しました: {output_path}")


def print_events_preview(events: List[Dict], event_type: str) -> None:
    """イベントリストの最初の数件をプレビュー表示する。"""
    if not events:
        print(f"[INFO] 表示できる有効な {event_type} イベントがありません。")
        return

    max_show = min(20, len(events))
    label_key = event_type.lower()

    print(f"\n--- {event_type} 予測プレビュー (先頭{max_show}件) ---")
    for i in range(max_show):
        event = events[i]
        start = event.get("start_time", 0.0)
        end = event.get("end_time", 0.0)
        label = event.get(label_key, "N/A")
        print(f"{start:8.3f}s - {end:8.3f}s | {label}")
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


def save_chord25_logits_image(chord25_logits: Any, output_path: Path, target_time_bins: int = 128) -> None:
    """
    chord25_logits (T, 25) をそのまま PNG で保存する。時間軸を最近傍で target_time_bins まで拡大する。
    """
    if chord25_logits is None:
        print("[WARN] chord25_logits が見つからないため画像を保存できません。")
        return

    logits_np = np.asarray(chord25_logits)
    if logits_np.ndim != 2:
        print(f"[WARN] chord25_logits の想定外の形状です: {logits_np.shape}")
        return

    logits_np = logits_np.astype(np.float32)
    upsampled = upscale_time_axis(logits_np, target_time_bins=target_time_bins)
    image_data = upsampled.T  # (25, T_upsampled) -> y: chord, x: time

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6.0, image_data.shape[1] / 20), 4))
    plt.imshow(image_data, aspect="auto", origin="lower", interpolation="nearest", cmap="gray", vmin=0)
    plt.xlabel("Time (upsampled)")
    plt.ylabel("Chord25 class")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n[OK] chord25 logitsイメージを保存しました: {output_path}")


def main():
    """引数を解析し、推論処理を実行するメイン関数。"""
    parser = argparse.ArgumentParser(description="音声からコード進行とキーをTSV形式で推論・保存するスクリプト")
    parser.add_argument("--config", type=str, required=True, help="モデル設定ファイル(YAML/JSON)へのパス")
    parser.add_argument("--checkpoint", type=str, required=True, help="学習済みモデルのチェックポイントへのパス")
    parser.add_argument("--audio", type=str, required=True, help="入力音声ファイルへのパス")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (例: 'cpu', 'cuda')",
    )
    parser.add_argument("--output_dir", type=str, default="predictions", help="出力ファイルを保存するディレクトリ")
    parser.add_argument(
        "--stems_dir", type=str, default="stems", help="分離されたステムを保存/読み込みするディレクトリ"
    )
    parser.add_argument("--no-reuse-stems", action="store_true", help="既存のステムを再利用せず、強制的に再分離する")
    parser.add_argument(
        "--quality_json", type=Path, default="./data/quality.json", help="qualityラベルのJSONファイルへのパス"
    )
    parser.add_argument("--use_segment_model", action="store_true")
    parser.add_argument("--use_hmm", action="store_true", help="HMMによる平滑化を適用する")
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
        "--min_duration_tempo",
        type=float,
        default=2.0,
        help="テンポイベントの最小継続時間(秒)。これより短いイベントは前のイベントに結合されます。",
    )
    parser.add_argument(
        "--tempo_change_ratio",
        type=float,
        default=0.05,
        help="テンポ変化判定の相対しきい値。例: 0.05 なら約5%以上の変化を境界候補にします。",
    )
    parser.add_argument(
        "--tempo_change_bpm",
        type=float,
        default=4.0,
        help="テンポ変化判定の絶対しきい値(BPM)。",
    )
    args = parser.parse_args()

    try:
        # セットアップ
        config = load_config(Path(args.config))
        transcriber = AudioTranscriber(
            config=config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            quality_json_path=args.quality_json,
            use_segment_model=args.use_segment_model,
        )

        # 推論の実行（フレームごと）
        predictions = transcriber.predict(
            audio_path=args.audio,
            stems_dir=args.stems_dir,
            reuse_stems=not args.no_reuse_stems,
            use_hmm=args.use_hmm,
        )

        # フレームごとの予測をイベント（コードチェンジ、キーチェンジ）に変換
        if predictions:
            chord_events, key_events, tempo_events = process_predictions_to_events(
                predictions,
                min_duration_chord=args.min_duration_chord,
                min_duration_key=args.min_duration_key,
                min_duration_tempo=args.min_duration_tempo,
                tempo_change_ratio=args.tempo_change_ratio,
                tempo_change_bpm=args.tempo_change_bpm,
            )

            # 結果の表示と保存
            audio_path = Path(args.audio)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            chord25_logits = predictions.get("final_chord25_logits")
            if chord25_logits is not None:
                chord25_img_path = output_dir / f"{audio_path.stem}_chord25_logits.png"
                save_chord25_logits_image(chord25_logits, chord25_img_path, target_time_bins=128)

            if chord_events:
                print_events_preview(chord_events, "Chord")
                chord_output_path = output_dir / f"{audio_path.stem}.chords.txt"
                save_events_tsv(chord_events, chord_output_path, ["start_time", "end_time", "chord"])

            if key_events:
                print_events_preview(key_events, "Key")
                key_output_path = output_dir / f"{audio_path.stem}.key.txt"
                save_events_tsv(key_events, key_output_path, ["start_time", "end_time", "key"])

            if tempo_events:
                print_events_preview(tempo_events, "Tempo")
                tempo_output_path = output_dir / f"{audio_path.stem}.tempo.txt"
                save_events_tsv(tempo_events, tempo_output_path, ["start_time", "end_time", "tempo"])

        else:
            print("[ERROR] 予測を生成できませんでした。")

    except (FileNotFoundError, KeyError, ImportError, AttributeError) as e:
        print(f"\n[エラー] 処理中にエラーが発生しました: {e}")
        print("設定ファイル、チェックポイント、または base_transcriber.py の内容が正しいか確認してください。")
    except Exception as e:
        print(f"\n[予期せぬエラー] 予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
