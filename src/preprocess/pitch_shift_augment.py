import os
import glob
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import soundfile as sf
import argparse

USE_PYRUBBERBAND = True

try:
    import pyrubberband as pyrb

    pyrb.pitch_shift(np.zeros(10, dtype=np.float32), 44100, 1)
    print("pyrubberband が利用可能です。高品質なピッチシフトを実行します。")
except Exception:
    print("pyrubberbandの利用に失敗しました。librosaをフォールバックとして使用します。")
    USE_PYRUBBERBAND = False
    try:
        import librosa
    except ImportError:
        print("エラー: librosaがインストールされていません。'pip install librosa' を実行してください。")
        exit()


PITCH_SEMITONES = [i for i in range(-6, 6) if i != 0]


def pitch_shift_file(args):
    """
    単一の音声ファイルに対して、指定された半音数でピッチシフトを実行する関数。
    並列処理のために、引数をタプルで受け取る。
    """
    input_path, semitone = args

    # 出力ファイルパスを動的に生成 (例: A_bass_pitch_-6st.wav)
    base, ext = os.path.splitext(input_path)
    suffix = f"_pitch_{semitone}st"
    output_path = f"{base}{suffix}{ext}"

    # 既存のファイルは処理しない
    if os.path.exists(output_path):
        return "skipped", os.path.basename(input_path), semitone

    try:
        # 音声ファイルを読み込み
        y, sr = sf.read(input_path)
        y_shifted = None

        if USE_PYRUBBERBAND:
            # pyrubberbandでピッチシフト
            y_shifted = pyrb.pitch_shift(y, sr, semitone)
        else:
            # librosaでピッチシフト
            # librosa.effects.pitch_shiftは (チャンネル数, サンプル数) の形式を期待する
            # soundfile.readは (サンプル数, チャンネル数) の形式で返すので、転置(.T)が必要
            if y.ndim > 1:  # ステレオか多チャンネル音源の場合
                y_shifted = librosa.effects.pitch_shift(y.T, sr=sr, n_steps=float(semitone)).T
            else:  # モノラル音源の場合
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitone))

        sf.write(output_path, y_shifted, sr)

        method = "pyrb" if USE_PYRUBBERBAND else "librosa"
        print(f"✓ 保存完了 ({method}): {os.path.basename(output_path)}")
        return None, os.path.basename(input_path), semitone

    except Exception as e:
        print(f"✗ エラー発生: {os.path.basename(input_path)} (semitone: {semitone}) - {e}")
        return str(e), os.path.basename(input_path), semitone


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定されたフォルダ内のWAVファイルのピッチを並列処理でシフトします。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./dataset/songs_separated",
        help="処理対象の.wavファイルが含まれるフォルダのパス。\n例: python pitch_shift_multi.py ./my_audio_files",
    )
    parser.add_argument("--num_processes", type=int, default=6)
    args = parser.parse_args()
    target_dir = args.target_dir

    if not os.path.isdir(target_dir):
        print(f"エラー: 指定されたフォルダが見つかりません: {target_dir}")
        exit()
    start_time = time.time()

    # ターゲットフォルダ内のすべての.wavファイルを取得
    search_pattern = os.path.join(target_dir, "**", "*.wav")
    all_files = glob.glob(search_pattern, recursive=True)

    # ピッチシフトされていないオリジナルのファイルのみを処理対象とする
    original_files = [p for p in all_files if "_pitch_" not in os.path.basename(p)]

    if not original_files:
        print(f"フォルダ内に処理対象のオリジナル.wavファイルが見つかりませんでした: {target_dir}")
    else:
        # 実行するタスクのリストを作成
        # (ファイルパス, 半音数) のタプルのリスト
        tasks = []
        for path in original_files:
            for semitone in PITCH_SEMITONES:
                # 出力ファイルが既に存在しない場合のみタスクに追加
                base, ext = os.path.splitext(path)
                suffix = f"_pitch_{semitone}st"
                output_path = f"{base}{suffix}{ext}"
                if not os.path.exists(output_path):
                    tasks.append((path, semitone))

        if not tasks:
            print("全てのピッチシフト済みファイルが既に存在します。処理を終了します。")
        else:
            print(
                f"{len(original_files)}個の元ファイルに対し、{len(PITCH_SEMITONES)}パターンのピッチシフトを実行します。"
            )
            print(f"合計 {len(tasks)} 件の新規タスクを実行します。")

            num_processes = args.num_processes
            print(f"{num_processes}個のCPUコアを使用して並列処理を開始します。")

            with Pool(processes=num_processes) as pool:
                results = pool.map(pitch_shift_file, tasks)

            print("\n--- 全ての処理が完了しました ---")

            success_count = 0
            skipped_count = (len(original_files) * len(PITCH_SEMITONES)) - len(tasks)
            error_details = {}

            for res in results:
                if res:  # 結果がNoneでない場合
                    err, filename, semitone = res
                    if err is None:
                        success_count += 1
                    elif err != "skipped":
                        if filename not in error_details:
                            error_details[filename] = []
                        error_details[filename].append(semitone)

            print(f"成功: {success_count}件")
            if skipped_count > 0:
                print(f"スキップ (既存): {skipped_count}件")
            if error_details:
                print(f"失敗: {len(error_details)}個のファイルでエラーが発生しました。")
                for filename, semitones in error_details.items():
                    print(f"  - {filename} (semitones: {semitones})")

    # 処理時間を計算して表示
    end_time = time.time()
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")
