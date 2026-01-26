import math
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from dlchordx import Tone
from dlchordx.const import CHORD_MAP


@dataclass
class EventSpan:
    """キーやテンポなど、単一のラベルを持つイベントを表す汎用クラス。"""

    start_time: float
    end_time: float
    label: Any  # 文字列だけでなく、数値なども扱えるようにAny型にしておく


@dataclass
class ChordEvent:
    """コードを表すイベント。root, bass, qualityを保持します。"""

    start_time: float
    end_time: float
    root: str
    bass: str
    quality: str


class LabelProcessor:
    """
    時間ベースのイベント（Chord, Keyなど）を、モデル入力と同期したフレーム系列に変換するクラス。
    """

    def __init__(self, sample_rate: int, hop_length: int, n_fft: int, ignore_index: int = -100):
        """
        Args:
            sample_rate: オーディオのサンプリングレート
            hop_length: STFTのホップサイズ（フレーム間のサンプル数）
            n_fft: STFTのFFTサイズ
            ignore_index: 損失計算で無視するインデックス
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.hop_sec = hop_length / float(sample_rate)
        self.ignore_index = ignore_index

        # CHORD_MAPのキーから空白を除去（前処理）
        self.chord_map = {key.replace(" ", ""): val for key, val in CHORD_MAP.items()}

    def get_num_frames(self, num_samples: int) -> int:
        """
        音声サンプル数から、モデルのSTFT出力と完全に一致するフレーム数を計算します。
        """
        # パディングを考慮しないSTFTのフレーム数計算
        return (num_samples - self.n_fft) // self.hop_length + 1

    def spans_to_frames(self, spans: List[Dict[str, Any]], num_frames: int, seg_start_sec: float) -> np.ndarray:
        """
        時間情報を持つイベントのリスト（spans）をフレーム系列に変換します。

        Args:
            spans: {"start_time": float, "end_time": float, "idx": int} のリスト
            num_frames: 出力フレーム数（get_num_framesで計算された値）
            seg_start_sec: オーディオセグメントの開始時間（秒）

        Returns:
            フレームごとのラベルインデックスを持つNumpy配列 (shape: [num_frames])
        """
        # すべてのフレームをignore_indexで初期化
        frame_targets = np.full((num_frames,), self.ignore_index, dtype=np.int64)
        if num_frames == 0:
            return frame_targets

        # 各イベント（span）をループ処理
        for span in spans:
            event_start_sec = float(span["start_time"])
            event_end_sec = float(span["end_time"])
            event_idx = int(span["idx"])

            # イベントの開始・終了時刻を、セグメントの開始時刻からの相対的な秒数に変換
            start_relative_sec = event_start_sec - seg_start_sec
            end_relative_sec = event_end_sec - seg_start_sec

            # 秒数をフレームインデックスに変換
            # フレームiの開始時刻は i * hop_sec なので、秒数をhop_secで割る
            start_frame = math.floor(start_relative_sec / self.hop_sec)
            end_frame = math.ceil(end_relative_sec / self.hop_sec)

            # 計算されたフレーム範囲を、セグメントの有効範囲 [0, num_frames] にクリップ
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)

            # 有効なフレームがあれば、その範囲のラベルをevent_idxで上書き
            if end_frame > start_frame:
                frame_targets[start_frame:end_frame] = event_idx

        return frame_targets

    def spans_to_float_frames(self, spans: List[Dict[str, Any]], num_frames: int, seg_start_sec: float) -> np.ndarray:
        """
        【回帰用】浮動小数点数の値を持つイベントをフレーム系列に変換します。
        (tempo など)
        """
        frame_targets = np.full((num_frames,), self.ignore_index, dtype=np.float32)
        if num_frames == 0:
            return frame_targets

        for span in spans:
            event_value = float(span["value"])

            start_relative_sec = float(span["start_time"]) - seg_start_sec
            end_relative_sec = float(span["end_time"]) - seg_start_sec

            start_frame = math.floor(start_relative_sec / self.hop_sec)
            end_frame = math.ceil(end_relative_sec / self.hop_sec)

            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)

            if end_frame > start_frame:
                frame_targets[start_frame:end_frame] = event_value

        return frame_targets

    def _create_chord_vector(self, event: ChordEvent) -> np.ndarray:
        """
        単一のChordEventから25次元のベクトルを生成するヘルパーメソッド。

        Returns:
            np.ndarray: 25次元のコードベクトル (12次元ピッチ + 13次元ベース)
        """
        # "N" (No Chord) の場合はゼロベクトルを返す
        if event.root == "N" or event.quality == "N":
            return np.zeros(25, dtype=np.float32)

        try:
            # ルート音を0-11のインデックスに変換 (C=0, C#=1, ...)
            root_idx = int(Tone(event.root).get_interval())

            # 12次元のピッチクラスベクトル（構成音）を作成
            pitch_vec = np.zeros(12, dtype=np.float32)
            if event.quality in self.chord_map:
                for interval in self.chord_map[event.quality]:
                    # ルート音からの相対音程を足して12で割った余りを計算
                    pitch_vec[(root_idx + int(interval)) % 12] = 1.0

            # 13次元のベース音ベクトル（ワンホット）を作成
            bass_vec = np.zeros(13, dtype=np.float32)
            # "N"は0、Cは1、... Bは12
            bass_idx = int(Tone(event.bass).get_interval()) + 1 if event.bass != "N" else 0
            bass_vec[bass_idx] = 1.0

            # ピッチベクトルとベース音ベクトルを連結して25次元ベクトルを返す
            return np.concatenate([pitch_vec, bass_vec], axis=0)

        except Exception:
            # Tone()が解析できない音名の場合もゼロベクトルを返す
            return np.zeros(25, dtype=np.float32)

    def chords_to_25d_frames(self, chord_events: List[ChordEvent], num_frames: int, seg_start_sec: float) -> np.ndarray:
        """
        ChordEventのリストを、25次元ベクトルで表現されたフレーム系列に変換します。

        Returns:
            np.ndarray: shapeが (num_frames, 25) の配列
        """
        # (num_frames, 25) のゼロで満たされた配列を準備
        frame_targets = np.zeros((num_frames, 25), dtype=np.float32)
        if num_frames == 0:
            return frame_targets

        # 各コードイベントをループ処理
        for event in chord_events:
            # 25次元ベクトルを生成
            vec25 = self._create_chord_vector(event)

            # イベントの時間範囲をフレームインデックスに変換
            start_relative_sec = event.start_time - seg_start_sec
            end_relative_sec = event.end_time - seg_start_sec

            start_frame = math.floor(start_relative_sec / self.hop_sec)
            end_frame = math.ceil(end_relative_sec / self.hop_sec)

            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)

            # 有効なフレーム範囲に、生成した25次元ベクトルを割り当てる
            if end_frame > start_frame:
                frame_targets[start_frame:end_frame, :] = vec25

        return frame_targets

    def chords_to_boundary_frames(
        self, chord_events: List[ChordEvent], num_frames: int, seg_start_sec: float, neighbor_weight: float = 0.5
    ) -> np.ndarray:
        """
        コードイベントのリストから、コードが変化する境界フレームを検出してターゲットを生成します。

        Args:
            chord_events (List[ChordEvent]): コードイベントのリスト
            num_frames (int): 出力フレーム数
            seg_start_sec (float): セグメント開始時間
            neighbor_weight (float): 境界フレームの隣接フレームに与える重み

        Returns:
            np.ndarray: 境界ターゲット配列 (shape: [num_frames, 1])
        """
        if num_frames == 0:
            return np.zeros((0, 1), dtype=np.float32)

        # 各フレームがどのコードに属するかを一時的に表現します。
        # コードの同一性は、root, quality, bassの組み合わせで判定します。
        chord_strings_to_id = {f"{e.root}:{e.quality}/{e.bass}": i for i, e in enumerate(chord_events)}

        temp_spans = [
            {
                "start_time": e.start_time,
                "end_time": e.end_time,
                "idx": chord_strings_to_id.get(f"{e.root}:{e.quality}/{e.bass}", -1),
            }
            for e in chord_events
        ]

        chord_frame_indices = self.spans_to_frames(temp_spans, num_frames, seg_start_sec)

        # 前後のフレームでコードIDが異なる箇所を境界とする
        boundary = np.zeros(num_frames, dtype=np.float32)

        # 隣接要素の差分を計算。差分が0でない箇所が変化点
        change_points = np.where(np.diff(chord_frame_indices) != 0)[0] + 1

        # 変化点の中心（変化した後ろのフレーム）の値を1.0にする
        boundary[change_points] = 1.0

        # 変化点の左右のフレームに重みを与える
        if neighbor_weight > 0:
            # 左隣
            left_indices = change_points - 1
            valid_left = left_indices >= 0
            boundary[left_indices[valid_left]] = np.maximum(boundary[left_indices[valid_left]], neighbor_weight)
            # 右隣
            right_indices = change_points + 1
            valid_right = right_indices < num_frames
            boundary[right_indices[valid_right]] = np.maximum(boundary[right_indices[valid_right]], neighbor_weight)

        # 最終的な形状を (num_frames, 1) にして返す
        return boundary.reshape(num_frames, 1)

    def sections_to_boundary_frames(
        self, spans: List[Dict[str, Any]], num_frames: int, seg_start_sec: float, neighbor_weight: float = 0.5
    ) -> np.ndarray:
        """
        セクションイベントのリストから、セクションが変化する境界フレームを検出してターゲットを生成します。
        """
        if num_frames == 0:
            return np.zeros((0, 1), dtype=np.float32)

        # セクションのインデックスフレームを取得
        section_frame_indices = self.spans_to_frames(spans, num_frames, seg_start_sec)

        # 前後のフレームでセクションIDが異なる箇所を境界とする
        boundary = np.zeros(num_frames, dtype=np.float32)

        # 隣接要素の差分を計算。差分が0でない箇所が変化点
        # インデックスが変わった瞬間を捉える
        diff = np.diff(section_frame_indices)
        change_points = np.where(diff != 0)[0] + 1

        # 変化点を1.0にする
        boundary[change_points] = 1.0

        # 隣接フレームへの重み付け
        if neighbor_weight > 0:
            left_indices = change_points - 1
            valid_left = left_indices >= 0
            boundary[left_indices[valid_left]] = np.maximum(boundary[left_indices[valid_left]], neighbor_weight)

            right_indices = change_points + 1
            valid_right = right_indices < num_frames
            boundary[right_indices[valid_right]] = np.maximum(boundary[right_indices[valid_right]], neighbor_weight)

        return boundary.reshape(num_frames, 1)
