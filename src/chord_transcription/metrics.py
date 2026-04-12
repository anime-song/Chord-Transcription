from __future__ import annotations

from typing import Any, Sequence, Tuple

# 区間（開始、終了）を表す型
Interval = Tuple[float, float]


class IntervalEvaluator:
    """
    区間（セグメント）の評価を行うクラス。
    IoU（Intersection over Union）に基づく二部グラフマッチングにより、
    適合率、再現率、F1スコアを計算します。
    """

    def __init__(self, iou_threshold: float = 0.5, inclusive_end: bool = False) -> None:
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")

        self.iou_threshold = float(iou_threshold)
        self.inclusive_end = inclusive_end

    def evaluate_batch(
        self,
        ground_truth_batch: Sequence[Sequence[Interval]],
        prediction_batch: Sequence[Sequence[Interval]],
    ) -> float:
        """
        バッチ全体のMacro F1スコア（各サンプルのF1スコアの平均）を計算します。

        Args:
            ground_truth_batch: 正解の区間リストのバッチ
            prediction_batch: 予測の区間リストのバッチ

        Returns:
            Macro F1スコア
        """
        if len(ground_truth_batch) != len(prediction_batch):
            raise ValueError(
                f"ground_truth_batch and prediction_batch must have the same length, "
                f"got {len(ground_truth_batch)} and {len(prediction_batch)}"
            )
        if not ground_truth_batch:
            return 0.0

        scores = [
            self.evaluate_single(ground_truth_intervals, prediction_intervals)
            for ground_truth_intervals, prediction_intervals in zip(ground_truth_batch, prediction_batch)
        ]
        return float(sum(scores) / len(scores))

    def evaluate_single(
        self,
        ground_truth_intervals: Sequence[Interval],
        prediction_intervals: Sequence[Interval],
    ) -> float:
        """単一サンプルのF1スコアを計算します。"""
        # 無効な区間を除外
        valid_ground_truths = [iv for iv in ground_truth_intervals if self._is_valid_interval(iv)]
        valid_predictions = [iv for iv in prediction_intervals if self._is_valid_interval(iv)]

        if not valid_ground_truths and not valid_predictions:
            return 1.0
        if not valid_ground_truths or not valid_predictions:
            return 0.0

        # IoU条件を満たすエッジ（隣接リスト）を構築
        adjacency = self._build_adjacency_list(valid_ground_truths, valid_predictions)

        # マッチング数を計算（True Positive）
        true_positives = self._calculate_maximum_bipartite_matches(adjacency, len(valid_predictions))

        false_positives = len(valid_predictions) - true_positives
        false_negatives = len(valid_ground_truths) - true_positives

        denominator = 2 * true_positives + false_positives + false_negatives
        if denominator <= 0:
            return 0.0

        return float((2.0 * true_positives) / float(denominator))

    def _is_valid_interval(self, interval: Interval) -> bool:
        begin, end = float(interval[0]), float(interval[1])
        if self.inclusive_end:
            return end >= begin
        return end > begin

    def _calculate_iou(self, left: Interval, right: Interval) -> float:
        left_begin, left_end = float(left[0]), float(left[1])
        right_begin, right_end = float(right[0]), float(right[1])

        # 重なり（Intersection）部分の計算
        if self.inclusive_end:
            intersection = min(left_end, right_end) - max(left_begin, right_begin) + 1.0
            if intersection <= 0.0:
                return 0.0
            left_length = left_end - left_begin + 1.0
            right_length = right_end - right_begin + 1.0
        else:
            intersection = min(left_end, right_end) - max(left_begin, right_begin)
            if intersection <= 0.0:
                return 0.0
            left_length = left_end - left_begin
            right_length = right_end - right_begin

        union = left_length + right_length - intersection
        if union <= 0.0:
            return 0.0

        return float(intersection / union)

    def _build_adjacency_list(
        self,
        ground_truths: Sequence[Interval],
        predictions: Sequence[Interval],
    ) -> list[list[int]]:
        """正解と予測の二部グラフの隣接リストを作成します。"""
        adjacency: list[list[int]] = [[] for _ in range(len(ground_truths))]
        for gt_index, gt_interval in enumerate(ground_truths):
            for pred_index, pred_interval in enumerate(predictions):
                if self._calculate_iou(gt_interval, pred_interval) >= self.iou_threshold:
                    adjacency[gt_index].append(pred_index)
        return adjacency

    def _calculate_maximum_bipartite_matches(self, adjacency: list[list[int]], num_prediction_nodes: int) -> int:
        """
        深さ優先探索(DFS)を用いて最大二部マッチングのペア数を計算します。

        Args:
            adjacency: 正解ノードから予測ノードへの隣接リスト
            num_prediction_nodes: 予測ノードの総数

        Returns:
            最大マッチング数
        """
        # prediction側のノードがどのground_truthノードとマッチしたかを記録 (-1は未マッチ)
        matched_ground_truth = [-1] * num_prediction_nodes

        def try_match(ground_truth_index: int, visited_predictions: list[bool]) -> bool:
            """ある正解ノードに対してマッチングできるかDFSで探索します。"""
            for prediction_index in adjacency[ground_truth_index]:
                # すでにこの探索で訪問済みの予測ノードはスキップ（循環防止）
                if visited_predictions[prediction_index]:
                    continue

                visited_predictions[prediction_index] = True
                previous_ground_truth_index = matched_ground_truth[prediction_index]

                # 予測ノードが未マッチ、あるいは元のマッチを別の予測ノードに振り替えられるならマッチ成功
                if previous_ground_truth_index < 0 or try_match(previous_ground_truth_index, visited_predictions):
                    matched_ground_truth[prediction_index] = ground_truth_index
                    return True
            return False

        matches = 0
        for current_ground_truth_index in range(len(adjacency)):
            visited = [False] * num_prediction_nodes
            if try_match(current_ground_truth_index, visited):
                matches += 1

        return matches


def extract_time_intervals_from_events(events: Sequence[Any]) -> list[Interval]:
    """
    辞書やオブジェクトのリストから開始・終了時間の区間リストを抽出します。
    """
    intervals: list[Interval] = []
    for event in events:
        if isinstance(event, dict):
            begin = float(event["start_time"])
            end = float(event["end_time"])
        else:
            begin = float(getattr(event, "start_time"))
            end = float(getattr(event, "end_time"))

        # NOTE: start_time < end_time の場合のみ有効な区間として抽出
        if end > begin:
            intervals.append((begin, end))

    return intervals


def segment_f1_score(
    ground_truth_intervals: Sequence[Interval],
    prediction_intervals: Sequence[Interval],
    *,
    iou_threshold: float = 0.5,
    inclusive_end: bool = False,
) -> float:
    """後方互換性のためのセグメントF1スコア計算へのラッパー関数。"""
    evaluator = IntervalEvaluator(iou_threshold=iou_threshold, inclusive_end=inclusive_end)
    return evaluator.evaluate_single(ground_truth_intervals, prediction_intervals)


def segment_macro_f1_score(
    ground_truth_batch: Sequence[Sequence[Interval]],
    prediction_batch: Sequence[Sequence[Interval]],
    *,
    iou_threshold: float = 0.5,
    inclusive_end: bool = False,
) -> float:
    """後方互換性のためのバッチMacroセグメントF1スコア計算へのラッパー関数。"""
    evaluator = IntervalEvaluator(iou_threshold=iou_threshold, inclusive_end=inclusive_end)
    return evaluator.evaluate_batch(ground_truth_batch, prediction_batch)
