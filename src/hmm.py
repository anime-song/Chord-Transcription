import numpy as np
from numba import njit


def viterbi_naive(log_em, log_tr, log_pi):
    """
    log_emission  : (T, C)  NN の softmax 出力に log を取ったもの
    log_transition: (C, C)  手動 or カウント推定した log 遷移行列
    log_init      : (C,)    初期状態 log 確率
    """
    T, C = log_em.shape
    score = np.full((T, C), -np.inf)
    backp = np.zeros((T, C), dtype=int)
    score[0] = log_pi + log_em[0]
    for t in range(1, T):
        for j in range(C):
            prev = score[t - 1] + log_tr[:, j]
            best = np.argmax(prev)
            score[t, j] = prev[best] + log_em[t, j]
            backp[t, j] = best
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(score[-1])
    for t in range(T - 2, -1, -1):
        path[t] = backp[t + 1, path[t + 1]]
    return path


@njit
def viterbi_jit(log_emission, log_transition, log_init):
    """
    log_emission  : (T, C)  NN の softmax 出力に log を取ったもの
    log_transition: (C, C)  手動 or カウント推定した log 遷移行列
    log_init      : (C,)    初期状態 log 確率
    """
    T, C = log_emission.shape
    score = np.full((T, C), -1e9, dtype=np.float32)
    backp = np.zeros((T, C), dtype=np.int32)
    # 初期化
    for j in range(C):
        score[0, j] = log_init[j] + log_emission[0, j]
    # DP
    for t in range(1, T):
        for j in range(C):
            max_score = -1e9
            max_i = 0
            for i in range(C):
                v = score[t - 1, i] + log_transition[i, j]
                if v > max_score:
                    max_score = v
                    max_i = i
            score[t, j] = max_score + log_emission[t, j]
            backp[t, j] = max_i
    # 経路復元
    path = np.empty(T, dtype=np.int32)
    # 終端
    last = 0
    max_last = -1e9
    for j in range(C):
        if score[-1, j] > max_last:
            max_last = score[-1, j]
            last = j
    path[-1] = last
    for t in range(T - 2, -1, -1):
        path[t] = backp[t + 1, path[t + 1]]
    return path


def make_sticky_transition(C: int, stay_prob: float) -> np.ndarray:
    """
    C 状態の HMM 用に、
    自己遷移を stay_prob、他への遷移を (1−stay_prob)/(C−1) に設定した行列を返す。
    """
    off_diag = (1.0 - stay_prob) / (C - 1)
    A = np.full((C, C), off_diag, dtype=np.float32)
    np.fill_diagonal(A, stay_prob)
    return A


def decode_viterbi_from_probs(
    probs: np.ndarray, transition_matrix: np.ndarray = None, init_probs: np.ndarray = None, method: str = "jit"
) -> np.ndarray:
    """
    確率分布を渡すだけで Viterbi を実行するラッパー関数。

    Parameters
    ----------
    probs : np.ndarray, shape=(T, C)
        各時刻の観測確率（softmax 出力）。
    transition_matrix : np.ndarray, shape=(C, C), optional
        状態間の遷移確率行列。None の場合、
        自己遷移 0.9、他は (0.1/(C-1)) で初期化。
    init_probs : np.ndarray, shape=(C,), optional
        初期状態確率。None の場合、一様分布を使用。

    Returns
    -------
    best_labels : np.ndarray, shape=(T,)
        最尤状態系列のラベル配列。
    """
    # ログ領域に変換
    log_emission = np.log(probs + 1e-9)

    T, C = probs.shape

    # 遷移行列の初期化（デフォルト）
    if transition_matrix is None:
        A = make_sticky_transition(C, stay_prob=0.9)
    else:
        A = transition_matrix
    log_transition = np.log(A + 1e-9)

    # 初期状態確率の初期化（デフォルト）
    if init_probs is None:
        pi = np.ones(C) / C
    else:
        pi = init_probs
    log_init = np.log(pi + 1e-9)

    if method == "native":
        return viterbi_naive(log_emission, log_transition, log_init)

    elif method == "jit":
        return viterbi_jit(log_emission, log_transition, log_init)
    else:
        raise ValueError("存在しないタイプです")
