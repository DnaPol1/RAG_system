import numpy as np
from typing import List


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max нормализация
    """
    if not scores:
        return scores

    arr = np.array(scores, dtype=np.float32)
    mx, mn = arr.max(), arr.min()

    if mx == mn:
        return [1.0] * len(arr)

    return ((arr - mn) / (mx - mn)).tolist()
