import numpy as np

from typing import Tuple


def intersect(v1: Tuple[np.ndarray], v2: Tuple[np.ndarray]) -> np.ndarray:
    da = v1[1] - v1[0]
    db = v2[1] - v2[0]
    dp = v1[0] - v2[0]

    dap = np.asarray([-da[1], da[0]])
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)

    return np.asarray((num / denom.astype(float)) * db + v2[0])
