import numpy as np

from typing import Tuple


def circumcenter(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = 2 * (a[0] * (b[1] - c[1]) + \
                b[0] * (c[1] - a[1]) + \
                c[0] * (a[1] - b[1]))
    ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) + \
            (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1]) + \
            (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
    uy = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) + \
            (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0]) + \
            (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d

    return np.asarray((ux, uy))
