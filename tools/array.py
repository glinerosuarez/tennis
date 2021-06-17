import numpy as np
import scipy.signal
from typing import Optional, Tuple, Union


def combined_shape(length: int, shape: Optional[int] = None) -> Union[Tuple[int, None], Tuple[int]]:
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]