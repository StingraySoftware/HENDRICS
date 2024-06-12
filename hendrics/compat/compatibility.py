from functools import wraps
import warnings
import numpy as np
from astropy import log
from collections.abc import Iterable as iterable

try:
    from numba import jit, njit, prange, vectorize
    from numba import float32, float64, int32, int64
    from numba import types
    from numba.extending import overload_method

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(**kwargs):
        """Dummy decorator in case jit cannot be imported."""

        def true_decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                r = func(*args, **kwargs)
                return r

            return wrapped

        return true_decorator

    jit = njit

    def prange(*args):
        """Dummy decorator in case jit cannot be imported."""
        return range(*args)

    class vectorize(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            wrapped_f = np.vectorize(func)

            return wrapped_f

    float32 = float64 = int32 = int64 = lambda x, y: None


def array_take(arr, indices):  # pragma: no cover
    """Adapt np.take to arrays"""
    warnings.warn("array_take is deprecated. Use np.take instead, also with Numba.")
    return np.take(arr, indices)
