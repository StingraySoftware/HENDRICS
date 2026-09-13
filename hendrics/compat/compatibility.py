from functools import wraps

import numpy as np

try:
    from numba import (
        float32,
        float64,
        int32,
        int64,
        jit,
        njit,
        prange,
        types,
        vectorize,
    )
    from numba.extending import overload_method

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Dummy decorator in case jit cannot be imported.

        Works both bare (``@njit``) and called (``@njit(cache=True)``); the
        bare form used to raise ``TypeError`` here.
        """

        def true_decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                r = func(*args, **kwargs)
                return r

            return wrapped

        if len(args) == 1 and not kwargs and callable(args[0]):
            return true_decorator(args[0])

        return true_decorator

    jit = njit

    def prange(*args):
        """Dummy decorator in case jit cannot be imported."""
        return range(*args)

    class vectorize:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            wrapped_f = np.vectorize(func)

            return wrapped_f

    float32 = float64 = int32 = int64 = lambda x, y: None
