import copy
import os
from functools import wraps
import numpy as np
from astropy.table import Table
from astropy import log
import stingray.utils
from stingray.events import EventList

try:
    from numba import jit, njit, prange, vectorize
    from numba import float32, float64, int32, int64
    from numba import types
    from numba.extending import overload_method

    @overload_method(types.Array, "take")  # pragma: no cover
    def array_take(arr, indices):
        """Adapt np.take to arrays"""
        if isinstance(indices, types.Array):

            def take_impl(arr, indices):
                n = indices.shape[0]
                res = np.empty(n, arr.dtype)
                for i in range(n):
                    res[i] = arr[indices[i]]
                return res

            return take_impl

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

    def array_take(arr, indices):
        """Adapt np.take to arrays"""
        return np.take(arr, indices)


class _MonkeyPatchedEventList(EventList):
    def apply_mask(self, mask, inplace=False):  # pragma: no cover
        """For compatibility with old stingray version.
        Examples
        --------
        >>> evt = _MonkeyPatchedEventList(time=[0, 1, 2])
        >>> newev0 = evt.apply_mask([True, True, False], inplace=False);
        >>> newev1 = evt.apply_mask([True, True, False], inplace=True);
        >>> np.allclose(newev0.time, [0, 1])
        True
        >>> np.allclose(newev1.time, [0, 1])
        True
        >>> evt is newev1
        True
        """
        if inplace:
            new_ev = self
        else:
            new_ev = copy.deepcopy(self)
        for attr in "time", "energy", "pi", "cal_pi", "detector_id":
            if hasattr(new_ev, attr) and getattr(new_ev, attr) is not None:
                setattr(new_ev, attr, getattr(new_ev, attr)[mask])
        return new_ev
