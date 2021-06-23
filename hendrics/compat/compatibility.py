import copy
import os
from functools import wraps
import numpy as np
from astropy.table import Table
from astropy import log
import stingray.utils
from scipy import stats
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


def power_confidence_limits(preal, n=1, alpha=0.16):
    """Confidence limits on power, given a signal power in the PDS/Z search.

    Adapted from Vaughan et al. 1994, noting that, after appropriate
    normalization of the spectral stats, the distribution of powers in the PDS
    and the Z^2_n searches is always described by a noncentral chi squared
    distribution.

    Parameters
    ----------
    preal: float
        The underlying, signal-generated value of power

    Other Parameters
    ----------------
    n: int
        The number of summed powers to obtain pmeas. It can be multiple
        harmonics of the PDS, adjacent bins in a PDS summed to collect all the
        power in a QPO, or the n in Z^2_n
    alpha: float
        The p-value (e.g. 0.16 = 16% for 1-sigma)

    Results
    -------
    pmeas: [float, float]
        The upper and lower confidence interval (a, 1-a) on the measured power

    Examples
    --------
    >>> cl = power_confidence_limits(150)
    >>> np.allclose(cl, [127, 176], atol=1)
    True
    """
    rv = stats.ncx2(2 * n, preal)
    return rv.ppf([alpha, 1 - alpha])


def power_upper_limit(pmeas, n=1, c=0.95):
    """Upper limit on signal power, given a measured power in the PDS/Z search.

    Adapted from Vaughan et al. 1994, noting that, after appropriate
    normalization of the spectral stats, the distribution of powers in the PDS
    and the Z^2_n searches is always described by a noncentral chi squared
    distribution.

    Note that Vaughan+94 gives p(pmeas | preal), while we are interested in
    p(real | pmeas), which is not described by the NCX2 stat. Rather than
    integrating the CDF of this probability distribution, we start from a
    reasonable approximation and fit to find the preal that gives pmeas as
    a (e.g.95%) confidence limit.

    As Vaughan+94 shows, this power is always larger than the observed one.
    This is because we are looking for the maximum signal power that,
    combined with noise powers, would give the observed power. This involves
    the possibility that noise powers partially cancel out some signal power.

    Parameters
    ----------
    pmeas: float
        The measured value of power

    Other Parameters
    ----------------
    n: int
        The number of summed powers to obtain pmeas. It can be multiple
        harmonics of the PDS, adjacent bins in a PDS summed to collect all the
        power in a QPO, or the n in Z^2_n
    c: float
        The confidence value for the probability (e.g. 0.95 = 95%)

    Results
    -------
    psig: float
        The signal power that could produce P>pmeas with 1 - c probability

    Examples
    --------
    >>> pup = power_upper_limit(40, 1, 0.99)
    >>> np.isclose(pup, 75, atol=2)
    True
    """

    def ppf(x):
        rv = stats.ncx2(2 * n, x)
        return rv.ppf(1 - c)

    def isf(x):
        rv = stats.ncx2(2 * n, x)
        return rv.ppf(c)

    def func_to_minimize(x, xmeas):
        return np.abs(ppf(x) - xmeas)

    from scipy.optimize import minimize

    initial = isf(pmeas)

    res = minimize(
        func_to_minimize, [initial], pmeas, bounds=[(0, initial * 2)]
    )

    return res.x[0]


def pf_upper_limit(pmeas, counts, n=1, c=0.95):
    """Upper limit on pulsed fraction, given a measured power in the PDS/Z search.

    See `power_upper_limit` and `pf_from_ssig`.

    Parameters
    ----------
    pmeas: float
        The measured value of power

    counts: int
        The number of counts in the light curve used to calculate the spectrum

    Other Parameters
    ----------------
    n: int
        The number of summed powers to obtain pmeas. It can be multiple
        harmonics of the PDS, adjacent bins in a PDS summed to collect all the
        power in a QPO, or the n in Z^2_n
    c: float
        The confidence value for the probability (e.g. 0.95 = 95%)

    Results
    -------
    pf: float
        The pulsed fraction that could produce P>pmeas with 1 - c probability

    Examples
    --------
    >>> pfup = pf_upper_limit(40, 30000, 1, 0.99)
    >>> np.isclose(pfup, 0.13, atol=0.01)
    True
    """

    uplim = power_upper_limit(pmeas, n, c)
    return pf_from_ssig(uplim, counts)


def pf_from_a(a):
    """Pulsed fraction from fractional amplitude of modulation.

    If the pulsed profile is defined as
    p = mean * (1 + a * sin(phase)),

    we define "pulsed fraction" as 2a/b, where b = mean + a is the maximum and
    a is the amplitude of the modulation.

    Hence, pulsed fraction = 2a/(1+a)

    Examples
    --------
    >>> pf_from_a(1)
    1.0
    >>> pf_from_a(0)
    0.0
    """
    return 2 * a / (1 + a)


def a_from_pf(p):
    """Fractional amplitude of modulation from pulsed fraction

    If the pulsed profile is defined as
    p = mean * (1 + a * sin(phase)),

    we define "pulsed fraction" as 2a/b, where b = mean + a is the maximum and
    a is the amplitude of the modulation.

    Hence, a = pf / (2 - pf)

    Examples
    --------
    >>> a_from_pf(1)
    1.0
    >>> a_from_pf(0)
    0.0
    """
    return p / (2 - p)


def ssig_from_a(a, ncounts):
    """Theoretical power in the Z or PDS search for a sinusoid of amplitude a.

    From Leahy et al. 1983, given a pulse profile
    p = lambda * (1 + a * sin(phase)),
    The theoretical value of Z^2_n is Ncounts / 2 * a^2

    Note that if there are multiple sinusoidal components, one can use
    a = sqrt(sum(a_l))
    (Bachetti+2021b)

    Examples
    --------
    >>> round(ssig_from_a(0.1, 30000), 1)
    150.0
    """
    return ncounts / 2 * a ** 2


def a_from_ssig(ssig, ncounts):
    """Amplitude of a sinusoid corresponding to a given Z/PDS value

    From Leahy et al. 1983, given a pulse profile
    p = lambda * (1 + a * sin(phase)),
    The theoretical value of Z^2_n is Ncounts / 2 * a^2

    Note that if there are multiple sinusoidal components, one can use
    a = sqrt(sum(a_l))
    (Bachetti+2021b)

    Examples
    --------
    >>> a_from_ssig(150, 30000)
    0.1
    """
    return np.sqrt(2 * ssig / ncounts)


def ssig_from_pf(pf, ncounts):
    """Theoretical power in the Z or PDS for a sinusoid of pulsed fraction pf.

    See `ssig_from_a` and `a_from_pf` for more details

    Examples
    --------
    >>> round(ssig_from_pf(pf_from_a(0.1), 30000), 1)
    150.0
    """
    a = a_from_pf(pf)
    return ncounts / 2 * a ** 2


def pf_from_ssig(ssig, ncounts):
    """Estimate pulsed fraction for a sinusoid from a given Z or PDS power.

    See `a_from_ssig` and `pf_from_a` for more details

    Examples
    --------
    >>> round(a_from_pf(pf_from_ssig(150, 30000)), 1)
    0.1
    """
    a = a_from_ssig(ssig, ncounts)
    return pf_from_a(a)
