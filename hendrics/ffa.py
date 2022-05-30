import functools
import numpy as np

from .base import njit, vectorize, float32, float64, int32, int64
from .base import show_progress

"""
prof_n  step0  step1  step2
0        0+1    0+2    0+4
1        0+1'   0+2'   0+4'
2        2+3    1+3'   1+5'
3        2+3'   1+3''  1+5''
4        4+5    4+6    2+6''
5        4+5'   4+6'   2+6'''
6        6+7    5+7'   3+7'''
7        6+7'   5+7''  3+7''''
8        8+9    8+10   8+12
9        8+9'   8+10'  8+12'
10     10+11    9+11'  9+13'
11     10+11'   9+11'' 9+13''
...

Each element is a full profile.
Each profile number in the sums refers to the profile created in the _previous_
step. Primes indicate the amount of shift.

Let's call step_pow the quantity (2**(step+1)) So, in each sum:

+ The _jump_ between the summed profiles is equal to step_pow / 2
+ The _shift_ in each element is equal to (prof_n % step_pow(step) + 1) // 2
+ The starting number is obtained as
    prof_n // step_pow * step_pow + (prof_n - prof_n // step_pow) // 2
"""


@functools.lru_cache(maxsize=128)
def cached_sin_harmonics(nbin, z_n_n):
    twopiphases = np.pi * 2 * np.arange(0, 1, 1.0 / nbin)

    cached_sin = np.zeros(z_n_n * nbin)
    for i in range(z_n_n):
        cached_sin[i * nbin : (i + 1) * nbin] = np.sin(twopiphases)

    return cached_sin


@functools.lru_cache(maxsize=128)
def cached_cos_harmonics(nbin, z_n_n):
    twopiphases = np.pi * 2 * np.arange(0, 1, 1.0 / nbin)
    cached_cos = np.zeros(z_n_n * nbin)
    for i in range(z_n_n):
        cached_cos[i * nbin : (i + 1) * nbin] = np.cos(twopiphases)

    return cached_cos


@njit()
def _z_n_fast_cached(norm, cached_sin, cached_cos, n=2):
    """Z^2_n statistics, a` la Buccheri+03, A&A, 128, 245, eq. 2.

    Here in a fast implementation based on numba.
    Assumes that nbin != 0 and norm is an array.

    Parameters
    ----------
    norm : array of floats
        The pulse profile
    cached_sin : array of floats
        This array contains the values of the sine at each phase of
        ``norm``, repeated ``n`` times
    cached_cos : array of floats
        This array contains the values of the cosine at each phase of
        ``norm``, repeated ``n`` times
    n : int, default 2
        The ``n`` in $Z^2_n$.

    Returns
    -------
    z2_n : float
        The Z^2_n statistics of the events.

    Examples
    --------
    >>> phase = 2 * np.pi * np.arange(0, 1, 0.01)
    >>> norm = np.sin(phase) + 1
    >>> cached_sin = np.sin(np.concatenate((phase, phase, phase, phase)))
    >>> cached_cos = np.cos(np.concatenate((phase, phase, phase, phase)))
    >>> np.isclose(_z_n_fast_cached(norm, cached_sin, cached_cos, n=4), 50)
    True
    >>> np.isclose(_z_n_fast_cached(norm, cached_sin, cached_cos, n=2), 50)
    True
    """

    total_norm = np.sum(norm)

    result = 0
    N = norm.size

    for k in range(1, n + 1):
        result += (
            np.sum(cached_cos[: N * k : k] * norm) ** 2
            + np.sum(cached_sin[: N * k : k] * norm) ** 2
        )

    return 2 / total_norm * result


def z_n_fast_cached(norm, n=2):
    """Z^2_n statistics, a` la Buccheri+03, A&A, 128, 245, eq. 2.

    This allocates the cached_sin and cached_cos first, then calls
    `_z_n_fast_cached`
    Assumes that nbin != 0 and norm is an array.

    Parameters
    ----------
    norm : array of floats
        The pulse profile
    n : int, default 2
        The ``n`` in $Z^2_n$.

    Returns
    -------
    z2_n : float
        The Z^2_n statistics of the events.

    Examples
    --------
    >>> phase = 2 * np.pi * np.arange(0, 1, 0.01)
    >>> norm = np.sin(phase) + 1
    >>> np.isclose(z_n_fast_cached(norm, n=4), 50)
    True
    >>> np.isclose(z_n_fast_cached(norm, n=2), 50)
    True
    """

    cached_sin = cached_sin_harmonics(norm.size, n)
    cached_cos = cached_cos_harmonics(norm.size, n)

    return _z_n_fast_cached(norm, cached_sin, cached_cos, n=2)


@njit()
def _z_n_fast_cached_all(norm, cached_sin, cached_cos, ks):
    """Numba-compiled core of z_n_fast_cached_all"""

    total_norm = np.sum(norm)
    all_zs = np.zeros(ks.size)

    N = norm.size

    total_sum = 0
    for k in ks:
        local_z = (
            np.sum(cached_cos[: N * k : k] * norm) ** 2
            + np.sum(cached_sin[: N * k : k] * norm) ** 2
        )
        all_zs[k - 1] = local_z + total_sum
        total_sum += local_z

    result = 2 / total_norm * all_zs
    return result


def z_n_fast_cached_all(norm, nmax=20):
    """Z^2_n statistics, a` la Buccheri+03, A&A, 128, 245, eq. 2.

    Here in a fast implementation based on numba.
    Assumes that nbin != 0 and norm is an array.

    Parameters
    ----------
    norm : array of floats
        The pulse profile
    n : int, default 2
        The ``n`` in $Z^2_n$.

    Returns
    -------
    z2_ns : dict
        Dict containing the Z^2_n statistics of the events for each n in
        (1..nmax)

    Examples
    --------
    >>> phase = 2 * np.pi * np.arange(0, 1, 0.01)
    >>> norm = np.sin(phase) + 1
    >>> allks, allzs = z_n_fast_cached_all(norm, nmax=4)
    >>> allks[1]
    2
    >>> np.isclose(allzs[1], 50)
    True
    >>> np.isclose(allzs[3], 50)
    True
    """

    cached_sin = cached_sin_harmonics(norm.size, nmax)
    cached_cos = cached_cos_harmonics(norm.size, nmax)
    ks = np.arange(1, nmax + 1, dtype=int)

    all_zs = _z_n_fast_cached_all(norm, cached_sin, cached_cos, ks)

    return ks, all_zs


def h_test(norm, nmax=20):
    """H statistics, a` la de Jager+89, A&A, 221, 180, eq. 11.


    Examples
    --------
    >>> phase = 2 * np.pi * np.arange(0, 1, 0.01)
    >>> norm = np.sin(phase) + 1
    >>> h, m = h_test(norm, nmax=4)
    >>> np.isclose(h, 50)
    True
    >>> m
    1
    """
    ks, zs = z_n_fast_cached_all(norm, nmax=nmax)
    hs = zs - 4 * ks + 4
    idx = np.argmax(hs)
    return hs[idx], ks[idx]


@njit()
def roll(a, shift):
    n = a.size
    reshape = True

    if n == 0:
        return a

    shift %= n

    indexes = np.concatenate((np.arange(n - shift, n), np.arange(n - shift)))

    res = a.take(indexes)
    if reshape:
        res = res.reshape(a.shape)
    return res


@njit()
def step_pow(step):
    return 2 ** (step + 1)


@njit()
def shift(prof_n, step):
    return (prof_n % step_pow(step) + 1) // 2


@njit()
def start_value(prof_n, step):
    """

    Examples
    --------
    >>> start_value(0, 0)
    0
    >>> start_value(4, 0)
    4
    >>> start_value(5, 2)
    2
    >>> start_value(8, 1)
    8
    >>> start_value(7, 1)
    5
    >>> start_value(10, 2)
    9
    """

    n = prof_n

    val = 0
    sp = step_pow(step)
    val += n // sp * sp

    if step >= 1:
        val += (prof_n - val) // 2

    return val


@vectorize([(float64, float64), (int64, int64), (float32, float32), (int32, int32)])
def sum_arrays(arr1, arr2):
    return arr1 + arr2


def sum_rolled(arr1, arr2, out, shift):
    """Sum arr1 with a rolled version of arr2

    Examples
    --------
    >>> arr1 = np.random.random(10000)
    >>> arr2 = np.random.random(10000)
    >>> shift = np.random.randint(0, 10000)
    >>> out = sum_rolled(arr1, arr2, np.zeros_like(arr1), shift)
    >>> np.all(out == arr1 + np.roll(arr2, -shift))
    True
    """
    out[:-shift] = sum_arrays(arr1[:-shift], arr2[shift:])
    out[-shift:] = sum_arrays(arr1[-shift:], arr2[:shift])
    return out


@njit()
def ffa_step(array, step, ntables):
    array_reshaped_dum = np.copy(array)
    jump = 2**step
    # dum = np.zeros_like(array_reshaped_dum[0, :])

    for prof_n in range(ntables):
        start = start_value(prof_n, step)
        sh = shift(prof_n, step)
        jumpstart = start + jump
        if sh > 0:
            rolled = roll(array[start + jump, :], -sh)
            array_reshaped_dum[prof_n, :] = sum_arrays(array[start, :], rolled[:])

        else:
            array_reshaped_dum[prof_n, :] = sum_arrays(
                array[start, :], array[jumpstart, :]
            )

    return array_reshaped_dum


@njit()
def _ffa(array_reshaped, bin_period, ntables, z_n_n=2):
    """Fast folding algorithm search."""
    periods = np.array([bin_period + n / (ntables - 1) for n in range(ntables)])

    for step in range(0, int(np.log2(ntables))):
        array_reshaped = ffa_step(array_reshaped, step, ntables)

    twopiphases = np.pi * 2 * np.arange(0, 1, 1 / array_reshaped.shape[1])

    nbin = twopiphases.size
    cached_cos = np.zeros(z_n_n * nbin)
    cached_sin = np.zeros(z_n_n * nbin)
    for i in range(z_n_n):
        cached_cos[i * nbin : (i + 1) * nbin] = np.cos(twopiphases)
        cached_sin[i * nbin : (i + 1) * nbin] = np.sin(twopiphases)

    stats = np.zeros(ntables)
    for i in range(array_reshaped.shape[0]):
        stats[i] = _z_n_fast_cached(
            array_reshaped[i, :], cached_cos, cached_sin, n=z_n_n
        )

    return periods, stats


def ffa(array, bin_period, z_n_n=2):
    """Fast folding algorithm search"""
    N_raw = len(array)
    ntables = int(2 ** np.ceil(np.log2(N_raw // bin_period + 1)))
    if ntables <= 1:
        return np.zeros(1), np.zeros(1)
    N = ntables * bin_period
    new_arr = np.zeros(N)
    new_arr[:N_raw] = array

    array_reshaped = new_arr.reshape([ntables, bin_period])

    return _ffa(array_reshaped, bin_period, ntables, z_n_n=z_n_n)


def _quick_rebin(counts, current_rebin):
    """

    Examples
    --------
    >>> counts = np.arange(1, 11)
    >>> reb = _quick_rebin(counts, 2)
    >>> np.allclose(reb, [3, 7, 11, 15, 19])
    True
    """
    n = int(counts.size // current_rebin)
    rebinned_counts = np.sum(
        counts[: n * current_rebin].reshape(n, current_rebin), axis=1
    )
    return rebinned_counts


def ffa_search(counts, dt, period_min, period_max):
    counts = np.array(counts)
    pmin = np.floor(period_min / dt)
    pmax = np.ceil(period_max / dt)
    bin_periods = None
    stats = None

    current_rebin = 1
    rebinned_counts = counts
    for bin_period in show_progress(np.arange(pmin, pmax + 1, dtype=int)):
        # Only powers of two
        rebin = int(2 ** np.floor(np.log2(bin_period / pmin)))
        if rebin > current_rebin:
            current_rebin = rebin
            rebinned_counts = _quick_rebin(counts, current_rebin)

        # When rebinning, bin_period // current_rebin is the same for nearby
        # periods
        if bin_period % current_rebin != 0:
            continue

        per, st = ffa(rebinned_counts, bin_period // current_rebin)

        per *= current_rebin

        if per[0] == 0:
            continue
        elif bin_periods is None:
            bin_periods = per[:-1] * dt
            stats = st[:-1]
        else:
            bin_periods = np.concatenate((bin_periods, per[:-1] * dt))
            stats = np.concatenate((stats, st[:-1]))

    return bin_periods, stats
