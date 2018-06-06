import numpy as np

from .base import jit, vectorize, HAS_NUMBA

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
+ The starting number is obtained as prof_n // step_pow * step_pow + (prof_n - prof_n // step_pow) // 2
"""

@jit(nopython=True)
def stat(profile):
    """Calculate the epoch folding statistics \'a la Leahy et al. (1983).

    Parameters
    ----------
    profile : array
        The pulse profile

    Other Parameters
    ----------------
    err : float or array
        The uncertainties on the pulse profile

    Returns
    -------
    stat : float
        The epoch folding statistics
    """
    mean = np.mean(profile)
    err = np.sqrt(mean)
    sum = 0
    for p in profile:
        sum += (p - mean) ** 2
    return sum / err ** 2


if HAS_NUMBA:
    from numba import types
    from numba.extending import overload_method

    @overload_method(types.Array, 'take')
    def array_take(arr, indices):
        if isinstance(indices, types.Array):
            def take_impl(arr, indices):
                n = indices.shape[0]
                res = np.empty(n, arr.dtype)
                for i in range(n):
                    res[i] = arr[indices[i]]
                return res

            return take_impl
else:
    array_take = np.take


@jit(nopython=True)
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


@jit(nopython=True)
def step_pow(step):
    return 2 ** (step + 1)


@jit(nopython=True)
def jump(step):
    return 2 ** step


@jit(nopython=True)
def shift(prof_n, step):
    return (prof_n % step_pow(step) + 1) // 2


@jit(nopython=True)
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


@vectorize
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
    out[:-shift] = \
        sum_arrays(arr1[:-shift], arr2[shift:])
    out[-shift:] = \
        sum_arrays(arr1[-shift:], arr2[:shift])
    return out


@jit(nopython=True)
def ffa_step(array, step, ntables):
    array_reshaped_dum = np.copy(array)
    jump = 2 ** step
    # dum = np.zeros_like(array_reshaped_dum[0, :])

    for prof_n in range(ntables):
        start = start_value(prof_n, step)
        sh = shift(prof_n, step)
        jumpstart = start + jump
        if sh > 0:
            # FOR SOME REASON THIS DOESN'T WORK. DAMN
            # array_reshaped_dum[prof_n, :] = \
            #     sum_rolled(array[start, :], array[jumpstart, :], dum, sh)

            rolled = roll(array[start + jump, :], -sh)
            array_reshaped_dum[prof_n, :] = \
                sum_arrays(array[start, :], rolled[:])

        else:
            array_reshaped_dum[prof_n, :] = \
                sum_arrays(array[start, :], array[jumpstart, :])

    return array_reshaped_dum


def ffa_step_old(array, step, ntables):
    array_reshaped_dum = np.copy(array)
    jump = 2 ** step
    for chunk_no in range(ntables // 2):
        for half in [0, 1]:
            prof_n = 2 * chunk_no + half
            start = start_value(prof_n, step)
            sh = shift(prof_n, step)
            if sh == 0:
                array_reshaped_dum[prof_n, :] = \
                    sum_arrays(array[start, :], array[start + jump, :])
            else:
                rolled = roll(array[start + jump, :], -sh)
                array_reshaped_dum[prof_n, :] = \
                    sum_arrays(array[start, :], rolled[:])

    return array_reshaped_dum


@jit(nopython=True)
def ffa(array, bin_period, nave=1):
    """Fast folding algorithm search
    """
    N_raw = len(array)
    ntables = np.int(2**np.ceil(np.log2(N_raw // bin_period + 1)))
    if ntables <= 1:
        return np.zeros(1), np.zeros(1)
    N = ntables * bin_period
    new_arr = np.zeros(N)
    new_arr[:N_raw] = array
    nave = int(2**np.rint(np.log2(nave)))

    if nave > 1:
        array_reshaped = \
            new_arr.reshape(ntables // nave, nave, bin_period).sum(axis=1)
        # array_reshaped = array_reshaped
        ntables = ntables // nave
    else:
        array_reshaped = new_arr.reshape((ntables, bin_period))

    periods = \
        np.array([bin_period + n / (ntables - 1) for n in range(ntables)])

    for step in range(0, np.int(np.log2(ntables))):
        array_reshaped = ffa_step(array_reshaped, step, ntables)

    stats = np.zeros(ntables)
    for i in range(array_reshaped.shape[0]):
        stats[i] = stat(array_reshaped[i, :])

    # Here I'm subtracting the degrees of freedom from stats to flatten the
    # periodogram
    return periods, stats - bin_period + 1


def ffa_search(counts, dt, period_min, period_max, nave=1):
    pmin = np.floor(period_min / dt)
    pmax = np.ceil(period_max / dt)
    bin_periods = None
    stats = None

    for bin_period in np.arange(pmin, pmax + 1, dtype=int):
        per, st = ffa(counts, bin_period, nave=nave)
        if per[0] == 0:
            continue
        elif bin_periods is None:
            bin_periods = per
            stats = st
        else:
            bin_periods = np.concatenate((bin_periods, per))
            stats = np.concatenate((stats, st))

    return bin_periods * dt, stats
