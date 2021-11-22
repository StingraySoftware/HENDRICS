import copy
import os
from functools import wraps
from typing import Iterable
import numpy as np
from astropy.table import Table
from astropy import log
import stingray
from scipy import stats
from collections.abc import Iterable as iterable

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


class EventList(stingray.EventList):
    def array_attrs(self):
        return [
            attr
            for attr in dir(self)
            if (
                isinstance(getattr(self, attr), Iterable)
                and np.shape(getattr(self, attr)) == self.time.shape
            )
        ]

    def apply_mask(self, mask, inplace=False):
        """For compatibility with old stingray version.
        Examples
        --------
        >>> evt = EventList(time=[0, 1, 2], mission="nustar")
        >>> evt.bubuattr = [222, 111, 333]
        >>> newev0 = evt.apply_mask([True, True, False], inplace=False);
        >>> newev1 = evt.apply_mask([True, True, False], inplace=True);
        >>> newev0.mission == "nustar"
        True
        >>> np.allclose(newev0.time, [0, 1])
        True
        >>> np.allclose(newev0.bubuattr, [222, 111])
        True
        >>> np.allclose(newev1.time, [0, 1])
        True
        >>> evt is newev1
        True
        """
        array_attrs = self.array_attrs()

        if inplace:
            new_ev = self
        else:
            new_ev = EventList()
            for attr in dir(self):
                if not attr.startswith("_") and attr not in array_attrs:
                    setattr(new_ev, attr, getattr(self, attr))

        for attr in array_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(new_ev, attr, np.asarray(getattr(self, attr))[mask])
        return new_ev

    def to_astropy_timeseries(self):
        from astropy.timeseries import TimeSeries
        from astropy.time import TimeDelta
        from astropy import units as u

        data = {}
        array_attrs = self.array_attrs()

        for attr in array_attrs:
            if attr == "time":
                continue
            data[attr] = np.asarray(getattr(self, attr))

        if data == {}:
            data = None

        if self.time is not None and self.time.size > 0:
            times = TimeDelta(self.time * u.s)
            ts = TimeSeries(data=data, time=times)
        else:
            ts = TimeSeries()
        ts.meta["gti"] = self.gti
        ts.meta["mjdref"] = self.mjdref
        ts.meta["instr"] = self.instr
        ts.meta["mission"] = self.mission
        ts.meta["header"] = self.header
        return ts

    @staticmethod
    def from_astropy_timeseries(ts):
        from astropy.timeseries import TimeSeries
        from astropy import units as u

        kwargs = dict([(key.lower(), val) for (key, val) in ts.meta.items()])
        ev = EventList(time=ts.time, **kwargs)
        array_attrs = ts.colnames

        for attr in array_attrs:
            if attr == "time":
                continue
            setattr(ev, attr, ts[attr])

        return ev

    def to_astropy_table(self):
        data = {}
        array_attrs = self.array_attrs()

        for attr in array_attrs:
            data[attr] = np.asarray(getattr(self, attr))

        ts = Table(data)

        ts.meta["gti"] = self.gti
        ts.meta["mjdref"] = self.mjdref
        ts.meta["instr"] = self.instr
        ts.meta["mission"] = self.mission
        ts.meta["header"] = self.header
        return ts

    @staticmethod
    def from_astropy_table(ts):
        kwargs = dict([(key.lower(), val) for (key, val) in ts.meta.items()])
        ev = EventList(time=ts["time"], **kwargs)
        array_attrs = ts.colnames

        for attr in array_attrs:
            if attr == "time":
                continue
            setattr(ev, attr, ts[attr])

        return ev


def power_confidence_limits(preal, n=1, c=0.95):
    """Confidence limits on power, given a (theoretical) signal power.

    This is to be used when we *expect* a given power (e.g. from the pulsed
    fraction measured in previous observations) and we want to know the
    range of values the measured power could take to a given confidence level.
    Adapted from Vaughan et al. 1994, noting that, after appropriate
    normalization of the spectral stats, the distribution of powers in the PDS
    and the Z^2_n searches is always described by a noncentral chi squared
    distribution.

    Parameters
    ----------
    preal: float
        The theoretical signal-generated value of power

    Other Parameters
    ----------------
    n: int
        The number of summed powers to obtain the result. It can be multiple
        harmonics of the PDS, adjacent bins in a PDS summed to collect all the
        power in a QPO, or the n in Z^2_n
    c: float
        The confidence level (e.g. 0.95=95%)

    Results
    -------
    pmeas: [float, float]
        The upper and lower confidence interval (a, 1-a) on the measured power

    Examples
    --------
    >>> cl = power_confidence_limits(150, c=0.84)
    >>> np.allclose(cl, [127, 176], atol=1)
    True
    """
    rv = stats.ncx2(2 * n, preal)
    return rv.ppf([1 - c, c])


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


def amplitude_upper_limit(
    pmeas, counts, n=1, c=0.95, fft_corr=False, nyq_ratio=0
):
    """Upper limit on a sinusoidal modulation, given a measured power in the PDS/Z search.

    Eq. 10 in Vaughan+94 and `a_from_ssig`: they are equivalent but Vaughan+94
    corrects further for the response inside an FFT bin and at frequencies close
    to Nyquist. These two corrections are added by using fft_corr=True and
    nyq_ratio to the correct :math:`f / f_{Nyq}` of the FFT peak

    To understand the meaning of this amplitude: if the modulation is described by:

    ..math:: p = \overline{p} (1 + a * \sin(x))

    this function returns a.

    If it is a sum of sinusoidal harmonics instead
    ..math:: p = \overline{p} (1 + \sum_l a_l * \sin(lx))
    a is equivalent to :math:`\sqrt(\sum_l a_l^2)`.

    See `power_upper_limit`

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
    fft_corr: bool
        Apply a correction for the expected power concentrated in an FFT bin,
        which is about 0.773 on average (it's 1 at the center of the bin, 2/pi
        at the bin edge.
    nyq_ratio: float
        Ratio of the frequency of this feature with respect to the Nyquist
        frequency. Important to know when dealing with FFTs, because the FFT
        response decays between 0 and f_Nyq similarly to the response inside
        a frequency bin: from 1 at 0 Hz to ~2/pi at f_Nyq

    Results
    -------
    a: float
        The modulation amplitude that could produce P>pmeas with 1 - c probability

    Examples
    --------
    >>> aup = amplitude_upper_limit(40, 30000, 1, 0.99)
    >>> aup_nyq = amplitude_upper_limit(40, 30000, 1, 0.99, nyq_ratio=1)
    >>> np.isclose(aup_nyq, aup / (2 / np.pi))
    True
    >>> aup_corr = amplitude_upper_limit(40, 30000, 1, 0.99, fft_corr=True)
    >>> np.isclose(aup_corr, aup / np.sqrt(0.773))
    True
    """

    uplim = power_upper_limit(pmeas, n, c)
    a = a_from_ssig(uplim, counts)
    if fft_corr:
        factor = 1 / np.sqrt(0.773)
        a *= factor
    if nyq_ratio > 0:
        factor = np.pi / 2 * nyq_ratio
        sinc_factor = np.sin(factor) / factor
        a /= sinc_factor
    return a


def pf_upper_limit(*args, **kwargs):
    """Upper limit on pulsed fraction, given a measured power in the PDS/Z search.

    See `power_upper_limit` and `pf_from_ssig`.
    All arguments are the same as `amplitude_upper_limit`

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
    fft_corr: bool
        Apply a correction for the expected power concentrated in an FFT bin,
        which is about 0.773 on average (it's 1 at the center of the bin, 2/pi
        at the bin edge.
    nyq_ratio: float
        Ratio of the frequency of this feature with respect to the Nyquist
        frequency. Important to know when dealing with FFTs, because the FFT
        response decays between 0 and f_Nyq similarly to the response inside
        a frequency bin: from 1 at 0 Hz to ~2/pi at f_Nyq

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

    return pf_from_a(amplitude_upper_limit(*args, **kwargs))


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
