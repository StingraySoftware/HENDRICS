# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A miscellaneous collection of basic functions."""

import sys
import copy
import os
import warnings
from functools import wraps
from collections.abc import Iterable
from pathlib import Path
import tempfile

import numpy as np
from numpy import histogram2d as histogram2d_np
from numpy import histogram as histogram_np
from astropy.logger import AstropyUserWarning
from astropy import log

from stingray.pulse.pulsar import get_orbital_correction_from_ephemeris_file

try:
    from skimage.feature import peak_local_max

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from numba import jit, njit, prange, vectorize
    from numba import float32, float64, int32, int64
    from numba import types
    from numba.extending import overload_method

    @overload_method(types.Array, "take")  # pragma: no cover
    def array_take(arr, indices):
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

    array_take = np.take


try:
    from tqdm import tqdm as show_progress
except ImportError:

    def show_progress(a):
        return a


try:
    from stingray.stats import pds_probability, pds_detection_level
    from stingray.stats import z2_n_detection_level, z2_n_probability
    from stingray.stats import fold_detection_level, fold_profile_probability
except ImportError:
    from stingray.pulse.pulsar import fold_detection_level
    from stingray.pulse.pulsar import fold_profile_probability

    def z2_n_detection_level(n=2, epsilon=0.01, n_summed_spectra=1, ntrial=1):
        """Return the detection level for the Z^2_n statistics.

        See Buccheri et al. (1983), Bendat and Piersol (1971).

        Parameters
        ----------
        n : int, default 2
            The ``n`` in $Z^2_n$
        epsilon : float, default 0.01
            The fractional probability that the signal has been produced by noise

        Other Parameters
        ----------------
        ntrial : int
            The number of trials executed to find this profile
        n_summed_spectra : int
            Number of Z_2^n periodograms that are being averaged

        Returns
        -------
        detlev : float
            The epoch folding statistics corresponding to a probability
            epsilon * 100 % that the signal has been produced by noise

        Examples
        --------
        >>> np.isclose(z2_n_detection_level(2), 13.276704135987625)
        True
        """

        from scipy import stats

        retlev = stats.chi2.isf(
            epsilon / ntrial, 2 * int(n_summed_spectra) * n
        ) / (n_summed_spectra)

        return retlev

    def z2_n_probability(z2, n=2, ntrial=1, n_summed_spectra=1):
        """Calculate the probability of a certain folded profile, due to noise.

        Parameters
        ----------
        z2 : float
            A Z^2_n statistics value
        n : int, default 2
            The ``n`` in $Z^2_n$

        Other Parameters
        ----------------
        ntrial : int
            The number of trials executed to find this profile
        n_summed_spectra : int
            Number of Z_2^n periodograms that were averaged to obtain z2

        Returns
        -------
        p : float
            The probability that the Z^2_n value has been produced by noise

        Examples
        --------
        >>> detlev = z2_n_detection_level(2, 0.1)
        >>> np.isclose(z2_n_probability(detlev, 2), 0.1)
        True
        """
        if ntrial > 1:
            import warnings

            warnings.warn(
                "Z2_n: The treatment of ntrial is very rough. "
                "Use with caution",
                AstropyUserWarning,
            )
        from scipy import stats

        epsilon = ntrial * stats.chi2.sf(
            z2 * n_summed_spectra, 2 * n * n_summed_spectra
        )
        return epsilon

    def pds_probability(level, ntrial=1, n_summed_spectra=1, n_rebin=1):
        r"""Give the probability of a given power level in PDS.

        Return the probability of a certain power level in a Power Density
        Spectrum of nbins bins, normalized a la Leahy (1983), based on
        the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
        and multiple PDS averaging (n_summed_spectra)

        Parameters
        ----------
        level : float or array of floats
            The power level for which we are calculating the probability

        Other Parameters
        ----------------
        ntrial : int
            The number of *independent* trials (the independent bins of the PDS)
        n_summed_spectra : int
            The number of power density spectra that have been averaged to obtain
            this power level
        n_rebin : int
            The number of power density bins that have been averaged to obtain
            this power level

        Returns
        -------
        epsilon : float
            The probability value(s)
        """
        from scipy import stats

        epsilon_1 = stats.chi2.sf(
            level * n_summed_spectra * n_rebin, 2 * n_summed_spectra * n_rebin
        )

        epsilon = epsilon_1 * ntrial
        return epsilon

    def pds_detection_level(
        epsilon=0.01, ntrial=1, n_summed_spectra=1, n_rebin=1
    ):
        r"""Detection level for a PDS.

        Return the detection level (with probability 1 - epsilon) for a Power
        Density Spectrum of nbins bins, normalized a la Leahy (1983), based on
        the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
        and multiple PDS averaging (n_summed_spectra)

        Parameters
        ----------
        epsilon : float
            The single-trial probability value(s)

        Other Parameters
        ----------------
        ntrial : int
            The number of *independent* trials (the independent bins of the PDS)
        n_summed_spectra : int
            The number of power density spectra that have been averaged to obtain
            this power level
        n_rebin : int
            The number of power density bins that have been averaged to obtain
            this power level

        Examples
        --------
        >>> np.isclose(pds_detection_level(0.1), 4.6, atol=0.1)
        True
        >>> np.allclose(pds_detection_level(0.1, n_rebin=[1]), [4.6], atol=0.1)
        True
        """
        from scipy import stats

        epsilon = epsilon / ntrial
        if isinstance(n_rebin, Iterable):
            retlev = [
                stats.chi2.isf(epsilon, 2 * n_summed_spectra * r)
                / (n_summed_spectra * r)
                for r in n_rebin
            ]
            retlev = np.array(retlev)
        else:
            r = n_rebin
            retlev = stats.chi2.isf(epsilon, 2 * n_summed_spectra * r) / (
                n_summed_spectra * r
            )
        return retlev


__all__ = [
    "array_take",
    "njit",
    "prange",
    "show_progress",
    "z2_n_detection_level",
    "z2_n_probability",
    "pds_detection_level",
    "pds_probability",
    "fold_detection_level",
    "fold_profile_probability",
    "r_in",
    "r_det",
    "_assign_value_if_none",
    "_look_for_array_in_array",
    "is_string",
    "_order_list_of_arrays",
    "mkdir_p",
    "common_name",
    "hen_root",
    "optimal_bin_time",
    "gti_len",
    "deorbit_events",
    "_add_default_args",
    "check_negative_numbers_in_args",
    "interpret_bintime",
    "get_bin_edges",
    "compute_bin",
    "hist1d_numba_seq",
    "hist2d_numba_seq",
    "hist3d_numba_seq",
    "hist2d_numba_seq_weight",
    "hist3d_numba_seq_weight",
    "index_arr",
    "index_set_arr",
    "histnd_numba_seq",
    "histogram2d",
    "histogram",
    "touch",
    "log_x",
    "get_list_of_small_powers",
    "adjust_dt_for_power_of_two",
    "adjust_dt_for_small_power",
    "memmapped_arange",
    "nchars_in_int_value",
]


DEFAULT_PARSER_ARGS = {}
DEFAULT_PARSER_ARGS["loglevel"] = dict(
    args=["--loglevel"],
    kwargs=dict(
        help=(
            "use given logging level (one between INFO, "
            "WARNING, ERROR, CRITICAL, DEBUG; "
            "default:WARNING)"
        ),
        default="WARNING",
        type=str,
    ),
)
DEFAULT_PARSER_ARGS["nproc"] = dict(
    args=["--nproc"],
    kwargs=dict(help=("Number of processors to use"), default=1, type=int),
)
DEFAULT_PARSER_ARGS["debug"] = dict(
    args=["--debug"],
    kwargs=dict(
        help=("set DEBUG logging level"), default=False, action="store_true"
    ),
)
DEFAULT_PARSER_ARGS["bintime"] = dict(
    args=["-b", "--bintime"],
    kwargs=dict(help="Bin time", type=np.longdouble, default=1),
)
DEFAULT_PARSER_ARGS["energies"] = dict(
    args=["-e", "--energy-interval"],
    kwargs=dict(
        help="Energy interval used for filtering",
        nargs=2,
        type=float,
        default=None,
    ),
)
DEFAULT_PARSER_ARGS["pi"] = dict(
    args=["--pi-interval"],
    kwargs=dict(
        help="PI interval used for filtering",
        nargs=2,
        type=int,
        default=[-1, -1],
    ),
)
DEFAULT_PARSER_ARGS["deorbit"] = dict(
    args=["-p", "--deorbit-par"],
    kwargs=dict(
        help=(
            "Deorbit data with this parameter file (requires PINT installed)"
        ),
        default=None,
        type=str,
    ),
)
DEFAULT_PARSER_ARGS["output"] = dict(
    args=["-o", "--outfile"],
    kwargs=dict(help="Output file", default=None, type=str),
)
DEFAULT_PARSER_ARGS["usepi"] = dict(
    args=["--use-pi"],
    kwargs=dict(
        help="Use the PI channel instead of energies",
        default=False,
        action="store_true",
    ),
)
DEFAULT_PARSER_ARGS["test"] = dict(
    args=["--test"],
    kwargs=dict(
        help="Only used for tests", default=False, action="store_true"
    ),
)
DEFAULT_PARSER_ARGS["pepoch"] = dict(
    args=["--pepoch"],
    kwargs=dict(
        type=float,
        required=False,
        help="Reference epoch for timing parameters (MJD)",
        default=None,
    ),
)


def r_in(td, r_0):
    """Calculate incident countrate given dead time and detected countrate."""
    tau = 1 / r_0
    return 1.0 / (tau - td)


def r_det(td, r_i):
    """Calculate detected countrate given dead time and incident countrate."""
    tau = 1 / r_i
    return 1.0 / (tau + td)


def _assign_value_if_none(value, default):
    if value is None:
        return default
    return value


def _look_for_array_in_array(array1, array2):
    """
    Examples
    --------
    >>> _look_for_array_in_array([1, 2], [2, 3, 4])
    2
    >>> _look_for_array_in_array([1, 2], [3, 4, 5]) is None
    True
    """
    for a1 in array1:
        if a1 in array2:
            return a1
    return None


def is_string(s):
    """Portable function to answer this question."""
    return isinstance(s, str)  # NOQA


def _order_list_of_arrays(data, order):
    """
    Examples
    --------
    >>> order = [1, 2, 0]
    >>> new = _order_list_of_arrays({'a': [4, 5, 6], 'b':[7, 8, 9]}, order)
    >>> np.all(new['a'] == [5, 6, 4])
    True
    >>> np.all(new['b'] == [8, 9, 7])
    True
    >>> new = _order_list_of_arrays([[4, 5, 6], [7, 8, 9]], order)
    >>> np.all(new[0] == [5, 6, 4])
    True
    >>> np.all(new[1] == [8, 9, 7])
    True
    >>> _order_list_of_arrays(2, order) is None
    True
    """
    if hasattr(data, "items"):
        data = dict((i[0], np.asarray(i[1])[order]) for i in data.items())
    elif hasattr(data, "index"):
        data = [np.asarray(i)[order] for i in data]
    else:
        data = None
    return data


class _empty:
    def __init__(self):
        pass


def mkdir_p(path):
    """Safe mkdir function."""
    return os.makedirs(path, exist_ok=True)


def common_name(str1, str2, default="common"):
    """Strip two strings of the letters not in common.

    Filenames must be of same length and only differ by a few letters.

    Parameters
    ----------
    str1 : str
    str2 : str

    Returns
    -------
    common_str : str
        A string containing the parts of the two names in common

    Other Parameters
    ----------------
    default : str
        The string to return if common_str is empty

    Examples
    --------
    >>> common_name('strAfpma', 'strBfpmb')
    'strfpm'
    >>> common_name('strAfpma', 'strBfpmba')
    'common'
    >>> common_name('asdfg', 'qwerr')
    'common'
    >>> common_name('A_3-50_A.nc', 'B_3-50_B.nc')
    '3-50'
    """
    if not len(str1) == len(str2):
        return default
    common_str = ""
    # Extract the HEN root of the name (in case they're event files)
    str1 = hen_root(str1)
    str2 = hen_root(str2)
    for i, letter in enumerate(str1):
        if str2[i] == letter:
            common_str += letter
    # Remove leading and trailing underscores and dashes
    common_str = common_str.rstrip("_").rstrip("-")
    common_str = common_str.lstrip("_").lstrip("-")
    if common_str == "":
        common_str = default
    # log.debug('common_name: %s %s -> %s', str1, str2, common_str)
    return common_str


def hen_root(filename):
    """Return the root file name (without _ev, _lc, etc.).

    Parameters
    ----------
    filename : str
    """
    fname = filename.replace(".gz", "")
    fname = os.path.splitext(fname)[0]
    fname = fname.replace("_ev", "").replace("_lc", "")
    fname = fname.replace("_calib", "")
    return fname


def optimal_bin_time(fftlen, tbin):
    """Vary slightly the bin time to have a power of two number of bins.

    Given an FFT length and a proposed bin time, return a bin time
    slightly shorter than the original, that will produce a power-of-two number
    of FFT bins.

    Examples
    --------
    >>> optimal_bin_time(512, 1.1)
    1.0
    """
    current_nbin = fftlen / tbin
    new_nbin = 2 ** np.ceil(np.log2(current_nbin))
    return fftlen / new_nbin


def gti_len(gti):
    """Return the total good time from a list of GTIs.

    Examples
    --------
    >>> gti_len([[0, 1], [2, 4]])
    3
    """
    return np.sum(np.diff(gti, axis=1))


def deorbit_events(events, parameter_file=None):
    """Refer arrival times to the center of mass of binary system.

    Parameters
    ----------
    events : `stingray.events.EventList` object
        The event list
    parameter_file : str
        The parameter file, in Tempo-compatible format, containing
        the orbital solution (e.g. a BT model)
    """
    events = copy.deepcopy(events)
    if parameter_file is None:
        warnings.warn(
            "No parameter file specified for deorbit. Returning"
            " unaltered event list"
        )
        return events
    if not os.path.exists(parameter_file):
        raise FileNotFoundError(
            "Parameter file {} does not exist".format(parameter_file)
        )

    if events.mjdref < 33282.0:
        raise ValueError(
            "MJDREF is very low (<01-01-1950), " "this is unsupported."
        )

    pepoch = events.gti[0, 0]
    pepoch_mjd = pepoch / 86400 + events.mjdref

    length = np.max(events.time) - np.min(events.time)
    if length > 200000:
        warnings.warn(
            "The observation is very long. The barycentric correction "
            "will be rough"
        )

    length_d = length / 86400
    results = get_orbital_correction_from_ephemeris_file(
        pepoch_mjd - 1,
        pepoch_mjd + length_d + 1,
        parameter_file,
        ntimes=min(int(length // 10), 10000),
    )
    orbital_correction_fun = results[0]
    events.time = orbital_correction_fun(events.time, mjdref=events.mjdref)
    events.gti = orbital_correction_fun(events.gti, mjdref=events.mjdref)
    return events


def _add_default_args(parser, list_of_args):
    for key in list_of_args:
        arg = DEFAULT_PARSER_ARGS[key]
        a = arg["args"]
        k = arg["kwargs"]
        parser.add_argument(*a, **k)

    return parser


def check_negative_numbers_in_args(args):
    """If there are negative numbers in args, prepend a space.

    Examples
    --------
    >>> args = ['events.nc', '-f', '103', '--fdot', '-2e-10']
    >>> newargs = check_negative_numbers_in_args(args)
    >>> args[:4] == newargs[:4]
    True
    >>> newargs[4] == ' -2e-10'
    True
    """
    if args is None:
        args = sys.argv[1:]
    newargs = []
    for arg in args:
        try:
            # Has to be a number, has to be negative
            assert -float(arg) > 0
        except (ValueError, AssertionError):
            newargs.append(arg)
            continue

        newargs.append(" " + arg)

    return newargs


def interpret_bintime(bintime):
    """If bin time is negative, interpret as power of two.

    Examples
    --------
    >>> interpret_bintime(2)
    2
    >>> interpret_bintime(-2) == 0.25
    True
    >>> interpret_bintime(0)
    Traceback (most recent call last):
        ...
    ValueError: Bin time cannot be = 0
    """
    if bintime < 0:
        return 2 ** bintime
    elif bintime > 0:
        return bintime
    raise ValueError("Bin time cannot be = 0")


@njit(nogil=True, parallel=False)
def _get_bin_edges(a, bins, a_min, a_max):
    bin_edges = np.zeros(bins + 1, dtype=np.float64)

    delta = (a_max - a_min) / bins
    for i in range(bin_edges.size):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


def get_bin_edges(a, bins):
    """

    Examples
    --------
    >>> array = np.array([0., 10.])
    >>> bins = 2
    >>> np.allclose(get_bin_edges(array, bins), [0, 5, 10])
    True
    """
    a_min = np.min(a)
    a_max = np.max(a)
    return _get_bin_edges(a, bins, a_min, a_max)


@njit(nogil=True, parallel=False)
def compute_bin(x, bin_edges):
    """

    Examples
    --------
    >>> bin_edges = np.array([0, 5, 10])
    >>> compute_bin(1, bin_edges)
    0
    >>> compute_bin(5, bin_edges)
    1
    >>> compute_bin(10, bin_edges)
    1
    """

    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1  # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@njit(nogil=True, parallel=False)
def _hist1d_numba_seq(H, tracks, bins, ranges):
    delta = 1 / ((ranges[1] - ranges[0]) / bins)

    for t in range(tracks.size):
        i = (tracks[t] - ranges[0]) * delta
        if 0 <= i < bins:
            H[int(i)] += 1

    return H


def hist1d_numba_seq(a, bins, ranges, use_memmap=False, tmp=None):
    """
    Examples
    --------
    >>> if os.path.exists('out.npy'): os.unlink('out.npy')
    >>> x = np.random.uniform(0., 1., 100)
    >>> H, xedges = np.histogram(x, bins=5, range=[0., 1.])
    >>> Hn = hist1d_numba_seq(x, bins=5, ranges=[0., 1.], tmp='out.npy',
    ...                       use_memmap=True)
    >>> assert np.all(H == Hn)
    >>> # The number of bins is small, memory map was not used!
    >>> assert not os.path.exists('out.npy')
    >>> H, xedges = np.histogram(x, bins=10**8, range=[0., 1.])
    >>> Hn = hist1d_numba_seq(x, bins=10**8, ranges=[0., 1.], tmp='out.npy',
    ...                       use_memmap=True)
    >>> assert np.all(H == Hn)
    >>> assert os.path.exists('out.npy')
    """
    if bins > 10 ** 7 and use_memmap:
        if tmp is None:
            tmp = tempfile.NamedTemporaryFile("w+")
        hist_arr = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=a.dtype, shape=(bins,)
        )
    else:
        hist_arr = np.zeros((bins,), dtype=a.dtype)

    return _hist1d_numba_seq(hist_arr, a, bins, np.asarray(ranges))


@njit(nogil=True, parallel=False)
def _hist2d_numba_seq(H, tracks, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1

    return H


def hist2d_numba_seq(x, y, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(5, 5),
    ...                                    range=[(0., 1.), (2., 3.)])
    >>> Hn = hist2d_numba_seq(x, y, bins=(5, 5),
    ...                       ranges=[[0., 1.], [2., 3.]])
    >>> assert np.all(H == Hn)
    """
    H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
    return _hist2d_numba_seq(
        H, np.array([x, y]), np.asarray(list(bins)), np.asarray(ranges)
    )


@njit(nogil=True, parallel=False)
def _hist3d_numba_seq(H, tracks, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        k = (tracks[2, t] - ranges[2, 0]) * delta[2]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j), int(k)] += 1

    return H


def hist3d_numba_seq(tracks, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> z = np.random.uniform(4., 5., 100)
    >>> H, _ = np.histogramdd((x, y, z), bins=(5, 6, 7),
    ...                       range=[(0., 1.), (2., 3.), (4., 5)])
    >>> Hn = hist3d_numba_seq((x, y, z), bins=(5, 6, 7),
    ...                       ranges=[[0., 1.], [2., 3.], [4., 5.]])
    >>> assert np.all(H == Hn)
    """

    H = np.zeros((bins[0], bins[1], bins[2]), dtype=np.uint64)
    return _hist3d_numba_seq(
        H, np.asarray(tracks), np.asarray(list(bins)), np.asarray(ranges)
    )


@njit(nogil=True, parallel=False)
def _hist2d_numba_seq_weight(H, tracks, weights, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += weights[t]

    return H


def hist2d_numba_seq_weight(x, y, weights, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> weight = np.random.uniform(0, 1, 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(5, 5),
    ...                                    range=[(0., 1.), (2., 3.)],
    ...                                    weights=weight)
    >>> Hn = hist2d_numba_seq_weight(x, y, bins=(5, 5),
    ...                              ranges=[[0., 1.], [2., 3.]],
    ...                              weights=weight)
    >>> assert np.all(H == Hn)
    """
    H = np.zeros((bins[0], bins[1]), dtype=np.double)
    return _hist2d_numba_seq_weight(
        H,
        np.array([x, y]),
        weights,
        np.asarray(list(bins)),
        np.asarray(ranges),
    )


@njit(nogil=True, parallel=False)
def _hist3d_numba_seq_weight(H, tracks, weights, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        k = (tracks[2, t] - ranges[2, 0]) * delta[2]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j), int(k)] += weights[t]

    return H


def hist3d_numba_seq_weight(tracks, weights, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> z = np.random.uniform(4., 5., 100)
    >>> weights = np.random.uniform(0, 1., 100)
    >>> H, _ = np.histogramdd((x, y, z), bins=(5, 6, 7),
    ...                       range=[(0., 1.), (2., 3.), (4., 5)],
    ...                       weights=weights)
    >>> Hn = hist3d_numba_seq_weight(
    ...    (x, y, z), weights, bins=(5, 6, 7),
    ...    ranges=[[0., 1.], [2., 3.], [4., 5.]])
    >>> assert np.all(H == Hn)
    """

    H = np.zeros((bins[0], bins[1], bins[2]), dtype=np.double)
    return _hist3d_numba_seq_weight(
        H,
        np.asarray(tracks),
        weights,
        np.asarray(list(bins)),
        np.asarray(ranges),
    )


@njit(nogil=True, parallel=False)
def index_arr(a, ix_arr):
    strides = np.array(a.strides) / a.itemsize
    ix = int((ix_arr * strides).sum())
    return a.ravel()[ix]


@njit(nogil=True, parallel=False)
def index_set_arr(a, ix_arr, val):
    strides = np.array(a.strides) / a.itemsize
    ix = int((ix_arr * strides).sum())
    a.ravel()[ix] = val


@njit(nogil=True, parallel=False)
def _histnd_numba_seq(H, tracks, bins, ranges, slice_int):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        slicearr = np.array(
            [
                (tracks[dim, t] - ranges[dim, 0]) * delta[dim]
                for dim in range(tracks.shape[0])
            ]
        )

        good = np.all((slicearr < bins) & (slicearr >= 0))
        slice_int[:] = slicearr

        if good:
            curr = index_arr(H, slice_int)
            index_set_arr(H, slice_int, curr + 1)

    return H


def histnd_numba_seq(tracks, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> z = np.random.uniform(4., 5., 100)
    >>> # 2d example
    >>> H, _, _ = np.histogram2d(x, y, bins=np.array((5, 5)),
    ...                          range=[(0., 1.), (2., 3.)])
    >>> alldata = np.array([x, y])
    >>> Hn = histnd_numba_seq(alldata, bins=np.array([5, 5]),
    ...                       ranges=np.array([[0., 1.], [2., 3.]]))
    >>> assert np.all(H == Hn)
    >>> # 3d example
    >>> H, _ = np.histogramdd((x, y, z), bins=np.array((5, 6, 7)),
    ...                       range=[(0., 1.), (2., 3.), (4., 5)])
    >>> alldata = np.array([x, y, z])
    >>> Hn = hist3d_numba_seq(alldata, bins=np.array((5, 6, 7)),
    ...                       ranges=np.array([[0., 1.], [2., 3.], [4., 5.]]))
    >>> assert np.all(H == Hn)
    """
    H = np.zeros(tuple(bins), dtype=np.uint64)
    slice_int = np.zeros(len(bins), dtype=np.uint64)

    return _histnd_numba_seq(H, tracks, bins, ranges, slice_int)


if HAS_NUMBA:

    def histogram2d(*args, **kwargs):
        if "range" in kwargs:
            kwargs["ranges"] = kwargs.pop("range")
        return hist2d_numba_seq(*args, **kwargs)

    def histogram(*args, **kwargs):
        if "range" in kwargs:
            kwargs["ranges"] = kwargs.pop("range")
        return hist1d_numba_seq(*args, **kwargs)


else:

    def histogram2d(*args, **kwargs):
        return histogram2d_np(*args, **kwargs)[0]

    def histogram(*args, **kwargs):
        return histogram_np(*args, **kwargs)[0]


def touch(fname):
    """Mimick the same shell command.

    Examples
    --------
    >>> touch('bububu')
    >>> os.path.exists('bububu')
    True
    >>> os.unlink('bububu')
    """
    Path(fname).touch()


def log_x(a, base):
    # Logb x = Loga x/Loga b
    return np.log(a) / np.log(base)


def get_list_of_small_powers(maxno=100000000000):
    powers_of_two = 2 ** np.arange(0, np.ceil(np.log2(maxno)))
    powers_of_three = 3 ** np.arange(0, np.ceil(log_x(maxno, 3)))
    powers_of_five = 5 ** np.arange(0, np.ceil(log_x(maxno, 5)))
    list_of_powers = []
    for p2 in powers_of_two:
        for p3 in powers_of_three:
            for p5 in powers_of_five:
                newno = p2 * p3 * p5
                if newno > maxno:
                    break
                list_of_powers.append(p2 * p3 * p5)
    return sorted(list_of_powers)


def adjust_dt_for_power_of_two(dt, length, strict=False):
    """
    Examples
    --------
    >>> length, dt = 10, 0.1
    >>> # There are 100 bins there. I want them to be 128.
    >>> new_dt = adjust_dt_for_power_of_two(dt, length)
    INFO: ...
    INFO: ...
    >>> np.isclose(new_dt, 0.078125)
    True
    >>> length / new_dt == 128
    True
    >>> length, dt = 6.5, 0.1
    >>> # There are 100 bins there. I want them to be 128.
    >>> new_dt = adjust_dt_for_power_of_two(dt, length)
    INFO: ...
    INFO: Too many ...
    INFO: ...
    >>> length / new_dt == 72
    True
    """
    log.info("Adjusting bin time to closest power of 2 of bins.")
    nbin = length / dt
    closest_to_pow2 = 2 ** np.ceil(np.log2(nbin))
    if closest_to_pow2 > 1.5 * nbin and not strict:
        log.info("Too many bins: using powers of 2, 3, and 5.")
        return adjust_dt_for_small_power(dt, length)
    new_dt = length / closest_to_pow2
    log.info(f"New bin time: {new_dt} (nbin {nbin} -> {closest_to_pow2})")
    return new_dt


def adjust_dt_for_small_power(dt, length):
    """
    Examples
    --------
    >>> length, dt = 9.9, 0.1
    >>> # There are 99 bins there. I want them to be 100 (2**2 * 5**2).
    >>> new_dt = adjust_dt_for_small_power(dt, length)
    INFO:...
    >>> np.isclose(new_dt, 0.099)
    True
    >>> np.isclose(length / new_dt, 100)
    True
    """
    nbin = length / dt
    losp = get_list_of_small_powers(2 * nbin)
    nbin_new = np.searchsorted(losp, nbin)
    if losp[nbin_new] < nbin:
        nbin_new += 1

    new_dt = length / losp[nbin_new]
    log.info(f"New bin time: {new_dt} (nbin {nbin} -> {losp[nbin_new]})")
    return new_dt


def memmapped_arange(
    i0, i1, istep, fname=None, nbin_threshold=10 ** 7, dtype=float
):
    """Arange plus memory mapping.

    Examples
    --------
    >>> i0, i1, istep = 0, 10, 1e-2
    >>> np.allclose(np.arange(i0, i1, istep), memmapped_arange(i0, i1, istep))
    True
    >>> i0, i1, istep = 0, 10, 1e-7
    >>> np.allclose(np.arange(i0, i1, istep), memmapped_arange(i0, i1, istep))
    True

    """
    import tempfile

    chunklen = 10 ** 6
    Nbins = int((i1 - i0) / istep)
    if Nbins < nbin_threshold:
        return np.arange(i0, i1, istep)
    if fname is None:
        _, fname = tempfile.mkstemp(suffix=".npy")

    hist_arr = np.lib.format.open_memmap(
        fname, mode="w+", dtype=dtype, shape=(Nbins,)
    )

    for start in range(0, Nbins, chunklen):
        stop = min(start + chunklen, Nbins)
        hist_arr[start:stop] = np.arange(start, stop) * istep

    return hist_arr


def nchars_in_int_value(value):
    """Number of characters to write an integer number

    Examples
    --------
    >>> nchars_in_int_value(2)
    1
    >>> nchars_in_int_value(1356)
    4
    >>> nchars_in_int_value(9999)
    4
    >>> nchars_in_int_value(10000)
    5
    """
    #  "+1" because, e.g., 10000 would return 4
    return int(np.ceil(np.log10(value + 1)))


def find_peaks_in_image(image, n=5, rough=False, **kwargs):
    """

    Parameters
    ----------
    image : :class:`np.array`
        An image

    Other Parameters
    ----------------
    n: int
        Number of best peaks to find
    kwargs: dict
        Additional parameters to be passed to skimage

    Returns
    -------
    idxs : list of tuples
        List of indices of the maxima

    Examples
    --------
    >>> image = np.random.normal(0, 0.01, (10, 10))
    >>> image[5, 5] = 2
    >>> image[7, 2] = 1
    >>> image[8, 1] = 3
    >>> idxs = find_peaks_in_image(image, n=2)
    >>> np.allclose(idxs, [(8, 1), (5, 5)])
    True
    >>> idxs = find_peaks_in_image(image, n=2, rough=True)
    >>> np.allclose(idxs, [(8, 1), (5, 5)])
    True
    """
    if not HAS_SKIMAGE or rough:
        best_cands = [
            np.unravel_index(idx, image.shape)
            for idx in np.argpartition(image.flatten(), -n)[-n:]
        ]
        __best_stats = [image[bci] for bci in best_cands]
        best_cands = np.asarray(best_cands)[np.argsort(__best_stats)][::-1]
        return best_cands

    return peak_local_max(image, num_peaks=n, **kwargs)


def force_iterable(val):
    """Force a number to become an array with one element.

    Arrays are preserved.

    Examples
    --------
    >>> val = 5.
    >>> force_iterable(val)[0] == val
    True
    >>> val = None
    >>> force_iterable(val) is None
    True
    >>> val = np.array([5., 5])
    >>> np.all(force_iterable(val) == val)
    True
    """
    if val is None:
        return val

    if not isinstance(val, Iterable):
        return np.array([val])

    return np.asarray(val)
