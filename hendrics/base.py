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
from astropy.logger import AstropyUserWarning
from stingray.pulse.pulsar import get_orbital_correction_from_ephemeris_file

try:
    from numba import jit, njit, prange, vectorize
    from numba import float32, float64, int32, int64

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

DEFAULT_PARSER_ARGS = {}
DEFAULT_PARSER_ARGS['loglevel'] = dict(
    args=['--loglevel'],
    kwargs=dict(help=("use given logging level (one between INFO, "
                      "WARNING, ERROR, CRITICAL, DEBUG; "
                      "default:WARNING)"),
                default='WARNING', type=str))
DEFAULT_PARSER_ARGS['nproc'] = dict(
    args=['--nproc'],
    kwargs=dict(help=("Number of processors to use"),
                default=1, type=int))
DEFAULT_PARSER_ARGS['debug'] = dict(
    args=['--debug'],
    kwargs=dict(help=("set DEBUG logging level"),
                default=False, action='store_true'))
DEFAULT_PARSER_ARGS['bintime'] = dict(
    args=["-b", "--bintime"],
    kwargs=dict(help="Bin time",
                type=np.longdouble, default=1))
DEFAULT_PARSER_ARGS['energies'] = dict(
    args=["-e", "--energy-interval"],
    kwargs=dict(help="Energy interval used for filtering",
                nargs=2, type=float, default=None))
DEFAULT_PARSER_ARGS['pi'] = dict(
    args=["--pi-interval"],
    kwargs=dict(help="PI interval used for filtering",
                nargs=2, type=int, default=[-1, -1]))
DEFAULT_PARSER_ARGS['deorbit'] = dict(
    args=["-p", "--deorbit-par"],
    kwargs=dict(
        help=("Deorbit data with this parameter file (requires PINT installed)"),
        default=None,
        type=str))
DEFAULT_PARSER_ARGS['output'] = dict(
    args=["-o", "--outfile"],
    kwargs=dict(help='Output file',
                default=None, type=str))
DEFAULT_PARSER_ARGS['usepi'] = dict(
    args=['--use-pi'],
    kwargs=dict(help="Use the PI channel instead of energies",
                default=False, action='store_true'))
DEFAULT_PARSER_ARGS['test'] = dict(
    args=["--test"],
    kwargs=dict(help="Only used for tests",
                default=False, action='store_true'))
DEFAULT_PARSER_ARGS['pepoch'] = dict(
    args=["--pepoch"],
    kwargs=dict(type=float, required=False,
                help="Reference epoch for timing parameters (MJD)",
                default=None))


def r_in(td, r_0):
    """Calculate incident countrate given dead time and detected countrate."""
    tau = 1 / r_0
    return 1. / (tau - td)


def r_det(td, r_i):
    """Calculate detected countrate given dead time and incident countrate."""
    tau = 1 / r_i
    return 1. / (tau + td)


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
    if hasattr(data, 'items'):
        data = dict((i[0], np.asarray(i[1])[order])
                    for i in data.items())
    elif hasattr(data, 'index'):
        data = [np.asarray(i)[order] for i in data]
    else:
        data = None
    return data


class _empty():
    def __init__(self):
        pass


def mkdir_p(path):
    """Safe mkdir function."""
    return os.makedirs(path, exist_ok=True)


def common_name(str1, str2, default='common'):
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
    """
    if not len(str1) == len(str2):
        return default
    common_str = ''
    # Extract the HEN root of the name (in case they're event files)
    str1 = hen_root(str1)
    str2 = hen_root(str2)
    for i, letter in enumerate(str1):
        if str2[i] == letter:
            common_str += letter
    # Remove leading and trailing underscores and dashes
    common_str = common_str.rstrip('_').rstrip('-')
    common_str = common_str.lstrip('_').lstrip('-')
    if common_str == '':
        common_str = default
    # log.debug('common_name: %s %s -> %s', str1, str2, common_str)
    return common_str


def hen_root(filename):
    """Return the root file name (without _ev, _lc, etc.).

    Parameters
    ----------
    filename : str
    """
    fname = filename.replace('.gz', '')
    fname = os.path.splitext(fname)[0]
    fname = fname.replace('_ev', '').replace('_lc', '')
    fname = fname.replace('_calib', '')
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


def detection_level(nbins, epsilon=0.01, n_summed_spectra=1, n_rebin=1):
    r"""Detection level for a PDS.

    Return the detection level (with probability 1 - epsilon) for a Power
    Density Spectrum of nbins bins, normalized a la Leahy (1983), based on
    the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
    and multiple PDS averaging (n_summed_spectra)
    Examples
    --------
    >>> np.isclose(detection_level(1, 0.1), 4.6, atol=0.1)
    True
    >>> np.allclose(detection_level(1, 0.1, n_rebin=[1]), [4.6], atol=0.1)
    True
    """
    try:
        from scipy import stats
    except Exception:  # pragma: no cover
        raise Exception('You need Scipy to use this function')

    if not isinstance(n_rebin, Iterable):
        r = n_rebin
        retlev = stats.chi2.isf(epsilon / nbins, 2 * n_summed_spectra * r) \
            / (n_summed_spectra * r)
    else:
        retlev = [stats.chi2.isf(epsilon / nbins, 2 * n_summed_spectra * r) /
                  (n_summed_spectra * r) for r in n_rebin]
        retlev = np.array(retlev)
    return retlev


def probability_of_power(level, nbins, n_summed_spectra=1, n_rebin=1):
    r"""Give the probability of a given power level in PDS.

    Return the probability of a certain power level in a Power Density
    Spectrum of nbins bins, normalized a la Leahy (1983), based on
    the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
    and multiple PDS averaging (n_summed_spectra)

    Examples
    --------
    >>> np.isclose(probability_of_power(4.6, 1), 0.1, atol=0.1)
    True
    >>> np.allclose(probability_of_power([4.6], 1), [0.1], atol=0.1)
    True
    """
    try:
        from scipy import stats
    except Exception:  # pragma: no cover
        raise Exception('You need Scipy to use this function')

    epsilon = nbins * stats.chi2.sf(level * n_summed_spectra * n_rebin,
                                    2 * n_summed_spectra * n_rebin)
    return epsilon


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
        warnings.warn("No parameter file specified for deorbit. Returning"
                      " unaltered event list")
        return events
    if not os.path.exists(parameter_file):
        raise FileNotFoundError(
            "Parameter file {} does not exist".format(parameter_file))

    pepoch = events.gti[0, 0]
    pepoch_mjd = pepoch / 86400 + events.mjdref
    if events.mjdref < 10000:
        warnings.warn("MJDREF is very low. Are you sure everything is "
                      "correct?", AstropyUserWarning)

    length = np.max(events.time) - np.min(events.time)
    length_d = length / 86400
    results = get_orbital_correction_from_ephemeris_file(
        pepoch_mjd - 1,
        pepoch_mjd + length_d + 1,
        parameter_file,
        ntimes=int(
            length // 10))
    orbital_correction_fun = results[0]
    events.time = orbital_correction_fun(events.time, mjdref=events.mjdref)
    events.gti = orbital_correction_fun(events.gti, mjdref=events.mjdref)
    return events


def _add_default_args(parser, list_of_args):
    for key in list_of_args:
        arg = DEFAULT_PARSER_ARGS[key]
        a = arg['args']
        k = arg['kwargs']
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
        return 2**bintime
    elif bintime > 0:
        return bintime
    raise ValueError("Bin time cannot be = 0")


@njit(nogil=True, parallel=False)
def _get_bin_edges(a, bins, a_min, a_max):
    bin_edges = np.zeros(bins+1, dtype=np.float64)

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
        return n - 1 # a_max always in last bin

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
    if bins > 10**7 and use_memmap:
        if tmp is None:
            tmp = tempfile.NamedTemporaryFile('w+')
        hist_arr = np.lib.format.open_memmap(
            tmp, mode='w+', dtype=a.dtype, shape=(bins,))
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
    return _hist2d_numba_seq(H, np.array([x, y]), np.asarray(list(bins)),
                             np.asarray(ranges))


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
    return _hist3d_numba_seq(H, np.asarray(tracks), np.asarray(list(bins)),
                             np.asarray(ranges))


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
        H, np.array([x, y]), weights, np.asarray(list(bins)),
        np.asarray(ranges))


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
        H, np.asarray(tracks), weights, np.asarray(list(bins)),
        np.asarray(ranges))


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
        slicearr = np.array([(tracks[dim, t] - ranges[dim, 0]) * delta[dim]
                             for dim in range(tracks.shape[0])])

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
