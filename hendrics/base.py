# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A miscellaneous collection of basic functions."""

import os.path
import sys
import copy
import os
import urllib
import warnings
from collections.abc import Iterable
from pathlib import Path
import tempfile
from astropy.io.registry import identify_format
from astropy.table import Table

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import ncx2
import numpy as np
from numpy import histogram2d as histogram2d_np
from numpy import histogram as histogram_np
from astropy.logger import AstropyUserWarning
from astropy import log
from stingray.stats import pds_probability, pds_detection_level
from stingray.stats import z2_n_detection_level, z2_n_probability
from stingray.stats import fold_detection_level, fold_profile_probability
from stingray.pulse.pulsar import _load_and_prepare_TOAs

try:
    import pint.toa as toa
    import pint
    from pint.models import get_model

    HAS_PINT = True
except (ImportError, urllib.error.URLError):
    HAS_PINT = False
    get_model = None

try:
    from skimage.feature import peak_local_max

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from tqdm import tqdm as show_progress
except ImportError:

    def show_progress(a):
        return a


from . import (
    prange,
    array_take,
    HAS_NUMBA,
    njit,
    vectorize,
    float32,
    float64,
    int32,
    int64,
)

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


HENDRICS_STAR_VALUE = "**" if os.name != "nt" else "XX"

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
    kwargs=dict(help=("set DEBUG logging level"), default=False, action="store_true"),
)
DEFAULT_PARSER_ARGS["colormap"] = dict(
    args=["--colormap"],
    kwargs=dict(
        help=("Change the color map of the image. Any matplotlib colormap is valid"),
        default="cubehelix",
        type=str,
    ),
)
DEFAULT_PARSER_ARGS["dynprofnorm"] = dict(
    args=["--norm"],
    kwargs=dict(
        help=(
            "Normalization for the dynamical phase plot. Can be:\n"
            "  'to1' (each profile normalized from 0 to 1);\n"
            "  'std' (subtract the mean and divide by the standard "
            "deviation);\n"
            "  'sub' (just subtract the mean of each profile);\n"
            "  'ratios' (divide by the average profile, to highlight "
            "changes).\n"
            "Prepending 'median' to any of those options uses the median "
            "in place of the mean. Appending '_smooth' smooths the 2d "
            "array with a Gaussian filter.\n"
            "E.g. mediansub_smooth subtracts the median and smooths the "
            "image"
            "default None"
        ),
        default=None,
        type=str,
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
        help=("Deorbit data with this parameter file (requires PINT installed)"),
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
    kwargs=dict(help="Only used for tests", default=False, action="store_true"),
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

    Examples
    --------
    >>> fname = "blabla_ev_calib.nc"
    >>> hen_root(fname)
    'blabla'
    >>> fname = "blablu_ev_bli.fits.gz"
    >>> hen_root(fname)
    'blablu_ev_bli'
    >>> fname = "blablu_ev_lc.nc"
    >>> hen_root(fname)
    'blablu'
    >>> fname = "blablu_lc_asrd_ev_lc.nc"
    >>> hen_root(fname)
    'blablu_lc_asrd'
    """
    fname, _ = splitext_improved(filename)
    todo = True
    while todo:
        todo = False
        for ending in ["_ev", "_lc", "_pds", "_cpds", "_calib"]:
            if fname.endswith(ending):
                fname = fname[: -len(ending)]
                todo = True

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


def simple_orbit_fun_from_parfile(
    mjdstart, mjdstop, parfile, ntimes=1000, ephem="DE421", invert=False
):
    """Get a correction for orbital motion from pulsar parameter file.

    Parameters
    ----------
    mjdstart, mjdstop : float
        Start and end of the time interval where we want the orbital solution
    parfile : str
        Any parameter file understood by PINT (Tempo or Tempo2 format)

    Other parameters
    ----------------
    ntimes : int
        Number of time intervals to use for interpolation. Default 1000
    invert : bool
        Invert the solution (e.g. to apply an orbital model instead of
        subtracting it)

    Returns
    -------
    correction_mjd : function
        Function that accepts times in MJDs and returns the deorbited times.
    """
    from scipy.interpolate import interp1d
    from astropy import units

    if not HAS_PINT:
        raise ImportError(
            "You need the optional dependency PINT to use this "
            "functionality: github.com/nanograv/pint"
        )

    mjds = np.linspace(mjdstart, mjdstop, ntimes)
    toalist = _load_and_prepare_TOAs(mjds, ephem=ephem)
    m = get_model(parfile)
    delays = m.delay(toalist)
    if invert:
        delays = -delays

    correction = interp1d(
        mjds,
        (toalist.table["tdbld"] * units.d - delays).to(units.d).value,
        fill_value="extrapolate",
    )

    return correction


def deorbit_events(events, parameter_file=None, invert=False, ephem=None):
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
            "No parameter file specified for deorbit. Returning" " unaltered event list"
        )
        return events
    if not os.path.exists(parameter_file):
        raise FileNotFoundError(
            "Parameter file {} does not exist".format(parameter_file)
        )

    if events.mjdref < 33282.0:
        raise ValueError("MJDREF is very low (<01-01-1950), " "this is unsupported.")

    if not HAS_PINT:
        raise ImportError(
            "You need the optional dependency PINT to use this "
            "functionality: github.com/nanograv/pint"
        )

    model = get_model(parameter_file)
    porb = model.PB.value
    pepoch = events.gti[0, 0]
    pepoch_mjd = pepoch / 86400 + events.mjdref

    length = np.max(events.time) - np.min(events.time)

    length_d = length / 86400
    ntimes = max(100, int(length // 60), int(length_d / porb * 100))
    log.info(f"Interpolating orbital solution with {ntimes} points")

    if ephem is None and hasattr(events, "ephem") and events.ephem is not None:
        ephem = events.ephem
        log.info(f"Using default ephemeris: {ephem}")

    elif ephem is None:
        ephem = "DE421"

    orbital_correction_fun = simple_orbit_fun_from_parfile(
        pepoch_mjd - 1,
        pepoch_mjd + length_d + 1,
        parameter_file,
        ntimes=ntimes,
        invert=invert,
        ephem=ephem,
    )

    mjdtimes = events.time / 86400 + events.mjdref
    mjdgti = events.gti / 86400 + events.mjdref

    outtime = (orbital_correction_fun(mjdtimes) - events.mjdref) * 86400
    outgti = (orbital_correction_fun(mjdgti) - events.mjdref) * 86400

    events.time = outtime
    events.gti = outgti
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
        return 2**bintime
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
    >>> # Now use memmap but do not specify a tmp file
    >>> Hn = hist1d_numba_seq(x, bins=10**8, ranges=[0., 1.],
    ...                       use_memmap=True)
    >>> assert np.all(H == Hn)
    """
    if bins > 10**7 and use_memmap:
        if tmp is None:
            tmp = tempfile.NamedTemporaryFile("w+").name
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
def _hist1d_numba_seq_weight(H, tracks, weights, bins, ranges):
    delta = 1 / ((ranges[1] - ranges[0]) / bins)

    for t in range(tracks.size):
        i = (tracks[t] - ranges[0]) * delta
        if 0 <= i < bins:
            H[int(i)] += weights[t]

    return H


def hist1d_numba_seq_weight(a, weights, bins, ranges, use_memmap=False, tmp=None):
    """
    Examples
    --------
    >>> if os.path.exists('out.npy'): os.unlink('out.npy')
    >>> x = np.random.uniform(0., 1., 100)
    >>> weights = np.random.uniform(0, 1, 100)
    >>> H, xedges = np.histogram(x, bins=5, range=[0., 1.], weights=weights)
    >>> Hn = hist1d_numba_seq_weight(x, weights, bins=5, ranges=[0., 1.], tmp='out.npy',
    ...                              use_memmap=True)
    >>> assert np.all(H == Hn)
    >>> # The number of bins is small, memory map was not used!
    >>> assert not os.path.exists('out.npy')
    >>> H, xedges = np.histogram(x, bins=10**8, range=[0., 1.], weights=weights)
    >>> Hn = hist1d_numba_seq_weight(x, weights, bins=10**8, ranges=[0., 1.], tmp='out.npy',
    ...                              use_memmap=True)
    >>> assert np.all(H == Hn)
    >>> assert os.path.exists('out.npy')
    >>> # Now use memmap but do not specify a tmp file
    >>> Hn = hist1d_numba_seq_weight(x, weights, bins=10**8, ranges=[0., 1.],
    ...                              use_memmap=True)
    >>> assert np.all(H == Hn)
    """
    if bins > 10**7 and use_memmap:
        if tmp is None:
            tmp = tempfile.NamedTemporaryFile("w+").name
        hist_arr = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=a.dtype, shape=(bins,)
        )
    else:
        hist_arr = np.zeros((bins,), dtype=a.dtype)

    return _hist1d_numba_seq_weight(hist_arr, a, weights, bins, np.asarray(ranges))


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
        """
        Examples
        --------
        >>> x = np.random.uniform(0., 1., 100)
        >>> y = np.random.uniform(2., 3., 100)
        >>> weight = np.random.uniform(0, 1, 100)
        >>> H, xedges, yedges = np.histogram2d(x, y, bins=(5, 5),
        ...                                    range=[(0., 1.), (2., 3.)],
        ...                                    weights=weight)
        >>> Hn = histogram2d(x, y, bins=(5, 5),
        ...                  ranges=[[0., 1.], [2., 3.]],
        ...                  weights=weight)
        >>> assert np.all(H == Hn)
        >>> Hn1 = histogram2d(x, y, bins=(5, 5),
        ...                   ranges=[[0., 1.], [2., 3.]],
        ...                   weights=None)
        >>> Hn2 = histogram2d(x, y, bins=(5, 5),
        ...                   ranges=[[0., 1.], [2., 3.]])
        >>> assert np.all(Hn1 == Hn2)
        """
        if "range" in kwargs:
            kwargs["ranges"] = kwargs.pop("range")

        if "weights" not in kwargs:
            return hist2d_numba_seq(*args, **kwargs)

        weights = kwargs.pop("weights")

        if weights is not None:
            return hist2d_numba_seq_weight(*args, weights, **kwargs)

        return hist2d_numba_seq(*args, **kwargs)

    def histogram(*args, **kwargs):
        """
        Examples
        --------
        >>> x = np.random.uniform(0., 1., 100)
        >>> weights = np.random.uniform(0, 1, 100)
        >>> H, xedges = np.histogram(x, bins=5, range=[0., 1.], weights=weights)
        >>> Hn = histogram(x, weights=weights, bins=5, ranges=[0., 1.], tmp='out.npy',
        ...                use_memmap=True)
        >>> assert np.all(H == Hn)
        >>> Hn1 = histogram(x, weights=None, bins=5, ranges=[0., 1.])
        >>> Hn2 = histogram(x, bins=5, ranges=[0., 1.])
        >>> assert np.all(Hn1 == Hn2)
        """
        if "range" in kwargs:
            kwargs["ranges"] = kwargs.pop("range")

        if "weights" not in kwargs:
            return hist1d_numba_seq(*args, **kwargs)

        weights = kwargs.pop("weights")

        if weights is not None:
            return hist1d_numba_seq_weight(*args, weights, **kwargs)

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


def memmapped_arange(i0, i1, istep, fname=None, nbin_threshold=10**7, dtype=float):
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

    chunklen = 10**6
    Nbins = int((i1 - i0) / istep)
    if Nbins < nbin_threshold:
        return np.arange(i0, i1, istep)
    if fname is None:
        _, fname = tempfile.mkstemp(suffix=".npy")

    hist_arr = np.lib.format.open_memmap(fname, mode="w+", dtype=dtype, shape=(Nbins,))

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


def normalize_dyn_profile(dynprof, norm):
    """Normalize a dynamical profile (e.g. a phaseogram).

    Parameters
    ----------
    dynprof : np.ndarray
        The dynamical profile has to be a 2d array structured as:
        `dynprof = [profile0, profile1, profile2, ...]`
        where each `profileX` is a pulse profile.
    norm : str
        The chosen normalization. If it ends with `_smooth`, a
        simple Gaussian smoothing is applied to the image.
        Besides the smoothing string, the options are:
        1. to1: make each profile normalized between 0 and 1
        2. std: subtract the mean and divide by standard deviation
            in each row
        3. ratios: divide by the average profile (particularly
            useful in energy vs phase plots)
        4. mediansub, meansub: just subtract the median or the mean
            from each profile
        5. mediannorm, meannorm: subtract the median or the norm
            and divide by it to get fractional amplitude

    Examples
    --------
    >>> hist = [[1, 2], [2, 3], [3, 4]]
    >>> hnorm = normalize_dyn_profile(hist, "meansub")
    >>> np.allclose(hnorm[0], [-0.5, 0.5])
    True
    >>> hnorm = normalize_dyn_profile(hist, "meannorm")
    >>> np.allclose(hnorm[0], [-1/3, 1/3])
    True
    >>> hnorm = normalize_dyn_profile(hist, "ratios")
    >>> np.allclose(hnorm[1], [1, 1])
    True
    """
    dynprof = np.array(dynprof, dtype=float)

    if norm is None:
        norm = ""

    if norm.endswith("_smooth"):
        dynprof = gaussian_filter(dynprof, 1, mode=("constant", "wrap"))
        norm = norm.replace("_smooth", "")

    if norm.startswith("median"):
        y_mean = np.median(dynprof, axis=1)
        prof_mean = np.median(dynprof, axis=0)
        norm = norm.replace("median", "")
    else:
        y_mean = np.mean(dynprof, axis=1)
        prof_mean = np.mean(dynprof, axis=0)
        norm = norm.replace("mean", "")

    if "ratios" in norm:
        dynprof /= prof_mean[np.newaxis, :]
        norm = norm.replace("ratios", "")
        y_mean = np.mean(dynprof, axis=1)

    y_min = np.min(dynprof, axis=1)
    y_max = np.max(dynprof, axis=1)
    y_std = np.std(np.diff(dynprof, axis=0)) / np.sqrt(2)

    if norm in ("", "none"):
        pass
    elif norm == "to1":
        dynprof -= y_min[:, np.newaxis]
        dynprof /= (y_max - y_min)[:, np.newaxis]
    elif norm == "std":
        dynprof -= y_mean[:, np.newaxis]
        dynprof /= y_std
    elif norm == "sub":
        dynprof -= y_mean[:, np.newaxis]
    elif norm == "norm":
        dynprof -= y_mean[:, np.newaxis]
        dynprof /= y_mean[:, np.newaxis]
    else:
        warnings.warn(f"Profile normalization {norm} not known. Using default")
    return dynprof


def get_file_extension(fname):
    """Get the file extension, including (if any) the compression format.

    Examples
    --------
    >>> get_file_extension('bu.p')
    '.p'
    >>> get_file_extension('bu.nc')
    '.nc'
    >>> get_file_extension('bu.evt.Z')
    '.evt.Z'
    >>> get_file_extension('bu.ecsv')
    '.ecsv'
    >>> get_file_extension('bu.fits.Gz')
    '.fits.Gz'
    """

    raw_ext = os.path.splitext(fname)[1]
    if raw_ext.lower() in [".gz", ".bz", ".z", ".bz2"]:
        fname = fname.replace(raw_ext, "")
        return os.path.splitext(fname)[1] + raw_ext

    return raw_ext


def splitext_improved(fname):
    """Get the file name and extension, including compression format.

    Examples
    --------
    >>> splitext_improved('bu.p')
    ('bu', '.p')
    >>> splitext_improved('bu.nc')
    ('bu', '.nc')
    >>> splitext_improved('bu.evt.Z')
    ('bu', '.evt.Z')
    >>> splitext_improved('bu.ecsv')
    ('bu', '.ecsv')
    >>> splitext_improved('bu.fits.Gz')
    ('bu', '.fits.Gz')
    """
    ext = get_file_extension(fname)
    root = fname.replace(ext, "")
    return root, ext


def get_file_format(fname):
    """Decide the file format of the file.

    Examples
    --------
    >>> get_file_format('bu.p')
    'pickle'
    >>> get_file_format('bu.nc')
    'nc'
    >>> get_file_format('bu.evt')
    'ogip'
    >>> get_file_format('bu.ecsv')
    'ascii.ecsv'
    >>> get_file_format('bu.fits.gz')
    'ogip'
    >>> get_file_format('bu.pdfghj')
    Traceback (most recent call last):
        ...
    RuntimeError: File format pdfghj not recognized
    """
    ext = get_file_extension(fname)
    if ext in [".p", ".pickle"]:
        return "pickle"

    if ext == ".nc":
        return "nc"

    if ext in [".evt", ".fits", ".fits.gz"]:
        return "ogip"

    # For the rest of formats, use Astropy
    fmts = identify_format("write", Table, fname, None, [], {})
    if len(fmts) > 0:
        return fmts[0]

    raise RuntimeError(f"File format {ext[1:]} " f"not recognized")
