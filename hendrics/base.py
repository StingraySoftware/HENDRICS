# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A miscellaneous collection of basic functions."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import logging
import sys
import copy
from stingray.pulse.pulsar import get_orbital_correction_from_ephemeris_file


def r_in(td, r_0):
    """Calculate incident countrate given dead time and detected countrate."""
    tau = 1 / r_0
    return 1. / (tau - td)


def r_det(td, r_in):
    """Calculate detected countrate given dead time and incident countrate."""
    tau = 1 / r_in
    return 1. / (tau + td)


def _assign_value_if_none(value, default):
    if value is None:
        return default
    else:
        return value


def _look_for_array_in_array(array1, array2):
    for a1 in array1:
        if a1 in array2:
            return a1


def is_string(s):
    """Portable function to answer this question."""
    PY3 = sys.version_info[0] == 3
    if PY3:
        return isinstance(s, str)  # NOQA
    else:
        return isinstance(s, basestring)  # NOQA


def _order_list_of_arrays(data, order):
    if hasattr(data, 'items'):
        data = dict([(i[0], i[1][order])
                     for i in data.items()])
    elif hasattr(data, 'index'):
        data = [i[order] for i in data]
    else:
        data = None
    return data


class _empty():
    def __init__(self):
        pass


def mkdir_p(path):  # pragma: no cover
    """Safe mkdir function.

    Parameters
    ----------
    path : str
        Name of the directory/ies to create

    Notes
    -----
    Found at
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_header_key(fits_file, key, hdu=1):
    """Read the header key key from HDU hdu of the file fits_file.

    Parameters
    ----------
    fits_file: str
    key: str
        The keyword to be read

    Other Parameters
    ----------------
    hdu : int
    """
    from astropy.io import fits as pf

    hdulist = pf.open(fits_file)
    try:
        value = hdulist[hdu].header[key]
    except:  # pragma: no cover
        value = ''
    hdulist.close()
    return value


def ref_mjd(fits_file, hdu=1):
    """Read MJDREFF+ MJDREFI or, if failed, MJDREF, from the FITS header.

    Parameters
    ----------
    fits_file : str

    Returns
    -------
    mjdref : numpy.longdouble
        the reference MJD

    Other Parameters
    ----------------
    hdu : int
    """
    import collections

    if isinstance(fits_file, collections.Iterable) and\
            not is_string(fits_file):
        fits_file = fits_file[0]
        logging.info("opening %s" % fits_file)

    try:
        ref_mjd_int = np.long(read_header_key(fits_file, 'MJDREFI'))
        ref_mjd_float = np.longdouble(read_header_key(fits_file, 'MJDREFF'))
        ref_mjd_val = ref_mjd_int + ref_mjd_float
    except:
        ref_mjd_val = np.longdouble(read_header_key(fits_file, 'MJDREF'))
    return ref_mjd_val


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
    logging.debug('common_name: %s %s -> %s' % (str1, str2, common_str))
    return common_str


def hen_root(filename):
    """Return the root file name (without _ev, _lc, etc.).

    Parameters
    ----------
    filename : str
    """
    import os.path
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
    """
    import numpy as np
    return fftlen / (2 ** np.ceil(np.log2(fftlen / tbin)))


def detection_level(nbins, epsilon=0.01, n_summed_spectra=1, n_rebin=1):
    r"""Detection level for a PDS.

    Return the detection level (with probability 1 - epsilon) for a Power
    Density Spectrum of nbins bins, normalized a la Leahy (1983), based on
    the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
    and multiple PDS averaging (n_summed_spectra)
    """
    try:
        from scipy import stats
    except:  # pragma: no cover
        raise Exception('You need Scipy to use this function')

    import collections
    if not isinstance(n_rebin, collections.Iterable):
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
    """
    try:
        from scipy import stats
    except:  # pragma: no cover
        raise Exception('You need Scipy to use this function')

    epsilon = nbins * stats.chi2.sf(level * n_summed_spectra * n_rebin,
                                    2 * n_summed_spectra * n_rebin)
    return 1 - epsilon


def gti_len(gti):
    """Return the total good time from a list of GTIs."""
    return np.sum([g[1] - g[0] for g in gti])


def deorbit_events(events, parameter_file=None):
    events = copy.deepcopy(events)
    if parameter_file is None:
        return events

    pepoch = events.gti[0, 0]
    pepoch_mjd = pepoch / 86400 + events.mjdref

    length = np.max(events.time) - np.min(events.time)
    length_d = length / 86400
    results = get_orbital_correction_from_ephemeris_file(pepoch_mjd - 1,
                                                         pepoch_mjd + length_d + 1,
                                                         parameter_file,
                                                         ntimes=int(length // 10))
    orbital_correction_fun = results[0]
    events.time = orbital_correction_fun(events.time, mjdref=events.mjdref)
    return events
