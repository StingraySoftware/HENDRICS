# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to calculate frequency spectra."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .base import mp_root, cross_gtis, create_gti_mask
from .base import common_name, _empty
from .rebin import const_rebin
from .io import sort_files, get_file_type, load_lcurve, save_pds
from .io import MP_FILE_EXTENSION
import numpy as np
import logging
import warnings
from multiprocessing import Pool
import os


def _wrap_fun_cpds(arglist):
    f1, f2, outname, kwargs = arglist
    return calc_cpds(f1, f2, outname=outname, **kwargs)


def _wrap_fun_pds(argdict):
    fname = argdict["fname"]
    argdict.pop("fname")
    return calc_pds(fname, **argdict)


def fft(lc, bintime):
    """A wrapper for the fft function. Just numpy for now.

    Parameters
    ----------
    lc : array-like
    bintime : float

    Returns
    -------
    freq : array-like
    ft : array-like
        the Fourier transform.
    """
    nbin = len(lc)

    ft = np.fft.fft(lc)
    freqs = np.fft.fftfreq(nbin, bintime)

    return freqs.astype(np.double), ft


def leahy_pds(lc, bintime):
    r"""Calculate the power density spectrum.

    Calculates the Power Density Spectrum a la Leahy+1983, ApJ 266, 160,
    given the lightcurve and its bin time.
    Assumes no gaps are present! Beware!

    Parameters
    ----------
    lc : array-like
        the light curve
    bintime : array-like
        the bin time of the light curve

    Returns
    -------
    freqs : array-like
        Frequencies corresponding to PDS
    pds : array-like
        The power density spectrum
    """
    nph = sum(lc)

    # Checks must be done before. At this point, only good light curves have to
    # be provided
    assert (nph > 0), 'Invalid interval. Light curve is empty'

    freqs, ft = fft(lc, bintime)
    # I'm pretty sure there is a faster way to do this.
    pds = np.absolute(ft.conjugate() * ft) * 2. / nph

    good = freqs >= 0
    freqs = freqs[good]
    pds = pds[good]

    return freqs, pds


def welch_pds(time, lc, bintime, fftlen, gti=None, return_all=False):
    r"""Calculate the PDS, averaged over equal chunks of data.

    Calculates the Power Density Spectrum \'a la Leahy (1983), given the
    lightcurve and its bin time, over equal chunks of length fftlen, and
    returns the average of all PDSs, or the sum PDS and the number of chunks

    Parameters
    ----------
    time : array-like
        Central times of light curve bins
    lc : array-like
        Light curve
    bintime : float
        Bin time of the light curve
    fftlen : float
        Length of each FFT
    gti : [[g0_0, g0_1], [g1_0, g1_1], ...]
         Good time intervals. Defaults to
         [[time[0] - bintime/2, time[-1] + bintime/2]]

    Returns
    -------
    return_str : object, optional
        An Object containing all values below.
    freq : array-like
        array of frequencies corresponding to PDS bins
    pds : array-like
        the values of the PDS
    pds_err : array-like
        the values of the PDS
    n_chunks : int
        the number of summed PDSs (if normalize is False)
    ctrate : float
        the average count rate in the two lcs

    Other parameters
    ----------------
    return_all : bool
        if True, return everything, including the dynamical PDS
    """
    if gti is None:
        gti = [[time[0] - bintime / 2, time[-1] + bintime / 2]]

    start_bins, stop_bins = \
        decide_spectrum_lc_intervals(gti, fftlen, time)

    results = _empty()
    if return_all:
        results.dynpds = []
        results.edynpds = []
        results.dynctrate = []
        results.times = []

    pds = 0
    npds = len(start_bins)

    mask = np.zeros(len(lc), dtype=np.bool)

    for start_bin, stop_bin in zip(start_bins, stop_bins):
        l = lc[start_bin:stop_bin]
        if np.sum(l) == 0:
            logging.warning('Interval starting at' +
                            ' time %.7f' % time[start_bin] +
                            ' is bad. Check GTIs')
            npds -= 1
            continue
        try:
            f, p = leahy_pds(l, bintime)
        except Exception as e:
            warnings.warn(str(e))

        if return_all:
            results.dynpds.append(p)
            results.edynpds.append(p)
            results.dynctrate.append(np.mean(l) / bintime)
            results.times.append(time[start_bin])

        pds += p
        mask[start_bin:stop_bin] = True

    pds /= npds
    epds = pds / np.sqrt(npds)
    ctrate = np.mean(lc[mask]) / bintime

    results.f = f
    results.pds = pds
    results.epds = epds
    results.npds = npds
    results.ctrate = ctrate

    return results


def leahy_cpds(lc1, lc2, bintime):
    """Calculate the cross power density spectrum.

    Calculates the Cross Power Density Spectrum, normalized similarly to the
    PDS in Leahy+1983, ApJ 266, 160., given the lightcurve and its bin time.
    Assumes no gaps are present! Beware!

    Parameters
    ----------
    lc1 : array-like
        The first light curve
    lc2 : array-like
        The light curve
    bintime : array-like
        The bin time of the light curve

    Returns
    -------
    freqs : array-like
        Frequencies corresponding to PDS
    cpds : array-like
        The cross power density spectrum

    """
    assert len(lc1) == len(lc2), 'Light curves MUST have the same length!'
    nph1 = sum(lc1)
    nph2 = sum(lc2)
    # Checks must be done before. At this point, only good light curves have to
    # be provided
    assert (nph1 > 0 and nph2 > 0), ('Invalid interval. At least one light '
                                     'curve is empty')

    freqs, ft1 = fft(lc1, bintime)
    freqs, ft2 = fft(lc2, bintime)

    pds1 = np.absolute(ft1.conjugate() * ft1) * 2. / nph1
    pds2 = np.absolute(ft2.conjugate() * ft2) * 2. / nph2
    pds1e = np.copy(pds1)
    pds2e = np.copy(pds2)

    # The "effective" count rate is the geometrical mean of the count rates
    # of the two light curves
    nph = np.sqrt(nph1 * nph2)
    # I'm pretty sure there is a faster way to do this.
    if nph != 0:
        cpds = ft1.conjugate() * ft2 * 2. / nph
    else:
        cpds = np.zeros(len(freqs))

    # Justification in timing paper! (Bachetti et al. arXiv:1409.3248)
    # This only works for cospectrum. For the cross spectrum, I *think*
    # it's irrelevant
    cpdse = np.sqrt(pds1e * pds2e) / np.sqrt(2.)

    good = freqs >= 0
    freqs = freqs[good]
    cpds = cpds[good]
    cpdse = cpdse[good]

    return freqs, cpds, cpdse, pds1, pds2


def welch_cpds(time, lc1, lc2, bintime, fftlen, gti=None, return_all=False):
    """Calculate the CPDS, averaged over equal chunks of data.

    Calculates the Cross Power Density Spectrum normalized like PDS, given the
    lightcurve and its bin time, over equal chunks of length fftlen, and
    returns the average of all PDSs, or the sum PDS and the number of chunks

    Parameters
    ----------
    time : array-like
        Central times of light curve bins
    lc1 : array-like
        Light curve 1
    lc2 : array-like
        Light curve 2
    bintime : float
        Bin time of the light curve
    fftlen : float
        Length of each FFT
    gti : [[g0_0, g0_1], [g1_0, g1_1], ...]
         Good time intervals. Defaults to
         [[time[0] - bintime/2, time[-1] + bintime/2]]

    Returns
    -------
    return_str : object, optional
        An Object containing all return values below
    freq : array-like
        array of frequencies corresponding to PDS bins
    pds : array-like
        the values of the PDS
    pds_err : array-like
        the values of the PDS
    n_chunks : int
        the number of summed PDSs (if normalize is False)
    ctrate : float
        the average count rate in the two lcs

    Other parameters
    ----------------
    return_all : bool
        if True, return everything, including the dynamical PDS
    """
    if gti is None:
        gti = [[time[0] - bintime / 2, time[-1] + bintime / 2]]

    start_bins, stop_bins = \
        decide_spectrum_lc_intervals(gti, fftlen, time)

    cpds = 0
    ecpds = 0
    npds = len(start_bins)
    mask = np.zeros(len(lc1), dtype=np.bool)

    results = _empty()
    if return_all:
        results.dyncpds = []
        results.edyncpds = []
        results.dynctrate = []
        results.times = []

    cpds = 0
    ecpds = 0
    npds = len(start_bins)

    for start_bin, stop_bin in zip(start_bins, stop_bins):
        l1 = lc1[start_bin:stop_bin]
        l2 = lc2[start_bin:stop_bin]

        if np.sum(l1) == 0 or np.sum(l2) == 0:
            logging.warning('Interval starting at' +
                            ' time %.7f' % time[start_bin] +
                            ' is bad. Check GTIs')
            npds -= 1
            continue

        try:
            f, p, pe, p1, p2 = leahy_cpds(l1, l2, bintime)
        except Exception as e:
            warnings.warn(str(e))

        cpds += p
        ecpds += pe ** 2
        if return_all:
            results.dyncpds.append(p)
            results.edyncpds.append(pe)
            results.dynctrate.append(
                np.sqrt(np.mean(l1)*np.mean(l2)) / bintime)
            results.times.append(time[start_bin])

        mask[start_bin:stop_bin] = True

    cpds /= npds
    ecpds = np.sqrt(ecpds) / npds

    ctrate = np.sqrt(np.mean(lc1[mask])*np.mean(lc2[mask])) / bintime

    results.f = f
    results.cpds = cpds
    results.ecpds = ecpds
    results.ncpds = npds
    results.ctrate = ctrate

    return results


def rms_normalize_pds(pds, pds_err, source_ctrate, back_ctrate=None):
    """Normalize a Leahy PDS with RMS normalization ([1]_, [2]_).

    Parameters
    ----------
    pds : array-like
        The Leahy-normalized PDS
    pds_err : array-like
        The uncertainties on the PDS values
    source_ctrate : float
        The source count rate
    back_ctrate: float, optional
        The background count rate

    Returns
    -------
    pds : array-like
        the RMS-normalized PDS
    pds_err : array-like
        the uncertainties on the PDS values

    References
    ----------
    .. [1] Belloni & Hasinger 1990, A&A, 230, 103
    .. [2] Miyamoto+1991, ApJ, 383, 784

    """
    if back_ctrate is None:
        logging.warning("Assuming background level 0")
        back_ctrate = 0
    factor = (source_ctrate + back_ctrate) / source_ctrate ** 2
    return pds * factor, pds_err * factor


def decide_spectrum_intervals(gtis, fftlen):
    """Decide the start times of PDSs.

    Start each FFT/PDS/cospectrum from the start of a GTI, and stop before the
    next gap.
    Only use for events! This will give problems with binned light curves.

    Parameters
    ----------
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    fftlen : float
        Length of the chunks

    Returns
    -------
    spectrum_start_times : array-like
        List of starting times to use in the spectral calculations.

    """
    spectrum_start_times = np.array([], dtype=np.longdouble)
    for g in gtis:
        if g[1] - g[0] < fftlen:
            logging.info("GTI at %g--%g is Too short. Skipping." %
                         (g[0], g[1]))
            continue

        newtimes = np.arange(g[0], g[1] - fftlen, np.longdouble(fftlen),
                             dtype=np.longdouble)
        spectrum_start_times = \
            np.append(spectrum_start_times,
                      newtimes)

    assert len(spectrum_start_times) > 0, \
        "No GTIs are equal to or longer than fftlen. " + \
        "Choose shorter fftlen (MPfspec -f <fftlen> <options> <filename>)"
    return spectrum_start_times


def decide_spectrum_lc_intervals(gtis, fftlen, time):
    """Similar to decide_spectrum_intervals, but dedicated to light curves.

    In this case, it is necessary to specify the time array containing the
    times of the light curve bins.
    Returns start and stop bins of the intervals to use for the PDS

    Parameters
    ----------
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    fftlen : float
        Length of the chunks
    time : array-like
        Times of light curve bins
    """
    bintime = time[1] - time[0]
    nbin = np.long(fftlen / bintime)

    spectrum_start_bins = np.array([], dtype=np.long)
    for g in gtis:
        if g[1] - g[0] < fftlen:
            logging.info("GTI at %g--%g is Too short. Skipping." %
                         (g[0], g[1]))
            continue
        startbin = np.argmin(np.abs(time - g[0]))
        stopbin = np.argmin(np.abs(time - g[1]))

        newbins = np.arange(startbin, stopbin - nbin, nbin,
                            dtype=np.long)
        spectrum_start_bins = \
            np.append(spectrum_start_bins,
                      newbins)

    assert len(spectrum_start_bins) > 0, \
        "No GTIs are equal to or longer than fftlen. " + \
        "Choose shorter fftlen (MPfspec -f <fftlen> <options> <filename>)"
    return spectrum_start_bins, spectrum_start_bins + nbin


def calc_pds(lcfile, fftlen,
             save_dyn=False,
             bintime=1,
             pdsrebin=1,
             normalization='Leahy',
             back_ctrate=0.,
             noclobber=False,
             outname=None):
    """Calculate the PDS from an input light curve file.

    Parameters
    ----------
    lcfile : str
        The light curve file
    fftlen : float
        The length of the chunks over which FFTs will be calculated, in seconds

    Other Parameters
    ----------------
    save_dyn : bool
        If True, save the dynamical power spectrum
    bintime : float
        The bin time. If different from that of the light curve, a rebinning is
        performed
    pdsrebin : int
        Rebin the PDS of this factor.
    normalization : str
        'Leahy' or 'rms'
    back_ctrate : float
        The non-source count rate
    noclobber : bool
        If True, do not overwrite existing files
    outname : str
        If speficied, output file name. If not specified or None, the new file
        will have the same root as the input light curve and the '_pds' suffix
    """
    root = mp_root(lcfile)
    outname = root + '_pds' + MP_FILE_EXTENSION
    if noclobber and os.path.exists(outname):
        print('File exists, and noclobber option used. Skipping')
        return

    logging.info("Loading file %s..." % lcfile)
    lcdata = load_lcurve(lcfile)
    time = lcdata['time']
    mjdref = lcdata['MJDref']
    try:
        lc = lcdata['lccorr']
    except:
        lc = lcdata['lc']
    dt = lcdata['dt']
    gti = lcdata['GTI']
    instr = lcdata['Instr']
    tctrate = lcdata['total_ctrate']

    if bintime <= dt:
        bintime = dt
    else:
        lcrebin = np.rint(bintime / dt)
        bintime = lcrebin * dt
        logging.info("Rebinning lc by a factor %d" % lcrebin)
        time, lc, dum = \
            const_rebin(time, lc, lcrebin, normalize=False)

    try:
        results = welch_pds(time, lc, bintime, fftlen, gti, return_all=True)
        freq = results.f
        pds = results.pds
        epds = results.epds
        npds = results.npds
        ctrate = results.ctrate
    except Exception as e:
        # If it fails, exit cleanly
        logging.error("{0} failed ({1}: {2})".format('Problem with the PDS.',
                                                     type(e), e))
        raise Exception("{0} failed ({1}: {2})".format('Problem with the PDS.',
                                                       type(e), e))

    freq, pds, epds = const_rebin(freq[1:], pds[1:], pdsrebin,
                                  epds[1:])

    if normalization == 'rms':
        logging.info('Applying %s normalization' % normalization)

        pds, epds = \
            rms_normalize_pds(pds, epds,
                              source_ctrate=ctrate,
                              back_ctrate=back_ctrate)

        for ic, pd in enumerate(results.dynpds):
            ep = results.edynpds[ic].copy()
            ct = results.dynctrate[ic].copy()

            pd, ep = rms_normalize_pds(pd, ep, source_ctrate=ct,
                                       back_ctrate=back_ctrate)
            results.edynpds[ic][:] = ep
            results.dynpds[ic][:] = pd

    outdata = {'time': time[0], 'pds': pds, 'epds': epds, 'npds': npds,
               'fftlen': fftlen, 'Instr': instr, 'freq': freq,
               'rebin': pdsrebin, 'norm': normalization, 'ctrate': ctrate,
               'total_ctrate': tctrate,
               'back_ctrate': back_ctrate, 'MJDref': mjdref}
    if 'Emin' in lcdata.keys():
        outdata['Emin'] = lcdata['Emin']
        outdata['Emax'] = lcdata['Emax']
    if 'PImin' in lcdata.keys():
        outdata['PImin'] = lcdata['PImin']
        outdata['PImax'] = lcdata['PImax']

    logging.debug(repr(results.dynpds))

    if save_dyn:
        outdata["dynpds"] = np.array(results.dynpds)[:, 1:]
        outdata["edynpds"] = np.array(results.edynpds)[:, 1:]
        outdata["dynctrate"] = np.array(results.dynctrate)

        outdata["dyntimes"] = np.array(results.times)

    logging.info('Saving PDS to %s' % outname)
    save_pds(outdata, outname)


def calc_cpds(lcfile1, lcfile2, fftlen,
              save_dyn=False,
              bintime=1,
              pdsrebin=1,
              outname='cpds' + MP_FILE_EXTENSION,
              normalization='Leahy',
              back_ctrate=0.,
              noclobber=False):
    """Calculate the CPDS from a pair of input light curve files.

    Parameters
    ----------
    lcfile1 : str
        The first light curve file
    lcfile2 : str
        The second light curve file
    fftlen : float
        The length of the chunks over which FFTs will be calculated, in seconds

    Other Parameters
    ----------------
    save_dyn : bool
        If True, save the dynamical power spectrum
    bintime : float
        The bin time. If different from that of the light curve, a rebinning is
        performed
    pdsrebin : int
        Rebin the PDS of this factor.
    normalization : str
        'Leahy' or 'rms'. Default 'Leahy'
    back_ctrate : float
        The non-source count rate
    noclobber : bool
        If True, do not overwrite existing files
    outname : str
        Output file name for the cpds. Default: cpds.[nc|p]
    """
    if noclobber and os.path.exists(outname):
        print('File exists, and noclobber option used. Skipping')
        return

    logging.info("Loading file %s..." % lcfile1)
    lcdata1 = load_lcurve(lcfile1)
    logging.info("Loading file %s..." % lcfile2)
    lcdata2 = load_lcurve(lcfile2)

    time1 = lcdata1['time']
    try:
        lc1 = lcdata1['lccorr']
    except:
        lc1 = lcdata1['lc']
    dt1 = lcdata1['dt']
    gti1 = lcdata1['GTI']
    instr1 = lcdata1['Instr']
    tctrate1 = lcdata1['total_ctrate']
    mjdref = lcdata1['MJDref']

    time2 = lcdata2['time']
    try:
        lc2 = lcdata2['lccorr']
    except:
        lc2 = lcdata2['lc']
    dt2 = lcdata2['dt']
    gti2 = lcdata2['GTI']
    instr2 = lcdata2['Instr']
    tctrate2 = lcdata2['total_ctrate']

    tctrate = np.sqrt(tctrate1 * tctrate2)

    assert instr1 != instr2, ('Did you check the ordering of files? '
                              'These are both ' + instr1)

    assert dt1 == dt2, 'Light curves are sampled differently'
    dt = dt1

    if bintime <= dt:
        bintime = dt
    else:
        lcrebin = np.rint(bintime / dt)
        dt = bintime
        logging.info("Rebinning lcs by a factor %d" % lcrebin)
        time1, lc1, dum = \
            const_rebin(time1, lc1, lcrebin, normalize=False)
        time2, lc2, dum = \
            const_rebin(time2, lc2, lcrebin, normalize=False)

    gti = cross_gtis([gti1, gti2])

    mask1 = create_gti_mask(time1, gti)
    mask2 = create_gti_mask(time2, gti)
    time1 = time1[mask1]
    time2 = time2[mask2]

    assert np.all(time1 == time2), "Something's not right in GTI filtering"
    time = time1
    del time2

    lc1 = lc1[mask1]
    lc2 = lc2[mask2]

    try:
        results = welch_cpds(time, lc1, lc2, bintime, fftlen, gti,
                             return_all=True)
        freq = results.f
        cpds = results.cpds
        ecpds = results.ecpds
        ncpds = results.ncpds
        ctrate = results.ctrate
    except Exception as e:
        # If it fails, exit cleanly
        logging.error("{0} failed ({1}: {2})".format(
            'Problem with the CPDS.', type(e), e))
        raise Exception("{0} failed ({1}: {2})".format(
            'Problem with the CPDS.', type(e), e))

    freq, cpds, ecpds = const_rebin(freq[1:], cpds[1:], pdsrebin,
                                    ecpds[1:])

    if normalization == 'rms':
        logging.info('Applying %s normalization' % normalization)
        cpds, ecpds = \
            rms_normalize_pds(cpds, ecpds,
                              source_ctrate=ctrate,
                              back_ctrate=back_ctrate)
        for ic, cp in enumerate(results.dyncpds):
            ec = results.edyncpds[ic].copy()
            ct = results.dynctrate[ic].copy()

            cp, ec = rms_normalize_pds(cp, ec, source_ctrate=ct,
                                       back_ctrate=back_ctrate)
            results.edyncpds[ic][:] = ec
            results.dyncpds[ic][:] = cp

    outdata = {'time': gti[0][0], 'cpds': cpds, 'ecpds': ecpds, 'ncpds': ncpds,
               'fftlen': fftlen, 'Instrs': instr1 + ',' + instr2,
               'freq': freq, 'rebin': pdsrebin, 'norm': normalization,
               'ctrate': ctrate, 'total_ctrate': tctrate,
               'back_ctrate': back_ctrate, 'MJDref': mjdref}

    if 'Emin' in lcdata1.keys():
        outdata['Emin1'] = lcdata1['Emin']
        outdata['Emax1'] = lcdata1['Emax']
    if 'Emin' in lcdata2.keys():
        outdata['Emin2'] = lcdata2['Emin']
        outdata['Emax2'] = lcdata2['Emax']

    if 'PImin' in lcdata1.keys():
        outdata['PImin1'] = lcdata1['PImin']
        outdata['PImax1'] = lcdata1['PImax']
    if 'PImin' in lcdata2.keys():
        outdata['PImin2'] = lcdata2['PImin']
        outdata['PImax2'] = lcdata2['PImax']

    logging.debug(repr(results.dyncpds))
    if save_dyn:
        outdata["dyncpds"] = np.array(results.dyncpds)[:, 1:]
        outdata["edyncpds"] = np.array(results.edyncpds)[:, 1:]
        outdata["dynctrate"] = np.array(results.dynctrate)
        outdata["dyntimes"] = np.array(results.times)

    logging.info('Saving CPDS to %s' % outname)
    save_pds(outdata, outname)


def calc_fspec(files, fftlen,
               do_calc_pds=True,
               do_calc_cpds=True,
               do_calc_cospectrum=True,
               do_calc_lags=True,
               save_dyn=False,
               bintime=1,
               pdsrebin=1,
               outroot=None,
               normalization='Leahy',
               nproc=1,
               back_ctrate=0.,
               noclobber=False):
    r"""Calculate the frequency spectra: the PDS, the cospectrum, ...

    Parameters
    ----------
    files : list of str
        List of input file names
    fftlen : float
        length of chunks to perform the FFT on.

    Other Parameters
    ----------------
    save_dyn : bool
        If True, save the dynamical power spectrum
    bintime : float
        The bin time. If different from that of the light curve, a rebinning is
        performed
    pdsrebin : int
        Rebin the PDS of this factor.
    normalization : str
        'Leahy' [3] or 'rms' [4] [5]. Default 'Leahy'.
    back_ctrate : float
        The non-source count rate
    noclobber : bool
        If True, do not overwrite existing files
    outroot : str
        Output file name root
    nproc : int
        Number of processors to use to parallelize the processing of multiple
        files

    References
    ----------
    [3] Leahy et al. 1983, ApJ, 266, 160.

    [4] Belloni & Hasinger 1990, A&A, 230, 103

    [5] Miyamoto et al. 1991, ApJ, 383, 784

    """
    if normalization not in ['Leahy', 'rms']:
        logging.warning('Beware! Unknown normalization!')
        normalization = 'Leahy'

    logging.info('Using %s normalization' % normalization)

    if do_calc_pds:
        wrapped_file_dicts = []
        for f in files:
            wfd = {"fftlen": fftlen,
                   "save_dyn": save_dyn,
                   "bintime": bintime,
                   "pdsrebin": pdsrebin,
                   "normalization": normalization,
                   "back_ctrate": back_ctrate,
                   "noclobber": noclobber}
            wfd["fname"] = f
            wrapped_file_dicts.append(wfd)

        if os.name == 'nt' or nproc == 1:
            [_wrap_fun_pds(w) for w in wrapped_file_dicts]
        else:
            pool = Pool(processes=nproc)
            for i in pool.imap_unordered(_wrap_fun_pds, wrapped_file_dicts):
                pass
            pool.close()

    if not do_calc_cpds or len(files) < 2:
        return

    logging.info('Sorting file list')
    sorted_files = sort_files(files)

    logging.warning('Beware! For cpds and derivatives, I assume that the'
                    'files are from only two instruments and in pairs'
                    '(even in random order)')

    instrs = list(sorted_files.keys())
    files1 = sorted_files[instrs[0]]
    files2 = sorted_files[instrs[1]]

    assert len(files1) == len(files2), 'An even number of files is needed'

    argdict = {"fftlen": fftlen,
               "save_dyn": save_dyn,
               "bintime": bintime,
               "pdsrebin": pdsrebin,
               "normalization": normalization,
               "back_ctrate": back_ctrate,
               "noclobber": noclobber}

    funcargs = []

    for i_f, f in enumerate(files1):
        f1, f2 = f, files2[i_f]

        outdir = os.path.dirname(f1)
        if outdir == '':
            outdir = os.getcwd()

        if outroot is None:
            outr = common_name(f1, f2, default='%d' % i_f)
        else:
            outr = outroot

        outname = os.path.join(outdir,
                               outr.replace(MP_FILE_EXTENSION, '') +
                               '_cpds' + MP_FILE_EXTENSION)

        funcargs.append([f1, f2, outname, argdict])

    if os.name == 'nt' or nproc == 1:
        [_wrap_fun_cpds(fa) for fa in funcargs]
    else:
        pool = Pool(processes=nproc)
        for i in pool.imap_unordered(_wrap_fun_cpds, funcargs):
            pass
        pool.close()


def read_fspec(fname):
    """Read the frequency spectrum from a file.

    Parameters
    ----------
    fname : str
        The input file name

    Returns
    -------
    ftype : str
        File type
    freq : array-like
        Frequency array
    fspec : array-like
        Frequency spectrum array
    efspec : array-like
        Errors on spectral bins
    nchunks : int
        Number of spectra that have been summed to obtain fspec
    rebin : array-like or int
        Rebin factor in each bin. Might be irregular in case of geometrical
        binning

    """
    ftype, contents = get_file_type(fname)
    if 'freq' in list(contents.keys()):
        freq = contents['freq']
    elif 'flo' in list(contents.keys()):
        flo = contents['flo']
        fhi = contents['fhi']
        freq = [flo, fhi]

    ft = ftype.replace('reb', '')
    pds = contents[ft]
    epds = contents['e' + ft]
    nchunks = contents['n' + ft]
    rebin = contents['rebin']

    return ftype, freq, pds, epds, nchunks, rebin, contents


def _normalize(array, ref=0):
    m = ref
    std = np.std(array)
    newarr = np.zeros_like(array)
    good = array > m
    newarr[good] = (array[good] - ref) / std
    return newarr


def dumpdyn(fname, plot=False):
    """Dump a dynamical frequency spectrum in text files.

    Parameters
    ----------
    fname : str
        The file name

    Other Parameters
    ----------------
    plot : bool
        if True, plot the spectrum

    """
    ftype, pdsdata = get_file_type(fname, specify_reb=False)

    dynpds = pdsdata['dyn' + ftype]
    edynpds = pdsdata['edyn' + ftype]

    try:
        freq = pdsdata['freq']
    except:
        flo = pdsdata['flo']
        fhi = pdsdata['fhi']
        freq = (fhi + flo) / 2

    time = pdsdata['dyntimes']
    freqs = np.zeros_like(dynpds)
    times = np.zeros_like(dynpds)

    for i, im in enumerate(dynpds):
        freqs[i, :] = freq
        times[i, :] = time[i]

    t = times.real.flatten()
    f = freqs.real.flatten()
    d = dynpds.real.flatten()
    e = edynpds.real.flatten()

    np.savetxt('{0}_dumped_{1}.txt'.format(mp_root(fname), ftype),
               np.array([t, f, d, e]).T)
    size = _normalize(d)
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(t, f, s=size)
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')

        plt.show()


def dumpdyn_main(args=None):
    import argparse

    description = ('Dump dynamical (cross) power spectra')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help=("List of files in any valid MaLTPyNT "
                                       "format for PDS or CPDS"), nargs='+')
    parser.add_argument("--noplot", help="plot results",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    fnames = args.files

    for f in fnames:
        dumpdyn(f, plot=not args.noplot)


def main(args=None):
    import argparse
    description = ('Create frequency spectra (PDS, CPDS, cospectrum) '
                   'starting from well-defined input ligthcurves')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of light curve files", nargs='+')
    parser.add_argument("-b", "--bintime", type=float, default=1/4096,
                        help="Light curve bin time; if negative, interpreted" +
                        " as negative power of 2." +
                        " Default: 2^-10, or keep input lc bin time" +
                        " (whatever is larger)")
    parser.add_argument("-r", "--rebin", type=int, default=1,
                        help="(C)PDS rebinning to apply. Default: none")
    parser.add_argument("-f", "--fftlen", type=float, default=512,
                        help="Length of FFTs. Default: 512 s")
    parser.add_argument("-k", "--kind", type=str, default="PDS,CPDS,cos",
                        help='Spectra to calculate, as comma-separated list' +
                        ' (Accepted: PDS and CPDS;' +
                        ' Default: "PDS,CPDS")')
    parser.add_argument("--norm", type=str, default="Leahy",
                        help='Normalization to use' +
                        ' (Accepted: Leahy and rms;' +
                        ' Default: "Leahy")')
    parser.add_argument("--noclobber", help="Do not overwrite existing files",
                        default=False, action='store_true')
    parser.add_argument("-o", "--outroot", type=str, default=None,
                        help='Root of output file names for CPDS only')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--nproc",
                        help=("Number of processors to use"),
                        default=1,
                        type=int)
    parser.add_argument("--back",
                        help=("Estimated background (non-source) count rate"),
                        default=0.,
                        type=float)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--save-dyn", help="save dynamical power spectrum",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPfspec.log', level=numeric_level,
                        filemode='w')

    bintime = args.bintime
    fftlen = args.fftlen
    pdsrebin = args.rebin
    normalization = args.norm

    do_cpds = do_pds = do_cos = do_lag = False
    kinds = args.kind.split(',')
    for k in kinds:
        if k == 'PDS':
            do_pds = True
        elif k == 'CPDS':
            do_cpds = True
        elif k == 'cos' or k == 'cospectrum':
            do_cos = True
            do_cpds = True
        elif k == 'lag':
            do_lag = True
            do_cpds = True

    calc_fspec(args.files, fftlen,
               do_calc_pds=do_pds,
               do_calc_cpds=do_cpds,
               do_calc_cospectrum=do_cos,
               do_calc_lags=do_lag,
               save_dyn=args.save_dyn,
               bintime=bintime,
               pdsrebin=pdsrebin,
               outroot=args.outroot,
               normalization=normalization,
               nproc=args.nproc,
               back_ctrate=args.back,
               noclobber=args.noclobber)
