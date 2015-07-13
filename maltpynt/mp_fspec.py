# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .mp_base import mp_root, mp_cross_gtis, mp_create_gti_mask
from .mp_base import mp_sort_files, common_name
from .mp_rebin import mp_const_rebin
from .mp_io import mp_get_file_type, mp_load_lcurve, mp_save_pds
from .mp_io import MP_FILE_EXTENSION
import numpy as np
import logging
from multiprocessing import Pool
import functools


def _wrap_calc_cpds(arglist, **kwargs):
    f1, f2, outname = arglist
    return mp_calc_cpds(f1, f2, outname=outname, **kwargs)


class _empty():
    def __init__(self):
        pass


def mp_fft(lc, bintime):
    '''A wrapper for the fft function. Just numpy for now'''
    nbin = len(lc)

    ft = np.fft.fft(lc)
    freqs = np.fft.fftfreq(nbin, bintime)

    return freqs.astype(np.double), ft


def mp_leahy_pds(lc, bintime, return_freq=True):
    '''
    Calculates the Power Density Spectrum \'a la Leahy+1983, ApJ 266, 160,
    given the lightcurve and its bin time.
    Assumes no gaps are present! Beware!
    Keyword arguments:
        return_freq: (bool, default True) Return the frequencies corresponding
                     to the PDS bins?
    '''

    nph = sum(lc)

    # Checks must be done before. At this point, only good light curves have to
    # be provided
    assert (nph > 0), 'Invalid interval. Light curve is empty'

    freqs, ft = mp_fft(lc, bintime)
    # I'm pretty sure there is a faster way to do this.
    pds = np.absolute(ft.conjugate() * ft) * 2. / nph

    good = freqs >= 0
    freqs = freqs[good]
    pds = pds[good]

    if return_freq:
        return freqs, pds
    else:
        return pds


def mp_welch_pds(time, lc, bintime, fftlen, gti=None, return_ctrate=False,
                 return_all=False):
    '''
    Calculates the Power Density Spectrum \'a la Leahy (1983), given the
    lightcurve and its bin time, over equal chunks of length fftlen, and
    returns the average of all PDSs, or the sum PDS and the number of chunks
    Arguments:
        time:         central times of light curve bins
        lc:           light curve
        bintime:      bin time of the light curve
        fftlen:       length of each FFT

    Keyword arguments:
        gti:          good time intervals [[g1s, g1e], [g2s,g2e], [g3s,g3e],..]
                      defaults to [[time[0] - bintime/2, time[-1] + bintime/2]]
        return_ctrate:if True, return also the count rate

    Return values:
        freq:         array of frequencies corresponding to PDS bins
        pds:          the values of the PDS
        pds_err:      the values of the PDS
        n_chunks:     the number of summed PDSs (if normalize is False)
        ctrate:       the average count rate in the two lcs
    '''
    if gti is None:
        gti = [[time[0] - bintime / 2, time[-1] + bintime / 2]]

    start_bins, stop_bins = \
        mp_decide_spectrum_lc_intervals(gti, fftlen, time)

    if return_all:
        results = _empty()
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

        f, p = mp_leahy_pds(l, bintime)

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

    if return_all:
        results.f = f
        results.pds = pds
        results.epds = epds
        results.npds = npds
        results.ctrate = ctrate

        return results

    if return_ctrate:
        return f, pds, epds, npds, ctrate
    else:
        return f, pds, epds, npds


def mp_leahy_cpds(lc1, lc2, bintime, return_freq=True, return_pdss=False):
    '''
    Calculates the Cross Power Density Spectrum, normalized similarly to the
    PDS in Leahy+1983, ApJ 266, 160., given the lightcurve and its bin time.
    Assumes no gaps are present! Beware!
    Keyword arguments:
        return_freq: (bool, default True) Return the frequencies corresponding
                     to the CPDS bins?
    '''
    assert len(lc1) == len(lc2), 'Light curves MUST have the same length!'
    nph1 = sum(lc1)
    nph2 = sum(lc2)
    # Checks must be done before. At this point, only good light curves have to
    # be provided
    assert (nph1 > 0 and nph2 > 0), ('Invalid interval. At least one light '
                                     'curve is empty')

    freqs, ft1 = mp_fft(lc1, bintime)
    freqs, ft2 = mp_fft(lc2, bintime)

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

    if return_freq:
        result = [freqs, cpds, cpdse]
    else:
        result = [cpds, cpdse]
    if return_pdss:
        result.extend([pds1, pds2])
    return result


def mp_welch_cpds(time, lc1, lc2, bintime, fftlen, gti=None,
                  return_ctrate=False, return_all=False):
    '''
    Calculates the Cross Power Density Spectrum normalized like PDS, given the
    lightcurve and its bin time, over equal chunks of length fftlen, and
    returns the average of all PDSs, or the sum PDS and the number of chunks
    Arguments:
        time:         central times of light curve bins
        lc1:          light curve 1
        lc2:          light curve 2
        bintime:      bin time of the light curve
        fftlen:       length of each FFT

    Keyword arguments:
        gti:          good time intervals [[g1s, g1e], [g2s,g2e], [g3s,g3e],..]
                      defaults to [[time[0] - bintime/2, time[-1] + bintime/2]]
        return_ctrate:if True, return also the count rate

    Return values:
        freq:         array of frequencies corresponding to CPDS bins
        pds:          the values of the CPDS
        pds_err:      the values of the CPDS
        n_chunks:     the number of summed CPDSs
        ctrate:       the average count rate in the two lcs
    '''
    if gti is None:
        gti = [[time[0] - bintime / 2, time[-1] + bintime / 2]]

    start_bins, stop_bins = \
        mp_decide_spectrum_lc_intervals(gti, fftlen, time)

    cpds = 0
    ecpds = 0
    npds = len(start_bins)
    mask = np.zeros(len(lc1), dtype=np.bool)

    if return_all:
        results = _empty()
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

        f, p, pe, p1, p2 = mp_leahy_cpds(l1, l2, bintime, return_pdss=True)
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

    if return_all:
        results.f = f
        results.cpds = cpds
        results.ecpds = ecpds
        results.ncpds = npds
        results.ctrate = ctrate

        return results

    if return_ctrate:
        return f, cpds, ecpds, npds, ctrate
    else:
        return f, cpds, ecpds, npds


def mp_rms_normalize_pds(pds, pds_err, source_ctrate, back_ctrate=None):
    '''
    Normalize a Leahy PDS with RMS normalization
    (Belloni & Hasinger 1990, A&A, 230, 103; Miyamoto+1991, ApJ, 383, 784).
    Inputs:
        pds:           the Leahy-normalized PDS
        pds_err:       the uncertainties on the PDS values
        source_ctrate: the source count rate
        back_ctrate:   (optional) the background count rate
    Outputs:
        pds:           the RMS-normalized PDS
        pds_err:       the uncertainties on the PDS values
    '''
    if back_ctrate is None:
        logging.warning("Assuming background level 0")
        back_ctrate = 0
    factor = (source_ctrate + back_ctrate) / source_ctrate ** 2
    return pds * factor, pds_err * factor


def mp_decide_spectrum_intervals(gtis, fftlen):
    '''A way to avoid gaps. Start each FFT/PDS/cospectrum from the start of
    a GTI, and stop before the next gap.
    Only use for events! This will give problems with binned light curves'''

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

    return spectrum_start_times


def mp_decide_spectrum_lc_intervals(gtis, fftlen, time):
    """Similar to mp_decide_spectrum_intervals, but dedicated to light curves.

    In this case, it is necessary to specify the time array containing the
    times of the light curve bins.
    Returns start and stop bins of the intervals to use for the PDS
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

    return spectrum_start_bins, spectrum_start_bins + nbin


def mp_calc_pds(lcfile, fftlen,
                save_dyn=False,
                bintime=1,
                pdsrebin=1,
                normalization='Leahy',
                back_ctrate=0.):
    '''Calculates the PDS from an input light curve file'''

    logging.info("Loading file %s..." % lcfile)
    lcdata = mp_load_lcurve(lcfile)
    time = lcdata['time']
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
            mp_const_rebin(time, lc, lcrebin, normalize=False)

    try:
        results = mp_welch_pds(time, lc, bintime, fftlen, gti, return_all=True)
        freq = results.f
        pds = results.pds
        epds = results.epds
        npds = results.npds
        ctrate = results.ctrate
    except Exception as e:
        # If it fails, exit cleanly
        logging.error("{} failed ({}: {})".format('Problems with the PDS.',
                                                  type(e), e))
        raise Exception("{} failed ({}: {})".format('Problems with the PDS.',
                                                    type(e), e))

    freq, pds, epds = mp_const_rebin(freq[1:], pds[1:], pdsrebin,
                                     epds[1:])

    if normalization == 'rms':
        logging.info('Applying %s normalization' % normalization)

        pds, epds = \
            mp_rms_normalize_pds(pds, epds,
                                 source_ctrate=ctrate,
                                 back_ctrate=back_ctrate)

        for ic, pd in enumerate(results.dynpds):
            ep = results.edynpds[ic].copy()
            ct = results.dynctrate[ic].copy()

            pd, ep = mp_rms_normalize_pds(pd, ep, source_ctrate=ct,
                                          back_ctrate=back_ctrate)
            results.edynpds[ic][:] = ep
            results.dynpds[ic][:] = pd

    root = mp_root(lcfile)
    outdata = {'time': time[0], 'pds': pds, 'epds': epds, 'npds': npds,
               'fftlen': fftlen, 'Instr': instr, 'freq': freq,
               'rebin': pdsrebin, 'norm': normalization, 'ctrate': ctrate,
               'total_ctrate': tctrate,
               'back_ctrate': back_ctrate}
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

    outname = root + '_pds' + MP_FILE_EXTENSION
    logging.info('Saving PDS to %s' % outname)
    mp_save_pds(outdata, outname)


def mp_calc_cpds(lcfile1, lcfile2, fftlen,
                 save_dyn=False,
                 bintime=1,
                 pdsrebin=1,
                 outname='cpds' + MP_FILE_EXTENSION,
                 normalization='Leahy',
                 back_ctrate=0.):
    '''Calculates the Cross Power Density Spectrum from a pair of
    input light curve files'''

    logging.info("Loading file %s..." % lcfile1)
    lcdata1 = mp_load_lcurve(lcfile1)
    logging.info("Loading file %s..." % lcfile2)
    lcdata2 = mp_load_lcurve(lcfile2)

    time1 = lcdata1['time']
    lc1 = lcdata1['lc']
    dt1 = lcdata1['dt']
    gti1 = lcdata1['GTI']
    instr1 = lcdata1['Instr']
    tctrate1 = lcdata1['total_ctrate']

    time2 = lcdata2['time']
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
            mp_const_rebin(time1, lc1, lcrebin, normalize=False)
        time2, lc2, dum = \
            mp_const_rebin(time2, lc2, lcrebin, normalize=False)

    gti = mp_cross_gtis([gti1, gti2])

    mask1 = mp_create_gti_mask(time1, gti)
    mask2 = mp_create_gti_mask(time2, gti)
    time1 = time1[mask1]
    time2 = time2[mask2]

    assert np.all(time1 == time2), "Something's not right in GTI filtering"
    time = time1
    del time2

    lc1 = lc1[mask1]
    lc2 = lc2[mask2]

    try:
        results = mp_welch_cpds(time, lc1, lc2, bintime, fftlen, gti,
                                return_all=True)
        freq = results.f
        cpds = results.cpds
        ecpds = results.ecpds
        ncpds = results.ncpds
        ctrate = results.ctrate
    except Exception as e:
        # If it fails, exit cleanly
        logging.error("{} failed ({}: {})".format('Problems with the CPDS.',
                                                  type(e), e))
        raise Exception("{} failed ({}: {})".format('Problems with the CPDS.',
                                                    type(e), e))

    freq, cpds, ecpds = mp_const_rebin(freq[1:], cpds[1:], pdsrebin,
                                       ecpds[1:])

    if normalization == 'rms':
        logging.info('Applying %s normalization' % normalization)
        cpds, ecpds = \
            mp_rms_normalize_pds(cpds, ecpds,
                                 source_ctrate=ctrate,
                                 back_ctrate=back_ctrate)
        for ic, cp in enumerate(results.dyncpds):
            ec = results.edyncpds[ic].copy()
            ct = results.dynctrate[ic].copy()

            cp, ec = mp_rms_normalize_pds(cp, ec, source_ctrate=ct,
                                          back_ctrate=back_ctrate)
            results.edyncpds[ic][:] = ec
            results.dyncpds[ic][:] = cp

    outdata = {'time': gti[0][0], 'cpds': cpds, 'ecpds': ecpds, 'ncpds': ncpds,
               'fftlen': fftlen, 'Instrs': instr1 + ',' + instr2,
               'freq': freq, 'rebin': pdsrebin, 'norm': normalization,
               'ctrate': ctrate, 'total_ctrate': tctrate,
               'back_ctrate': back_ctrate}

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
    mp_save_pds(outdata, outname)


def mp_calc_fspec(files, fftlen,
                  calc_pds=True,
                  calc_cpds=True,
                  calc_cospectrum=True,
                  calc_lags=True,
                  save_dyn=False,
                  bintime=1,
                  pdsrebin=1,
                  outroot=None,
                  normalization='Leahy',
                  nproc=1,
                  back_ctrate=0.):
    '''Calculates the frequency spectra: the PDS, the cospectrum, ...'''

    import os

    if normalization not in ['Leahy', 'rms']:
        logging.warning('Beware! Unknown normalization!')
        normalization = 'Leahy'

    logging.info('Using %s normalization' % normalization)

    if calc_pds:
        wrap_fun = functools.partial(
            mp_calc_pds, fftlen=fftlen,
            save_dyn=save_dyn,
            bintime=bintime,
            pdsrebin=pdsrebin,
            normalization=normalization,
            back_ctrate=back_ctrate)

        with Pool(processes=nproc) as pool:
            for i in pool.imap_unordered(wrap_fun, files):
                pass

    if not calc_cpds or len(files) < 2:
        return

    if len(files) > 2:
        logging.info('Sorting file list')
        sorted_files = mp_sort_files(files)
        logging.warning('Beware! For cpds and derivatives, I assume that the'
                        'files are from only two instruments and in pairs'
                        '(even in random order)')

        instrs = list(sorted_files.keys())
        files1 = sorted_files[instrs[0]]
        files2 = sorted_files[instrs[1]]
    else:
        files1 = [files[0]]
        files2 = [files[1]]

    assert len(files1) == len(files2), 'An even number of files is needed'

    wrap_fun = functools.partial(
        _wrap_calc_cpds, fftlen=fftlen,
        save_dyn=save_dyn,
        bintime=bintime,
        pdsrebin=pdsrebin,
        normalization=normalization,
        back_ctrate=back_ctrate)

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

        funcargs.append([f1, f2, outname])

    with Pool(processes=nproc) as pool:
        for i in pool.imap_unordered(wrap_fun, funcargs):
            pass


def mp_read_fspec(fname):
    ftype, contents = mp_get_file_type(fname)
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


if __name__ == '__main__':
    import sys
    import subprocess as sp

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MPfspec'] + args)
