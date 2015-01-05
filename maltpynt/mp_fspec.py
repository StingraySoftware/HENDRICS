from __future__ import division, print_function
from .mp_base import mp_root, mp_cross_gtis, mp_create_gti_mask
from .mp_base import mp_sort_files, common_name
from .mp_rebin import mp_const_rebin
from .mp_io import mp_get_file_type, mp_load_lcurve, mp_save_pds
from .mp_io import MP_FILE_EXTENSION
import numpy as np
import logging


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


def mp_welch_pds(time, lc, bintime, fftlen, gti=None, return_ctrate=False):
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

        pds += p
        mask[start_bin:stop_bin] = True

    pds /= npds
    epds = pds / np.sqrt(npds)

    if return_ctrate:
        return f, pds, epds, npds, np.mean(lc[mask]) / bintime
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
                  return_ctrate=False):
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
        mask[start_bin:stop_bin] = True

    cpds /= npds
    ecpds = np.sqrt(ecpds) / npds

    if return_ctrate:
        return f, cpds, ecpds, npds, \
            np.sqrt(np.mean(lc1[mask])*np.mean(lc2[mask])) / bintime
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
    '''Similar to mp_decide_spectrum_intervals, but dedicated to light curves.
    In this case, it is necessary to specify the time array containing the
    times of the light curve bins.
    Returns start and stop bins of the intervals to use for the PDS'''

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
                normalization='Leahy'):
    '''Calculates the PDS from an input light curve file'''
    # TODO:Implement save_dyn
    if save_dyn:
        logging.warning('Beware! save_dyn not yet implemented')

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
        freq, pds, epds, npds, ctrate = \
            mp_welch_pds(time, lc, bintime, fftlen, gti, return_ctrate=True)
    except:
        # If it fails, exit cleanly
        logging.error('Problems with the PDS. Check input files!')
        return -1

    freq, pds, epds = mp_const_rebin(freq[1:], pds[1:], pdsrebin,
                                     epds[1:])

    if normalization == 'rms':
        logging.info('Applying %s normalization' % normalization)
        # TODO: allow to specify background ctrate.
        # Do not confuse with the count rate outside the source region.
        # Here we are talking about contamination inside the source region
        pds, epds = \
            mp_rms_normalize_pds(pds, epds,
                                 source_ctrate=ctrate,
                                 back_ctrate=0)

    root = mp_root(lcfile)
    outdata = {'time': time[0], 'pds': pds, 'epds': epds, 'npds': npds,
               'fftlen': fftlen, 'Instr': instr, 'freq': freq,
               'rebin': pdsrebin, 'norm': normalization, 'ctrate': ctrate,
               'total_ctrate': tctrate}

    outname = root + '_pds' + MP_FILE_EXTENSION
    logging.info('Saving PDS to %s' % outname)
    mp_save_pds(outdata, outname)


def mp_calc_cpds(lcfile1, lcfile2, fftlen,
                 save_dyn=False,
                 bintime=1,
                 pdsrebin=1,
                 outname='cpds' + MP_FILE_EXTENSION,
                 normalization='Leahy'):
    '''Calculates the Cross Power Density Spectrum from a pair of
    input light curve files'''
    # TODO:Implement save_dyn
    if save_dyn:
        logging.warning('Beware! save_dyn not yet implemented')

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
        freq, cpds, ecpds, ncpds, ctrate = \
            mp_welch_cpds(time, lc1, lc2, bintime, fftlen, gti,
                          return_ctrate=True)
    except:
        # If it fails, exit cleanly
        logging.error('Problems with the CPDS. Check input files!')
        return -1

    freq, cpds, ecpds = mp_const_rebin(freq[1:], cpds[1:], pdsrebin,
                                       ecpds[1:])

    if normalization == 'rms':
        logging.info('Applying %s normalization' % normalization)
        # TODO: allow to specify background ctrate
        cpds, ecpds = \
            mp_rms_normalize_pds(cpds, ecpds,
                                 source_ctrate=ctrate,
                                 back_ctrate=0)

    outdata = {'time': gti[0][0], 'cpds': cpds, 'ecpds': ecpds, 'ncpds': ncpds,
               'fftlen': fftlen, 'Instrs': instr1 + ',' + instr2,
               'freq': freq, 'rebin': pdsrebin, 'norm': normalization,
               'ctrate': ctrate, 'total_ctrate': tctrate}

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
                  normalization='Leahy'):
    '''Calculates the frequency spectra:
        the PDS, the CPDS, the cospectrum, ...'''
    # TODO: Implement cospectrum
    # TODO: Implement lags
    import os

    if normalization not in ['Leahy', 'rms']:
        logging.warning('Beware! Unknown normalization!')
        normalization = 'Leahy'

    logging.info('Using %s normalization' % normalization)

    if calc_pds:
        for lcf in files:
            mp_calc_pds(lcf, fftlen,
                        save_dyn=save_dyn,
                        bintime=bintime,
                        pdsrebin=pdsrebin,
                        normalization=normalization)

    if not calc_cpds:
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

    for i_f, f in enumerate(files1):
        f1, f2 = f, files2[i_f]

        outdir = os.path.dirname(f1)
        if outdir == '':
            outdir = os.getcwd()

        if outroot is None:
            outroot = common_name(f1, f2, default='cpds_%d' % i_f)

        outname = os.path.join(outdir,
                               outroot + MP_FILE_EXTENSION)
        mp_calc_cpds(f1, f2, fftlen,
                     save_dyn=save_dyn,
                     bintime=bintime,
                     pdsrebin=pdsrebin,
                     outname=outname,
                     normalization=normalization)


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
    parser.add_argument("-o", "--outroot", type=str, default="cpds",
                        help='Root of output file names for CPDS only')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG;"
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    args = parser.parse_args()

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

    mp_calc_fspec(args.files, fftlen,
                  calc_pds=do_pds,
                  calc_cpds=do_cpds,
                  calc_cospectrum=do_cos,
                  calc_lags=do_lag,
                  save_dyn=False,
                  bintime=bintime,
                  pdsrebin=pdsrebin,
                  outroot=args.outroot,
                  normalization=normalization)
