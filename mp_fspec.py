from __future__ import division, print_function
from mp_base import mp_root, mp_cross_gtis, mp_create_gti_mask
from mp_base import mp_sort_files
from mp_rebin import mp_const_rebin
from mp_io import mp_get_file_type, mp_load_lcurve, mp_save_pds
from mp_io import MP_FILE_EXTENSION
import numpy as np


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
    freqs, ft = mp_fft(lc, bintime)
    # I'm pretty sure there is a faster way to do this.
    if nph != 0:
        pds = np.absolute(ft.conjugate() * ft) * 2. / nph
    else:
        pds = np.zeros(len(freqs))

    good = freqs >= 0
    freqs = freqs[good]
    pds = pds[good]

    if return_freq:
        return freqs, pds
    else:
        return pds


def mp_welch_pds(time, lc, bintime, fftlen, gti):
    '''Calculates the PDS of a light curve with constant binning time.'''
    start_times = \
        mp_decide_spectrum_intervals(gti, fftlen, verbose=False)

    pds = 0
    npds = len(start_times)

    for t in start_times:
        good = np.logical_and(time >= t, time < t + fftlen)
        l = lc[good]
        f, p = mp_leahy_pds(l, bintime)
        pds += p
    pds /= npds
    epds = pds / np.sqrt(npds)
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
    freqs, ft1 = mp_fft(lc1, bintime)
    freqs, ft2 = mp_fft(lc2, bintime)
    if nph1 <= 0 or nph2 <= 0:
        dum = np.zeros(len(ft1[freqs > 0]))
        return freqs, dum, dum

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


def mp_welch_cpds(time, lc1, lc2, bintime, fftlen, gti):
    '''Calculates the CPDS of a light curve with constant binning time.'''
    start_times = \
        mp_decide_spectrum_intervals(gti, fftlen, verbose=False)

    cpds = 0
    ecpds = 0
    npds = len(start_times)

    for t in start_times:
        good = np.logical_and(time >= t, time < t + fftlen)
        l1 = lc1[good]
        l2 = lc2[good]
        if np.sum(l1) == 0 or np.sum(l2) == 0:
            print ('Interval starting at time %.7f is bad. Check GTIs' % t)
            npds -= 1
            continue
        f, p, pe, p1, p2 = mp_leahy_cpds(l1, l2, bintime)
        cpds += p
        ecpds += pe ** 2

    cpds /= npds
    ecpds = np.sqrt(ecpds) / npds

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
        print ("Assuming background level 0")
        back_ctrate = 0
    factor = (source_ctrate + back_ctrate) / source_ctrate ** 2
    return pds * factor, pds_err * factor


def mp_decide_spectrum_intervals(gtis, fftlen, verbose=False):
    '''A way to avoid gaps. Start each FFT/PDS/cospectrum from the start of
    a GTI, and stop before the next gap.'''

    spectrum_start_times = np.array([])
    for g in gtis:
        if g[1] - g[0] < fftlen:
            if verbose:
                print ("Too short. Skipping.")
            continue
        newtimes = np.arange(g[0], g[1] - fftlen, np.longdouble(fftlen),
                             dtype=np.longdouble)
        spectrum_start_times = \
            np.append(spectrum_start_times,
                      newtimes)
    return spectrum_start_times


def mp_calc_pds(lcfile, fftlen,
                save_dyn=False,
                bintime=1,
                pdsrebin=1,
                normalization='Leahy'):
    '''Calculates the PDS from an input light curve file'''
    # TODO:Implement save_dyn
    if save_dyn:
        print ('Beware! save_dyn not yet implemented')

    print ("Loading file %s..." % lcfile)
    lcdata = mp_load_lcurve(lcfile)
    time = lcdata['time']
    lc = lcdata['lc']
    dt = lcdata['dt']
    gti = lcdata['GTI']
    instr = lcdata['Instr']

    if bintime <= dt:
        bintime = dt
    else:
        lcrebin = np.rint(bintime / dt)
        bintime = lcrebin * dt
        print ("Rebinning lc by a factor %d" % lcrebin)
        time, lc, dum = \
            mp_const_rebin(time, lc, lcrebin, normalize=False)

    freq, pds, epds, npds = \
        mp_welch_pds(time, lc, bintime, fftlen, gti)

    freq, pds, epds = mp_const_rebin(freq[1:], pds[1:], pdsrebin,
                                     epds[1:])

    if normalization == 'rms':
        print ('Applying %s normalization' % normalization)
        mask = mp_create_gti_mask(time, gti)
        ctrate = np.mean(lc[mask]) / bintime
        # TODO: allow to specify background ctrate
        pds, epds = \
            mp_rms_normalize_pds(pds, epds,
                                 source_ctrate=ctrate,
                                 back_ctrate=0)
    root = mp_root(lcfile)
    outdata = {'time': time[0], 'pds': pds, 'epds': epds, 'npds': npds,
               'fftlen': fftlen, 'Instr': instr, 'freq': freq,
               'rebin': pdsrebin, 'norm': normalization}
    outname = root + '_pds' + MP_FILE_EXTENSION
    print ('Saving PDS to %s' % outname)
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
        print ('Beware! save_dyn not yet implemented')

    print ("Loading file %s..." % lcfile1)
    lcdata1 = mp_load_lcurve(lcfile1)
    print ("Loading file %s..." % lcfile2)
    lcdata2 = mp_load_lcurve(lcfile2)

    time1 = lcdata1['time']
    lc1 = lcdata1['lc']
    dt1 = lcdata1['dt']
    gti1 = lcdata1['GTI']
    instr1 = lcdata1['Instr']

    time2 = lcdata2['time']
    lc2 = lcdata2['lc']
    dt2 = lcdata2['dt']
    gti2 = lcdata2['GTI']
    instr2 = lcdata2['Instr']

    assert instr1 != instr2, 'Did you check the ordering of files? ' + \
        "These are both " + instr1

    assert dt1 == dt2, 'Light curves are sampled differently'
    dt = dt1

    if bintime <= dt:
        bintime = dt
    else:
        lcrebin = np.rint(bintime / dt)
        dt = bintime
        print ("Rebinning lcs by a factor %d" % lcrebin)
        time1, lc1, dum = \
            mp_const_rebin(time1, lc1, lcrebin, normalize=False)
        time2, lc2, dum = \
            mp_const_rebin(time2, lc2, lcrebin, normalize=False)

#    from test_cross_gtis import mp_cross_gtis
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
    freq, cpds, ecpds, ncpds = \
        mp_welch_cpds(time, lc1, lc2, bintime, fftlen, gti)

    freq, cpds, ecpds = mp_const_rebin(freq[1:], cpds[1:], pdsrebin,
                                       ecpds[1:])

    if normalization == 'rms':
        print ('Applying %s normalization' % normalization)
        mask = mp_create_gti_mask(time, gti)
        ctrate = np.mean(np.sqrt(lc1[mask] * lc2[mask])) / bintime
        # TODO: allow to specify background ctrate
        cpds, ecpds = \
            mp_rms_normalize_pds(cpds, ecpds,
                                 source_ctrate=ctrate,
                                 back_ctrate=0)

    outdata = {'time': gti[0][0], 'cpds': cpds, 'ecpds': ecpds, 'ncpds': ncpds,
               'fftlen': fftlen, 'Instrs': instr1 + ',' + instr2,
               'freq': freq, 'rebin': pdsrebin, 'norm': normalization}
    print ('Saving CPDS to %s' % outname)
    mp_save_pds(outdata, outname)


def mp_calc_lags(freqs, cpds, pds1, pds2, n_chunks, rebin):
    '''Calculates time lags'''
    lags = np.angle(cpds) / (2 * np.pi * freqs)
    sigcpd = np.absolute(cpds)

    rawcof = (sigcpd) ** 2 / ((pds1) * (pds1))

    dum = (1. - rawcof) / (2. * rawcof)

    lagse = np.sqrt(dum / n_chunks / rebin) / (2 * np.pi * freqs)
    return lags, lagse


def mp_calc_fspec(files, fftlen,
                  calc_pds=True,
                  calc_cpds=True,
                  calc_cospectrum=True,
                  calc_lags=True,
                  save_dyn=False,
                  bintime=1,
                  pdsrebin=1,
                  outroot='cpds',
                  normalization='Leahy'):
    '''Calculates the frequency spectra:
        the PDS, the CPDS, the cospectrum, ...'''
    # TODO: Implement cospectrum
    # TODO: Implement lags
    import os

    if normalization not in ['Leahy', 'rms']:
        print ('Beware! Unknown normalization!')
        normalization == 'Leahy'

    print ('Using %s normalization' % normalization)

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
        print ('Sorting file list')
        sorted_files = mp_sort_files(files)
        print ('Beware! For cpds and derivatives, I assume that the files are')
        print ('from only two instruments and in pairs (even in random order)')

        instrs = sorted_files.keys()
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

        outname = os.path.join(outdir,
                               outroot + "_%d" % i_f + MP_FILE_EXTENSION)
        mp_calc_cpds(f1, f2, fftlen,
                     save_dyn=save_dyn,
                     bintime=bintime,
                     pdsrebin=pdsrebin,
                     outname=outname,
                     normalization=normalization)


def mp_read_fspec(fname):
    ftype, contents = mp_get_file_type(fname)
    if 'freq' in contents.keys():
        freq = contents['freq']
    elif 'flo' in contents.keys():
        flo = contents['flo']
        fhi = contents['fhi']
        freq = [flo, fhi]

    ft = ftype.replace('reb', '')
    pds = contents[ft]
    epds = contents['e' + ft]
    nchunks = contents['n' + ft]
    rebin = contents['rebin']

    return ftype, freq, pds, epds, nchunks, rebin


if __name__ == '__main__':
    import argparse
    description = 'Creates frequency spectra (PDS, CPDS, cospectrum)' + \
        ' starting from well-defined input ligthcurves'
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

    args = parser.parse_args()

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
