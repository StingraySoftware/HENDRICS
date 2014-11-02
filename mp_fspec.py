from __future__ import division, print_function
from mp_base import mp_root, mp_cross_gtis, mp_create_gti_mask
from mp_rebin import mp_const_rebin
import numpy as np


def mp_fft(lc, bintime):
    '''A wrapper for the fft function. Just numpy for now'''
    nbin = len(lc)

    ft = np.fft.fft(lc)
    freqs = np.fft.fftfreq(nbin, bintime)

    return freqs, ft


def mp_leahy_pds(lc, bintime, return_freq=True):
    '''
    Calculates the Power Density Spectrum \'a la Leahy (1983), given the
    lightcurve and its bin time.
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


def mp_leahy_cpds(lc1, lc2, bintime, return_freq=True):
    '''
    Calculates the Cross Power Density Spectrum, normalized similarly to the
    PDS in Leahy (1983), given the lightcurve and its bin time.
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

    # The "effective" count rate is the geometrical mean of the count rates
    # of the two light curves
    nph = np.sqrt(nph1 * nph2)
    # I'm pretty sure there is a faster way to do this.
    if nph != 0:
        cpds = ft1.conjugate() * ft2 * 2. / nph
    else:
        cpds = np.zeros(len(freqs))

    good = freqs >= 0
    freqs = freqs[good]
    cpds = cpds[good]

    if return_freq:
        return freqs, cpds
    else:
        return cpds


def mp_welch_cpds(time, lc1, lc2, bintime, fftlen, gti):
    '''Calculates the PDS of a light curve with constant binning time.'''
    start_times = \
        mp_decide_spectrum_intervals(gti, fftlen, verbose=False)

    cpds = 0
    npds = len(start_times)

    for t in start_times:
        good = np.logical_and(time >= t, time < t + fftlen)
        l1 = lc1[good]
        l2 = lc2[good]
        f, p = mp_leahy_cpds(l1, l2, bintime)
        cpds += p
    cpds /= npds
    ecpds = cpds / np.sqrt(npds)
    return f, cpds, ecpds, npds


def rms_normalize_pds(pds, pds_err, source_ctrate, back_ctrate=None):
    '''
    Normalize a Leahy PDS with RMS normalization.
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
        if verbose:
            print("Calculating PDS over GTI %g--%g" % (g[0], g[1]))
        if g[1] - g[0] < fftlen:
            if verbose:
                print("Too short. Skipping.")
            continue
        spectrum_start_times = \
            np.append(spectrum_start_times,
                      np.arange(g[0], g[1] - fftlen, np.longdouble(fftlen),
                                dtype=np.longdouble))
    return spectrum_start_times


def mp_calc_pds(lcfile, fftlen,
                save_dyn=False,
                bintime=1,
                pdsrebin=1):
    '''Calculates the PDS from an input light curve file'''
    # TODO:Implement save_dyn

    print("Loading file %s..." % lcfile)
    lcdata = pickle.load(open(lcfile))
    time = lcdata['time']
    lc = lcdata['lc']
    dt = lcdata['dt']
    gti = lcdata['gti']
    instr = lcdata['Instr']

    if bintime <= dt:
        bintime = dt
    else:
        lcrebin = np.rint(bintime / dt)
        bintime = lcrebin * dt
        print("Rebinning lc by a factor %d" % lcrebin)
        time, lc, dum = \
            mp_const_rebin(time, lc, lcrebin, normalize=False)

    freq, pds, epds, npds = \
        mp_welch_pds(time, lc, bintime, fftlen, gti)

    freq, pds, epds = mp_const_rebin(freq[1:], pds[1:], pdsrebin,
                                     epds[1:])

    root = mp_root(lcfile)
    outdata = {'time': time[0], 'pds': pds, 'epds': epds, 'npds': npds,
               'fftlen': fftlen, 'Instr': instr, 'freq': freq,
               'rebin': pdsrebin}
    pickle.dump(outdata, open(root + '_pds.p', 'wb'))


def mp_calc_cpds(lcfile1, lcfile2, fftlen,
                 save_dyn=False,
                 bintime=1,
                 pdsrebin=1,
                 outname='cpds.p'):
    '''Calculates the Cross Power Density Spectrum from a pair of
    input light curve files'''
    # TODO:Implement save_dyn

    print("Loading file %s..." % lcfile1)
    lcdata1 = pickle.load(open(lcfile1))
    print("Loading file %s..." % lcfile2)
    lcdata2 = pickle.load(open(lcfile2))

    time1 = lcdata1['time']
    lc1 = lcdata1['lc']
    dt1 = lcdata1['dt']
    gti1 = lcdata1['gti']
    instr1 = lcdata1['Instr']

    time2 = lcdata2['time']
    lc2 = lcdata2['lc']
    dt2 = lcdata2['dt']
    gti2 = lcdata2['gti']
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
        print("Rebinning lcs by a factor %d" % lcrebin)
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

    freq, cpds, ecpds, ncpds = \
        mp_welch_cpds(time, lc1, lc2, bintime, fftlen, gti)

    freq, cpds, ecpds = mp_const_rebin(freq[1:], cpds[1:], pdsrebin,
                                       ecpds[1:])

    outdata = {'time': gti[0][0], 'cpds': cpds, 'ecpds': ecpds, 'ncpds': ncpds,
               'fftlen': fftlen, 'Instrs': instr1 + ',' + instr2,
               'freq': freq, 'rebin': pdsrebin}
    pickle.dump(outdata, open(outname, 'wb'))


def mp_calc_fspec(files, fftlen,
                  calc_pds=True,
                  calc_cpds=True,
                  calc_cospectrum=True,
                  calc_lags=True,
                  save_dyn=False,
                  bintime=1,
                  pdsrebin=1,
                  outroot='cpds'):
    '''Calculates the frequency spectra:
        the PDS, the CPDS, the cospectrum, ...'''
    # TODO: Implement cospectrum
    # TODO: Implement lags
    import os

    if calc_pds:
        for lcf in files:
            mp_calc_pds(lcf, fftlen,
                        save_dyn=save_dyn,
                        bintime=bintime,
                        pdsrebin=pdsrebin)

    if not calc_cpds:
        return
    print('Beware! For cpds and derivatives, I assume that the files are')
    print('ordered as follows: obs1_FPMA, obs1_FPMB, obs2_FPMA, obs2_FPMB...')
    files1 = files[:-1:2]
    files2 = files[1::2]

    assert len(files1) == len(files2), 'An even number of files is needed'

    for i_f, f in enumerate(files1):
        f1, f2 = f, files2[i_f]

        outdir = os.path.dirname(f1)

        mp_calc_cpds(f1, f2, fftlen,
                     save_dyn=save_dyn,
                     bintime=bintime,
                     pdsrebin=pdsrebin,
                     outname=outdir + "/" + outroot + "_%d.p" % i_f)


if __name__ == '__main__':
    import argparse
    import cPickle as pickle
    parser = argparse.ArgumentParser()

    parser.add_argument("files", help="List of light curve files", nargs='+')
    parser.add_argument("-b", "--bintime", type=float, default=1/4096,
                        help="Light curve bin time; if negative, interpreted" +
                        " as negative power of 2")
    parser.add_argument("-r", "--rebin", type=int, default=1,
                        help="(C)PDS rebinning to apply")
    parser.add_argument("-f", "--fftlen", type=float, default=512,
                        help="Length of FFTs")
    parser.add_argument("-k", "--kind", type=str, default="PDS,CPDS,cos",
                        help='Spectra to calculate, as comma-separated list' +
                        ' (Accepted: PDS, CPDS;' +
                        ' Default: PDS, CPDS)')
    parser.add_argument("-o", "--outroot", type=str, default="cpds",
                        help='Root of output file names for CPDS only')

    args = parser.parse_args()

    bintime = args.bintime
    fftlen = args.fftlen
    pdsrebin = args.rebin

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
                  outroot=args.outroot)
