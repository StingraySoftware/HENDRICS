from __future__ import division, print_function
from mp_base import mp_root, mp_cross_gtis
from mp_lcurve import mp_join_lightcurves, mp_scrunch_lightcurves
import numpy as np


def mp_fft(lc, bin_time):
    '''A wrapper for the fft function. Just numpy for now'''
    nbin = len(lc)

    ft = np.fft.fft(lc)
    freqs = np.fft.fftfreq(nbin, bin_time)

    return ft, freqs


def mp_leahy_pds(lc, bin_time, return_freq=True):
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
    #    print nph
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


def mp_leahy_cpds(lc1, lc2, bin_time, return_freq=True):
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


def mp_calc_fspec(files, fftlen,
                  calc_pds=True,
                  calc_cpds=True,
                  calc_cospectrum=True,
                  calc_lags=True,
                  save_dyn=False,
                  bintime=1):
    '''Calculates the frequency spectra:
        the PDS, the CPDS, the cospectrum, ...'''

#    if calc_pds:
#        for f in files:
#        start_times = \
#            mp_decide_spectrum_intervals(gti, fftlen, verbose=False)
#
#        for t in start_times:
#        for inst in instrs:
#            time = lcdata[inst]['time']
#            lc = lcdata[inst]['lc']
#            mp_leahy_pds()
#

    # TODO: Implement PDS
    # TODO: Implement CPDS
    # TODO: Implement cospectrum
    # TODO: Implement lag

if __name__ == '__main__':
    import argparse
    import cPickle as pickle
    parser = argparse.ArgumentParser()

    parser.add_argument("files", help="List of light curve files", nargs='+')
    parser.add_argument("-b", "--bintime", type=float, default=1/4096,
                        help="Bin time; if negative, negative power of 2")
    parser.add_argument("-f", "--fftlen", type=float, default=512,
                        help="Length of FFTs")
    parser.add_argument("-k", "--kind", type=str, default="PDS,CPDS,cos",
                        help='Spectra to calculate, as comma-separated list' +
                        ' (Accepted: PDS, CPDS, cos[pectrum], lag, all;' +
                        ' (Default: PDS, CPDS, cos[pectrum])')

    parser.add_argument("-o", "--outroot", type=str, default="out",
                        help='Root of output file names')

    args = parser.parse_args()

    bintime = args.bintime
    fftlen = args.fftlen

    do_cpds = do_pds = do_cos = do_lag = False
    kinds = args.kind.split()
    for k in kinds:
        if k == 'PDS':
            do_pds = True
        elif k == 'CPDS':
            do_cpds = True
        elif k == 'cos' or k == 'cospectrum':
            do_cos = True
        elif k == 'lag':
            do_lag = True

    mp_calc_fspec(args.files, fftlen,
                  calc_pds=do_pds,
                  calc_cpds=do_cpds,
                  calc_cospectrum=do_cos,
                  calc_lags=do_lag,
                  save_dyn=False,
                  bintime=bintime)
