from __future__ import division, print_function
import sys
import matplotlib.pyplot as plt
from maltpynt.mp_base import mp_detection_level
from maltpynt.mp_io import mp_load_data
import collections
import numpy as np


if __name__ == '__main__':
    pdsdata = mp_load_data(sys.argv[1])
    try:
        freq = pdsdata['freq']
    except:
        flo = pdsdata['flo']
        fhi = pdsdata['fhi']
        freq = (fhi + flo) / 2
        plt.semilogx()

    pds = pdsdata['pds']
    epds = pdsdata['epds']
    npds = pdsdata['npds']
    norm = pdsdata['norm']
    rebin = pdsdata['rebin']

    nbin = len(pds[1:])

    meanp = np.mean(pds[len(pds)//2:])
    lev = mp_detection_level(nbin, n_summed_spectra=npds, n_rebin=rebin)

    if meanp < 2:
        print ('Renormalizing PDS because of PDS average < 2')
        print ('The procedure is not failproof. Beware of wrong results')
#        lev = lev / 2 * meanp
        pds = pds / meanp * 2
    if isinstance(lev, collections.Iterable):
        plt.plot(freq, lev)
    else:
        plt.axhline(lev)

    plt.plot(freq[1:], pds[1:],
             drawstyle='steps-mid')
    plt.errorbar(freq[1:], pds[1:], yerr=epds[1:],
                 drawstyle='steps-mid', fmt='-')

    plt.xlabel('Frequency')
    if norm == 'rms':
        plt.ylabel('(rms/mean)^2')
    elif norm == 'Leahy':
        plt.ylabel('Leahy power')

    plt.show()
