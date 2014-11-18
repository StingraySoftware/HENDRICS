from __future__ import division, print_function
import cPickle as pickle
import sys
import matplotlib.pyplot as plt
from maltpynt.mp_base import mp_detection_level


if __name__ == '__main__':
    pdsdata = pickle.load(open(sys.argv[1]))
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

    nbin = len(pds[1:])

    plt.plot(freq[1:], pds[1:],
             drawstyle='steps-mid')
    plt.errorbar(freq[1:], pds[1:], yerr=epds[1:],
                 drawstyle='steps-mid', fmt='-')

    lev = mp_detection_level(nbin, n_summed_spectra=npds)
    plt.axhline(lev)

    plt.xlabel('Frequency')
    if norm == 'rms':
        plt.ylabel('(rms/mean)^2')
    elif norm == 'Leahy':
        plt.ylabel('Leahy power')

    plt.show()
