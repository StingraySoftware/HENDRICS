from __future__ import division, print_function
import cPickle as pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from ..mp_base import mp_detection_level

if __name__=='__main__':
    pdsdata = pickle.load(open(sys.argv[1]))
    freq = pdsdata['freq']
    pds = pdsdata['pds']
    npds = pdsdata['npds']
    norm = pdsdata['norm']

    nbin = len(pds[1:])
    #plt.loglog(freq[1:], freq[1:] * (pds[1:] - np.mean(pds[len(pds) / 2:])),
    #           drawstyle='steps-mid')
    plt.plot(freq[1:], pds[1:],
             drawstyle='steps-mid')

    lev = mp_detection_level(nbin, n_summed_spectra=npds)
    plt.axhline(lev)

    plt.xlabel('Frequency')
    if norm == 'rms':
        plt.ylabel('(rms/mean)^2')
    elif norm == 'Leahy':
        plt.ylabel('Leahy power')

    plt.show()
