from __future__ import division, print_function
import sys
import matplotlib.pyplot as plt
from maltpynt.mp_base import mp_detection_level
from maltpynt.mp_io import mp_load_data
import collections
import numpy as np

from scipy.optimize import curve_fit


def baseline_fun(x, a):
    return a


if __name__ == '__main__':
    ax = plt.subplot(1,1,1)
    rainbow = ax._get_lines.color_cycle
    for i, fname in enumerate(sys.argv[1:]):
        pdsdata = mp_load_data(fname)
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

        lev = mp_detection_level(nbin, n_summed_spectra=npds, n_rebin=rebin)

        color = rainbow.next()

        p, pcov = curve_fit(baseline_fun, freq, pds, p0=[2], sigma=epds)
        print ('White noise level is', p[0])
        pds -= p[0]
        if isinstance(lev, collections.Iterable):
            plt.plot(freq, lev - p[0], color=color)
        else:
            plt.axhline(lev - p[0], color=color)

        plt.errorbar(freq[1:], pds[1:], yerr=epds[1:], fmt='-',
                     drawstyle='steps-mid', color=color)

    plt.xlabel('Frequency')
    if norm == 'rms':
        plt.ylabel('(rms/mean)^2')
    elif norm == 'Leahy':
        plt.ylabel('Leahy power')

    plt.show()
