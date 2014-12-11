from __future__ import division, print_function
import sys
import matplotlib.pyplot as plt
from maltpynt.mp_io import mp_load_data

if __name__ == '__main__':
    for fname in sys.argv[1:]:
        pdsdata = mp_load_data(fname)

        try:
            freq = pdsdata['freq']
        except:
            flo = pdsdata['flo']
            fhi = pdsdata['fhi']
            freq = (fhi + flo) / 2
            plt.semilogx()

        cpds = pdsdata['cpds']

        cospectrum = cpds.real
        plt.figure('Log')
        plt.loglog(freq[1:], freq[1:] * cospectrum[1:], drawstyle='steps-mid')
        plt.figure('Lin')
        plt.plot(freq[1:], cospectrum[1:], drawstyle='steps-mid')

    plt.figure('Log')
    plt.xlabel('Frequency')
    plt.ylabel('Cospectrum')

    plt.figure('Lin')
    plt.axhline(0, lw=3, ls='--', color='k')
    plt.xlabel('Frequency')
    plt.ylabel('Cospectrum')
    plt.show()
