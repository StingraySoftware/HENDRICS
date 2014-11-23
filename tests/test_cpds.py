from __future__ import division, print_function
import sys
import matplotlib.pyplot as plt
from maltpynt.mp_io import mp_load_data

if __name__ == '__main__':
    pdsdata = mp_load_data(sys.argv[1])

    try:
        freq = pdsdata['freq']
    except:
        flo = pdsdata['flo']
        fhi = pdsdata['fhi']
        freq = (fhi + flo) / 2
        plt.semilogx()

    cpds = pdsdata['cpds']

    cospectrum = cpds.real

    plt.loglog(freq[1:], freq[1:] * cospectrum[1:], drawstyle='steps-mid')

    plt.xlabel('Frequency')
    plt.ylabel('Cospectrum')

    plt.show()
