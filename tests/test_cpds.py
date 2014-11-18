from __future__ import division, print_function
import cPickle as pickle
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pdsdata = pickle.load(open(sys.argv[1]))

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
