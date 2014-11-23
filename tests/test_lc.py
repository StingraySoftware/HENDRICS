from __future__ import division, print_function
from __future__ import unicode_literals
import sys
import matplotlib.pyplot as plt
from maltpynt.mp_io import mp_load_data


if __name__ == '__main__':
    for lcfile in sys.argv[1:]:
        print ('Loading %s...' % lcfile)
        lcdata = mp_load_data(lcfile)

        time = lcdata['time']
        lc = lcdata['lc']
        gti = lcdata['gti']

        plt.plot(time, lc, drawstyle='steps-mid', color='k')

        for g in gti:
            plt.axvline(g[0], ls='-', color='red')
            plt.axvline(g[1], ls='--', color='red')

    plt.xlabel('Time')
    plt.ylabel('light curve')

    plt.show()
