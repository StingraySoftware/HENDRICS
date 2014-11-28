from __future__ import division, print_function
from __future__ import unicode_literals
import sys
import matplotlib.pyplot as plt
from maltpynt.mp_io import mp_load_data
from maltpynt.mp_base import mp_create_gti_mask


if __name__ == '__main__':
    for lcfile in sys.argv[1:]:
        print ('Loading %s...' % lcfile)
        lcdata = mp_load_data(lcfile)

        time = lcdata['time']
        lc = lcdata['lc']
        gti = lcdata['GTI']
        instr = lcdata['Instr']
        if instr == 'PCA':
            # If RXTE, plot per PCU count rate
            npcus = lcdata['nPCUs']
            lc /= npcus

        for g in gti:
            plt.axvline(g[0], ls='-', color='red')
            plt.axvline(g[1], ls='--', color='red')

        good = mp_create_gti_mask(time, gti)
        plt.plot(time, lc, drawstyle='steps-mid', color='grey')
        plt.plot(time[good], lc[good], drawstyle='steps-mid', color='k')

    plt.xlabel('Time (s)')
    if instr == 'PCA':
        plt.ylabel('light curve (Ct/bin/PCU)')
    else:
        plt.ylabel('light curve (Ct/bin)')

    plt.show()
