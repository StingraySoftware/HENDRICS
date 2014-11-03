from __future__ import division, print_function
import cPickle as pickle
import sys
import matplotlib.pyplot as plt

print ('Loading lc file...')
lcdata = pickle.load(open(sys.argv[1]))

time = lcdata['time']
lc = lcdata['lc']
gti = lcdata['gti']

plt.plot(time, lc, drawstyle='steps-mid', color='k')

for g in gti:
    plt.axvline(g[0], ls='--', color='red')
    plt.axvline(g[1], ls='--', color='red')
plt.show()
