from __future__ import division, print_function
import cPickle as pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

pdsdata = pickle.load(open(sys.argv[1]))

freq = pdsdata['freq']
pds = pdsdata['pds']

#plt.loglog(freq[1:], freq[1:] * (pds[1:] - np.mean(pds[len(pds) / 2:])),
#           drawstyle='steps-mid')
plt.plot(freq[1:], pds[1:],
         drawstyle='steps-mid')

plt.show()
