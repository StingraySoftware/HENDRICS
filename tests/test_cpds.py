import cPickle as pickle
import sys
import matplotlib.pyplot as plt

pdsdata = pickle.load(open(sys.argv[1]))

freq = pdsdata['freq']
pds = pdsdata['cpds']

plt.loglog(freq[1:], freq[1:] * pds[1:].real)

plt.show()
