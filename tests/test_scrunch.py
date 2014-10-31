import cPickle as pickle
import sys
import matplotlib.pyplot as plt

jlc = pickle.load(open(sys.argv[1]))
scr = pickle.load(open(sys.argv[2]))

st = scr['time']
sl = scr['lc']

j1t = jlc['FPMA']['time']
j1l = jlc['FPMA']['lc']
j2t = jlc['FPMB']['time']
j2l = jlc['FPMB']['lc']

plt.plot(st, sl, label='Scrunch')
plt.plot(j1t, j1l, label='FPMA')
plt.plot(j2t, j2l, label='FPMB')

plt.legend()
plt.show()
