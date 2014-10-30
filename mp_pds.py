from __future__ import division, print_function
from mp_base import *
import numpy as np


def fft(lc):
    '''A wrapper for the fft function. Just numpy for now'''
    nbin = len(lc)

    ft = np.fft.fft(lc)
    freqs = np.fft.fftfreq(nbin, bin_time)

    return ft, freqs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("files", help="List of light curve files", nargs='+')
    parser.add_argument("-b", "--bintime", type=float, default=1/4096,
                        help="Bin time; if negative, negative power of 2")
    parser.add_argument("-f", "--fftlen", type=float, default=512,
                        help="Length of FFTs")

    args = parser.parse_args()

    bintime = args.bintime
    fftlen = args.fftlen

    for f in files:
        root = mp_root(f)
        print("Loading file %s..." % f)
        lcdata = pickle.load(open(f))
        print("Done.")

        time = lcdata['time']
        lc = lcdata['time']
        lcdt = lcdata['dt']
        if bintime < lcdt:
            bintime = lcdt
