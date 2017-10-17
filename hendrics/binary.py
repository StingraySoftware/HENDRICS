"""Save different input files in PRESTO-readable format."""
from __future__ import print_function, division
import argparse
import logging
from astropy.coordinates import SkyCoord
import numpy as np
from .io import high_precision_keyword_read, load_lcurve


def get_header_info(obj):
    from astropy.io.fits import Header
    header = Header.fromstring(obj.header)
    info = type('', (), {})()
    info.mjdref = high_precision_keyword_read(header, 'MJDREF')
    info.telescope = header['TELESCOP']
    info.instrument = header['INSTRUME']
    info.source = header['OBJECT']
    info.observer = header['USER']
    info.user = header['USER']
    info.tstart = header['TSTART']
    info.tstop = header['TSTOP']
    ra = header['RA_OBJ']
    dec = header['DEC_OBJ']
    a = SkyCoord(ra, dec, unit="degree")
    info.raj = \
        (a.ra.to_string("hourangle")
         ).replace("s", "").replace("h", ":").replace("m", ":")
    info.decj = (a.ra.to_string()
                 ).replace("s", "").replace("d", ":").replace("m", ":")
    if hasattr(obj, 'e_interval'):
        e0, e1 = obj.e_interval
    elif hasattr(obj, 'energy'):
        e0, e1 = np.min(obj.energy), np.max(obj.energy)
    else:
        e0, e1 = 0, 0
    info.centralE = (e0 + e1) / 2
    info.bandpass = e1 - e0

    return info


def _save_to_binary(lc, filename):
    """Save a light curve to binary format."""
    nc = len(lc.counts)
    lc.counts[:nc // 2 * 2].astype('float32').tofile(filename)
    return


def save_lc_to_binary(lc, filename):
    tstart = lc.tstart
    tstop = lc.tstart + lc.tseg
    nbin = lc.n
    bin_time = lc.dt
    _save_to_binary(lc, filename)

    lcinfo = type('', (), {})()
    lcinfo.bin_intervals_start = [0]
    lcinfo.bin_intervals_stop = [nbin]
    lcinfo.lclen = nbin
    lcinfo.tstart = tstart
    lcinfo.dt = bin_time
    lcinfo.tseg = tstop - tstart
    return lcinfo


MAXBIN = 100000000
def save_events_to_binary(events, filename, tstart, bin_time):
    import struct

    tstop = events.gti[-1, 1]
    nbin = (tstop - tstart) / bin_time

    print(nbin, nbin/MAXBIN)

    lclen = 0
    print("Saving")
    file = open(filename, 'wb')
    nphot = 0
    for i in np.arange(0, nbin, MAXBIN):
        t0 = i * bin_time + tstart

        lastbin = np.min([MAXBIN, (nbin - i) // 2 * 2])
        t1 = t0 + lastbin * bin_time

        good = (events.time >= t0)&(events.time < t1)

        goodev = events.time[good]
        hist, times = np.histogram(goodev, bins=np.linspace(t0, t1, lastbin + 1))

        lclen += lastbin
        s = struct.pack('f'*len(hist), *hist)
        print(s[:8])
        file.write(s)
        nphot += len(goodev)
    file.close()

    print(tstart, tstop, MAXBIN, bin_time, nbin)
    lcinfo = type('', (), {})()
    lcinfo.bin_intervals_start = np.floor((events.gti[:, 0] - tstart) / bin_time)
    lcinfo.bin_intervals_stop = np.floor((events.gti[:, 1] - tstart) / bin_time)
    lcinfo.lclen = lclen
    lcinfo.tstart = tstart
    lcinfo.dt = bin_time
    lcinfo.tseg = tstop - tstart
    lcinfo.nphot = nphot
    return lcinfo


def main_presto(args=None):
    import argparse
    from multiprocessing import Pool

    description = ('Save light curves in a format readable to PRESTO')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of input light curves", nargs='+')

    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--nproc",
                        help=("Number of processors to use"),
                        default=1,
                        type=int)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENbinary.log', level=numeric_level,
                        filemode='w')

