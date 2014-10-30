from __future__ import division, print_function
import numpy as np
from mp_base import *
import cPickle as pickle


def mp_lcurve(event_list,
              bin_time,
              start_time=None,
              stop_time=None,
              verbose=0):
    '''
        From a list of event times, it extracts a lightcurve
        usage:
        times, lc = bin_events(event_list, bin_time)
        '''
    if start_time is None:
        print ("mp_lcurve: Changing start time")
        start_time = np.floor(event_list[0])
    if stop_time is None:
        print ("mp_lcurve: Changing stop time")
        stop_time = np.ceil(event_list[-1])
    if verbose > 0:
        print ("mp_lcurve: Time limits: %g -- %g" % (start_time, stop_time))

    new_event_list = event_list[event_list >= start_time]
    new_event_list = new_event_list[new_event_list <= stop_time]
    # To compute the histogram, the times array must specify the bin edges.
    # therefore, if nbin is the length of the lightcurve, times will have
    # nbin + 1 elements
    new_event_list = ((new_event_list - start_time) / bin_time).astype(int)
    times = np.arange(start_time, stop_time, bin_time)
    lc = np.bincount(new_event_list, minlength=len(times))
    if verbose > 1:
        print ("mp_lcurve: Length of the lightcurve: %g" % len(times))
    #    print len(times), len (lc)
    return times, lc.astype(np.float)


def mp_filter_lc_gtis(time, lc, gti, safe_interval=None):
    mask, newgtis = create_gti_mask(time, gti, return_new_gtis=True,
                                    safe_interval=safe_interval)

    nomask = np.logical_not(mask)

    lc[nomask] = 0
    return time, lc, newgtis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-b", "--bintime", type=float, default=1/4096,
                        help="Bin time; if negative, negative power of 2")
    parser.add_argument("--safe_interval", nargs=2, type=float,
                        default=[0, 0],
                        help="Interval at start and stop of GTIs used" +
                        " for filtering")
    parser.add_argument("--pi_interval", type=long, default=[-1, -1],
                        nargs=2,
                        help="PI interval used for filtering")
    parser.add_argument('-e', "--e_interval", type=float, default=[-1, -1],
                        nargs=2,
                        help="Energy interval used for filtering")

    args = parser.parse_args()
    bintime = args.bintime
    if bintime < 0:
        bintime = 2 ** (-bintime)
    infiles = args.files
    safe_interval = args.safe_interval
    pi_interval = np.array(args.pi_interval)
    e_interval = np.array(args.e_interval)

    tag = ''

    for f in infiles:
        print("Loading file %s..." % f)
        evdata = pickle.load(open(f))
        print("Done.")
        out = {}
        tstart = evdata['Tstart']
        tstop = evdata['Tstop']
        events = evdata['Events']
        instr = evdata['Instr']
        gtis = evdata['GTI']
        mjdref = evdata['MJDref']

        if np.all(pi_interval > 0):
            pis = evdata['PI']
            good = np.logical_and(pis > pi_interval[0],
                                  pis <= pi_interval[1])
            events = events[good]
            tag = '_%g-%g' % (pi_interval[0], pi_interval[1])
            out['PImin'] = e_interval[0]
            out['PImax'] = e_interval[0]

        if np.all(e_interval > 0):
            try:
                es = evdata['E']
            except:
                raise \
                    ValueError("No energy information is present in the file."
                               + " Did you run mp_calibrate?")

            good = np.logical_and(es > e_interval[0],
                                  es <= e_interval[1])
            events = events[good]
            tag = '_%g-%g' % (e_interval[0], e_interval[1])
            out['Emin'] = e_interval[0]
            out['Emax'] = e_interval[0]

        time, lc = mp_lcurve(events, bintime, start_time=0,
                             stop_time=tstop - tstart)
        time, lc, newgtis = mp_filter_lc_gtis(time, lc, gtis - tstart,
                                              safe_interval=safe_interval)

        out['lc'] = lc
        out['time'] = time
        out['dt'] = bintime
        out['gti'] = newgtis
        out['Tstart'] = tstart
        out['Tstop'] = tstop
        out['Instr'] = instr

        outfile = mp_root(f) + tag + '_lc.p'
        print('Saving light curve to %s' % outfile)
        pickle.dump(out, open(outfile, 'wb'))

