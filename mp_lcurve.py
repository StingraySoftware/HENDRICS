from __future__ import division, print_function
import numpy as np
from mp_base import mp_root, mp_create_gti_mask, mp_cross_gtis
from mp_base import mp_contiguous_regions
from mp_io import mp_load_events, mp_load_lcurve, mp_save_lcurve
from mp_io import MP_FILE_EXTENSION


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


def mp_join_lightcurves(lcfilelist, outfile='out_lc' + MP_FILE_EXTENSION):
    lcdatas = []
    for lfc in lcfilelist:
        print ("Loading file %s..." % lfc)
        lcdata = mp_load_lcurve(lfc)
        print ("Done.")
        lcdatas.append(lcdata)

    # --------------- Check consistency of data --------------
    lcdts = [lcdata['dt'] for lcdata in lcdatas]
    # Find unique elements. If multiple bin times are used, throw an exception
    lcdts = list(set(lcdts))
    if len(lcdts) > 1:
        raise Exception('Light curves must have same dt for scrunching')

    instrs = [lcdata['Instr'] for lcdata in lcdatas]
    # Find unique elements. A lightcurve will be produced for each instrument
    instrs = list(set(instrs))
    outlcs = {}
    times = {}
    lcs = {}
    gtis = {}
    for instr in instrs:
        outlcs[instr] = {'dt': lcdts[0]}
        times[instr] = []
        lcs[instr] = []
        gtis[instr] = []
    # -------------------------------------------------------

    for lcdata in lcdatas:
        time = lcdata['time']
        lc = lcdata['lc']
        gti = lcdata['GTI']
        instr = lcdata['Instr']
        times[instr].extend(time)
        lcs[instr].extend(lc)
        gtis[instr].extend(gti)

    for instr in instrs:
        outlcs[instr]['time'] = np.array(times[instr])
        outlcs[instr]['lc'] = np.array(lcs[instr])
        outlcs[instr]['GTI'] = np.array(gtis[instr])

    if outfile is not None:
        print ('Saving joined light curve to %s' % outfile)
        mp_save_lcurve(outlcs, outfile)

    return outlcs


def mp_scrunch_lightcurves(lcfilelist, outfile='out_scrlc'+MP_FILE_EXTENSION):
    '''Create a single light curve from input light curves,
    regardless of the instrument'''
    lcdata = mp_join_lightcurves(lcfilelist)
    instrs = lcdata.keys()
    gti_lists = [lcdata[inst]['GTI'] for inst in instrs]
    gti = mp_cross_gtis(gti_lists)
    # Determine limits
    time0 = lcdata[instrs[0]]['time']
    mask = mp_create_gti_mask(time0, gti)

    time0 = time0[mask]
    lc0 = lcdata[instrs[0]]['lc']
    lc0 = lc0[mask]

    for inst in instrs[1:]:
        time1 = lcdata[inst]['time']
        mask = mp_create_gti_mask(time1, gti)
        time1 = time1[mask]
        assert np.all(time0 == time1), \
            'Something is not right with gti filtering'
        lc = lcdata[inst]['lc']
        lc0 += lc[mask]

    out = {}
    out['lc'] = lc0
    out['time'] = time0
    out['dt'] = lcdata[instrs[0]]['dt']

    print ('Saving scrunched light curve to %s' % outfile)
    mp_save_lcurve(out, outfile)

    return time0, lc0, gti


def mp_filter_lc_gtis(time, lc, gti, safe_interval=None, delete=False,
                      min_length=0, return_borders=False):

    mask, newgtis = mp_create_gti_mask(time, gti, return_new_gtis=True,
                                       safe_interval=safe_interval,
                                       min_length=min_length)

#    # test if newgti-created mask coincides with mask
#    newmask = mp_create_gti_mask(time, newgtis, safe_interval=0)
#    print ("Test: newly created gti is equivalent?", np.all(newmask == mask))

    nomask = np.logical_not(mask)

    if delete:
        time = time[mask]
        lc = lc[mask]
    else:
        lc[nomask] = 0

    if return_borders:
        # TODO: Check if this works with and without "delete" enabled
        mask = mp_create_gti_mask(time, newgtis)
        borders = mp_contiguous_regions(mask)
        return time, lc, newgtis, borders
    else:
        return time, lc, newgtis


def mp_lcurve_from_events(f, safe_interval=0,
                          pi_interval=None,
                          e_interval=None,
                          min_length=0,
                          gti_split=False,
                          ignore_gtis=False):
    print ("Loading file %s..." % f)
    evdata = mp_load_events(f)
    print ("Done.")
    tag = ''
    out = {}
    tstart = evdata['Tstart']
    tstop = evdata['Tstop']
    events = evdata['time']
    instr = evdata['Instr']
    gtis = evdata['GTI']
    if ignore_gtis:
        gtis = np.array([[tstart, tstop]])

    # make tstart and tstop multiples of bin times since MJDref
    tstart = np.ceil(tstart / bintime, dtype=np.longdouble) * bintime
    tstop = np.floor(tstop / bintime, dtype=np.longdouble) * bintime

    if pi_interval is not None and np.all(pi_interval > 0):
        pis = evdata['PI']
        good = np.logical_and(pis > pi_interval[0],
                              pis <= pi_interval[1])
        events = events[good]
        tag = '_PI%g-%g' % (pi_interval[0], pi_interval[1])
        out['PImin'] = e_interval[0]
        out['PImax'] = e_interval[0]

    if e_interval is not None and np.all(e_interval > 0):
        try:
            es = evdata['E']
        except:
            raise \
                ValueError("No energy information is present in the file."
                           + " Did you run mp_calibrate?")

        good = np.logical_and(es > e_interval[0],
                              es <= e_interval[1])
        events = events[good]
        tag = '_E%g-%g' % (e_interval[0], e_interval[1])
        out['Emin'] = e_interval[0]
        out['Emax'] = e_interval[0]

    time, lc = mp_lcurve(events, bintime, start_time=tstart,
                         stop_time=tstop)
    if gti_split:
        time, lc, newgtis, borders = \
            mp_filter_lc_gtis(time, lc, gtis,
                              safe_interval=safe_interval,
                              delete=False,
                              min_length=min_length,
                              return_borders=True)

        outfiles = []
        print (borders)
        for ib, b in enumerate(borders):
            print (b)
            local_tag = tag + '_gti%d' % ib
            local_out = out.copy()
            local_out['lc'] = lc[b[0]:b[1]]
            local_out['time'] = time[b[0]:b[1]]
            local_out['dt'] = bintime
            local_out['GTI'] = [[time[b[0]], time[b[1]]]]
            local_out['Tstart'] = time[b[0]]
            local_out['Tstop'] = time[b[1]]
            local_out['Instr'] = instr
            outfile = mp_root(f) + local_tag + '_lc' + MP_FILE_EXTENSION
            print ('Saving light curve to %s' % outfile)
            mp_save_lcurve(local_out, outfile)
            outfiles.append(outfile)
    else:
        time, lc, newgtis = mp_filter_lc_gtis(time, lc, gtis,
                                              safe_interval=safe_interval,
                                              delete=True,
                                              min_length=min_length,
                                              return_borders=False)

        out['lc'] = lc
        out['time'] = time
        out['dt'] = bintime
        out['GTI'] = newgtis
        out['Tstart'] = tstart
        out['Tstop'] = tstop
        out['Instr'] = instr
        outfile = mp_root(f) + tag + '_lc' + MP_FILE_EXTENSION
        print ('Saving light curve to %s' % outfile)
        mp_save_lcurve(out, outfile)
        outfiles = [outfile]

    # For consistency in return value
    return outfiles


if __name__ == "__main__":
    import argparse
    description = 'Creates lightcurves starting from event files. It is' + \
        ' possible to specify energy or channel filtering options'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')

    parser.add_argument("-b", "--bintime", type=float, default=1/4096,
                        help="Bin time; if negative, negative power of 2")
    parser.add_argument("--safe-interval", nargs=2, type=float,
                        default=[0, 0],
                        help="Interval at start and stop of GTIs used" +
                        " for filtering")
    parser.add_argument("--pi-interval", type=long, default=[-1, -1],
                        nargs=2,
                        help="PI interval used for filtering")
    parser.add_argument('-e', "--e-interval", type=float, default=[-1, -1],
                        nargs=2,
                        help="Energy interval used for filtering")
    parser.add_argument("-s", "--scrunch",
                        help="Create scrunched light curve",
                        default=False,
                        action="store_true")
    parser.add_argument("-g", "--gti-split",
                        help="Split light curve by GTI",
                        default=False,
                        action="store_true")
    parser.add_argument("--minlen",
                        help="Minimum length of acceptable GTIs (default:100)",
                        default=100, type=float)
    parser.add_argument("--ignore-gtis",
                        help="Ignore GTIs",
                        default=False,
                        action="store_true")
    args = parser.parse_args()
    bintime = args.bintime

    if bintime < 0:
        bintime = 2 ** (bintime)
    bintime = np.longdouble(bintime)

    infiles = args.files
    safe_interval = args.safe_interval
    pi_interval = np.array(args.pi_interval)
    e_interval = np.array(args.e_interval)

    outfiles = []
    for f in infiles:
        outfile = mp_lcurve_from_events(f, safe_interval=safe_interval,
                                        pi_interval=pi_interval,
                                        e_interval=e_interval,
                                        min_length=args.minlen,
                                        gti_split=args.gti_split,
                                        ignore_gtis=args.ignore_gtis)

        outfiles.extend(outfile)

    print (outfiles)
    # TODO: test if this still works!
    if args.scrunch:
        mp_scrunch_lightcurves(outfiles)
