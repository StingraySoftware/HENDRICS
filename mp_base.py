from __future__ import division, print_function
import numpy as np


def mp_mkdir_p(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def mp_read_header_key(fits_file, key, hdu=1):
    from astropy.io import fits as pf

    '''Reads the header key key from HDU hdu of the file fits_file'''
    hdulist = pf.open(fits_file)
    value = hdulist[hdu].header[key]
    hdulist.close()
    return value


def mp_ref_mjd(fits_file, hdu=1):
    '''
    Reads MJDREFF+ MJDREFI or, if failed, MJDREF, from the FITS header
    '''
    import collections
    import types
    if isinstance(fits_file, collections.Iterable) and\
            not isinstance("ba", types.StringTypes):
        fits_file = fits_file[0]
        print("opening %s" % fits_file)

    try:
        ref_mjd_int = np.long(mp_read_header_key(fits_file, 'MJDREFI'))
        ref_mjd_float = np.longdouble(mp_read_header_key(fits_file, 'MJDREFF'))
        ref_mjd_val = ref_mjd_int + ref_mjd_float
    except:
        ref_mjd_val = np.longdouble(mp_read_header_key(fits_file, 'MJDREF'))
    return ref_mjd_val


def mp_save_as_netcdf(vars, varnames, formats, fname):
    '''The future. Much faster than pickle'''
    import netCDF4 as nc
    import collections

    rootgrp = nc.Dataset(fname, 'w',
                         format='NETCDF4')

    for iv, v in enumerate(vars):
        if isinstance(v, collections.Iterable):
            dim = len(v)
        else:
            dim = 1
        rootgrp.createDimension(varnames[iv]+"dim", dim)
        vnc = rootgrp.createVariable(varnames[iv], formats[iv],
                                     (varnames[iv]+"dim",))
        vnc[:] = v
    rootgrp.close()


def mp_root(filename):
    filename = filename.replace('.evt', '').replace('.fits', '')
    filename = filename.replace('_ev.p', '').replace('_lc.p', '')
    filename = filename.replace('_ev_calib.p', '')
    return filename


def mp_contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
        a 2D array where the first column is the start index of the region and
        the second column is the end index.
        From http://stackoverflow.com/questions/4494404/
        find-large-number-of-consecutive-values-fulfilling-
        condition-in-a-numpy-array"""
    # Find the indicies of changes in "condition"
    diff = np.diff(condition)
    idx, = diff.nonzero()
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1
    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]
    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def mp_create_gti_mask(time, gtis, verbose=0, debug=False,
                       safe_interval=0, min_length=0,
                       return_new_gtis=False):
    '''Creates GTI mask under the assumption that no overlaps are present
        between GTIs
        '''
    import collections
    if verbose:
        print ("create_gti_mask: warning: this routine assumes that ")
        print ("                no overlaps are present between GTIs")

    mask = np.zeros(len(time), dtype=bool)

    if not isinstance(safe_interval, collections.Iterable):
        safe_interval = [safe_interval, safe_interval]

    newgtis = np.zeros_like(gtis)
    # Whose GTIs, including safe intervals, are longer than min_length
    newgtimask = np.zeros(len(newgtis), dtype=np.bool)

    for ig, gti in enumerate(gtis):
        limmin, limmax = gti
        limmin += safe_interval[0]
        limmax -= safe_interval[1]
        if limmax - limmin > min_length:
            newgtis[ig][:] = [limmin, limmax]
            cond1 = time >= limmin
            cond2 = time < limmax
            good = np.logical_and(cond1, cond2)
            mask[good] = True
            newgtimask[ig] = True

    res = mask
    if return_new_gtis:
        res = [res, newgtis[newgtimask]]
    return res


def mp_create_gti_from_condition(time, condition, verbose=False,
                                 safe_interval=0):
    '''Given a time array and a condition (e.g. obtained from lc > 0),
    it creates a GTI list'''
    import collections
    idxs = mp_contiguous_regions(condition)
    if not isinstance(safe_interval, collections.Iterable):
        safe_interval = [safe_interval, safe_interval]

    gtis = []
    for idx in idxs:
        if verbose:
            print (idx)
        t0 = time[idx[0]] + safe_interval[0]
        t1 = time[min(idx[1], len(time) - 1)] - safe_interval[1]
        if t1 - t0 < 0:
            continue
        gtis.append([t0, t1])
    return gtis


def mp_cross_gtis(gti_list, bin_time=1):
    '''From multiple GTI lists, it extracts the common intervals'''
    ninst = len(gti_list)
    if ninst == 1:
        return gti_list[0]

    start = np.min([g[0][0] for g in gti_list])
    stop = np.max([g[-1][-1] for g in gti_list])

    times = np.arange(start, stop, 1, dtype=np.longdouble)

    mask0 = mp_create_gti_mask(times, gti_list[0], verbose=0)

    for gti in gti_list:
        mask = mp_create_gti_mask(times, gti, verbose=0)
        mask0 = np.logical_and(mask0, mask)

    gtis = mp_create_gti_from_condition(times, mask0)

    return gtis


def mp_get_file_type(fname):
    import cPickle as pickle
    contents = pickle.load(open(fname))
    '''Gets file type'''
    # TODO: other file formats

    keys = contents.keys()
    if 'lc' in keys:
        ftype = 'lc'
    elif 'cpds' in keys:
        ftype = 'cpds'
        if 'fhi' in keys:
            ftype = 'rebcpds'
    elif 'pds' in keys:
        ftype = 'pds'
        if 'fhi' in keys:
            ftype = 'rebpds'
    return ftype, contents
