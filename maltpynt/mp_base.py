from __future__ import division, print_function
import numpy as np
from .mp_io import mp_get_file_type, is_string
import logging


def mp_mkdir_p(path):
    '''Found at http://stackoverflow.com/questions/600268/
    mkdir-p-functionality-in-python'''
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
    '''Read the header key key from HDU hdu of the file fits_file'''
    from astropy.io import fits as pf

    hdulist = pf.open(fits_file)
    value = hdulist[hdu].header[key]
    hdulist.close()
    return value


def mp_ref_mjd(fits_file, hdu=1):
    '''
    Read MJDREFF+ MJDREFI or, if failed, MJDREF, from the FITS header
    '''
    import collections

    if isinstance(fits_file, collections.Iterable) and\
            not is_string(fits_file):
        fits_file = fits_file[0]
        logging.info("opening %s" % fits_file)

    try:
        ref_mjd_int = np.long(mp_read_header_key(fits_file, 'MJDREFI'))
        ref_mjd_float = np.longdouble(mp_read_header_key(fits_file, 'MJDREFF'))
        ref_mjd_val = ref_mjd_int + ref_mjd_float
    except:
        ref_mjd_val = np.longdouble(mp_read_header_key(fits_file, 'MJDREF'))
    return ref_mjd_val


def common_name(str1, str2, default='common'):
    '''Strip two file names of the letters not in common. Filenames must be of
    same length and only differ by a few letters'''
    if not len(str1) == len(str2):
        return default
    common_str = ''
    # Extract the MP root of the name (in case they're event files)
    str1 = mp_root(str1)
    str2 = mp_root(str2)
    for i, letter in enumerate(str1):
        if str2[i] == letter:
            common_str += letter
    # Remove leading and trailing underscores and dashes
    common_str = common_str.rstrip('_').rstrip('-')
    common_str = common_str.lstrip('_').lstrip('-')
    if common_str == '':
        common_str = default
    logging.debug('common_name: %s %s -> %s' % (str1, str2, common_str))
    return common_str


def mp_root(filename):
    import os.path
    fname = filename.replace('.gz', '')
    fname = os.path.splitext(filename)[0]
    fname = fname.replace('_ev', '').replace('_lc', '')
    fname = fname.replace('_calib', '')
    return fname


def mp_contiguous_regions(condition):
    """Find contiguous True regions of the boolean array "condition".

    Return a 2D array where the first column is the start index of the region
    and the second column is the end index.
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


def mp_check_gtis(gti):
    '''Check if GTIs are well-behaved'''
    gti_start = gti[:, 0]
    gti_end = gti[:, 1]

    logging.debug('-- GTI: ' + repr(gti))
    # Check that GTIs are well-behaved
    assert np.all(gti_end >= gti_start), 'This GTI is incorrect'
    # Check that there are no overlaps in GTIs
    assert np.all(gti_start[1:] >= gti_end[:-1]), 'This GTI has overlaps'
    logging.debug('-- Correct')

    return


def mp_create_gti_mask(time, gtis, safe_interval=0, min_length=0,
                       return_new_gtis=False, dt=None):
    '''Create GTI mask under the assumption that no overlaps are present
    between GTIs
        '''
    import collections

    mp_check_gtis(gtis)

    if dt is None:
        dt = np.zeros_like(time) + (time[1] - time[0]) / 2

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
        if limmax - limmin >= min_length:
            newgtis[ig][:] = [limmin, limmax]
            cond1 = time - dt >= limmin
            cond2 = time + dt <= limmax
            good = np.logical_and(cond1, cond2)
            mask[good] = True
            newgtimask[ig] = True

    res = mask
    if return_new_gtis:
        res = [res, newgtis[newgtimask]]
    return res


def mp_create_gti_from_condition(time, condition,
                                 safe_interval=0, dt=None):
    '''Create a GTI list from a time array and a boolean mask ("condition").

    A possible condition can be, e.g., lc > 0.
    The length of the condition array and the time array must be the same.
    '''
    import collections

    assert len(time) == len(condition), \
        'The length of the condition and time arrays must be the same.'
    idxs = mp_contiguous_regions(condition)

    if not isinstance(safe_interval, collections.Iterable):
        safe_interval = [safe_interval, safe_interval]

    if dt is None:
        dt = np.zeros_like(time) + (time[1] - time[0]) / 2

    gtis = []
    for idx in idxs:
        logging.debug(idx)
        startidx = idx[0]
        stopidx = idx[1]-1

        t0 = time[startidx] - dt[startidx] + safe_interval[0]
        t1 = time[stopidx] + dt[stopidx] - safe_interval[1]
        if t1 - t0 < 0:
            continue
        gtis.append([t0, t1])
    return np.array(gtis)


def mp_cross_gtis_bin(gti_list, bin_time=1):
    '''From multiple GTI lists, extract the common intervals.

    Uses a very rough algorithm. Better to use mp_cross_gtis.'''
    ninst = len(gti_list)
    if ninst == 1:
        return gti_list[0]

    start = np.min([g[0][0] for g in gti_list])
    stop = np.max([g[-1][-1] for g in gti_list])

    times = np.arange(start + bin_time / 2,
                      stop + bin_time / 2,
                      bin_time, dtype=np.longdouble)

    mask0 = mp_create_gti_mask(times, gti_list[0],
                               safe_interval=[0, bin_time])

    for gti in gti_list[1:]:
        mask = mp_create_gti_mask(times, gti,
                                  safe_interval=[0, bin_time])
        mask0 = np.logical_and(mask0, mask)

    gtis = mp_create_gti_from_condition(times, mask0)

    return gtis


def mp_cross_two_gtis(gti0, gti1):
    '''Extract the common intervals from two GTI lists *EXACTLY*.'''

    # Check GTIs
    mp_check_gtis(gti0)
    mp_check_gtis(gti1)

    gti0_start = gti0[:, 0]
    gti0_end = gti0[:, 1]
    gti1_start = gti1[:, 0]
    gti1_end = gti1[:, 1]

    # Create a list that references to the two start and end series
    gti_start = [gti0_start, gti1_start]
    gti_end = [gti0_end, gti1_end]

    # Concatenate the series, while keeping track of the correct origin of
    # each start and end time
    gti0_tag = np.array([0 for g in gti0_start], dtype=bool)
    gti1_tag = np.array([1 for g in gti1_start], dtype=bool)
    conc_start = np.concatenate((gti0_start, gti1_start))
    conc_end = np.concatenate((gti0_end, gti1_end))
    conc_tag = np.concatenate((gti0_tag, gti1_tag))

    # Put in time order
    order = np.argsort(conc_end)
    conc_start = conc_start[order]
    conc_end = conc_end[order]
    conc_tag = conc_tag[order]

    last_end = conc_start[0] - 1
    final_gti = []
    for ie, e in enumerate(conc_end):
        # Is this ending in series 0 or 1?
        this_series = conc_tag[ie]
        other_series = not this_series

        # Check that this closes intervals in both series.
        # 1. Check that there is an opening in both series 0 and 1 lower than e
        try:
            st_pos = \
                np.argmax(gti_start[this_series][gti_start[this_series] < e])
            so_pos = \
                np.argmax(gti_start[other_series][gti_start[other_series] < e])
            st = gti_start[this_series][st_pos]
            so = gti_start[other_series][so_pos]

            s = max([st, so])
        except:
            continue

        # If this start is inside the last interval (It can happen for equal
        # GTI start times between the two series), then skip!
        if s <= last_end:
            continue
        # 2. Check that there is no closing before e in the "other series",
        # from intervals starting either after s, or starting and ending
        # between the last closed interval and this one
        cond1 = (gti_end[other_series] > s) * (gti_end[other_series] < e)
        cond2 = gti_end[other_series][so_pos] < s
        condition = np.any(np.logical_or(cond1, cond2))
        # Well, if none of the conditions at point 2 apply, then you can
        # create the new gti!
        if not condition:
            final_gti.append([s, e])
            last_end = e

    return np.array(final_gti, dtype=np.longdouble)


def mp_cross_gtis(gti_list):
    '''From multiple GTI lists, extract the common intervals *EXACTLY*'''
    ninst = len(gti_list)
    if ninst == 1:
        return gti_list[0]

    gti0 = gti_list[0]

    for gti in gti_list[1:]:
        gti0 = mp_cross_two_gtis(gti0, gti)

    return gti0


def get_btis(gtis, start_time=None, stop_time=None):
    '''From GTIs, obtain bad time intervals.

    GTIs have to be well-behaved! No overlaps, no other crap'''
    # Check GTIs
    if len(gtis) == 0:
        assert start_time is not None and stop_time is not None, \
            'Empty GTI and no valid start_time and stop_time. BAD!'

        return np.array([[start_time, stop_time]], dtype=np.longdouble)
    mp_check_gtis(gtis)

    if start_time is None:
        start_time = gtis[0][0]
    if stop_time is None:
        stop_time = gtis[-1][1]
    if gtis[0][0] - start_time <= 0:
        btis = []
    else:
        btis = [[gtis[0][0] - start_time]]
    # Transform GTI list in
    flat_gtis = gtis.flatten()
    new_flat_btis = zip(flat_gtis[1:-2:2], flat_gtis[2:-1:2])
    btis.extend(new_flat_btis)

    if stop_time - gtis[-1][1] > 0:
        btis.extend([[gtis[0][0] - stop_time]])

    return np.array(btis, dtype=np.longdouble)


def mp_optimal_bin_time(fftlen, tbin):
    '''Vary slightly the bin time to have a power of two number of bins.

    Given an FFT length and a proposed bin time, return a bin time
    slightly shorter than the original, that will produce a power-of-two number
    of FFT bins'''
    import numpy as np
    return fftlen / (2 ** np.ceil(np.log2(fftlen / tbin)))


def mp_detection_level(nbins, epsilon=0.01, n_summed_spectra=1, n_rebin=1):
    '''
    Return the detection level (with probability 1 - epsilon) for a Power
    Density Spectrum of nbins bins, normalized \'a la Leahy (1983), based on
    the 2 dof Chi^2 statistics, corrected for rebinning (n_rebin) and multiple
    PDS averaging (n_summed_spectra)
    '''
    try:
        from scipy import stats
    except:
        raise Exception('You need Scipy to use this function')

    import collections
    if not isinstance(n_rebin, collections.Iterable):
        r = n_rebin
        retlev = stats.chi2.isf(epsilon / nbins, 2 * n_summed_spectra * r) \
            / (n_summed_spectra * r)
    else:
        retlev = [stats.chi2.isf(epsilon / nbins, 2 * n_summed_spectra * r)
                  / (n_summed_spectra * r) for r in n_rebin]
        retlev = np.array(retlev)
    return retlev


def mp_probability_of_power(level, nbins, n_summed_spectra=1, n_rebin=1):
    '''
    Return the probability of a certain power level in a Power Density
    Spectrum of nbins bins, normalized \'a la Leahy (1983), based on
    the 2 dof Chi^2 statistics, corrected for rebinning (n_rebin) and multiple
    PDS averaging (n_summed_spectra)
    '''
    try:
        from scipy import stats
    except:
        raise Exception('You need Scipy to use this function')

    epsilon = nbins * stats.chi2.sf(level * n_summed_spectra * n_rebin,
                                    2 * n_summed_spectra * n_rebin)
    return 1 - epsilon


def mp_sort_files(files):
    '''Sort a list of MaLTPyNT files'''
    allfiles = {}
    ftypes = []
    for f in files:
        logging.info('Loading file', f)
        ftype, contents = mp_get_file_type(f)
        instr = contents['Instr']
        ftypes.append(ftype)
        if instr not in list(allfiles.keys()):
            allfiles[instr] = []
        # Add file name to the dictionary
        contents['FILENAME'] = f
        allfiles[instr].append(contents)

    # Check if files are all of the same kind (lcs, PDSs, ...)
    ftypes = list(set(ftypes))
    assert len(ftypes) == 1, 'Files are not all of the same kind.'

    instrs = list(allfiles.keys())
    for instr in instrs:
        contents = list(allfiles[instr])
        tstarts = [c['Tstart'] for c in contents]
        fnames = [c['FILENAME'] for c in contents]

        fnames = [x for (y, x) in sorted(zip(tstarts, fnames))]

        # Substitute dictionaries with the sorted list of files
        allfiles[instr] = fnames

    return allfiles


def mp_calc_countrate(time, lc, gtis=None, bintime=1):
    if gtis is not None:
        mask = mp_create_gti_mask(time, gtis)
        lc = lc[mask]
    return np.mean(lc) / bintime
