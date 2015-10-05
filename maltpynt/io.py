# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform input/output operations."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import logging
import warnings
try:
    import netCDF4 as nc
    MP_FILE_EXTENSION = '.nc'
except:
    msg = "Warning! NetCDF is not available. Using pickle format."
    logging.warning(msg)
    MP_FILE_EXTENSION = '.p'
    pass

try:
    # Python 3
    import pickle
except:
    # Python 2
    import cPickle as pickle

import collections
import numpy as np
import os.path
from .base import _order_list_of_arrays, _empty, is_string
from .base import _assign_value_if_none

cpl128 = np.dtype([(str('real'), np.double),
                   (str('imag'), np.double)])


def _get_key(dict_like, key):
    try:
        return dict_like[key]
    except:
        return ""


def high_precision_keyword_read(hdr, keyword):
    """Read FITS header keywords, also if split in two.

    In the case where the keyword is split in two, like

        MJDREF = MJDREFI + MJDREFF

    in some missions, this function returns the summed value. Otherwise, the
    content of the single keyword

    Parameters
    ----------
    hdr : dict_like
        The header structure, or a dictionary
    keyword : str
        The key to read in the header

    Returns
    -------
    value : long double
        The value of the key

    """
    try:
        value = np.longdouble(hdr[keyword])
        return value
    except:
        pass
    try:
        if len(keyword) == 8:
            keyword = keyword[:7]
        value = np.longdouble(hdr[keyword + 'I'])
        value += np.longdouble(hdr[keyword + 'F'])
        return value
    except:
        return None


def get_file_extension(fname):
    """Get the file extension."""
    return os.path.splitext(fname)[1]


def get_file_format(fname):
    """Decide the file format of the file."""
    ext = get_file_extension(fname)
    if ext == '.p':
        return 'pickle'
    elif ext == '.nc':
        return 'nc'
    elif ext in ['.evt', '.fits']:
        return 'fits'
    else:
        raise Exception("File format not recognized")


# ---- Base function to save NetCDF4 files
def save_as_netcdf(vars, varnames, formats, fname):
    """Save variables in a NetCDF4 file."""
    rootgrp = nc.Dataset(fname, 'w',
                         format='NETCDF4')

    for iv, v in enumerate(vars):
        dims = {}
        dimname = varnames[iv]+"dim"
        dimspec = (varnames[iv]+"dim", )

        if formats[iv] == 'c16':
            # unicode_literals breaks something, I need to specify str.

            if 'cpl128' not in rootgrp.cmptypes.keys():
                complex128_t = rootgrp.createCompoundType(cpl128, 'cpl128')
            vcomp = np.empty(v.shape, dtype=cpl128)
            vcomp['real'] = v.real
            vcomp['imag'] = v.imag
            v = vcomp
            formats[iv] = complex128_t

        if isinstance(v, collections.Iterable) and formats[iv] != str:
            dim = len(v)
            dims[dimname] = dim

            if isinstance(v[0], collections.Iterable):
                dim = len(v[0])
                dims[dimname + '_2'] = dim
                dimspec = (dimname, dimname + '_2')
        else:
            dims[dimname] = 1
            dim = 1

        for dimname in dims.keys():
            rootgrp.createDimension(dimname, dims[dimname])

        vnc = rootgrp.createVariable(varnames[iv], formats[iv],
                                     dimspec)
        if formats[iv] == str:
            vnc[0] = v
        else:
            vnc[:] = v
    rootgrp.close()


def read_from_netcdf(fname):
    """Read from a netCDF4 file."""
    rootgrp = nc.Dataset(fname)
    out = {}
    for k in rootgrp.variables.keys():
        dum = rootgrp.variables[k]
        values = dum.__array__()
        # Handle special case of complex
        if dum.dtype == cpl128:
            arr = np.empty(values.shape, dtype=np.complex128)
            arr.real = values[str('real')]
            arr.imag = values[str('imag')]
            values = arr

        if dum.dtype == str or dum.size == 1:
            to_save = values[0]
        else:
            to_save = values
        out[k] = to_save

    rootgrp.close()

    return out


# ----- Functions to handle file types
def get_file_type(fname, specify_reb=True):
    """Return the file type and its contents.

    Only works for maltpynt-format pickle or netcdf files.
    """
    contents = load_data(fname)

    keys = list(contents.keys())

    for i in 'lccorr,lc,cpds,pds,lag'.split(','):
        if i in keys and 'fhi' in keys and specify_reb:
            ftype = 'reb' + i
            break
        elif i in keys:
            ftype = i
            break
    else:  # If none of the above
        if 'time' in keys:
            ftype = 'events'
        elif 'GTI' in keys:
            ftype = 'GTI'

    return ftype, contents


# ----- functions to save and load EVENT data
def save_events(eventStruct, fname):
    """Save events in a file."""
    if get_file_format(fname) == 'pickle':
        _save_data_pickle(eventStruct, fname)
    elif get_file_format(fname) == 'nc':
        _save_data_nc(eventStruct, fname)


def load_events(fname):
    """Load events from a file."""
    if get_file_format(fname) == 'pickle':
        return _load_data_pickle(fname)
    elif get_file_format(fname) == 'nc':
        return _load_data_nc(fname)


# ----- functions to save and load LCURVE data
def save_lcurve(lcurveStruct, fname):
    """Save light curve in a file."""
    if get_file_format(fname) == 'pickle':
        return _save_data_pickle(lcurveStruct, fname)
    elif get_file_format(fname) == 'nc':
        return _save_data_nc(lcurveStruct, fname)


def load_lcurve(fname):
    """Load light curve from a file."""
    if get_file_format(fname) == 'pickle':
        return _load_data_pickle(fname)
    elif get_file_format(fname) == 'nc':
        return _load_data_nc(fname)


# ---- Functions to save PDSs

def save_pds(pdsStruct, fname):
    """Save PDS in a file."""
    if get_file_format(fname) == 'pickle':
        return _save_data_pickle(pdsStruct, fname)
    elif get_file_format(fname) == 'nc':
        return _save_data_nc(pdsStruct, fname)


def load_pds(fname):
    """Load PDS from a file."""
    if get_file_format(fname) == 'pickle':
        return _load_data_pickle(fname)
    elif get_file_format(fname) == 'nc':
        return _load_data_nc(fname)


# ---- GENERIC function to save stuff.
def _load_data_pickle(fname, kind="data"):
    """Load generic data in pickle format."""
    logging.info('Loading %s and info from %s' % (kind, fname))
    try:
        with open(fname, 'rb') as fobj:
            result = pickle.load(fobj)
        return result
    except Exception as e:
        raise Exception("{0} failed ({1}: {2})".format('_load_data_pickle',
                                                       type(e), e))


def _save_data_pickle(struct, fname, kind="data"):
    """Save generic data in pickle format."""
    logging.info('Saving %s and info to %s' % (kind, fname))
    try:
        with open(fname, 'wb') as fobj:
            pickle.dump(struct, fobj)
    except Exception as e:
        raise Exception("{0} failed ({1}: {2})".format('_save_data_pickle',
                                                       type(e), e))
    return


def _load_data_nc(fname):
    """Load generic data in netcdf format."""
    contents = read_from_netcdf(fname)
    keys = list(contents.keys())

    keys_to_delete = []
    for k in keys:
        if k[-2:] in ['_I', '_F']:
            kcorr = k[:-2]

            if kcorr not in list(contents.keys()):
                contents[kcorr] = np.longdouble(0)
            dum = contents[k]
            if isinstance(dum, collections.Iterable):
                dum = np.array(dum, dtype=np.longdouble)
            else:
                dum = np.longdouble(dum)
            contents[kcorr] += dum
            keys_to_delete.append(k)

    for k in keys_to_delete:
        del contents[k]

    return contents


def _save_data_nc(struct, fname, kind="data"):
    """Save generic data in netcdf format."""
    logging.info('Saving %s and info to %s' % (kind, fname))
    varnames = []
    values = []
    formats = []

    for k in struct.keys():
        var = struct[k]
        probe = var
        if isinstance(var, collections.Iterable):
            try:
                probe = var[0]
            except:
                logging.error('This failed: %s %s in file %s' %
                              (k, repr(var), fname))
                raise Exception('This failed: %s %s in file %s' %
                                (k, repr(var), fname))
        if is_string(var):
            probekind = str
            probesize = -1
        else:
            probekind = np.result_type(probe).kind
            probesize = np.result_type(probe).itemsize

        if probesize == 16 and probekind == 'f':
            # If a longdouble, split it in integer + floating part
            if isinstance(var, collections.Iterable):
                var_I = var.astype(np.long)
                var_F = np.array(var - var_I, dtype=np.double)
            else:
                var_I = np.long(var)
                var_F = np.double(var - var_I)
            values.extend([var_I, var_F])
            formats.extend(['i8', 'f8'])
            varnames.extend([k + '_I', k + '_F'])
        elif probekind == str:
            values.append(var)
            formats.append(probekind)
            varnames.append(k)
        else:
            values.append(var)
            formats.append(probekind + '%d' % probesize)
            varnames.append(k)

    save_as_netcdf(values, varnames, formats, fname)


def save_data(struct, fname, ftype='data'):
    """Save generic data in maltpynt format."""
    if get_file_format(fname) == 'pickle':
        _save_data_pickle(struct, fname)
    elif get_file_format(fname) == 'nc':
        _save_data_nc(struct, fname)


def load_data(fname):
    """Load generic data in maltpynt format."""
    if get_file_format(fname) == 'pickle':
        return _load_data_pickle(fname)
    elif get_file_format(fname) == 'nc':
        return _load_data_nc(fname)


# QDP format is often used in FTOOLS
def save_as_qdp(arrays, errors=None, filename="out.qdp"):
    """Save arrays in a QDP file.

    Saves an array of variables, and possibly their errors, to a QDP file.

    Parameters
    ----------
    arrays: [array1, array2]
        List of variables. All variables must be arrays and of the same length.
    errors: [array1, array2]
        List of errors. The order has to be the same of arrays; the value can
        be:
        - None if no error is assigned
        - an array of same length of variable for symmetric errors
        - an array of len-2 lists for non-symmetric errors (e.g.
        [[errm1, errp1], [errm2, errp2], [errm3, errp3], ...])
    """
    import numpy as np
    errors = _assign_value_if_none(errors, [None for i in arrays])

    data_to_write = []
    list_of_errs = []
    for ia, ar in enumerate(arrays):
        data_to_write.append(ar)
        if errors[ia] is None:
            continue
        shape = np.shape(errors[ia])
        assert shape[0] == len(ar), \
            'Errors and arrays must have same length'
        if len(shape) == 1:
            list_of_errs.append([ia, 'S'])
            data_to_write.append(errors[ia])
        elif shape[1] == 2:
            list_of_errs.append([ia, 'T'])
            mine = [k[0] for k in errors[ia]]
            maxe = [k[1] for k in errors[ia]]
            data_to_write.append(mine)
            data_to_write.append(maxe)
    outfile = open(filename, 'w')
    for l in list_of_errs:
        i, kind = l
        print('READ %s' % kind + 'ERR %d' % (i + 1), file=outfile)

    length = len(data_to_write[0])
    for i in range(length):
        for idw, d in enumerate(data_to_write):
            print(d[i], file=outfile, end=" ")
        print("", file=outfile)

    outfile.close()


def save_as_ascii(cols, filename="out.txt", colnames=None,
                  append=False):
    """Save arrays as TXT file with respective errors."""
    import numpy as np

    logging.debug('%s %s' % (repr(cols), repr(np.shape(cols))))
    if append:
        txtfile = open(filename, "a")
    else:
        txtfile = open(filename, "w")
    shape = np.shape(cols)
    ndim = len(shape)

    if ndim == 1:
        cols = [cols]
    elif ndim > 3 or ndim == 0:
        logging.error("Only one- or two-dim arrays accepted")
        return -1
    lcol = len(cols[0])

    if colnames is not None:
        print("#", file=txtfile, end=' ')
        for i_c, c in enumerate(cols):
            print(colnames[i_c], file=txtfile, end=' ')
        print('', file=txtfile)
    for i in range(lcol):
        for c in cols:
            print(c[i], file=txtfile, end=' ')

        print('', file=txtfile)
    txtfile.close()
    return 0


def print_fits_info(fits_file, hdu=1):
    """Print general info about an observation."""
    from astropy.io import fits as pf
    from astropy.units import Unit
    from astropy.time import Time

    lchdulist = pf.open(fits_file)

    datahdu = lchdulist[hdu]
    header = datahdu.header

    info = {}
    info['N. events'] = _get_key(header, 'NAXIS2')
    info['Telescope'] = _get_key(header, 'TELESCOP')
    info['Instrument'] = _get_key(header, 'INSTRUME')
    info['OBS_ID'] = _get_key(header, 'OBS_ID')
    info['Target'] = _get_key(header, 'OBJECT')
    info['Start'] = _get_key(header, 'DATE-OBS')
    info['Stop'] = _get_key(header, 'DATE-END')

    # Give time in MJD
    mjdref = high_precision_keyword_read(header, 'MJDREF')
    tstart = high_precision_keyword_read(header, 'TSTART')
    tstop = high_precision_keyword_read(header, 'TSTOP')
    tunit = _get_key(header, 'TIMEUNIT')
    start_mjd = Time(mjdref, format='mjd') + tstart * Unit(tunit)
    stop_mjd = Time(mjdref, format='mjd') + tstop * Unit(tunit)

    print('ObsID:         {0}\n'.format(info['OBS_ID']))
    print('Date:          {0} -- {1}\n'.format(info['Start'], info['Stop']))
    print('Date (MJD):    {0} -- {1}\n'.format(start_mjd, stop_mjd))
    print('Instrument:    {0}/{1}\n'.format(info['Telescope'],
                                            info['Instrument']))
    print('Target:        {0}\n'.format(info['Target']))
    print('N. Events:     {0}\n'.format(info['N. events']))

    lchdulist.close()
    return info


def _get_gti_from_extension(lchdulist, accepted_gtistrings=['GTI']):
    hdunames = [h.name for h in lchdulist]
    gtiextn = [ix for ix, x in enumerate(hdunames)
               if x in accepted_gtistrings][0]
    gtiext = lchdulist[gtiextn]
    gtitable = gtiext.data

    colnames = [col.name for col in gtitable.columns.columns]
    # Default: NuSTAR: START, STOP. Otherwise, try RXTE: Start, Stop
    if 'START' in colnames:
        startstr, stopstr = 'START', 'STOP'
    else:
        startstr, stopstr = 'Start', 'Stop'

    gtistart = np.array(gtitable.field(startstr), dtype=np.longdouble)
    gtistop = np.array(gtitable.field(stopstr), dtype=np.longdouble)
    gti_list = np.array([[a, b]
                         for a, b in zip(gtistart,
                                         gtistop)],
                        dtype=np.longdouble)
    return gti_list


def _get_additional_data(lctable, additional_columns):
    additional_data = {}
    if additional_columns is not None:
        for a in additional_columns:
            try:
                additional_data[a] = np.array(lctable.field(a))
            except:  # pragma: no cover
                if a == 'PI':
                    logging.warning('Column PI not found. Trying with PHA')
                    additional_data[a] = np.array(lctable.field('PHA'))
                else:
                    raise Exception('Column' + a + 'not found')

    return additional_data


def load_gtis(fits_file, gtistring=None):
    """Load GTI from HDU EVENTS of file fits_file."""
    from astropy.io import fits as pf
    import numpy as np

    gtistring = _assign_value_if_none(gtistring, 'GTI')
    logging.info("Loading GTIS from file %s" % fits_file)
    lchdulist = pf.open(fits_file, checksum=True)
    lchdulist.verify('warn')

    gtitable = lchdulist[gtistring].data
    gti_list = np.array([[a, b]
                         for a, b in zip(gtitable.field('START'),
                                         gtitable.field('STOP'))],
                        dtype=np.longdouble)
    lchdulist.close()
    return gti_list


def load_events_and_gtis(fits_file, additional_columns=None,
                         gtistring='GTI,STDGTI',
                         gti_file=None, hduname='EVENTS', column='TIME'):
    """Load event lists and GTIs from one or more files.

    Loads event list from HDU EVENTS of file fits_file, with Good Time
    intervals. Optionally, returns additional columns of data from the same
    HDU of the events.

    Parameters
    ----------
    fits_file : str
    return_limits: bool, optional
        Return the TSTART and TSTOP keyword values
    additional_columns: list of str, optional
        A list of keys corresponding to the additional columns to extract from
        the event HDU (ex.: ['PI', 'X'])

    Returns
    -------
    ev_list : array-like
    gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    additional_data: dict
        A dictionary, where each key is the one specified in additional_colums.
        The data are an array with the values of the specified column in the
        fits file.
    t_start : float
    t_stop : float
    """
    from astropy.io import fits as pf

    lchdulist = pf.open(fits_file)

    # Load data table
    try:
        lctable = lchdulist[hduname].data
    except:  # pragma: no cover
        logging.warning('HDU %s not found. Trying first extension' % hduname)
        lctable = lchdulist[1].data

    # Read event list
    ev_list = np.array(lctable.field(column), dtype=np.longdouble)

    # Read TIMEZERO keyword and apply it to events
    try:
        timezero = np.longdouble(lchdulist[1].header['TIMEZERO'])
    except:  # pragma: no cover
        logging.warning("No TIMEZERO in file")
        timezero = np.longdouble(0.)

    ev_list += timezero

    # Read TSTART, TSTOP from header
    try:
        t_start = np.longdouble(lchdulist[1].header['TSTART'])
        t_stop = np.longdouble(lchdulist[1].header['TSTOP'])
    except:  # pragma: no cover
        logging.warning("Tstart and Tstop error. using defaults")
        t_start = ev_list[0]
        t_stop = ev_list[-1]

    # Read and handle GTI extension
    accepted_gtistrings = gtistring.split(',')

    if gti_file is None:
        # Select first GTI with accepted name
        try:
            gti_list = \
                _get_gti_from_extension(
                    lchdulist, accepted_gtistrings=accepted_gtistrings)
        except:  # pragma: no cover
            warnings.warn("No extensions found with a valid name. "
                          "Please check the `accepted_gtistrings` values.")
            gti_list = np.array([[t_start, t_stop]],
                                dtype=np.longdouble)
    else:
        gti_list = load_gtis(gti_file, gtistring)

    additional_data = _get_additional_data(lctable, additional_columns)

    lchdulist.close()

    # Sort event list
    order = np.argsort(ev_list)
    ev_list = ev_list[order]

    additional_data = _order_list_of_arrays(additional_data, order)

    returns = _empty()
    returns.ev_list = ev_list
    returns.gti_list = gti_list
    returns.additional_data = additional_data
    returns.t_start = t_start
    returns.t_stop = t_stop

    return returns


def main(args=None):
    """Main function called by the `MPreadfile` command line script."""
    from astropy.time import Time
    import astropy.units as u
    import argparse

    description = \
        'Print the content of MaLTPyNT files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')

    args = parser.parse_args(args)

    for fname in args.files:
        print()
        print('-' * len(fname))
        print('{0}'.format(fname))
        print('-' * len(fname))
        if fname.endswith('.fits') or fname.endswith('.evt'):
            print('This FITS file contains:', end='\n\n')
            print_fits_info(fname)
            print('-' * len(fname))
            continue
        ftype, contents = get_file_type(fname)
        print('This file contains:', end='\n\n')
        mjdref = Time(contents['MJDref'], format='mjd')

        for k in sorted(contents.keys()):
            if k in ['Tstart', 'Tstop']:
                timeval = contents[k] * u.s
                val = '{0} (MJD {1})'.format(contents[k], mjdref + timeval)
            else:
                val = contents[k]
            if isinstance(val, collections.Iterable) and not is_string(val):
                length = len(val)
                if len(val) < 4:
                    val = repr(list(val[:4]))
                else:
                    val = repr(list(val[:4])).replace(']', '') + '...]'
                    val = '{} (len {})'.format(val, length)
            print((k + ':').ljust(15), val, end='\n\n')

        print('-' * len(fname))


def sort_files(files):
    """Sort a list of MaLTPyNT files, looking at `Tstart` in each."""
    allfiles = {}
    ftypes = []

    for f in files:
        logging.info('Loading file ' + f)
        ftype, contents = get_file_type(f)
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
