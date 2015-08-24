# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform input/output operations."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import logging
try:
    import netCDF4 as nc
    MP_FILE_EXTENSION = '.nc'
except:
    msg = "Warning! NetCDF is not available. Using pickle format."
    logging.warning(msg)
    print(msg)
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
import sys

cpl128 = np.dtype([(str('real'), np.double),
                   (str('imag'), np.double)])


def is_string(s):
    """Portable function to answer this question."""
    PY3 = sys.version_info[0] == 3
    if PY3:
        return isinstance(s, str)
    else:
        return isinstance(s, basestring)


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
    """Gets file type."""

    keys = list(contents.keys())
    if 'lc' in keys:
        ftype = 'lc'
    elif 'cpds' in keys:
        ftype = 'cpds'
        if 'fhi' in keys and specify_reb:
            ftype = 'rebcpds'
    elif 'pds' in keys:
        ftype = 'pds'
        if 'fhi' in keys and specify_reb:
            ftype = 'rebpds'
    elif 'lag' in keys:
        ftype = 'lag'
        if 'fhi' in keys and specify_reb:
            ftype = 'reblag'
    elif 'time' in keys:
        # If it has not lc, pds or cpds, but has time, ...
        ftype = 'events'
    elif 'GTI' in keys:
        # If nothing of the above, but has GTIs, than...
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
        print('Failed')


def _save_data_pickle(struct, fname, kind="data"):
    """Save generic data in pickle format."""
    logging.info('Saving %s and info to %s' % (kind, fname))
    try:
        with open(fname, 'wb') as fobj:
            pickle.dump(struct, fobj)
    except Exception as e:
        raise Exception("{0} failed ({1}: {2})".format('_save_data_pickle',
                                                       type(e), e))
        print('Failed')
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
    if errors is None:
        errors = [None for i in arrays]
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


def _get_key(dict_like, key):
    try:
        return dict_like[key]
    except:
        return ""


def print_fits_info(fits_file, hdu=1):
    """Print general info about an observation."""
    from astropy.io import fits as pf

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

    print('ObsID:         {0}\n'.format(info['OBS_ID']))
    print('Date:          {0} -- {1}\n'.format(info['Start'], info['Stop']))
    print('Instrument:    {0}/{1}\n'.format(info['Telescope'],
                                            info['Instrument']))
    print('Target:        {0}\n'.format(info['Target']))
    print('N. Events:     {0}\n'.format(info['N. events']))

    lchdulist.close()
    return info
