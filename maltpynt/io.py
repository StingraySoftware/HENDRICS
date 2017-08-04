# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform input/output operations."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import logging
import warnings
from stingray.gti import cross_gtis
from stingray.events import EventList
from stingray.lightcurve import Lightcurve
from stingray.powerspectrum import Powerspectrum, AveragedPowerspectrum
from stingray.crossspectrum import Crossspectrum, AveragedCrossspectrum
try:
    import netCDF4 as nc
    MP_FILE_EXTENSION = '.nc'
except:
    msg = "Warning! NetCDF is not available. Using pickle format."
    logging.warning(msg)
    MP_FILE_EXTENSION = '.p'
    pass

try:
    # Python 2
    import cPickle as pickle
except:
    # Python 3
    import pickle

import collections
import numpy as np
import os.path
from .base import _order_list_of_arrays, _empty, is_string
from stingray.utils import assign_value_if_none
import os
import glob
import copy

cpl128 = np.dtype([(str('real'), np.double),
                   (str('imag'), np.double)])
cpl256 = np.dtype([(str('real'), np.longdouble),
                   (str('imag'), np.longdouble)])


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
        The value of the key, or None if keyword not present

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

        if formats[iv] == 'c32':
            # Too complicated. Let's decrease precision
            warnings.warn("complex256 yet unsuported")
            formats[iv] = 'c16'

        if formats[iv] == 'c16':
            # unicode_literals breaks something, I need to specify str.
            if 'cpl128' not in rootgrp.cmptypes.keys():
                complex128_t = rootgrp.createCompoundType(cpl128, 'cpl128')
            vcomp = np.empty(v.shape, dtype=cpl128)
            vcomp['real'] = v.real.astype(np.float64)
            vcomp['imag'] = v.imag.astype(np.float64)
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
        try:
            if formats[iv] == str:
                vnc[0] = v
            else:
                vnc[:] = v
        except:
            print("Error:", varnames[iv], formats[iv], dimspec, v)
            raise
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
        # Handle special case of complex
        if dum.dtype == cpl256:
            arr = np.empty(values.shape, dtype=np.complex256)
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
def get_file_type(fname):
    """Return the file type and its contents.

    Only works for maltpynt-format pickle or netcdf files.
    """
    contents = load_data(fname)

    keys = list(contents.keys())

    ftype_raw = contents['__sr__class__type__']
    if 'Lightcurve' in ftype_raw:
        ftype = 'lc'
        fun = load_lcurve
    elif 'EventList' in ftype_raw:
        ftype = 'events'
        fun = load_events
    elif 'Crossspectrum' in ftype_raw:
        ftype = 'cpds'
        fun = load_pds
    elif 'Powerspectrum' in ftype_raw:
        ftype = 'pds'
        fun = load_pds
    else:
        raise ValueError('File format not understood')

    contents = fun(fname)

    return ftype, contents


# ----- functions to save and load EVENT data
def save_events(eventlist, fname):
    """Save events in a file.

    Parameters
    ----------
    eventlist: :class:`stingray.EventList` object
        Event list to be saved
    fname: str
        Name of output file
    """
    out = {'time': eventlist.time,
           'gti': eventlist.gti,
           'pi': eventlist.pi,
           'mjdref': eventlist.mjdref,
           'tstart': np.min(eventlist.gti),
           'tstop': np.max(eventlist.gti)
           }

    out['__sr__class__type__'] = str(type(eventlist))

    if hasattr(eventlist, 'instr') and eventlist.instr is not None:
        out["instr"] = eventlist.instr
    else:
        out["instr"] = 'unknown'
    if hasattr(eventlist, 'energy') and eventlist.energy is not None:
        out['energy'] = eventlist.energy
    if hasattr(eventlist, 'header') and eventlist.header is not None:
        out["header"] = eventlist.header

    if get_file_format(fname) == 'pickle':
        _save_data_pickle(out, fname)
    elif get_file_format(fname) == 'nc':
        _save_data_nc(out, fname)


def load_events(fname):
    """Load events from a file."""
    if get_file_format(fname) == 'pickle':
        out = _load_data_pickle(fname)
    elif get_file_format(fname) == 'nc':
        out = _load_data_nc(fname)

    eventlist = EventList()

    eventlist.time = out['time']
    eventlist.gti = out['gti']
    eventlist.pi = out['pi']
    eventlist.mjdref = out['mjdref']
    if 'instr' in list(out.keys()):
        eventlist.instr = out["instr"]
    if 'energy' in list(out.keys()):
        eventlist.energy = out["energy"]
    if 'header' in list(out.keys()):
        eventlist.header = out["header"]
    return eventlist


# ----- functions to save and load LCURVE data
def save_lcurve(lcurve, fname):
    """Save Light curve to file

    Parameters
    ----------
    lcurve: :class:`stingray.Lightcurve` object
        Event list to be saved
    fname: str
        Name of output file
    """
    out = {}

    out['__sr__class__type__'] = str(type(lcurve))

    out['counts'] = lcurve.counts
    out['counts_err'] = lcurve.counts_err
    out['time'] = lcurve.time
    out['dt'] = lcurve.dt
    out['gti'] = lcurve.gti
    out['err_dist'] = lcurve.err_dist
    out['mjdref'] = lcurve.mjdref
    if hasattr(lcurve, 'header'):
        out['header'] = lcurve.header

    if hasattr(lcurve, 'instr'):
        out["instr"] = lcurve.instr
    else:
        out["instr"] = 'unknown'

    if get_file_format(fname) == 'pickle':
        return _save_data_pickle(out, fname)
    elif get_file_format(fname) == 'nc':
        return _save_data_nc(out, fname)


def load_lcurve(fname):
    """Load light curve from a file."""
    if get_file_format(fname) == 'pickle':
        data =  _load_data_pickle(fname)
    elif get_file_format(fname) == 'nc':
        data = _load_data_nc(fname)

    lcurve = Lightcurve(data['time'], data['counts'], err=data['counts_err'],
                        gti=data['gti'], err_dist = data['err_dist'],
                        mjdref=data['mjdref'])
    if 'instr' in list(data.keys()):
        lcurve.instr = data["instr"]
    if 'header' in list(data.keys()):
        lcurve.header = data["header"]

    return lcurve

# ---- Functions to save PDSs

def save_pds(cpds, fname):
    """Save PDS in a file."""

    outdata = copy.copy(cpds.__dict__)
    outdata['__sr__class__type__'] = str(type(cpds))

    if not hasattr(cpds, 'instr'):
        outdata["instr"] = 'unknown'

    if 'lc1' in outdata:
        save_lcurve(cpds.lc1, fname.replace(MP_FILE_EXTENSION,
                                            '__lc1__' + MP_FILE_EXTENSION))
        outdata.pop('lc1')
    if 'lc2' in outdata:
        save_lcurve(cpds.lc2, fname.replace(MP_FILE_EXTENSION,
                                            '__lc2__' + MP_FILE_EXTENSION))
        outdata.pop('lc2')
    if 'pds1' in outdata:
        save_pds(cpds.pds1, fname.replace(MP_FILE_EXTENSION,
                                            '__pds1__' + MP_FILE_EXTENSION))
        outdata.pop('pds1')
    if 'pds2' in outdata:
        save_pds(cpds.pds2, fname.replace(MP_FILE_EXTENSION,
                                            '__pds2__' + MP_FILE_EXTENSION))
        outdata.pop('pds2')
    if 'cs_all' in outdata:
        for i, c in enumerate(cpds.cs_all):
            save_pds(c,
                     fname.replace(MP_FILE_EXTENSION,
                                   '__cs__{}__'.format(i) + MP_FILE_EXTENSION))
        outdata.pop('cs_all')

    if get_file_format(fname) == 'pickle':
        return _save_data_pickle(outdata, fname)
    elif get_file_format(fname) == 'nc':
        return _save_data_nc(outdata, fname)


def load_pds(fname):
    """Load PDS from a file."""
    if get_file_format(fname) == 'pickle':
        data = _load_data_pickle(fname)
    elif get_file_format(fname) == 'nc':
        data = _load_data_nc(fname)

    type_string = data['__sr__class__type__']
    if 'AveragedPowerspectrum' in type_string:
        cpds = AveragedPowerspectrum()
    elif 'Powerspectrum' in type_string:
        cpds = Powerspectrum()
    elif 'AveragedCrossspectrum' in type_string:
        cpds = AveragedCrossspectrum()
    elif 'Crossspectrum' in type_string:
        cpds = Crossspectrum()
    else:
        raise ValueError('Unrecognized data type in file')

    data.pop('__sr__class__type__')
    for key in data.keys():
        setattr(cpds, key, data[key])

    lc1_name = fname.replace(MP_FILE_EXTENSION, '__lc1__' + MP_FILE_EXTENSION)
    lc2_name = fname.replace(MP_FILE_EXTENSION, '__lc2__' + MP_FILE_EXTENSION)
    pds1_name = fname.replace(MP_FILE_EXTENSION, '__pds1__' + MP_FILE_EXTENSION)
    pds2_name = fname.replace(MP_FILE_EXTENSION, '__pds2__' + MP_FILE_EXTENSION)
    cs_all_names = glob.glob(
        fname.replace(MP_FILE_EXTENSION, '__cs__[0-9]__' + MP_FILE_EXTENSION))

    if os.path.exists(lc1_name):
        cpds.lc1 = load_lcurve(lc1_name)
    if os.path.exists(lc2_name):
        cpds.lc2 = load_lcurve(lc2_name)
    if os.path.exists(pds1_name):
        cpds.pds1 = load_pds(pds1_name)
    if os.path.exists(pds2_name):
        cpds.pds2 = load_pds(pds2_name)
    if len(cs_all_names) > 0:
        cs_all = []
        for c in cs_all_names:
            cs_all.append(load_pds(c))
        cpds.cs_all = cs_all

    return cpds

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
        if k in keys_to_delete:
            continue

        if contents[k] == '__mp__None__type__':
            contents[k] = None

        if k[-2:] in ['_I', '_L', '_F', '_k']:
            kcorr = k[:-2]

            integer_key = kcorr + '_I'
            float_key = kcorr + '_F'
            kind_key = kcorr + '_k'
            log10_key = kcorr + '_L'

            if not (integer_key in keys and float_key in keys):
                continue
            # Maintain compatibility with old-style files:
            if not (kind_key in keys and log10_key in keys):
                contents[kind_key] = "longdouble"
                contents[log10_key] = 0

            keys_to_delete.extend([integer_key, float_key])
            keys_to_delete.extend([kind_key, log10_key])

            if contents[kind_key] == 'longdouble':
                dtype = np.longdouble
            elif contents[kind_key] == 'double':
                dtype = np.double
            else:
                raise ValueError(contents[kind_key] +
                                 ": unrecognized kind string")

            log10_part = contents[log10_key]
            if isinstance(contents[integer_key], collections.Iterable):
                integer_part = np.array(contents[integer_key], dtype=dtype)
                float_part = np.array(contents[float_key], dtype=dtype)
            else:
                integer_part = dtype(contents[integer_key])
                float_part = dtype(contents[float_key])

            contents[kcorr] = (integer_part + float_part) * 10. ** log10_part

    for k in keys_to_delete:
        del contents[k]

    return contents


def _split_high_precision_number(varname, var, probesize):
    var_log10 = 0
    if probesize == 8:
        kind_str = 'double'
    if probesize == 16:
        kind_str = 'longdouble'

    if isinstance(var, collections.Iterable):
        dum = np.min(np.abs(var))
        if dum < 1 and dum > 0.:
            var_log10 = np.floor(np.log10(dum))

        var = var / (10. ** var_log10)
        var_I = np.floor(var).astype(np.long)
        var_F = np.array(var - var_I, dtype=np.double)
    else:
        if np.abs(var) < 1 and np.abs(var) > 0.:
            var_log10 = np.floor(np.log10(np.abs(var)))

        var = var / 10. ** var_log10
        var_I = np.long(np.floor(var))
        var_F = np.double(var - var_I)
    return var_I, var_F, var_log10, kind_str


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
        elif var is None:
            probekind = None
        else:
            probekind = np.result_type(probe).kind
            probesize = np.result_type(probe).itemsize

        if probesize >= 8 and probekind == 'f':
            # If a (long)double, split it in integer + floating part.
            # If the number is below zero, also use a logarithm of 10 before
            # that, so that we don't lose precision
            var_I, var_F, var_log10, kind_str = \
                _split_high_precision_number(k, var, probesize)
            values.extend([var_I, var_log10, var_F, kind_str])
            formats.extend(['i8', 'i8', 'f8', str])
            varnames.extend([k + '_I', k + '_L', k + '_F', k + '_k'])
        elif probekind == str:
            values.append(var)
            formats.append(probekind)
            varnames.append(k)
        elif probekind is None:
            values.append('__mp__None__type__')
            formats.append(str)
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
    errors = assign_value_if_none(errors, [None for i in arrays])

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


def _get_gti_from_all_extensions(lchdulist, accepted_gtistrings=['GTI'],
                                 det_numbers=None):
    if det_numbers is None:
        return _get_gti_from_extension(lchdulist, accepted_gtistrings)

    gti_lists = []
    for i in det_numbers:
        acc_gti_str = [x + '{:02d}'.format(i) for x in accepted_gtistrings]
        gti_lists.append(_get_gti_from_extension(lchdulist, acc_gti_str))

    return cross_gtis(gti_lists)


def _get_additional_data(lctable, additional_columns):
    additional_data = {}
    if additional_columns is not None:
        for a in additional_columns:
            try:
                additional_data[a] = np.array(lctable.field(a))
            except KeyError:
                warnings.warn("Column {} not found".format(a))
                additional_data[a] = np.zeros(len(lctable))

    return additional_data


def load_gtis(fits_file, gtistring=None):
    """Load GTI from HDU EVENTS of file fits_file."""
    from astropy.io import fits as pf
    import numpy as np

    gtistring = assign_value_if_none(gtistring, 'GTI')
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

    gtistring = assign_value_if_none(gtistring, 'GTI,STDGTI')
    lchdulist = pf.open(fits_file)

    # Load data table
    try:
        lctable = lchdulist[hduname].data
    except:  # pragma: no cover
        logging.warning('HDU %s not found. Trying first extension' % hduname)
        lctable = lchdulist[1].data
        hduname = 1

    # Read event list
    ev_list = np.array(lctable.field(column), dtype=np.longdouble)
    det_number = None
    detector_id = None
    if 'CCDNR' in lctable.columns.names:
        detector_id = np.array(lctable.field('CCDNR'), dtype=np.int)
        det_number = list(set(detector_id))
    if 'PCUID' in lctable.columns.names:
        detector_id = np.array(lctable.field('PCUID'), dtype=np.int)

    header = lchdulist[1].header
    # Read TIMEZERO keyword and apply it to events
    try:
        timezero = np.longdouble(header['TIMEZERO'])
    except:  # pragma: no cover
        logging.warning("No TIMEZERO in file")
        timezero = np.longdouble(0.)

    try:
        instr = header['INSTRUME']
    except:
        instr = 'unknown'

    ev_list += timezero

    # Read TSTART, TSTOP from header
    try:
        t_start = np.longdouble(header['TSTART'])
        t_stop = np.longdouble(header['TSTOP'])
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
                _get_gti_from_all_extensions(
                    lchdulist, accepted_gtistrings=accepted_gtistrings,
                    det_numbers=det_number)
        except:  # pragma: no cover
            warnings.warn("No extensions found with a valid name. "
                          "Please check the `accepted_gtistrings` values.")
            gti_list = np.array([[t_start, t_stop]],
                                dtype=np.longdouble)
    else:
        gti_list = load_gtis(gti_file, gtistring)

    if additional_columns is None:
        additional_columns = ['PI']
    if 'PI' not in additional_columns:
        additional_columns.append('PI')

    additional_data = _get_additional_data(lctable, additional_columns)

    lchdulist.close()

    # Sort event list
    order = np.argsort(ev_list)
    ev_list = ev_list[order]

    additional_data = _order_list_of_arrays(additional_data, order)
    pi = additional_data['PI'][order]
    additional_data.pop('PI')

    returns = _empty()
    returns.ev_list = EventList(ev_list, gti=gti_list, pi=pi)
    returns.ev_list.instr = instr
    returns.ev_list.header = header.tostring()
    returns.additional_data = additional_data
    returns.t_start = t_start
    returns.t_stop = t_stop
    returns.detector_id = detector_id

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
        mjdref = Time(contents['mjdref'], format='mjd')

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
        instr = contents.instr
        ftypes.append(ftype)
        if instr not in list(allfiles.keys()):
            allfiles[instr] = []
        # Add file name to the dictionary
        contents.__sort__filename__ = f
        allfiles[instr].append(contents)

    # Check if files are all of the same kind (lcs, PDSs, ...)
    ftypes = list(set(ftypes))
    assert len(ftypes) == 1, 'Files are not all of the same kind.'

    instrs = list(allfiles.keys())
    for instr in instrs:
        contents = list(allfiles[instr])
        tstarts = [np.min(c.gti) for c in contents]
        fnames = [c.__sort__filename__ for c in contents]

        fnames = [x for (y, x) in sorted(zip(tstarts, fnames))]

        # Substitute dictionaries with the sorted list of files
        allfiles[instr] = fnames

    return allfiles
