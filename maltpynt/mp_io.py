from __future__ import unicode_literals
from __future__ import print_function
try:
    import netCDF4 as nc
    MP_FILE_EXTENSION = '.nc'
except:
    MP_FILE_EXTENSION = '.p'
    pass
import cPickle as pickle
import collections
import numpy as np
import os.path


cpl128 = np.dtype([(str('real'), np.double),
                   (str('imag'), np.double)])


def is_string(s):
    import sys
    PY3 = sys.version_info[0] == 3
    if PY3:
        return isinstance(s, str)
    else:
        return isinstance(s, basestring)


def mp_get_file_extension(fname):
    return os.path.splitext(fname)[1]


def mp_get_file_format(fname):
    '''Decide the file format of the file'''
    ext = mp_get_file_extension(fname)
    if ext == '.p':
        return 'pickle'
    elif ext == '.nc':
        return 'nc'
    else:
        raise Exception("File format not recognized")


#---- Base function to save NetCDF4 files
def mp_save_as_netcdf(vars, varnames, formats, fname):
    '''The future. Much faster than pickle'''

    rootgrp = nc.Dataset(fname, 'w',
                         format='NETCDF4')

    for iv, v in enumerate(vars):
        dims = {}
        dimname = varnames[iv]+"dim"
        dimspec = (varnames[iv]+"dim", )

        if formats[iv] == 'c16':
            # unicode_literals breaks something, I need to specify str.
            complex128_t = rootgrp.createCompoundType(cpl128, 'cpl128')
            vcomp = np.empty(v.size, dtype=cpl128)
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


def mp_read_from_netcdf(fname):
    rootgrp = nc.Dataset(fname)
    out = {}
    for k in rootgrp.variables.keys():
        dum = rootgrp.variables[k]
        values = dum.__array__()
        # Handle special case of complex
        if dum.dtype == cpl128:
            arr = np.empty(values.size, dtype=np.complex128)
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


#----- Functions to handle file types
def mp_get_file_type(fname):
    contents = mp_load_data(fname)
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
    elif 'time' in keys:
        # If it has not lc, pds or cpds, but has time, ...
        ftype = 'events'
    elif 'GTI' in keys:
        # If nothing of the above, but has GTIs, than...
        ftype = 'GTI'

    return ftype, contents


#----- functions to save and load EVENT data
def mp_save_events(eventStruct, fname):
    if mp_get_file_format(fname) == 'pickle':
        save_data_pickle(eventStruct, fname)
    elif mp_get_file_format(fname) == 'nc':
        save_data_nc(eventStruct, fname)


def mp_load_events(fname):
    if mp_get_file_format(fname) == 'pickle':
        return load_data_pickle(fname)
    elif mp_get_file_format(fname) == 'nc':
        return load_data_nc(fname)


#----- functions to save and load LCURVE data
def mp_save_lcurve(lcurveStruct, fname):
    if mp_get_file_format(fname) == 'pickle':
        return save_data_pickle(lcurveStruct, fname)
    elif mp_get_file_format(fname) == 'nc':
        return save_data_nc(lcurveStruct, fname)


def mp_load_lcurve(fname):
    if mp_get_file_format(fname) == 'pickle':
        return load_data_pickle(fname)
    elif mp_get_file_format(fname) == 'nc':
        return load_data_nc(fname)


# ---- Functions to save PDSs

def mp_save_pds(pdsStruct, fname):
    if mp_get_file_format(fname) == 'pickle':
        return save_data_pickle(pdsStruct, fname)
    elif mp_get_file_format(fname) == 'nc':
        return save_data_nc(pdsStruct, fname)


def mp_load_pds(fname):
    if mp_get_file_format(fname) == 'pickle':
        return load_data_pickle(fname)
    elif mp_get_file_format(fname) == 'nc':
        return load_data_nc(fname)


# ---- GENERIC function to save stuff.
def load_data_pickle(fname, kind="data"):
    print ('Loading %s and info from %s' % (kind, fname))
    return pickle.load(open(fname))
    return


def save_data_pickle(struct, fname, kind="data"):
    print ('Saving %s and info to %s' % (kind, fname))
    pickle.dump(struct, open(fname, 'wb'))
    return


def load_data_nc(fname):
    contents = mp_read_from_netcdf(fname)
    keys = contents.keys()

    keys_to_delete = []
    for k in keys:
        if k[-2:] in ['_I', '_F']:
            kcorr = k[:-2]

            if kcorr not in contents.keys():
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


def save_data_nc(struct, fname, kind="data"):
    print ('Saving %s and info to %s' % (kind, fname))
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
                print ('This failed:', k, var, 'in file ', fname)
                return -1
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

    mp_save_as_netcdf(values, varnames, formats, fname)


def mp_save_data(struct, fname, ftype='data'):
    if mp_get_file_format(fname) == 'pickle':
        save_data_pickle(struct, fname)
    elif mp_get_file_format(fname) == 'nc':
        save_data_nc(struct, fname)


def mp_load_data(fname):
    if mp_get_file_format(fname) == 'pickle':
        return load_data_pickle(fname)
    elif mp_get_file_format(fname) == 'nc':
        return load_data_nc(fname)


# QDP format is often used in FTOOLS
def save_as_qdp(arrays, errors=None, filename="out.qdp"):
    '''
    Saves an array of variables, and possibly their errors, to a QDP file.
    input:
        arrays: list of variables. All variables must be arrays and of the same
                length!
        errors: list of errors. The order has to be the same of arrays; the
                value can be
                  - None if no error is assigned
                  - an array of same length of variable for symmetric errors
                  - an array of len-2 lists for non-symmetric errors (e.g.
                    [[errm1, errp1], [errm2, errp2], [errm3, errp3], ...])
    '''
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
        print ('READ %s' % kind + 'ERR %d' % (i + 1), file=outfile)

    length = len(data_to_write[0])
    for i in range(length):
        for idw, d in enumerate(data_to_write):
            print (d[i], file=outfile, end=" ")
        print ("", file=outfile)

    outfile.close()


def save_as_ascii(cols, filename="out.txt", colnames=None, verbose=-1,
                  append=False):
    '''
    Saves as TXT file with respective errors
    '''
    import numpy as np

    if verbose > 1:
        print (cols, np.shape(cols))
    if append:
        txtfile = open(filename, "a")
    else:
        txtfile = open(filename, "w")
    shape = np.shape(cols)
    ndim = len(shape)
    #if colnames is None:
        #colnames = range(len(cols))
    if ndim == 1:
        cols = [cols]
    elif ndim > 3 or ndim == 0:
        print ("Only one- or two-dim arrays accepted")
        return -1
    lcol = len(cols[0])

    if colnames is not None:
        print ("#", file=txtfile, end=' ')
        for i_c, c in enumerate(cols):
            print (colnames[i_c], file=txtfile, end=' ')
        print ('', file=txtfile)
    for i in range(lcol):
        for c in cols:
            print (c[i], file=txtfile, end=' ')

        print ('', file=txtfile)
    txtfile.close()
    return 0

