try:
    import netCDF4 as nc
except:
    pass
import cPickle as pickle


def mp_file_format(fname):
    '''Decide the file format of the file'''
    if fname[-2:] == '.p':
        return 'pickle'
    elif fname[-3:] == '.nc':
        return 'nc'
    else:
        raise Exception("File format not recognized")


#---- Base function to save NetCDF4 files
def mp_save_as_netcdf(vars, varnames, formats, fname):
    '''The future. Much faster than pickle'''
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


#----- Functions to handle file types
def mp_get_file_type_pickle(fname):
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


def mp_get_file_type_nc(fname):
    '''Gets file type'''
    # TODO: implement this

    raise Exception("NetCDF file type handling not yet implemented")


def mp_get_file_type(fname):
    '''Gets file type'''
    if mp_file_format(fname) == 'pickle':
        return mp_get_file_type_pickle(fname)
    elif mp_file_format(fname) == 'nc':
        return mp_get_file_type_nc(fname)
    else:
        raise Exception("File format not recognized")


#----- functions to save and load EVENT data
def save_events_nc(eventStruct, fname):
    # TODO: implement this

    raise Exception("NetCDF file type handling not yet implemented")


def mp_save_events(eventStruct, fname):
    if mp_file_format(fname) == 'pickle':
        save_data_pickle(eventStruct, fname)
    elif mp_file_format(fname) == 'nc':
        save_events_nc(eventStruct, fname)


def load_events_nc(fname):
    # TODO: implement this

    raise Exception("NetCDF file type handling not yet implemented")


def mp_load_events(fname):
    if mp_file_format(fname) == 'pickle':
        return load_data_pickle(fname)
    elif mp_file_format(fname) == 'nc':
        return load_events_nc(fname)


#----- functions to save and load LCURVE data
def save_lcurve_nc(lcurveStruct, fname):
    # TODO: implement this

    raise Exception("NetCDF file type handling not yet implemented")


def mp_save_lcurve(lcurveStruct, fname):
    if mp_file_format(fname) == 'pickle':
        return save_data_pickle(lcurveStruct, fname)
    elif mp_file_format(fname) == 'nc':
        return save_lcurve_nc(lcurveStruct, fname)


def load_lcurve_nc(fname):
    # TODO: implement this

    raise Exception("NetCDF file type handling not yet implemented")


def mp_load_lcurve(fname):
    if mp_file_format(fname) == 'pickle':
        return load_data_pickle(fname)
    elif mp_file_format(fname) == 'nc':
        return load_lcurve_nc(fname)


# ---- Functions to save PDSs

def save_pds_nc(pdsStruct, fname):
    # TODO: implement this

    raise Exception("NetCDF file type handling not yet implemented")


def mp_save_pds(pdsStruct, fname):
    if mp_file_format(fname) == 'pickle':
        return save_data_pickle(pdsStruct, fname)
    elif mp_file_format(fname) == 'nc':
        return save_pds_nc(pdsStruct, fname)


def load_pds_nc(fname):
    # TODO: implement this

    raise Exception("NetCDF file type handling not yet implemented")


def mp_load_pds(fname):
    if mp_file_format(fname) == 'pickle':
        return load_data_pickle(fname)
    elif mp_file_format(fname) == 'nc':
        return load_pds_nc(fname)


# ---- GENERIC function to save stuff.
def load_data_pickle(fname):
    print ('Loading pds and info from %s' % fname)
    return pickle.load(open(fname))
    return


def save_data_pickle(struct, fname, kind="data"):
    print ('Saving %s and info to %s' % (kind, fname))
    pickle.dump(struct, open(fname, 'wb'))
    return


def mp_save_data(struct, fname, ftype):
    if mp_file_format(fname) == 'pickle':
        save_data_pickle(struct, fname)
    else:
        raise Exception("NetCDF file type handling not yet implemented")
