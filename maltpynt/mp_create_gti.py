# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to create GTIs."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .mp_io import MP_FILE_EXTENSION, mp_save_data
from .mp_base import mp_create_gti_from_condition, mp_root, mp_create_gti_mask
from .mp_base import mp_cross_gtis, mp_get_file_type
import logging


def mp_create_gti(fname, filter_expr, safe_interval=[0, 0]):
    """Create a GTI list by using boolean operations on file data."""
    # Necessary as nc variables are sometimes defined as array
    from numpy import array

    if filter_expr is None:
        sys.exit('Please specify a filter expression')
    ftype, data = mp_get_file_type(fname)

    instr = data['Instr']
    if ftype == 'lc' and instr == 'PCA':
        logging.warning('RXTE/PCA data; normalizing lc per no. PCUs')
        # If RXTE, plot per PCU count rate
        data['lc'] /= data['nPCUs']
    # Map all entries of data to local variables
    locals().update(data)

    good = eval(filter_expr)

    gtis = mp_create_gti_from_condition(locals()['time'], good,
                                        safe_interval=safe_interval)

    outfile = mp_root(fname) + '_gti' + MP_FILE_EXTENSION
    mp_save_data({'GTI': gtis}, outfile)

    return gtis


def mp_apply_gti(fname, gti, outname=None):
    """Apply a GTI list to the data contained in a file.

    File MUST have a GTI extension already, and an extension called `time`.
    """
    ftype, data = mp_get_file_type(fname)

    try:
        datagti = data['GTI']
        newgtis = mp_cross_gtis([gti, datagti])
    except:
        logging.warning('Data have no GTI extension')
        newgtis = gti

    data['GTI'] = newgtis
    good = mp_create_gti_mask(data['time'], newgtis)
    data['time'] = data['time'][good]
    if ftype == 'lc':
        data['lc'] = data['lc'][good]
    elif ftype == 'events':
        data['PI'] = data['PI'][good]
        if data['Instr'] == 'PCA':
            data['PCU'] = data['PCU'][good]

    if outname is None:
        outname = fname.replace(MP_FILE_EXTENSION, '') + \
            '_gtifilt' + MP_FILE_EXTENSION
    mp_save_data(data, outname)

    return newgtis


if __name__ == '__main__':
    import sys
    import subprocess as sp

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MPcreategti'] + args)
