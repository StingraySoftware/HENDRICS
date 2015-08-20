# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to create and apply GTIs."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import MP_FILE_EXTENSION, save_data
from .base import create_gti_from_condition, root, create_gti_mask
from .base import cross_gtis, get_file_type
import logging


def create_gti(fname, filter_expr, safe_interval=[0, 0], outfile=None):
    """Create a GTI list by using boolean operations on file data.

    Parameters
    ----------
    fname : str
        File name. The file must be in MaLTPyNT format.
    filter_expr : str
        A boolean condition on one or more of the arrays contained in the data.
        E.g. '(lc > 10) & (lc < 20)'

    Returns
    -------
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        The newly created GTIs

    Other parameters
    ----------------
    safe_interval : float or [float, float]
        A safe interval to exclude at both ends (if single float) or the start
        and the end (if pair of values) of GTIs.
    outfile : str
        The output file name. If None, use a default root + '_gti' combination
    """
    # Necessary as nc variables are sometimes defined as array
    from numpy import array

    if filter_expr is None:
        sys.exit('Please specify a filter expression')
    ftype, data = get_file_type(fname)

    instr = data['Instr']
    if ftype == 'lc' and instr == 'PCA':
        logging.warning('RXTE/PCA data; normalizing lc per no. PCUs')
        # If RXTE, plot per PCU count rate
        data['lc'] /= data['nPCUs']
    # Map all entries of data to local variables
    locals().update(data)

    good = eval(filter_expr)

    gtis = create_gti_from_condition(locals()['time'], good,
                                        safe_interval=safe_interval)

    if outfile is None:
        outfile = root(fname) + '_gti' + MP_FILE_EXTENSION
    save_data({'GTI': gtis}, outfile)

    return gtis


def apply_gti(fname, gti, outname=None):
    """Apply a GTI list to the data contained in a file.

    File MUST have a GTI extension already, and an extension called `time`.
    """
    ftype, data = get_file_type(fname)

    try:
        datagti = data['GTI']
        newgtis = cross_gtis([gti, datagti])
    except:  # pragma: no cover
        logging.warning('Data have no GTI extension')
        newgtis = gti

    data['GTI'] = newgtis
    good = create_gti_mask(data['time'], newgtis)
    data['time'] = data['time'][good]
    if ftype == 'lc':
        data['lc'] = data['lc'][good]
    elif ftype == 'events':
        data['PI'] = data['PI'][good]
        if data['Instr'] == 'PCA':  # pragma: no cover
            data['PCU'] = data['PCU'][good]

    if outname is None:
        outname = fname.replace(MP_FILE_EXTENSION, '') + \
            '_gtifilt' + MP_FILE_EXTENSION
    save_data(data, outname)

    return newgtis


if __name__ == '__main__':  # pragma: no cover
    import sys
    import subprocess as sp

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MPcreategti'] + args)
