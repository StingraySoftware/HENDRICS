# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to save data in a Xspec-readable format."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import get_file_type
import numpy as np
from .io import get_file_extension
import subprocess as sp


def save_as_xspec(fname, direct_save=False):
    """Save frequency spectra in a format readable to FTOOLS and Xspec.

    Parameters
    ----------
    fname : str
        Input MaLTPyNT frequency spectrum file name
    direct_save : bool
        If True, also call `flx2xsp` to produce the output .pha and .rsp files.
        If False (default), flx2xsp has to be called from the user

    Notes
    -----
    Uses method described here:
    https://asd.gsfc.nasa.gov/XSPECwiki/fitting_timing_power_spectra_in_XSPEC
    """
    ftype, contents = get_file_type(fname)

    outroot = fname.replace(get_file_extension(fname), '')
    outname = outroot + '_xsp.dat'

    if 'freq' in list(contents.keys()):
        freq = contents['freq']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = freq[1] - freq[0]

        np.savetxt(outname, np.transpose([freq - df / 2,
                                          freq + df / 2,
                                          pds.real * df,
                                          epds * df]))
    elif 'flo' in list(contents.keys()):
        ftype = ftype.replace('reb', '')
        flo = contents['flo']
        fhi = contents['fhi']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = fhi - flo
        np.savetxt(outname, np.transpose([flo, fhi,
                                          pds.real * df,
                                          epds * df]))
    else:
        raise Exception('File type not recognized')

    if direct_save:
        sp.check_call('flx2xsp {} {}.pha {}.rsp'.format(
            outname, outroot, outroot).split())


if __name__ == '__main__':  # pragma: no cover
    import sys

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MP2xspec'] + args)
