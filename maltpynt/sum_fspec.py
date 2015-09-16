# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Function to sum frequency spectra."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import save_data, get_file_type
from .io import MP_FILE_EXTENSION
from .base import _assign_value_if_none
import numpy as np
import logging


def sum_fspec(files, outname=None):
    """Take a bunch of (C)PDSs and sums them."""
    # Read first file
    ftype0, contents = get_file_type(files[0])
    pdstype = ftype0.replace('reb', '')
    freq0 = contents['freq']
    pds0 = contents[pdstype]
    epds0 = contents['e' + pdstype]
    nchunks0 = contents['n' + pdstype]
    rebin0 = contents['rebin']
    norm0 = contents['norm']

    tot_pds = pds0 * nchunks0
    tot_epds = epds0 ** 2 * nchunks0
    tot_npds = nchunks0
    tot_contents = contents.copy()
    outname = _assign_value_if_none(outname,
                                    'tot_' + ftype0 + MP_FILE_EXTENSION)

    for f in files[1:]:
        ftype, contents = get_file_type(f)
        pdstype = ftype.replace('reb', '')
        freq = contents['freq']
        pds = contents[pdstype]
        epds = contents['e' + pdstype]
        nchunks = contents['n' + pdstype]
        rebin = contents['rebin']
        norm = contents['norm']

        assert ftype == ftype0, 'Files must all be of the same kind'
        assert np.all(rebin == rebin0), \
            'Files must be rebinned in the same way'
        assert (np.all(freq == freq0)), 'Frequencies must coincide'
        assert norm == norm0, 'Files must have the same normalization'

        tot_pds += pds * nchunks
        tot_epds += epds ** 2 * nchunks ** 2
        tot_npds += nchunks

    tot_contents[pdstype] = tot_pds / tot_npds
    tot_contents['e' + pdstype] = np.sqrt(tot_epds) / tot_npds
    tot_contents['n' + pdstype] = tot_npds

    logging.info('Saving %s to %s' % (pdstype, outname))
    save_data(tot_contents, outname)

    return tot_contents


def main(args=None):
    import argparse

    description = 'Sum (C)PDSs contained in different files'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of light curve files", nargs='+')

    parser.add_argument("-o", "--outname", type=str, default=None,
                        help='Output file name for summed (C)PDS. Default:' +
                        ' tot_(c)pds' + MP_FILE_EXTENSION)

    args = parser.parse_args(args)

    sum_fspec(args.files, args.outname)
