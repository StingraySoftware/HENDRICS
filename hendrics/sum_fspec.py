# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Function to sum frequency spectra."""

import copy
import numpy as np
from astropy import log
from .io import save_pds, get_file_type
from .io import HEN_FILE_EXTENSION
from .base import _assign_value_if_none


def sum_fspec(files, outname=None):
    """Take a bunch of (C)PDSs and sums them."""
    # Read first file
    ftype0, contents = get_file_type(files[0])
    pdstype = ftype0.replace("reb", "")

    freq0 = contents.freq
    pds0 = contents.power
    epds0 = contents.power_err
    nchunks0 = contents.m
    rebin0 = 1
    norm0 = contents.norm

    tot_pds = pds0 * nchunks0
    tot_epds = epds0 ** 2 * nchunks0
    tot_npds = nchunks0
    tot_contents = copy.copy(contents)
    outname = _assign_value_if_none(
        outname, "tot_" + ftype0 + HEN_FILE_EXTENSION
    )

    for f in files[1:]:
        ftype, contents = get_file_type(f)
        pdstype = ftype.replace("reb", "")

        freq = contents.freq
        pds = contents.power
        epds = contents.power_err
        nchunks = contents.m
        rebin = 1
        norm = contents.norm
        fftlen = contents.fftlen

        assert ftype == ftype0, "Files must all be of the same kind"
        assert np.all(
            rebin == rebin0
        ), "Files must be rebinned in the same way"
        np.testing.assert_array_almost_equal(
            freq,
            freq0,
            decimal=int(-np.log10(1 / fftlen) + 2),
            err_msg="Frequencies must coincide",
        )
        assert norm == norm0, "Files must have the same normalization"

        tot_pds += pds * nchunks
        tot_epds += epds ** 2 * nchunks ** 2
        tot_npds += nchunks

    tot_contents.power = tot_pds / tot_npds
    tot_contents.power_err = np.sqrt(tot_epds) / tot_npds
    tot_contents.m = tot_npds

    log.info("Saving %s to %s" % (pdstype, outname))
    save_pds(tot_contents, outname)

    return tot_contents


def main(args=None):
    """Main function called by the `HENsumfspec` command line script."""
    import argparse

    description = "Sum (C)PDSs contained in different files"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of light curve files", nargs="+")

    parser.add_argument(
        "-o",
        "--outname",
        type=str,
        default=None,
        help="Output file name for summed (C)PDS. Default:"
        + " tot_(c)pds"
        + HEN_FILE_EXTENSION,
    )

    args = parser.parse_args(args)

    sum_fspec(args.files, args.outname)
