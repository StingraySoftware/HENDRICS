# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Function to sum frequency spectra."""

from astropy import log
from .io import save_pds, get_file_type
from .io import HEN_FILE_EXTENSION
from .base import _assign_value_if_none
from .fspec import average_periodograms


def sum_fspec(files, outname=None):
    """Take a bunch of (C)PDSs and sums them."""
    # Read first file
    ftype0, contents = get_file_type(files[0])
    pdstype = ftype0.replace("reb", "")
    outname = _assign_value_if_none(outname, "tot_" + ftype0 + HEN_FILE_EXTENSION)

    def check_and_distribute_files(files):
        for i, f in enumerate(files):
            ftype, contents = get_file_type(f)
            if i == 0:
                contents0, ftype0 = contents, ftype
            else:
                assert ftype == ftype0, "Files must all be of the same kind"
            contents.fftlen = contents.segment_size
            yield contents

    tot_contents = average_periodograms(check_and_distribute_files(files))
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
