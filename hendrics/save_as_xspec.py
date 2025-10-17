# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to save data in a Xspec-readable format."""

import subprocess as sp

import numpy as np

from astropy import log

from .base import get_file_extension
from .io import get_file_type


def save_as_xspec(fname, direct_save=False, save_lags=True):
    """Save frequency spectra in a format readable to FTOOLS and Xspec.

    Parameters
    ----------
    fname : str
        Input HENDRICS frequency spectrum file name
    direct_save : bool
        If True, also call `flx2xsp` to produce the output .pha and .rsp files.
        If False (default), flx2xsp has to be called from the user

    Notes
    -----
    Uses method described by Ingram and Done in Appendix A of
    `this paper<https://arxiv.org/pdf/1108.0789>__`
    """
    ftype, contents = get_file_type(fname)

    outroot = fname.replace(get_file_extension(fname), "")
    outname = outroot + "_xsp.dat"
    outroot_lags = outroot + "_lags"
    outname_lags = outroot_lags + "_xsp.dat"

    if ftype.endswith("pds"):
        flo = contents.freq - contents.df / 2
        fhi = contents.freq + contents.df / 2
        power = contents.power.real * contents.df
        power_err = contents.power_err.real * contents.df
    else:
        raise ValueError("Data type not supported for Xspec")

    np.savetxt(outname, np.transpose([flo, fhi, power, power_err]))
    if direct_save:
        sp.check_call(f"flx2xsp {outname} {outroot}.pha {outroot}.rsp".split())

    if save_lags and ftype == "cpds":
        lags, lags_err = contents.time_lag()
        np.savetxt(
            outname_lags,
            np.transpose([flo, fhi, lags * contents.df, lags_err * contents.df]),
        )
        if direct_save:
            sp.check_call(f"flx2xsp {outname_lags} {outroot_lags}.pha {outroot_lags}.rsp".split())


def main(args=None):
    """Main function called by the `HEN2xspec` command line script."""
    import argparse

    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Save a frequency spectrum in a qdp file that can be "
        "read by flx2xsp and produce a XSpec-compatible spectrum"
        "file"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")

    parser.add_argument(
        "--flx2xsp",
        help="Also call flx2xsp at the end",
        default=False,
        action="store_true",
    )
    _add_default_args(parser, ["loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)
    files = args.files
    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    for f in files:
        save_as_xspec(f, direct_save=args.flx2xsp)
