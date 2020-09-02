# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to create and apply GTIs."""

import sys
import warnings
import numpy as np
from astropy import log
from astropy.logger import AstropyUserWarning
from stingray.gti import cross_gtis, create_gti_from_condition, create_gti_mask
from .io import HEN_FILE_EXTENSION, save_data, load_data, get_file_type
from .base import hen_root, _assign_value_if_none


def filter_gti_by_length(gti, minimum_length):
    """Filter a list of GTIs: keep those longer than `minimum_length`."""
    if minimum_length == 0 or minimum_length is None:
        return gti

    newgtis = []
    for g in gti:
        length = g[1] - g[0]
        if length >= minimum_length:
            newgtis.append(g)

    return np.array(newgtis)


def create_gti(
    fname, filter_expr, safe_interval=[0, 0], outfile=None, minimum_length=0
):
    """Create a GTI list by using boolean operations on file data.

    Parameters
    ----------
    fname : str
        File name. The file must be in HENDRICS format.
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
    from numpy import array  # NOQA

    ftype, data = get_file_type(fname, raw_data=True)

    instr = data["instr"]
    if ftype == "lc" and instr.lower() == "pca":
        warnings.warn(
            "RXTE/PCA data; normalizing lc per no. PCUs", AstropyUserWarning
        )
        # If RXTE, plot per PCU count rate
        data["counts"] /= data["nPCUs"]
    mjdref = data["mjdref"]
    # Map all entries of data to local variables
    locals().update(data)

    good = eval(filter_expr)

    gtis = create_gti_from_condition(
        locals()["time"], good, safe_interval=safe_interval
    )

    gtis = filter_gti_by_length(gtis, minimum_length)

    outfile = _assign_value_if_none(
        outfile, hen_root(fname) + "_gti" + HEN_FILE_EXTENSION
    )
    save_data(
        {"gti": gtis, "mjdref": mjdref, "__sr__class__type__": "gti"}, outfile
    )

    return gtis


def apply_gti(fname, gti, outname=None, minimum_length=0):
    """Apply a GTI list to the data contained in a file.

    File MUST have a GTI extension already, and an extension called `time`.
    """
    ftype, data = get_file_type(fname, raw_data=True)

    try:
        datagti = data["gti"]
        newgtis = cross_gtis([gti, datagti])
    except Exception:  # pragma: no cover
        warnings.warn("Data have no GTI extension", AstropyUserWarning)
        newgtis = gti

    newgtis = filter_gti_by_length(newgtis, minimum_length)

    data["gti"] = newgtis
    good = create_gti_mask(data["time"], newgtis)

    data["time"] = data["time"][good]
    if ftype == "lc":
        data["counts"] = data["counts"][good]
        data["counts_err"] = data["counts_err"][good]
    elif ftype == "events":
        for key in ["pi", "energy"]:
            if key in data:
                data[key] = data[key][good]

        if data["instr"] == "PCA":  # pragma: no cover
            data["PCU"] = data["PCU"][good]

    newext = "_gtifilt" + HEN_FILE_EXTENSION
    outname = _assign_value_if_none(
        outname, fname.replace(HEN_FILE_EXTENSION, "") + newext
    )
    save_data(data, outname)

    return newgtis


def main(args=None):
    """Main function called by the `HENcreategti` command line script."""
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Create GTI files from a filter expression, or applies "
        "previously created GTIs to a file"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs="+")

    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default=None,
        help="Filter expression, that has to be a valid "
        + "Python boolean operation on a data variable "
        + "contained in the files",
    )

    parser.add_argument(
        "-c",
        "--create-only",
        default=False,
        action="store_true",
        help="If specified, creates GTIs withouth applying"
        + "them to files (Default: False)",
    )

    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite original file (Default: False)",
    )

    parser.add_argument(
        "-a",
        "--apply-gti",
        type=str,
        default=None,
        help="Apply a GTI from this file to input files",
    )

    parser.add_argument(
        "-l",
        "--minimum-length",
        type=float,
        default=0,
        help=(
            "Minimum length of GTIs (below this length, they"
            " will be discarded)"
        ),
    )

    parser.add_argument(
        "--safe-interval",
        nargs=2,
        type=float,
        default=[0, 0],
        help="Interval at start and stop of GTIs used" + " for filtering",
    )

    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["loglevel", "debug"])

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    with log.log_to_file("HENcreategti.log"):
        filter_expr = args.filter
        if filter_expr is None and args.apply_gti is None:
            sys.exit(
                "Please specify filter expression (-f option) or input "
                "GTI file (-a option)"
            )

        for fname in files:
            if args.apply_gti is not None:
                data = load_data(args.apply_gti)
                gtis = data["gti"]
            else:
                gtis = create_gti(
                    fname,
                    filter_expr,
                    safe_interval=args.safe_interval,
                    minimum_length=args.minimum_length,
                )
            if args.create_only:
                continue
            if args.overwrite:
                outname = fname
            else:
                # Use default
                outname = None
            apply_gti(
                fname,
                gtis,
                outname=outname,
                minimum_length=args.minimum_length,
            )
