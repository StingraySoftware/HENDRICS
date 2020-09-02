# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to calculate colors and hardness."""

import os
from astropy import log
from stingray.lightcurve import Lightcurve
import numpy as np
from .io import HEN_FILE_EXTENSION, load_lcurve, save_lcurve
from .base import hen_root
from .lcurve import main as henlcurve


def colors():
    pass


def main(args=None):
    """Main function called by the `HENcolors` command line script."""
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Calculate color light curves"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-e",
        "--energies",
        nargs=4,
        required=True,
        type=str,
        default=None,
        help="The energy boundaries in keV used to calculate "
        "the color. E.g. -e 2 3 4 6 means that the "
        "color will be calculated as 4.-6./2.-3. keV. "
        "If --use-pi is specified, these are interpreted "
        "as PI channels",
    )

    args = check_negative_numbers_in_args(args)
    _add_default_args(
        parser, ["bintime", "usepi", "output", "loglevel", "debug"]
    )
    args = parser.parse_args(args)
    files = args.files
    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    with log.log_to_file("HENcolors.log"):
        option = "--energy-interval"
        if args.use_pi:
            option = "--pi-interval"

        if args.outfile is not None and len(files) > 1:
            raise ValueError(
                "Specify --output only when processing " "a single file"
            )
        for f in files:
            henlcurve(
                [f]
                + [option]
                + args.energies[:2]
                + [
                    "-b",
                    str(args.bintime),
                    "-d",
                    ".",
                    "-o",
                    "lc0" + HEN_FILE_EXTENSION,
                ]
            )
            lc0 = load_lcurve("lc0" + HEN_FILE_EXTENSION)
            henlcurve(
                [f]
                + [option]
                + args.energies[2:]
                + [
                    "-b",
                    str(args.bintime),
                    "-d",
                    ".",
                    "-o",
                    "lc1" + HEN_FILE_EXTENSION,
                ]
            )
            lc1 = load_lcurve("lc1" + HEN_FILE_EXTENSION)

            time = lc0.time
            counts = lc1.countrate / lc0.countrate
            counts_err = np.sqrt(
                lc1.countrate_err ** 2 + lc0.countrate_err ** 2
            )
            scolor = Lightcurve(
                time=time,
                counts=counts,
                err=counts_err,
                input_counts=False,
                err_dist="gauss",
                gti=lc0.gti,
                dt=args.bintime,
            )
            del lc0
            del lc1
            os.unlink("lc0" + HEN_FILE_EXTENSION)
            os.unlink("lc1" + HEN_FILE_EXTENSION)

            if args.outfile is None:
                label = "_E_"
                if args.use_pi:
                    label = "_PI_"
                label += "{3}-{2}_over_{1}-{0}".format(*args.energies)
                args.outfile = hen_root(f) + label + HEN_FILE_EXTENSION
            scolor.e_intervals = np.asarray([float(k) for k in args.energies])
            scolor.use_pi = args.use_pi
            save_lcurve(scolor, args.outfile, lctype="Color")
