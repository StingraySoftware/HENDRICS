# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to calculate colors and hardness."""

import os
from astropy import log
from stingray.lightcurve import Lightcurve
import numpy as np
from .io import HEN_FILE_EXTENSION, load_events, save_lcurve
from .base import hen_root


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
        type=float,
        help="The energy boundaries in keV used to calculate "
        "the color. E.g. -e 2 3 4 6 means that the "
        "color will be calculated as 4.-6./2.-3. keV. "
        "If --use-pi is specified, these are interpreted "
        "as PI channels",
    )

    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["bintime", "usepi", "output", "loglevel", "debug"])
    args = parser.parse_args(args)
    files = args.files
    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    with log.log_to_file("HENcolors.log"):
        energies = [
            [args.energies[0], args.energies[1]],
            [args.energies[2], args.energies[3]],
        ]
        if args.outfile is not None and len(files) > 1:
            raise ValueError("Specify --output only when processing " "a single file")
        for f in files:
            events = load_events(f)
            if not args.use_pi and events.energy is None:
                raise ValueError(
                    "Energy information not found in file {0}. "
                    "Use --use-pi if you want to use PI channels "
                    "instead.".format(f)
                )
            h_starts, h_stops, colors, color_errs = events.get_color_evolution(
                energy_ranges=energies, segment_size=args.bintime, use_pi=args.use_pi
            )

            time = (h_starts + h_stops) / 2

            scolor = Lightcurve(
                time=time,
                counts=colors,
                err=color_errs,
                input_counts=False,
                err_dist="gauss",
                gti=events.gti,
                dt=args.bintime,
                skip_checks=True,
            )

            if args.outfile is None:
                label = "_E_"
                if args.use_pi:
                    label = "_PI_"
                label += "{3:g}-{2:g}_over_{1:g}-{0:g}".format(*args.energies)
                args.outfile = hen_root(f) + label + HEN_FILE_EXTENSION
            scolor.e_intervals = np.asarray([float(k) for k in args.energies])
            scolor.use_pi = args.use_pi
            save_lcurve(scolor, args.outfile, lctype="Color")
            print(args.outfile)
