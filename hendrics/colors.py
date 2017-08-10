# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to calculate colors and hardness."""

from __future__ import (absolute_import, division,
                        print_function)

from .io import MP_FILE_EXTENSION, load_lcurve, save_lcurve
from .base import mp_root
import logging
from .lcurve import main as henlcurve
from stingray import Lightcurve
import numpy as np
import os


def colors():
    pass


def main(args):
    """Main function called by the `HENcolors` command line script."""
    import argparse
    description = \
        'Calculate color light curves'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-e", "--energies", nargs=4, required=True,
                        type=str, default=None,
                        help="The energy boundaries in keV used to calculate "
                             "the color. E.g. -e 2 3 4 6 means that the "
                             "color will be calculated as 4.-6./2.-3. keV. "
                             "If --use-pi is specified, these are interpreted "
                             "as PI channels")
    parser.add_argument("-b", "--bintime", type=str, default='100',
                        help="Bin time; if negative, negative power of 2")
    parser.add_argument("-o", "--out", type=str,
                        default=None,
                        help='Output file')
    parser.add_argument('--use-pi', type=bool, default=False,
                        help="Use the PI channel instead of energies")
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPscrunchlc.log', level=numeric_level,
                        filemode='w')

    option = '--e-interval'
    if args.use_pi:
        option = '--pi-interval'

    for f in files:
        henlcurve([f] + [option] + args.energies[:2] +
                  ['-b', args.bintime, '-d', '.', '-o',
                   'lc0' + MP_FILE_EXTENSION])
        lc0 = load_lcurve('lc0' + MP_FILE_EXTENSION)
        henlcurve([f] + [option] + args.energies[2:] +
                  ['-b', args.bintime, '-d', '.', '-o',
                   'lc1' + MP_FILE_EXTENSION])
        lc1 = load_lcurve('lc1' + MP_FILE_EXTENSION)

        time = lc0.time
        counts = lc1.counts / lc0.counts
        counts_err = np.sqrt(lc1.counts_err ** 2 + lc0.counts_err ** 2)
        scolor = Lightcurve(time=time, counts=counts, err=counts_err,
                            input_counts=False, err_dist='gauss',
                            gti=lc0.gti)
        del lc0
        del lc1
        os.unlink('lc0' + MP_FILE_EXTENSION)
        os.unlink('lc1' + MP_FILE_EXTENSION)

        if args.out is None:
            label = '_E_'
            if args.use_pi:
                label = '_PI_'
            label += '{3}-{2}_over_{1}-{0}'.format(*args.energies)
            args.out = mp_root(f) + label + MP_FILE_EXTENSION
        scolor.e_intervals = np.asarray([float(k) for k in args.energies])
        scolor.use_pi = args.use_pi
        save_lcurve(scolor, args.out, lctype='Color')