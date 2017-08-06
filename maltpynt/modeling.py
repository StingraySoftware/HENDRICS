from __future__ import division, print_function
import logging
import os
from .io import load_model, load_pds, save_model

import numpy as np
from stingray.modeling import fit_powerspectrum

def main(args=None):
    """Main function called by the `MPfspec` command line script."""
    import argparse
    description = ('Fit frequency spectra (PDS, CPDS, cospectrum) '
                   'with user-defined models')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of light curve files", nargs='+')
    parser.add_argument("-m", "--modelfile", type=str,
                        help="File containing an Astropy model with or without"
                             " constraints")
    parser.add_argument('--fitmethod', type=str, default="L-BFGS-B",
                        help='Any scipy-compatible fit method')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPmodel.log', level=numeric_level,
                        filemode='w')

    model, kind, constraints = load_model(args.modelfile)
    if kind != 'Astropy':
        raise TypeError('At the moment, only Astropy models are accepted')

    for f in args.files:
        root = os.path.splitext(f)[0]
        spectrum = load_pds(f)

        priors = None
        max_post = False

        if constraints is not None and 'priors' in constraints:
            priors = constraints['priors']
            max_post = True

        parest, res = fit_powerspectrum(spectrum, model, model.parameters,
                                        max_post=max_post, priors=priors,
                                        fitmethod=args.fitmethod)

        save_model(res.model, root + '_bestfit.p')
