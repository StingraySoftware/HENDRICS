from __future__ import division, print_function
import logging
import os
import copy
from .io import load_model, load_pds, save_model

import numpy as np
from stingray.modeling import fit_powerspectrum

def main_model(args=None):
    """Main function called by the `HENfspec` command line script."""
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
    parser.add_argument('--frequency-interval', type=float, nargs='+',
                        default=None,
                        help='Select frequency interval(s) to fit. Must be '
                             'an even number of frequencies in Hz, like '
                             '"--frequency-interval 0 2" or '
                             '"--frequency-interval 0 2 5 10", meaning that '
                             'the spectrum will be fitted between 0 and 2 Hz, '
                             'or using the intervals 0-2 Hz and 5-10 Hz.')
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

    freqs = args.frequency_interval
    if freqs is not None and len(freqs) % 2 != 0:
        raise ValueError("Invalid number of frequencies specified")

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENmodel.log', level=numeric_level,
                        filemode='w')

    model, kind, constraints = load_model(args.modelfile)
    if kind != 'Astropy':
        raise TypeError('At the moment, only Astropy models are accepted')

    for f in args.files:
        root = os.path.splitext(f)[0]
        spectrum = load_pds(f)

        if freqs is not None:
            good = np.zeros(len(spectrum.freq), dtype=bool)
            for f0, f1 in zip(freqs[::2], freqs[1::2]):
                local_good = (spectrum.freq >= f0)&(spectrum.freq < f1)
                good[local_good] = True

            spectrum_filt = copy.copy(spectrum)
            spectrum_filt.power = spectrum.power[good]
            spectrum_filt.freq = spectrum.freq[good]
            spectrum_filt.power_err = spectrum.power_err[good]

        priors = None
        max_post = False

        if constraints is not None and 'priors' in constraints:
            priors = constraints['priors']
            max_post = True

        parest, res = fit_powerspectrum(spectrum, model, model.parameters,
                                        max_post=max_post, priors=priors,
                                        fitmethod=args.fitmethod)

        save_model(res.model, root + '_bestfit.p')
