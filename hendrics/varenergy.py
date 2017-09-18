# -*- coding: utf-8 -*-
"""
@author: marta
"""
from __future__ import print_function, division
import logging
import numpy as np
from .base import hen_root
from .io import load_events
from .io import save_as_qdp
from stingray.varenergyspectrum import RmsEnergySpectrum
from stingray.varenergyspectrum import LagEnergySpectrum


def main(args=None):
    import argparse
    description = ('Calculates variability-energy spectra')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument('-f', "--freq-interval", nargs=2, type=float,
                        default= [0., 100], help="Frequence interval")
    parser.add_argument("--energy-values", nargs=4, type=str,
                        default="0.3 12 5 lin".split(" "),
                        help="Choose Emin, Emax, number of intervals,"
                        "interval spacing, lin or log")
    parser.add_argument("--segment-size", type=float,
                        default=None,
                        help="Length of the light curve intervals to be "
                             "averaged")
    parser.add_argument("-b", "--bin-time", type=float,
                        default=None,
                        help="Bin time for the light curve")
    parser.add_argument("--ref-band", nargs=2, type=float,
                        default=None, help="Reference band when relevant")
    parser.add_argument("--rms", default=False, action='store_true',
                        help="Calculate rms")
    parser.add_argument("--lag", default=False, action='store_true',
                        help="Calculate lag-energy")
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING', type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENvarenergy.log', level=numeric_level,
                        filemode='w')
    filelist=[]
    energy_spec = (float(args.energy_values[0]),
                   float(args.energy_values[1]),
                   int(args.energy_values[2]),
                   args.energy_values[3])

    for fname in args.files:
        events = load_events(fname)
        if args.rms:
            rms = RmsEnergySpectrum(events, args.freq_interval,
                                    energy_spec,
                                    segment_size=args.segment_size,
                                    bin_time=args.bin_time)
            out1 = hen_root(fname) + "_rms" + '.qdp'
            start_energy = np.asarray(rms.energy_intervals)[:,0]
            stop_energy = np.asarray(rms.energy_intervals)[:,1]
            save_as_qdp([start_energy, stop_energy, rms.spectrum],
                    [None, None, rms.spectrum_error], filename=out1)
            filelist.append(out1)

        if args.lag:
            lag = LagEnergySpectrum(events, args.freq_interval,
                                    energy_spec, args.ref_band,
                                    segment_size=args.segment_size,
                                    bin_time=args.bin_time)
            start_energy = np.asarray(lag.energy_intervals)[:,0]
            stop_energy = np.asarray(lag.energy_intervals)[:,1]
            out2 = hen_root(fname) + "_lag" + '.qdp'
            save_as_qdp([start_energy, stop_energy, lag.spectrum],
                    [None, None, lag.spectrum_error], filename=out2)
            filelist.append(out2)

    return filelist
