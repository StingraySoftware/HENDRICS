# -*- coding: utf-8 -*-
"""
@author: marta
"""

import warnings
from astropy import log
from astropy.logger import AstropyUserWarning
import numpy as np
from stingray.varenergyspectrum import RmsEnergySpectrum
from stingray.varenergyspectrum import LagEnergySpectrum

# from stingray.covariancespectrum import AveragedCovariancespectrum

from .base import hen_root, interpret_bintime
from .io import load_events
from .io import save_as_qdp


def main(args=None):
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Calculates variability-energy spectra"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-f",
        "--freq-interval",
        nargs=2,
        type=float,
        default=[0.0, 100],
        help="Frequence interval",
    )
    parser.add_argument(
        "--energy-values",
        nargs=4,
        type=str,
        default="0.3 12 5 lin".split(" "),
        help="Choose Emin, Emax, number of intervals,"
        "interval spacing, lin or log",
    )
    parser.add_argument(
        "--segment-size",
        type=float,
        default=512,
        help="Length of the light curve intervals to be " "averaged",
    )
    parser.add_argument(
        "--ref-band",
        nargs=2,
        type=float,
        default=None,
        help="Reference band when relevant",
    )
    parser.add_argument(
        "--rms", default=False, action="store_true", help="Calculate rms"
    )
    parser.add_argument(
        "--covariance",
        default=False,
        action="store_true",
        help="Calculate covariance spectrum",
    )
    parser.add_argument(
        "--use-pi",
        default=False,
        action="store_true",
        help="Energy intervals are specified as PI channels",
    )
    parser.add_argument(
        "--cross-instr",
        default=False,
        action="store_true",
        help="Use data files in pairs, for example with the"
        "reference band from one and the subbands from "
        "the  other (useful in NuSTAR and "
        "multiple-detector missions)",
    )
    parser.add_argument(
        "--lag",
        default=False,
        action="store_true",
        help="Calculate lag-energy",
    )

    _add_default_args(parser, ["bintime", "loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)
    args.bintime = np.longdouble(interpret_bintime(args.bintime))

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    with log.log_to_file("HENvarenergy.log"):
        filelist = []
        energy_spec = (
            float(args.energy_values[0]),
            float(args.energy_values[1]),
            int(args.energy_values[2]),
            args.energy_values[3],
        )

        from .io import sort_files

        if args.cross_instr:
            log.info("Sorting file list")
            sorted_files = sort_files(args.files)

            warnings.warn(
                "Beware! For cpds and derivatives, I assume that the "
                "files are from only two instruments and in pairs "
                "(even in random order)"
            )

            instrs = list(sorted_files.keys())

            files1 = sorted_files[instrs[0]]
            files2 = sorted_files[instrs[1]]
        else:
            files1 = args.files
            files2 = args.files

        for fnames in zip(files1, files2):
            fname = fnames[0]
            fname2 = fnames[1]

            events = load_events(fname)
            events2 = load_events(fname2)
            if not args.use_pi and (
                events.energy is None or events2.energy is None
            ):
                raise ValueError(
                    "If --use-pi is not specified, event lists must "
                    "be calibrated! Please use HENcalibrate."
                )

            if args.rms:
                rms = RmsEnergySpectrum(
                    events,
                    args.freq_interval,
                    energy_spec,
                    segment_size=args.segment_size,
                    bin_time=args.bintime,
                    events2=events2,
                    use_pi=args.use_pi,
                )
                out1 = hen_root(fname) + "_rms" + ".qdp"
                start_energy = np.asarray(rms.energy_intervals)[:, 0]
                stop_energy = np.asarray(rms.energy_intervals)[:, 1]
                save_as_qdp(
                    [start_energy, stop_energy, rms.spectrum],
                    [None, None, rms.spectrum_error],
                    filename=out1,
                )
                filelist.append(out1)

            if args.lag:
                lag = LagEnergySpectrum(
                    events,
                    args.freq_interval,
                    energy_spec,
                    args.ref_band,
                    segment_size=args.segment_size,
                    bin_time=args.bintime,
                    events2=events2,
                    use_pi=args.use_pi,
                )
                start_energy = np.asarray(lag.energy_intervals)[:, 0]
                stop_energy = np.asarray(lag.energy_intervals)[:, 1]
                out2 = hen_root(fname) + "_lag" + ".qdp"
                save_as_qdp(
                    [start_energy, stop_energy, lag.spectrum],
                    [None, None, lag.spectrum_error],
                    filename=out2,
                )
                filelist.append(out2)

            if args.covariance:
                try:
                    from stingray.varenergyspectrum import (
                        CovarianceEnergySpectrum,
                    )
                except Exception:
                    warnings.warn(
                        "This version of Stingray does not implement "
                        "the correct version of Covariance Spectrum.",
                        AstropyUserWarning,
                    )
                    continue
                cov = CovarianceEnergySpectrum(
                    events,
                    args.freq_interval,
                    energy_spec,
                    args.ref_band,
                    segment_size=args.segment_size,
                    bin_time=args.bintime,
                    events2=events2,
                    use_pi=args.use_pi,
                )
                start_energy = np.asarray(cov.energy_intervals)[:, 0]
                stop_energy = np.asarray(cov.energy_intervals)[:, 1]
                out2 = hen_root(fname) + "_cov" + ".qdp"
                save_as_qdp(
                    [start_energy, stop_energy, cov.spectrum],
                    [None, None, cov.spectrum_error],
                    filename=out2,
                )
                filelist.append(out2)

    return filelist
