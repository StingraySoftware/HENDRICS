"""Functions to calculate power colors."""

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import warnings
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from astropy import log
from stingray import StingrayTimeseries, DynamicalPowerspectrum, DynamicalCrossspectrum
from stingray.power_colors import hue_from_power_color
from stingray.gti import cross_two_gtis

from .io import HEN_FILE_EXTENSION, load_events, save_timeseries
from .base import hen_root, interpret_bintime, common_name


def treat_power_colors(
    fname,
    frequency_edges=[1 / 256, 1 / 32, 0.25, 2, 16],
    segment_size=256,
    bintime=1 / 32,
    rebin=5,
    outfile=None,
    poisson_noise=None,
):
    if isinstance(fname, Iterable) and not isinstance(fname, str) and len(fname) == 2:
        events1 = load_events(fname[0])
        events2 = load_events(fname[1])
        dynps = DynamicalCrossspectrum(
            events1,
            events2,
            segment_size=segment_size,
            sample_time=bintime,
            norm="leahy",
        )

        gti = cross_two_gtis(events1.gti, events2.gti)
        local_poisson_noise = 0 if poisson_noise is None else poisson_noise
        base_name = hen_root(
            common_name(
                fname[0],
                fname[1],
                default=f"power_colors_{np.random.randint(0, 10000)}",
            )
        )
    else:
        events = load_events(fname)
        dynps = DynamicalPowerspectrum(
            events,
            segment_size=segment_size,
            sample_time=bintime,
            norm="leahy",
        )
        gti = events.gti

        local_poisson_noise = 2 if poisson_noise is None else poisson_noise
        base_name = hen_root(fname)

    dynps_reb = dynps.rebin_by_n_intervals(rebin, method="average")
    p1, p1e, p2, p2e = dynps_reb.power_colors(
        freq_edges=frequency_edges, poisson_power=local_poisson_noise
    )

    hues = hue_from_power_color(p1, p2)

    rms, rmse = dynps_reb.compute_rms(
        frequency_edges[0],
        frequency_edges[-1],
        poisson_noise_level=local_poisson_noise,
    )
    rms = np.asarray(rms).real
    rmse = np.abs(np.asarray(rmse))
    times = dynps_reb.time

    scolor = StingrayTimeseries(
        time=times,
        pc1=p1,
        pc1_err=np.abs(p1e),
        pc2=p2,
        pc2_err=np.abs(p2e),
        hue=hues,
        rms=rms,
        rms_err=rmse,
        input_counts=False,
        err_dist="gauss",
        gti=gti,
        dt=segment_size,
        skip_checks=True,
    )
    good = (scolor.pc1 > 0) & (scolor.pc2 > 0)
    if np.any(~good):
        warnings.warn(
            "Some (non-log) power colors are negative. Neglecting them", UserWarning
        )
        scolor = scolor.apply_mask(good)

    if outfile is None:
        label = "_edges_" + "_".join([f"{f:g}" for f in frequency_edges])
        outfile = base_name + label + "_pc" + HEN_FILE_EXTENSION

    scolor.f_intervals = np.asarray([float(k) for k in frequency_edges])
    scolor.__sr__class__type__ = "Powercolor"
    save_timeseries(scolor, outfile)
    return outfile


def main(args=None):
    """Main function called by the `HENcolors` command line script."""
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Calculate color light curves"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-f",
        "--frequency-edges",
        nargs=5,
        default=[1 / 256, 1 / 32, 0.25, 2, 16],
        type=float,
        help=(
            "Five frequency edges in Hz, delimiting four frequency ranges used to calculate "
            "the power colors"
        ),
    )

    parser.add_argument(
        "-r",
        "--rebin",
        type=int,
        default=5,
        help=(
            "Dynamical power spectrum rebinning (how many nearby segments to average"
            " before calculating the colors) to apply. Default: 5"
        ),
    )
    parser.add_argument(
        "-s",
        "--segment-size",
        type=float,
        default=512,
        help="Length of FFTs. Default: 512 s",
    )
    parser.add_argument(
        "--poisson-noise",
        type=float,
        default=None,
        help=(
            "Poisson noise level of the periodograms. Default: 2 for powerspectrum, 0 for "
            "crossspectrum"
        ),
    )
    parser.add_argument(
        "-b",
        "--bintime",
        type=float,
        default=1 / 64,
        help="Light curve bin time; if negative, interpreted"
        + " as negative power of 2."
        + " Default: 2^-10, or keep input lc bin time"
        + " (whatever is larger)",
    )
    parser.add_argument(
        "--cross",
        default=False,
        action="store_true",
        help="Use cross spectrum from pairs of files",
    )
    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["output", "loglevel", "debug"])
    args = parser.parse_args(args)
    files = args.files
    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    files = args.files
    if args.cross:
        files = [frange for frange in zip(files[::2], files[1::2])]

    outfiles = []
    with log.log_to_file("HENcolors.log"):
        if args.outfile is not None and len(files) > 1:
            raise ValueError("Specify --output only when processing " "a single file")
        bintime = np.longdouble(interpret_bintime(args.bintime))

        for f in files:
            outfile = treat_power_colors(
                f,
                args.frequency_edges,
                args.segment_size,
                bintime,
                args.rebin,
                args.outfile,
                args.poisson_noise,
            )
            outfiles.append(outfile)

    return outfiles
