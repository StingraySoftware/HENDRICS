"""Functions to calculate power colors."""
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from astropy import log
from stingray import StingrayTimeseries, DynamicalPowerspectrum, DynamicalCrossspectrum
from stingray.fourier import hue_from_power_color

from .io import HEN_FILE_EXTENSION, load_events, save_timeseries
from .base import hen_root, interpret_bintime, common_name


def trace_hue_line(center, angle):
    plot_angle = (-angle + 3 * np.pi / 4) % (np.pi * 2)

    m = np.tan(plot_angle)
    if np.isinf(m):
        x = np.zeros_like(x) + center[0]
        y = np.linspace(-4, 4, 20)
    else:
        x = np.linspace(0, 4, 20) * np.sign(np.cos(plot_angle)) + center[0]
        y = center[1] + m * (x - center[0])
    return x, y


state_hue_limits = {
    "HSS": [300, 360],
    "LHS": [-20, 140],
    "HIMS": [140, 220],
    "SIMS": [220, 300],
}


def trace_state_areas(center, state, color, alpha=0.5):
    hue0, hue1 = state_hue_limits[state]
    x0, y0 = trace_hue_line(center, np.radians(hue0))

    next_angle = hue0 + 5.0
    x1, y1 = trace_hue_line(center, np.radians(hue0))
    previous_angle = hue0
    while next_angle <= hue1:
        x0, y0 = x1, y1
        x1, y1 = trace_hue_line(center, np.radians(next_angle))
        t1 = plt.Polygon(
            [[x0[0], y0[0]], [x0[-1], y0[-1]], [x1[-1], y1[-1]]],
            alpha=alpha,
            color=color,
            ls=None,
            lw=0,
        )
        plt.gca().add_patch(t1)
        #         previous_angle += 5.
        next_angle += 5.0


def trace_state_name(center, state):
    hue0, hue1 = state_hue_limits[state]
    x0, y0 = trace_hue_line(center, np.radians(hue0))
    hue_mean = (hue0 + hue1) / 2
    hue_angle = (-np.radians(hue_mean) + 3 * np.pi / 4) % (np.pi * 2)

    radius = 1.4
    txt_x = radius * np.cos(hue_angle) + center[0]
    txt_y = radius * np.sin(hue_angle) + center[1]
    plt.text(txt_x, txt_y, state, color="k", ha="center", va="center")


def create_pc_plot(center, xrange=[-2, 2], yrange=[-2, 2]):
    """Creates an empty power color plot with labels in the right place."""
    fig = plt.figure()

    plt.gca().set_aspect("equal")

    plt.xlabel(r"log$_{10}$PC1")
    plt.ylabel(r"log$_{10}$PC2")
    plt.scatter(*center, marker="+", color="k")
    for angle in range(0, 360, 20):
        color = "k"
        x, y = trace_hue_line(center, np.radians(angle))
        alpha = 0.3
        lw = 0.2
        if angle in [0, 140, 220, 300, 340]:
            color = "k"
            alpha = 1
            lw = 1
        plt.plot(x, y, lw=lw, ls=":", color=color, alpha=alpha, zorder=10)

    trace_state_areas(center, "LHS", "blue", 0.1)
    trace_state_areas(center, "HIMS", "green", 0.1)
    trace_state_areas(center, "SIMS", "yellow", 0.1)
    trace_state_areas(center, "HSS", "red", 0.1)
    for state in state_hue_limits.keys():
        trace_state_name(center, state)

    plt.xlim(center[0] + np.asarray(xrange))
    plt.ylim(center[1] + np.asarray(yrange))
    plt.grid(False)
    return fig


def plot_power_colors(p1, p1e, p2, p2e, center=(4.51920, 0.453724)):
    hues = hue_from_power_color(p1, p2)

    p1e = np.abs(1 / p1 * p1e)
    p2e = np.abs(1 / p2 * p2e)
    p1 = np.log10(p1)
    p2 = np.log10(p2)
    center = np.log10(np.asarray(center))
    # Create empty power color plot
    fig = create_pc_plot(center)
    ax = fig.gca()
    ax.errorbar(p1, p2, xerr=p1e, yerr=p2e, alpha=0.4, color="k")
    ax.scatter(p1, p2, zorder=10, color="k")


heil_et_al_rms_span = {
    -20: [0.3, 0.7],
    0: [0.3, 0.7],
    10: [0.3, 0.6],
    40: [0.25, 0.4],
    100: [0.25, 0.35],
    150: [0.2, 0.3],
    170: [0.0, 0.3],
    200: [0, 0.15],
    220: [0, 0.15],
    250: [0, 0.15],
    300: [0, 0.15],
    360: [0, 0.15],
}
heil_et_al_x = list(heil_et_al_rms_span.keys())
heil_et_al_ymin = list([v[0] for v in heil_et_al_rms_span.values()])
heil_et_al_ymax = list([v[1] for v in heil_et_al_rms_span.values()])

ymin_func = interp1d(heil_et_al_x, heil_et_al_ymin, kind="linear")
ymax_func = interp1d(heil_et_al_x, heil_et_al_ymax, kind="linear")


def create_rms_hue_plot():
    plt.figure()
    plt.xlim(0, 360)
    plt.ylim(0, 0.7)
    plt.ylabel("Fractional rms")
    plt.xlabel("Hue")
    # state_hue_limits = {"HSS": [300, 360], "LHS": [-20, 140], "HIMS": [140, 220], "SIMS":[220, 300]}

    plt.fill_between(
        np.linspace(300, 360, 20),
        ymin_func(np.linspace(300, 360, 20)),
        ymax_func(np.linspace(300, 360, 20)),
        color="red",
        alpha=0.1,
    )
    plt.fill_between(
        np.linspace(340, 360, 20),
        ymin_func(np.linspace(-20, 0, 20)),
        ymax_func(np.linspace(-20, 0, 20)),
        color="b",
        alpha=0.1,
    )
    plt.fill_between(
        np.linspace(0, 140, 20),
        ymin_func(np.linspace(0, 140, 20)),
        ymax_func(np.linspace(0, 140, 20)),
        color="b",
        alpha=0.1,
    )
    plt.fill_between(
        np.linspace(140, 220, 20),
        ymin_func(np.linspace(140, 220, 20)),
        ymax_func(np.linspace(140, 220, 20)),
        color="g",
        alpha=0.1,
    )
    plt.fill_between(
        np.linspace(220, 300, 20),
        ymin_func(np.linspace(220, 300, 20)),
        ymax_func(np.linspace(220, 300, 20)),
        color="yellow",
        alpha=0.1,
    )
    return plt.gca()


def create_rms_hue_polar_plot():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_rmax(0.75)
    ax.set_rticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True)
    ax.set_rlim([0, 0.75])
    plt.fill_between(
        np.radians(np.linspace(300, 360, 20)),
        ymin_func(np.linspace(300, 360, 20)),
        ymax_func(np.linspace(300, 360, 20)),
        color="red",
        alpha=0.1,
    )
    ax.fill_between(
        np.radians(np.linspace(340, 360, 20)),
        ymin_func(np.linspace(-20, 0, 20)),
        ymax_func(np.linspace(-20, 0, 20)),
        color="b",
        alpha=0.1,
    )
    ax.fill_between(
        np.radians(np.linspace(0, 140, 20)),
        ymin_func(np.linspace(0, 140, 20)),
        ymax_func(np.linspace(0, 140, 20)),
        color="b",
        alpha=0.1,
    )
    ax.fill_between(
        np.radians(np.linspace(140, 220, 20)),
        ymin_func(np.linspace(140, 220, 20)),
        ymax_func(np.linspace(140, 220, 20)),
        color="g",
        alpha=0.1,
    )
    ax.fill_between(
        np.radians(np.linspace(220, 300, 20)),
        ymin_func(np.linspace(220, 300, 20)),
        ymax_func(np.linspace(220, 300, 20)),
        color="yellow",
        alpha=0.1,
    )
    return ax


def plot_hues_rms(hues, rms, rmse):
    ax = create_rms_hue_plot()
    hues = hues % (np.pi * 2)
    ax.errorbar(np.degrees(hues), rms, yerr=rmse, fmt="o", alpha=0.5)


def plot_hues_rms_polar(hues, rms, rmse):
    ax = create_rms_hue_polar_plot()
    hues = hues % (np.pi * 2)
    ax.errorbar(hues, rms, yerr=rmse, fmt="o", alpha=0.5)


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
        gti = events1.gti
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
