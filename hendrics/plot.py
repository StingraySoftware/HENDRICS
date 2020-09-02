# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Quicklook plots."""

import warnings
import os
import copy
from collections.abc import Iterable
import numpy as np
from stingray.gti import create_gti_mask
from astropy.modeling.models import Const1D
from astropy.modeling import Model
from astropy.stats import poisson_conf_interval
from astropy import log
from astropy.table import Table

from .fold import fold_events, filter_energy
from .base import z2_n_detection_level
from .base import fold_detection_level
from .base import deorbit_events
from .io import load_events
from .io import load_data, get_file_type, load_pds
from .io import is_string, save_as_qdp, load_folding
from .io import HEN_FILE_EXTENSION
from .io import find_file_in_allowed_paths
from .base import _assign_value_if_none, find_peaks_in_image
from .base import pds_detection_level as detection_level


def _next_color(ax):
    try:
        return next(ax._get_lines.color_cycle)
    except Exception:
        return next(ax._get_lines.prop_cycler)["color"]


def _baseline_fun(x, a):
    """A constant function."""
    return a


def _value_or_none(dict_like, key):
    try:
        return dict_like[key]
    except KeyError:
        return None


def plot_generic(
    fnames,
    vars,
    errs=None,
    figname=None,
    xlog=None,
    ylog=None,
    output_data_file=None,
):
    """Generic plotting function."""
    import matplotlib.pyplot as plt

    if is_string(fnames):
        fnames = [fnames]
    figname = _assign_value_if_none(
        figname, "{0} vs {1}".format(vars[1], vars[0])
    )
    plt.figure(figname)
    ax = plt.gca()
    if xlog:
        ax.set_xscale("log", nonposx="clip")
    if ylog:
        ax.set_yscale("log", nonposy="clip")

    xlabel, ylabel = vars
    xlabel_err, ylabel_err = None, None
    if errs is not None:
        xlabel_err, ylabel_err = errs

    for i, fname in enumerate(fnames):
        data = load_data(fname)
        color = _next_color(ax)
        xdata = data[xlabel]
        ydata = data[ylabel]
        xdata_err = _value_or_none(data, xlabel_err)
        ydata_err = _value_or_none(data, ylabel_err)
        plt.errorbar(
            xdata,
            ydata,
            yerr=ydata_err,
            xerr=xdata_err,
            fmt="-",
            drawstyle="steps-mid",
            color=color,
            label=fname,
        )

        if output_data_file is not None:
            save_as_qdp(
                [xdata, ydata],
                errors=[xdata_err, ydata_err],
                filename=output_data_file,
                mode="a",
            )

    plt.xlabel(vars[0])
    plt.ylabel(vars[1])
    plt.legend()


def _get_const(models):
    """Get constant from Astropy model, list of models or compound model.

    Return None if no Const1D objects are in ``models``.
    Return the value of the first Const1D object found.

    Examples
    --------
    >>> from astropy.modeling.models import Const1D, Gaussian1D
    >>> model = Const1D(2) + Gaussian1D(1, 4, 5)
    >>> _get_const(model)
    2.0
    >>> _get_const(model[0])
    2.0
    >>> _get_const([model[0]])
    2.0
    >>> _get_const([[model]])
    2.0
    >>> _get_const(model[1])

    >>> _get_const(None)

    >>> _get_const(1)

    >>> _get_const('avdsfa')

    """

    if isinstance(models, Const1D):
        return models.amplitude.value

    if hasattr(models, "submodel_names"):
        for subm in models:
            if isinstance(subm, Const1D):
                return subm.amplitude.value

    if models is None:
        return None

    if (
        isinstance(models, Iterable)
        and not is_string(models)
        and len(models) != 0
    ):
        for model in models:
            return _get_const(model)

    return None


def plot_pds(
    fnames,
    figname=None,
    xlog=None,
    ylog=None,
    output_data_file=None,
    white_sub=False,
):
    """Plot a list of PDSs, or a single one."""

    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    if is_string(fnames):
        fnames = [fnames]

    figlabel = fnames[0]

    for i, fname in enumerate(fnames):
        pds_obj = load_pds(fname, nosub=True)
        if np.allclose(np.diff(pds_obj.freq), pds_obj.df):
            freq = pds_obj.freq
            xlog = _assign_value_if_none(xlog, False)
            ylog = _assign_value_if_none(ylog, False)
        else:
            flo = pds_obj.freq - pds_obj.df / 2
            fhi = pds_obj.freq + pds_obj.df / 2
            freq = (fhi + flo) / 2
            xlog = _assign_value_if_none(xlog, True)
            ylog = _assign_value_if_none(ylog, True)

        models = []
        if hasattr(pds_obj, "best_fits") and pds_obj.best_fits is not None:
            models = pds_obj.best_fits
        if isinstance(models, Model):
            models = [models]

        pds = pds_obj.power
        epds = pds_obj.power_err
        npds = pds_obj.m
        norm = pds_obj.norm

        lev = detection_level(
            epsilon=0.015, n_summed_spectra=npds, ntrial=pds.size
        )

        if norm == "rms":
            # we need the unnormalized power
            lev = lev / 2 * pds_obj.nphots
            lev, _ = pds_obj._normalize_crossspectrum(lev, pds_obj.fftlen)

        if xlog and ylog:
            plt.figure("PDS - Loglog " + figlabel)
        else:
            plt.figure("PDS " + figlabel)
        ax = plt.gca()
        color = _next_color(ax)

        if xlog:
            ax.set_xscale("log", nonposx="clip")
        if ylog:
            ax.set_yscale("log", nonposy="clip")

        level = lev  # Can be modified below
        y = pds[1:]
        yerr = epds[1:]

        if norm.lower() == "leahy" or (
            norm.lower() in ["rms", "frac"] and (not xlog or not ylog)
        ):
            plt.plot(
                freq[1:], y, drawstyle="steps-mid", color=color, label=fname
            )
            for i, func in enumerate(models):
                plt.plot(
                    freq,
                    func(freq),
                    label="Model {}".format(i + 1),
                    zorder=20,
                    color="k",
                )

        elif norm.lower() in ["rms", "frac"] and xlog and ylog:
            # TODO: Very rough! Use new machinery
            const = _get_const(models)
            if const is None:
                p, pcov = curve_fit(
                    _baseline_fun, freq, pds, p0=[2], sigma=epds
                )
                log.info("White noise level is {0}".format(p[0]))
                const = p[0]

            pds -= const
            level = lev - const

            y = pds[1:] * freq[1:]
            yerr = epds[1:] * freq[1:]
            plt.plot(
                freq[1:], y, drawstyle="steps-mid", color=color, label=fname
            )
            level *= freq
            for i, func in enumerate(models):
                const = _get_const(func)
                plt.plot(
                    freq,
                    freq * (func(freq) - const),
                    label="Model {}".format(i + 1),
                    zorder=20,
                    color="k",
                )

        if np.any(level < 0):
            continue
        if isinstance(level, Iterable):
            plt.plot(freq, level, ls="--", color=color)
        else:
            plt.axhline(level, ls="--", color=color)
        if output_data_file is not None:
            save_as_qdp(
                [freq[1:], y],
                errors=[None, yerr],
                filename=output_data_file,
                mode="a",
            )

    plt.xlabel("Frequency")
    if norm.lower() == "rms":
        plt.ylabel("(rms/mean)^2")
    elif norm.lower() == "leahy":
        plt.ylabel("Leahy power")

    plt.legend()

    if figname is not None:
        plt.savefig(figname)


def plot_cospectrum(
    fnames, figname=None, xlog=None, ylog=None, output_data_file=None
):
    """Plot the cospectra from a list of CPDSs, or a single one."""
    import matplotlib.pyplot as plt

    if is_string(fnames):
        fnames = [fnames]

    figlabel = fnames[0]

    for i, fname in enumerate(fnames):
        pds_obj = load_pds(fname, nosub=True)
        models = []
        if hasattr(pds_obj, "best_fits") and pds_obj.best_fits is not None:
            models = pds_obj.best_fits

        if np.allclose(np.diff(pds_obj.freq), pds_obj.df):
            freq = pds_obj.freq
            xlog = _assign_value_if_none(xlog, False)
            ylog = _assign_value_if_none(ylog, False)
        else:
            flo = pds_obj.freq - pds_obj.df / 2
            fhi = pds_obj.freq + pds_obj.df / 2
            freq = (fhi + flo) / 2
            xlog = _assign_value_if_none(xlog, True)
            ylog = _assign_value_if_none(ylog, True)

        cpds = pds_obj.power

        cospectrum = cpds.real
        if xlog and ylog:
            plt.figure("Cospectrum - Loglog " + figlabel)
        else:
            plt.figure("Cospectrum " + figlabel)
        ax = plt.gca()
        if xlog:
            ax.set_xscale("log", nonposx="clip")
        if ylog:
            ax.set_yscale("log", nonposy="clip")

        plt.xlabel("Frequency")
        if xlog and ylog:
            y = freq[1:] * cospectrum[1:]
            plt.plot(freq[1:], y, drawstyle="steps-mid", label=fname)
            for i, func in enumerate(models):
                plt.plot(
                    freq, freq * func(freq), label="Model {}".format(i + 1)
                )

            plt.ylabel("Cospectrum * Frequency")
        else:
            y = cospectrum[1:]
            plt.plot(
                freq[1:], cospectrum[1:], drawstyle="steps-mid", label=fname
            )

            plt.ylabel("Cospectrum")
            for i, func in enumerate(models):
                plt.plot(freq, func(freq), label="Model {}".format(i + 1))
        if output_data_file is not None:
            save_as_qdp([freq[1:], y], filename=output_data_file, mode="a")

    plt.legend()

    if figname is not None:
        plt.savefig(figname)


def plot_folding(
    fnames, figname=None, xlog=None, ylog=None, output_data_file=None
):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt

    if is_string(fnames):
        fnames = [fnames]

    for fname in fnames:
        ef = load_folding(fname)

        if not hasattr(ef, "M") or ef.M is None:
            ef.M = 1
        label = "Stat"
        ntrial = ef.stat.size
        if hasattr(ef, "oversample") and ef.oversample is not None:
            ntrial /= ef.oversample
            ntrial = np.int(ntrial)
        if ef.kind == "Z2n":
            vmin = ef.N - 1
            vmax = z2_n_detection_level(
                epsilon=0.001,
                n=int(ef.N),
                ntrial=ntrial,
                n_summed_spectra=int(ef.M),
            )
            nbin = ef.N * 8
            label = "$" + f"Z^2_{ef.N}" + "$"
        else:
            vmin = ef.nbin
            vmax = fold_detection_level(
                nbin=int(ef.nbin), epsilon=0.001, ntrial=ntrial
            )
            nbin = ef.nbin

        best_cands = find_peaks_in_image(ef.stat, n=5)

        fddot = 0
        if hasattr(ef, "fddots") and ef.fddots is not None:
            fddot = ef.fddots

        print("Best candidates:")
        best_cand_table = Table(names=["mjd", "power", "f", "fdot", "fddot"])
        for idx in best_cands[::-1]:
            if len(ef.stat.shape) > 1 and ef.stat.shape[0] > 1:
                f, fdot = ef.freq[idx[0], idx[1]], ef.fdots[idx[0], idx[1]]
                max_stat = ef.stat[idx[0], idx[1]]
            elif len(ef.stat.shape) == 1:
                f = ef.freq[idx[0]]
                max_stat = ef.stat[idx[0]]
                fdot = 0
            else:
                raise ValueError("Did not understand stats shape.")
            best_cand_table.add_row([ef.pepoch, max_stat, f, fdot, fddot])

        print(best_cand_table)
        best_cand_table.write(fname + "_best_cands.csv", overwrite=True)
        plt.figure(fname, figsize=(8, 8))

        if (
            hasattr(ef, "filename")
            and ef.filename is not None
            and os.path.exists(ef.filename)
        ):
            external_gs = gridspec.GridSpec(2, 1)
            search_gs_no = 1

            events = load_events(ef.filename)
            if ef.emin is not None or ef.emax is not None:
                events, elabel = filter_energy(events, ef.emin, ef.emax)

            if hasattr(ef, "parfile") and ef.parfile is not None:
                root = os.path.split(fname)[0]
                parfile = find_file_in_allowed_paths(ef.parfile, [".", root])
                if not parfile:
                    warnings.warn("{} does not exist".format(ef.parfile))
                else:
                    ef.parfile = parfile

                if parfile and os.path.exists(parfile):
                    events = deorbit_events(events, parfile)

            if hasattr(ef, "ref_time") and ef.ref_time is not None:
                ref_time = ef.ref_time
            elif hasattr(ef, "pepoch") and ef.pepoch is not None:
                ref_time = (ef.pepoch - events.mjdref) * 86400
            else:
                ref_time = (events.time[0] + events.time[-1]) / 2

            pepoch = ref_time / 86400 + events.mjdref

            phase, profile, profile_err = fold_events(
                copy.deepcopy(events.time),
                f,
                fdot,
                ref_time=ref_time,
                # gtis=copy.deepcopy(events.gti),
                expocorr=False,
                nbin=nbin,
            )
            ax = plt.subplot(external_gs[0])

            # print(df, dfdot)
            # # noinspection PyPackageRequirements
            # ax.text(0.1, 0.9, "Profile for F0={} Hz, F1={} Hz/s".format(
            #     round(f, -np.int(np.floor(np.log10(np.abs(df))))),
            #     round(fdot, -np.int(np.floor(np.log10(np.abs(dfdot)))))),
            #     horizontalalignment='left', verticalalignment='center',
            #     transform=ax.transAxes)
            ax.plot(
                np.concatenate((phase, phase + 1)),
                np.concatenate((profile, profile)),
                drawstyle="steps-mid",
            )

            mean = np.mean(profile)

            low, high = poisson_conf_interval(
                mean, interval="frequentist-confidence", sigma=1
            )

            ax.axhline(mean)
            ax.fill_between(
                [0, 2],
                [low, low],
                [high, high],
                label=r"1-$\sigma c.l.$",
                alpha=0.5,
            )
            low, high = poisson_conf_interval(
                mean, interval="frequentist-confidence", sigma=3
            )
            ax.fill_between(
                [0, 2],
                [low, low],
                [high, high],
                label=r"3-$\sigma c.l.$",
                alpha=0.5,
            )
            ax.set_xlabel("Phase")
            ax.set_ylabel("Counts")
            ax.set_xlim([0, 2])
            ax.legend(loc=4)
            phascommand = (
                "HENphaseogram -f {} "
                "--fdot {} {} -n {} --norm to1".format(
                    f, fdot, ef.filename, nbin
                )
            )
            if ef.parfile and os.path.exists(ef.parfile):
                phascommand += " --deorbit-par {}".format(parfile)
            if hasattr(ef, "emin") and ef.emin is not None:
                phascommand += " --emin {}".format(ef.emin)
            if hasattr(ef, "emin") and ef.emin is not None:
                phascommand += " --emax {}".format(ef.emax)

            if hasattr(events, "mjdref") and events.mjdref is not None:
                phascommand += " --pepoch {}".format(pepoch)

            log.info(
                "To see the detailed phaseogram, " "run {}".format(phascommand)
            )

        elif not os.path.exists(ef.filename):
            warnings.warn(ef.filename + " does not exist")
            external_gs = gridspec.GridSpec(1, 1)
            search_gs_no = 0
        else:
            external_gs = gridspec.GridSpec(1, 1)
            search_gs_no = 0

        if len(ef.stat.shape) > 1 and ef.stat.shape[0] > 1:
            gs = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                height_ratios=(1, 3),
                width_ratios=(3, 1),
                hspace=0,
                wspace=0,
                subplot_spec=external_gs[search_gs_no],
            )

            axf = plt.subplot(gs[0, 0])
            axfdot = plt.subplot(gs[1, 1])
            if vmax is not None:
                axf.axhline(vmax, ls="--", label=r"99.9\% c.l.")
                axfdot.axvline(vmax)
            axffdot = plt.subplot(gs[1, 0], sharex=axf, sharey=axfdot)
            axffdot.pcolormesh(
                ef.freq, np.asarray(ef.fdots), ef.stat, vmin=vmin, vmax=vmax
            )
            maximum_idx = 0
            maximum = 0

            for ix in range(ef.stat.shape[0]):
                if ef.stat.shape[0] < 100:
                    axf.plot(
                        ef.freq[ix, :],
                        ef.stat[ix, :],
                        alpha=0.5,
                        lw=0.2,
                        color="k",
                    )
                if np.max(ef.stat[ix, :]) > maximum:
                    maximum = np.max(ef.stat[ix, :])
                    maximum_idx = ix
            if vmax is not None and maximum_idx > 0:
                axf.plot(
                    ef.freq[maximum_idx, :],
                    ef.stat[maximum_idx, :],
                    lw=1,
                    color="k",
                )
            maximum_idx = -1
            maximum = 0
            for iy in range(ef.stat.shape[1]):
                if ef.stat.shape[1] < 100:
                    axfdot.plot(
                        ef.stat[:, iy],
                        np.asarray(ef.fdots)[:, iy],
                        alpha=0.5,
                        lw=0.2,
                        color="k",
                    )
                if np.max(ef.stat[:, iy]) > maximum:
                    maximum = np.max(ef.stat[:, iy])
                    maximum_idx = iy
            if vmax is not None and maximum_idx > 0:
                axfdot.plot(
                    ef.stat[:, maximum_idx],
                    np.asarray(ef.fdots)[:, maximum_idx],
                    lw=1,
                    color="k",
                )
            axf.set_ylabel(label)
            axfdot.set_xlabel(label)

            # plt.colorbar()
            axffdot.set_xlabel("Frequency (Hz)")
            axffdot.set_ylabel("Fdot (Hz/s)")
            axffdot.set_xlim([np.min(ef.freq), np.max(ef.freq)])
            axffdot.set_ylim([np.min(ef.fdots), np.max(ef.fdots)])
            axffdot.axvline(f, ls="--", color="white")
            axffdot.axhline(fdot, ls="--", color="white")
            axf.legend(loc=4)
        else:
            axf = plt.subplot(external_gs[search_gs_no])
            axf.plot(ef.freq, ef.stat, drawstyle="steps-mid", label=fname)
            axf.set_xlabel("Frequency (Hz)")
            axf.set_ylabel(ef.kind + " stat")
            axf.legend(loc=4)

        if (
            hasattr(ef, "best_fits")
            and ef.best_fits is not None
            and not len(ef.stat.shape) > 1
        ):

            for f in ef.best_fits:
                xs = np.linspace(
                    np.min(ef.freq), np.max(ef.freq), len(ef.freq) * 2
                )
                plt.plot(xs, f(xs))

        if output_data_file is not None:
            fdots = ef.fdots
            if not isinstance(fdots, Iterable) or len(fdots) == 1:
                fdots = fdots + np.zeros_like(ef.freq.flatten())
            # print(fdots.shape, ef.freq.shape, ef.stat.shape)
            out = [ef.freq.flatten(), fdots.flatten(), ef.stat.flatten()]
            out_err = [None, None, None]

            if (
                hasattr(ef, "best_fits")
                and ef.best_fits is not None
                and not len(ef.stat.shape) > 1
            ):
                for f in ef.best_fits:
                    out.append(f(ef.freq.flatten()))
                    out_err.append(None)

            save_as_qdp(out, out_err, filename=output_data_file, mode="a")

    ax = plt.gca()
    if xlog:
        ax.set_xscale("log", nonposx="clip")
    if ylog:
        ax.set_yscale("log", nonposy="clip")
    plt.tight_layout()

    if figname is not None:
        plt.savefig(figname)


def plot_color(
    file0, file1, xlog=None, ylog=None, figname=None, output_data_file=None
):
    import matplotlib.pyplot as plt

    type0, lc0 = get_file_type(file0)
    type1, lc1 = get_file_type(file1)
    xlabel, ylabel = "Count rate", "Count rate"
    if type0 == "color":
        xlabel = "{3}-{2}/{1}-{0}".format(*lc0.e_intervals)
    if type1 == "color":
        ylabel = "{3}-{2}/{1}-{0}".format(*lc1.e_intervals)
    plt.errorbar(
        lc0.counts,
        lc1.counts,
        xerr=lc0.counts_err,
        yerr=lc1.counts_err,
        fmt="o",
        color="k",
        alpha=0.5,
    )
    plt.scatter(lc0.counts, lc1.counts, zorder=10)

    if output_data_file is not None:
        save_as_qdp(
            [lc0.counts, lc1.counts],
            errors=[lc0.counts_err, lc1.counts_err],
            filename=output_data_file,
            mode="a",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    if xlog:
        ax.set_xscale("log", nonposx="clip")
    if ylog:
        ax.set_yscale("log", nonposy="clip")
    if figname is not None:
        plt.savefig(figname)


def plot_lc(
    lcfiles,
    figname=None,
    fromstart=False,
    xlog=None,
    ylog=None,
    output_data_file=None,
):
    """Plot a list of light curve files, or a single one."""
    import matplotlib.pyplot as plt

    if is_string(lcfiles):
        lcfiles = [lcfiles]

    figlabel = lcfiles[0]

    plt.figure("LC " + figlabel)
    for lcfile in lcfiles:
        log.info("Loading %s..." % lcfile)
        lcdata = load_data(lcfile)

        time = lcdata["time"]
        lc = lcdata["counts"]
        gti = lcdata["gti"]
        instr = lcdata["instr"]

        if fromstart:
            time -= lcdata["Tstart"]
            gti -= lcdata["Tstart"]

        if instr == "PCA":
            # If RXTE, plot per PCU count rate
            npcus = lcdata["nPCUs"]
            lc /= npcus

        for g in gti:
            plt.axvline(g[0], ls="-", color="red")
            plt.axvline(g[1], ls="--", color="red")

        good = create_gti_mask(time, gti)
        plt.plot(time, lc, drawstyle="steps-mid", color="grey")
        plt.plot(time[good], lc[good], drawstyle="steps-mid", label=lcfile)
        if "base" in lcdata:
            plt.plot(time, lcdata["base"], color="r")

        if output_data_file is not None:
            outqdpdata = [time[good], lc[good]]
            if "base" in lcdata:
                outqdpdata.append(lcdata["base"][good])
            save_as_qdp(outqdpdata, filename=output_data_file, mode="a")

    plt.xlabel("Time (s)")
    if instr == "PCA":
        plt.ylabel("light curve (Ct/bin/PCU)")
    else:
        plt.ylabel("light curve (Ct/bin)")

    plt.legend()
    if figname is not None:
        plt.savefig(figname)


def main(args=None):
    """Main function called by the `HENplot` command line script."""
    import argparse
    from .base import check_negative_numbers_in_args

    description = (
        "Plot the content of HENDRICS light curves and frequency spectra"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "--noplot",
        help="Only create images, do not plot",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--CCD",
        help="This is a color-color diagram. In this case, the"
        " list of files is expected to be given as "
        "soft0.nc, hard0.nc, soft1.nc, hard1.nc, ...",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--HID",
        help="This is a hardness-intensity diagram. In this "
        "case, the list of files is expected to be given "
        "as color0.nc, intensity0.nc, color1.nc, "
        "intensity1.nc, ...",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--figname", help="Figure name", default=None, type=str
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output data file in QDP format",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--xlog",
        help="Use logarithmic X axis",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--ylog",
        help="Use logarithmic Y axis",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--xlin", help="Use linear X axis", default=None, action="store_true"
    )
    parser.add_argument(
        "--ylin", help="Use linear Y axis", default=None, action="store_true"
    )
    parser.add_argument(
        "--fromstart",
        help="Times are measured from the start of the "
        "observation (only relevant for light curves)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--axes",
        nargs=2,
        type=str,
        help="Plot two variables contained in the file",
        default=None,
    )

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)
    if args.noplot and args.figname is None:
        args.figname = args.files[0].replace(HEN_FILE_EXTENSION, ".png")
        import matplotlib

        matplotlib.use("Agg")
    if args.xlin is not None:
        args.xlog = False
    if args.ylin is not None:
        args.ylog = False

    if args.CCD or args.HID:
        args.files = zip(args.files[:-1:2], args.files[1::2])

    for fname in args.files:

        if args.CCD or args.HID:
            plot_color(
                fname[0],
                fname[1],
                xlog=args.xlog,
                ylog=args.ylog,
                figname=args.figname,
                output_data_file=args.outfile,
            )
            continue
        ftype, contents = get_file_type(fname)
        if args.axes is not None:
            plot_generic(
                fname,
                args.axes,
                xlog=args.xlog,
                ylog=args.ylog,
                figname=args.figname,
                output_data_file=args.outfile,
            )
            continue
        if ftype == "lc":
            plot_lc(
                fname,
                fromstart=args.fromstart,
                xlog=args.xlog,
                ylog=args.ylog,
                figname=args.figname,
                output_data_file=args.outfile,
            )
        elif ftype == "folding":
            plot_folding(
                fname,
                xlog=args.xlog,
                ylog=args.ylog,
                figname=args.figname,
                output_data_file=args.outfile,
            )
        elif ftype[-4:] == "cpds":
            plot_cospectrum(
                fname,
                xlog=args.xlog,
                ylog=args.ylog,
                figname=args.figname,
                output_data_file=args.outfile,
            )
        elif ftype[-3:] == "pds":
            plot_pds(
                fname,
                xlog=args.xlog,
                ylog=args.ylog,
                figname=args.figname,
                output_data_file=args.outfile,
            )

    if not args.noplot:
        import matplotlib.pyplot as plt

        plt.show()
