# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Quicklook plots."""

import warnings
import os
import copy
from collections.abc import Iterable
import numpy as np
import matplotlib.colors as colors
from stingray.gti import create_gti_mask
from astropy.modeling.models import Const1D
from astropy.modeling import Model
from astropy.stats import poisson_conf_interval
from astropy import log
from astropy.table import Table

from .efsearch import analyze_qffa_results
from .fold import fold_events, filter_energy
from .io import load_events, load_lcurve, load_pds
from .io import load_data, get_file_type
from .io import is_string, save_as_qdp
from .io import HEN_FILE_EXTENSION
from .io import find_file_in_allowed_paths
from .base import _assign_value_if_none
from .base import pds_detection_level as detection_level
from .base import deorbit_events
from stingray.power_colors import plot_hues, plot_power_colors


def _next_color(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    p = ax.plot(xlim, ylim)
    color = p[0].get_color()
    p[0].remove()
    return color


def _baseline_fun(x, a):
    """A constant function."""
    return a


def _value_or_none(dict_like, key):
    try:
        return dict_like[key]
    except KeyError:
        return None


def rescale_plot_units(values):
    """Rescale the values to an order of magnitude that allows better plotting.

    Subtracts the mean ``mean`` from the values, then rescales the residuals
    to a comfortable order of magnitude ``oom``.
    If ``out`` are the rescaled values, this should always work::

        out * 10**oom + mean == values

    Parameters
    ----------
    values: array-like
        Input values to be rescaled

    Returns
    -------
    mean: float
        The mean of the input values, rounded to the order of magnitude of the data span
    oom : int
        The order of magnitude of the data span
    values : array-like
        The rescaled values

    Examples
    --------
    >>> values = np.arange(-0.003, 0.0032, 0.0002) + 5.0001
    >>> mean, oom, rescaled = rescale_plot_units(values)
    >>> assert mean == 5.0
    >>> oom
    -3
    >>> assert np.allclose(rescaled * 10**oom + mean, values)
    >>> values = np.arange(-3, 3.2, 0.2) + 5.0001
    >>> mean, oom, rescaled = rescale_plot_units(values)
    >>> assert oom == 0
    >>> assert mean == 0.0
    >>> assert np.allclose(rescaled, values)
    """
    span = values.max() - values.min()

    oom = int(np.log10((span))) - 1
    if abs(oom) <= 2:
        return 0.0, 0, values

    mean = round(values.mean(), -oom)
    return mean, oom, (values - mean) / 10**oom


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
    figname = _assign_value_if_none(figname, "{0} vs {1}".format(vars[1], vars[0]))
    plt.figure(figname)
    ax = plt.gca()
    if xlog:
        ax.set_xscale("log", nonpositive="clip")
    if ylog:
        ax.set_yscale("log", nonpositive="clip")

    xlabel, ylabel = vars
    xlabel_err, ylabel_err = None, None
    if errs is not None:
        xlabel_err, ylabel_err = errs

    for i, fname in enumerate(fnames):
        data = get_file_type(fname)[1].dict()
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
    >>> assert _get_const(model) == 2.0
    >>> assert _get_const(model[0]) == 2.0
    >>> assert _get_const([model[0]]) == 2.0
    >>> assert _get_const([[model]]) == 2.0
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

    if isinstance(models, Iterable) and not is_string(models) and len(models) != 0:
        for model in models:
            return _get_const(model)

    return None


def plot_powercolors(fnames):
    if isinstance(fnames, Iterable) and not is_string(fnames):
        outs = []
        for fname in fnames:
            outs.append(plot_powercolors(fname))
        return outs

    ts = load_data(fnames)

    plot_power_colors(
        ts["pc1"], ts["pc1_err"], ts["pc2"], ts["pc2_err"], plot_spans=True
    )
    plot_hues(
        ts["rms"], ts["rms_err"], ts["pc1"], ts["pc2"], polar=True, plot_spans=True
    )
    plot_hues(
        ts["rms"], ts["rms_err"], ts["pc1"], ts["pc2"], polar=False, plot_spans=True
    )
    return ts


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
        if pds_obj.df is None:
            pds_obj.df = pds_obj.freq[1] - pds_obj.freq[0]
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

        lev = detection_level(epsilon=0.015, n_summed_spectra=npds, ntrial=pds.size)

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
            ax.set_xscale("log", nonpositive="clip")
        if ylog:
            ax.set_yscale("log", nonpositive="clip")

        level = lev  # Can be modified below
        y = pds[1:]
        yerr = yerr = None if epds is None else epds[1:]

        if not white_sub:
            white_sub = norm.lower() in ["rms", "frac"] and xlog and ylog

        if not white_sub:
            plt.plot(freq[1:], y, drawstyle="steps-mid", color=color, label=fname)
            for i, func in enumerate(models):
                plt.plot(
                    freq,
                    func(freq),
                    label="Model {}".format(i + 1),
                    zorder=20,
                    color="k",
                )
        else:
            # TODO: Very rough! Use new machinery
            const = _get_const(models)
            if const is None:
                p, pcov = curve_fit(_baseline_fun, freq, pds, p0=[2], sigma=epds)
                log.info("White noise level is {0}".format(p[0]))
                const = p[0]

            pds -= const
            level = lev - const

            y = pds[1:] * freq[1:]
            yerr = None if epds is None else epds[1:] * freq[1:]
            plt.plot(freq[1:], y, drawstyle="steps-mid", color=color, label=fname)
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


def plot_cospectrum(fnames, figname=None, xlog=None, ylog=None, output_data_file=None):
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
            ax.set_xscale("log", nonpositive="clip")
        if ylog:
            ax.set_yscale("log", nonpositive="clip")

        plt.xlabel("Frequency")
        if xlog and ylog:
            y = freq[1:] * cospectrum[1:]
            plt.plot(freq[1:], y, drawstyle="steps-mid", label=fname)
            for i, func in enumerate(models):
                plt.plot(freq, freq * func(freq), label="Model {}".format(i + 1))

            plt.ylabel("Cospectrum * Frequency")
        else:
            y = cospectrum[1:]
            plt.plot(freq[1:], cospectrum[1:], drawstyle="steps-mid", label=fname)

            plt.ylabel("Cospectrum")
            for i, func in enumerate(models):
                plt.plot(freq, func(freq), label="Model {}".format(i + 1))
        if output_data_file is not None:
            save_as_qdp([freq[1:], y], filename=output_data_file, mode="a")

    plt.legend()

    if figname is not None:
        plt.savefig(figname)


def plot_folding(fnames, figname=None, xlog=None, ylog=None, output_data_file=None):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt

    if is_string(fnames):
        fnames = [fnames]

    for fname in fnames:
        plt.figure(fname, figsize=(7, 7))
        plt.clf()
        ef, best_cand_table = analyze_qffa_results(fname)
        nbin = best_cand_table.meta["nbin"]
        label = best_cand_table.meta["label"]
        detlev = best_cand_table.meta["detlev"]
        ndof = best_cand_table.meta["ndof"]
        # Get these from the first row of the table
        f, fdot, fddot, max_stat, max_stat_cl_90, f_idx, fdot_idx = (
            best_cand_table["f"][0],
            best_cand_table["fdot"][0],
            best_cand_table["fddot"][0],
            best_cand_table["power"][0],
            best_cand_table["power_cl_0.9"][0],
            best_cand_table["f_idx"][0],
            best_cand_table["fdot_idx"][0],
        )

        if (filename := best_cand_table.meta["filename"]) is not None:
            external_gs = gridspec.GridSpec(2, 1)
            search_gs_no = 1

            events = load_events(filename)
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
                # gti=copy.deepcopy(events.gti),
                expocorr=False,
                nbin=nbin,
            )
            ax = plt.subplot(external_gs[0])
            Table(
                {
                    "phase": np.concatenate((phase, phase + 1)),
                    "profile": np.concatenate((profile, profile)),
                    "err": np.concatenate((profile_err, profile_err)),
                }
            ).write(
                f'{fname.replace(HEN_FILE_EXTENSION, "")}_folded.csv',
                overwrite=True,
                format="ascii",
            )

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
            ntimes = max(8, np.rint(max_stat / 20).astype(int))
            phascommand = (
                f"HENphaseogram -f {f} "
                f"--fdot {fdot} {ef.filename} -n {nbin} --ntimes {ntimes} --norm meansub"
            )
            if ef.parfile and os.path.exists(ef.parfile):
                phascommand += " --deorbit-par {}".format(parfile)
            if hasattr(ef, "emin") and ef.emin is not None:
                phascommand += " --emin {}".format(ef.emin)
            if hasattr(ef, "emin") and ef.emin is not None:
                phascommand += " --emax {}".format(ef.emax)

            if hasattr(events, "mjdref") and events.mjdref is not None:
                phascommand += " --pepoch {}".format(pepoch)

            log.info("To see the detailed phaseogram, " "run {}".format(phascommand))

        elif not os.path.exists(ef.filename):
            warnings.warn(ef.filename + " does not exist")
            external_gs = gridspec.GridSpec(1, 1)
            search_gs_no = 0
        else:
            external_gs = gridspec.GridSpec(1, 1)
            search_gs_no = 0

        f_mean, f_oom, f_rescale = rescale_plot_units(ef.freq)

        if f_oom != 0:
            flabel = f"Frequency"
            if f_mean != 0.0:
                flabel = "(" + flabel + f"- {f_mean})"
            flabel += rf" ($10^{{{f_oom}}}$ Hz)"
        else:
            flabel = f"Frequency (Hz)"

        if len(ef.stat.shape) > 1 and ef.stat.shape[0] > 1:
            fd_mean, fd_oom, fd_rescale = rescale_plot_units(ef.fdots)
            if fd_oom != 0:
                fdlabel = f"Fdot"
                if fd_mean != 0.0:
                    fdlabel = "(" + flabel + f" - {fd_mean:g})"
                fdlabel += rf" ($10^{{{fd_oom}}}$ Hz/s)"
            else:
                fdlabel = f"Fdot (Hz/s)"

            gs = gridspec.GridSpecFromSubplotSpec(
                2,
                3,
                height_ratios=(1, 3),
                width_ratios=(3, 1, 0.2),
                hspace=0,
                wspace=0,
                subplot_spec=external_gs[search_gs_no],
            )

            axf = plt.subplot(gs[0, 0])
            axfdot = plt.subplot(gs[1, 1])
            axcolor = plt.subplot(gs[:, 2])

            plt.setp(axf.get_xticklabels(), visible=False)
            plt.setp(axfdot.get_yticklabels(), visible=False)

            if detlev is not None and ef.stat.max() < 20 * detlev:
                axf.axhline(detlev, ls="--", label=r"99.9% det. lev.")
                axfdot.axvline(detlev)

            axffdot = plt.subplot(gs[1, 0], sharex=axf, sharey=axfdot)
            vmin = ndof
            vcenter = detlev
            vmax = max(detlev + 1, ef.stat.max())

            divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

            pcol = axffdot.pcolormesh(
                f_rescale,
                fd_rescale,
                ef.stat,
                shading="nearest",
                norm=divnorm,
                cmap="twilight",
            )

            cs = axffdot.contour(
                f_rescale,
                fd_rescale,
                ef.stat,
                [max_stat_cl_90],
                colors="white",
                zorder=20,
            )
            colorticks = list(
                set(
                    np.concatenate(
                        (
                            np.linspace(vmin, vcenter, 3),
                            np.linspace(vcenter, vmax, 3),
                        )
                    ).astype(int)
                )
            )

            cbar = plt.colorbar(pcol, cax=axcolor, ticks=colorticks)

            if len(cs.allsegs[0]) > 1:
                warnings.warn(
                    "More than one contour found. " "Frequency estimates might be wrong"
                )
            else:
                for ax in (axffdot, axf):
                    ax.axvline(cs.allsegs[0][0][:, 0].min(), label=f"90% conf. lim.")
                    ax.axvline(cs.allsegs[0][0][:, 0].max())

                for ax in (axffdot, axfdot):
                    ax.axhline(cs.allsegs[0][0][:, 1].max())
                    ax.axhline(cs.allsegs[0][0][:, 1].min())

            if detlev is not None:
                axf.plot(
                    f_rescale[f_idx, :],
                    ef.stat[f_idx, :],
                    lw=1,
                    color="k",
                )
            for cand_row in best_cand_table:
                axfdot.plot(
                    ef.stat[:, cand_row["fdot_idx"]],
                    fd_rescale[:, cand_row["fdot_idx"]],
                    alpha=0.5,
                    lw=0.2,
                    color="k",
                )
                axf.plot(
                    np.asarray(f_rescale)[cand_row["f_idx"], :],
                    ef.stat[cand_row["f_idx"], :],
                    alpha=0.5,
                    lw=0.2,
                    color="k",
                )

            if detlev is not None:
                axfdot.plot(
                    ef.stat[:, fdot_idx],
                    fd_rescale[:, fdot_idx],
                    lw=1,
                    color="k",
                )
            axf.set_ylabel(label)
            axfdot.set_xlabel(label)

            # plt.colorbar()
            axffdot.set_xlabel(flabel)
            axffdot.set_ylabel(fdlabel)
            axffdot.set_xlim([np.min(f_rescale), np.max(f_rescale)])
            axffdot.set_ylim([np.min(fd_rescale), np.max(fd_rescale)])
            axffdot.axvline((f - f_mean) / 10**f_oom, ls="--", color="white")
            axffdot.axhline((fdot - fd_mean) / 10**fd_oom, ls="--", color="white")
            axf.legend(loc=4)
        else:
            axf = plt.subplot(external_gs[search_gs_no])
            axf.plot(f_rescale, ef.stat, drawstyle="steps-mid", label=fname)
            axf.set_xlabel(flabel)
            axf.set_ylabel(ef.kind + " stat")
            axf.legend(loc=4)

        if (
            hasattr(ef, "best_fits")
            and ef.best_fits is not None
            and not len(ef.stat.shape) > 1
        ):
            for f in ef.best_fits:
                xs = np.linspace(np.min(ef.freq), np.max(ef.freq), len(ef.freq) * 2)
                plt.plot(xs, f(xs))

        if output_data_file is not None:
            fdots = ef.fdots
            if not isinstance(fdots, Iterable) or len(fdots) == 1:
                fdots = fdots + np.zeros_like(ef.freq.flatten())

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
        ax.set_xscale("log", nonpositive="clip")
    if ylog:
        ax.set_yscale("log", nonpositive="clip")
    try:
        plt.tight_layout()
    except Exception:  # pragma: no cover
        pass

    if figname is not None:
        plt.savefig(figname)


def plot_color(file0, file1, xlog=None, ylog=None, figname=None, output_data_file=None):
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
        ax.set_xscale("log", nonpositive="clip")
    if ylog:
        ax.set_yscale("log", nonpositive="clip")
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
        lcdata = load_lcurve(lcfile)

        time = lcdata.time
        lc = lcdata.counts
        gti = lcdata.gti
        instr = lcdata.instr
        if not hasattr(lcdata, "mjdref") or lcdata.mjdref is None:
            lcdata.mjdref = 0

        mjd = lcdata.mjdref + np.mean(lcdata.time) / 86400
        mjd = np.round(mjd, 2)

        if fromstart:
            tref = lcdata.gti[0, 0]
            mjd = lcdata.gti[0, 0] / 86400 + lcdata.mjdref
        else:
            tref = (mjd - lcdata.mjdref) * 86400

        time -= tref
        gti -= tref

        if instr == "PCA":
            # If RXTE, plot per PCU count rate
            if hasattr(lcdata, "nPCUs"):
                npcus = lcdata.nPCUs
                lc /= npcus

        bti = list(zip(gti[:-1, 1], gti[1:, 0]))

        for g in bti:
            plt.axvspan(g[0], g[1], color="red", alpha=0.5)

        good = create_gti_mask(time, gti)
        plt.plot(time, lc, drawstyle="steps-mid", color="grey")
        plt.plot(time[good], lc[good], drawstyle="steps-mid", label=lcfile)
        if hasattr(lcdata, "base"):
            plt.plot(time, lcdata.base, color="r")

        if output_data_file is not None:
            outqdpdata = [time[good], lc[good]]
            if hasattr(lcdata, "base"):
                outqdpdata.append(lcdata.base[good])
            save_as_qdp(outqdpdata, filename=output_data_file, mode="a")

    plt.xlabel(f"Time (s from MJD {mjd}, MET {tref})")
    print(f"Time (s from MJD {mjd}, MET {tref})")
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

    description = "Plot the content of HENDRICS light curves and frequency spectra"
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
    parser.add_argument("--figname", help="Figure name", default=None, type=str)
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
        "--xlin", help="Use linear X axis", default=False, action="store_true"
    )
    parser.add_argument(
        "--ylin", help="Use linear Y axis", default=False, action="store_true"
    )
    parser.add_argument(
        "--white-sub",
        help="Subtract Poisson noise (only applies to PDS)",
        default=False,
        action="store_true",
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
    if args.xlin:
        args.xlog = False
    if args.ylin:
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
        elif ftype == "powercolor":
            plot_powercolors(fname)
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
                white_sub=args.white_sub,
            )

    if not args.noplot:
        import matplotlib.pyplot as plt

        plt.show()
