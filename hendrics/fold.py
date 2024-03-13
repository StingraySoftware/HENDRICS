"""Interactive phaseogram."""

import warnings
import copy
import argparse
import numpy as np
import urllib
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import optimize
from astropy.stats import poisson_conf_interval
from astropy import log
from stingray.pulse.pulsar import fold_events, pulse_phase, get_TOA
from stingray.pulse.pulsar import pulse_phase, htest
from stingray.utils import assign_value_if_none, fft, fftfreq, ifft

from stingray.events import EventList

from hendrics.ml_timing import get_template_func

from .base import hen_root, normalize_dyn_profile
from .io import load_events, filter_energy

try:
    from tqdm import tqdm as show_progress
except ImportError:

    def show_progress(a):
        return a


try:
    import pint.toa as toa

    # import pint
    HAS_PINT = True
except (ImportError, urllib.error.URLError):
    warnings.warn(
        "PINT is not installed. " "Some pulsar functionality will not be available"
    )
    HAS_PINT = False
from .base import deorbit_events


def _load_and_prepare_TOAs(mjds, errs_us=None, ephem="DE421"):
    errs_us = assign_value_if_none(errs_us, np.zeros_like(mjds))

    toalist = [None] * len(mjds)
    for i, m in enumerate(mjds):
        toalist[i] = toa.TOA(
            m,
            error=errs_us[i],
            obs="Barycenter",
            scale="tdb",
        )

    toalist = toa.TOAs(toalist=toalist)
    if "tdb" not in toalist.table.colnames:
        toalist.compute_TDBs(ephem=ephem)
    if "ssb_obs_pos" not in toalist.table.colnames:
        toalist.compute_posvels(ephem, False)
    return toalist


def create_template_from_profile_sins(
    phase, profile, profile_err, imagefile="template.png", norm=1
):
    """
    Parameters
    ----------
    phase: :class:`np.array`
    profile: :class:`np.array`
    profile_err: :class:`np.array`
        Phase, pulse profile, and error bars
    imagefile: str
    norm: float or :class:`np.array`

    Returns
    -------
    template: :class:`np.array`
        The calculated template
    additional_phase: float

    Examples
    --------
    >>> phase = np.arange(0.0, 1, 0.001)
    >>> profile = np.cos(2 * np.pi * phase)
    >>> profile_err = profile * 0
    >>> template, additional_phase = create_template_from_profile_sins(
    ...     phase, profile, profile_err)
    ...
    >>> assert np.allclose(template, profile, atol=0.01)
    """
    import matplotlib.pyplot as plt

    prof = np.concatenate((profile, profile, profile))
    proferr = np.concatenate((profile_err, profile_err, profile_err))
    fit_pars_save, _, _ = fit_profile_with_sinusoids(
        prof, proferr, nperiods=3, baseline=True, debug=False
    )
    template = std_fold_fit_func(fit_pars_save, phase)
    fig = plt.figure()
    plt.plot(phase, profile, drawstyle="steps-mid")
    plt.plot(phase, template, drawstyle="steps-mid")
    plt.savefig(imagefile)
    plt.close(fig)
    # start template from highest bin!
    template *= norm
    template_fine = std_fold_fit_func(fit_pars_save, np.arange(0, 1, 0.001))
    additional_phase = np.argmax(template_fine) / len(template_fine)
    return template, additional_phase


def create_template_from_profile(
    phase, profile, profile_err, imagefile="template.png", norm=1
):
    """
    Parameters
    ----------
    phase: :class:`np.array`
    profile: :class:`np.array`
    profile_err: :class:`np.array`
        Phase, pulse profile, and error bars
    imagefile: str
    norm: float or :class:`np.array`

    Returns
    -------
    template: :class:`np.array`
        The calculated template
    additional_phase: float

    Examples
    --------
    >>> phase = np.arange(0.0, 1, 0.01)
    >>> profile = np.cos(2 * np.pi * phase)
    >>> profile_err = profile * 0
    >>> template, additional_phase = create_template_from_profile(
    ...     phase, profile, profile_err)
    ...
    >>> assert np.allclose(template, profile, atol=0.001)
    """
    from scipy.interpolate import splrep, splev
    import matplotlib.pyplot as plt

    ph = np.concatenate((phase - 1, phase, phase + 1))
    prof = np.concatenate((profile, profile, profile))
    proferr = np.concatenate((profile_err, profile_err, profile_err))

    weights = 1 / proferr if np.all(proferr != 0) else None
    # template = savgol_filter(profile, 5, 3, mode='wrap')
    spl = splrep(ph, prof, w=weights, s=0)
    phases_fine = np.arange(0, 1, 0.001)
    template_fine = splev(phases_fine, spl)
    template = splev(phase, spl)

    fig = plt.figure()
    plt.plot(phase, profile, drawstyle="steps-mid")
    plt.plot(phase, template, drawstyle="steps-mid")
    plt.savefig(imagefile)
    plt.close(fig)

    additional_phase = np.argmax(template_fine) / len(template_fine)
    return template, additional_phase


def create_template_from_profile_harm(
    phase,
    profile,
    profile_err=0,
    imagefile="template.png",
    norm=1,
    nharm=None,
    final_nbin=None,
):
    """
    Parameters
    ----------
    phase: :class:`np.array`
    profile: :class:`np.array`
    profile_err: :class:`np.array`
        Phase, pulse profile, and error bars
    imagefile: str
    norm: float or :class:`np.array`
    final_nbin: int

    Returns
    -------
    template: :class:`np.array`
        The calculated template
    additional_phase: float

    Examples
    --------
    >>> phase = np.arange(0.005, 1, 0.01)
    >>> profile = np.cos(2 * np.pi * phase)
    >>> profile_err = profile * 0
    >>> template, additional_phase = create_template_from_profile_harm(
    ...     phase, profile, profile_err)
    ...
    >>> assert np.allclose(template, profile, atol=0.001)
    """
    import matplotlib.pyplot as plt

    nbin = profile.size
    prof = np.concatenate((profile, profile, profile))
    dph = 1 / profile.size
    ft = fft(prof)
    freq = fftfreq(prof.size, dph)

    if nharm is None:
        nharm = max(1, int(prof.size / 16))

    if final_nbin is None:
        final_nbin = nbin

    new_ft = np.zeros(final_nbin * 3, dtype=complex)
    new_ft_freq = fftfreq(final_nbin * 3, 1 / final_nbin)

    new_ft[np.abs(new_ft_freq) <= nharm] = ft[np.abs(freq) <= nharm]

    template = ifft(new_ft)
    dph = 1 / final_nbin
    phas = np.arange(dph / 2, 3, dph)
    templ_func = interp1d(phas, template, kind="cubic", assume_sorted=True)
    phases_fine = np.linspace(1, 2, 1_000 * nharm)
    dph_fine = phases_fine[1] - phases_fine[0]
    # phases_fine += 0.5 * dph_fine
    template_fine = templ_func(phases_fine)

    additional_phase = np.argmax(template_fine) / len(template_fine) + dph_fine / 2
    template = template[:final_nbin].real
    fig = plt.figure()
    plt.plot(phase, profile, drawstyle="steps-mid")
    plt.plot(phas[:final_nbin], template, drawstyle="steps-mid")
    plt.savefig(imagefile)
    plt.close(fig)
    return template * final_nbin / nbin, additional_phase


def create_default_template(template_raw):
    """Create a smooth-ish template from an input folded profile.

    An initial H test assesses the kind of profile. If M > 5, the profile is
    approximated through a Savitzky-Golay filter. Otherwise, through a Fourier
    interpolation with M harmonics.

    Parameters
    ----------
    template: :class:`np.array`

    Returns
    -------
    template: :class:`np.array`
        The calculated template
    additional_phase: float

    Examples
    --------
    >>> phase = np.arange(0.005, 1, 0.001)
    >>> profile = np.cos(2 * np.pi * phase)
    >>> profile_err = profile * 0
    >>> template, additional_phase = create_default_template(
    ...     profile)
    ...
    >>> assert np.allclose(template.max(), profile.max(), atol=0.1)
    >>> profile = np.exp(-(phase - 0.5)**2 / (2 * 0.0001))
    >>> profile_err = profile * 0
    >>> template, additional_phase = create_default_template(
    ...     profile)
    ...
    >>> assert np.allclose(template.max(), profile.max(), atol=0.1)
    """
    nbin = template_raw.size
    nharm = min(max(1, nbin // 16), htest(template_raw)[0])
    dph = 1 / nbin
    phase = np.arange(0.5 * dph, 1, dph)
    if nharm <= 5:
        template, _ = create_template_from_profile_harm(
            phase, template_raw, nharm=nharm, final_nbin=nbin * 100
        )
    else:
        from scipy.signal import savgol_filter

        template = savgol_filter(template_raw, 4, 2, mode="wrap")
        phase = np.concatenate((phase - 1, phase, phase + 1))
        template = np.concatenate((template, template, template))

        templ_func = interp1d(phase, template, kind="cubic", assume_sorted=True)

        phases_fine = np.linspace(0, 1, nbin * 100 + 1)[:-1]
        phases_fine += (phases_fine[1] - phases_fine[0]) / 2
        template = templ_func(phases_fine)

    # print(template_raw.max(), template.max())
    additional_phase = np.argmax(template) / len(template)

    return template, additional_phase


def get_TOAs_from_events(events, folding_length, *frequency_derivatives, **kwargs):
    """Get TOAs of pulsation.

    Parameters
    ----------
    events : array-like
        event arrival times
    folding_length : float
        length of sub-intervals to fold
    *frequency_derivatives : floats
        pulse frequency, first derivative, second derivative, etc.

    Other parameters
    ----------------
    pepoch : float, default None
        Epoch of timing solution, in the same units as ev_times. If none, the
        first event time is used.
    mjdref : float, default None
        Reference MJD
    template : array-like, default None
        The pulse template
    nbin : int, default 16
        The number of bins in the profile (overridden by the dimension of the
        template)
    timfile : str, default 'out.tim'
        file to save the TOAs to (if PINT is installed)
    gti: [[g0_0, g0_1], [g1_0, g1_1], ...]
         Good time intervals. Defaults to None
    quick: bool
         If True, use a quicker fitting algorithms for TOAs. Defaults to False
    position: `astropy.SkyCoord` object
         Position of the object
    ephem : str
        Ephemeris. Default DE421

    Returns
    -------
    toas : array-like
        list of times of arrival. If ``mjdref`` is specified, they are
        expressed as MJDs, otherwise in MET
    toa_err : array-like
        errorbars on TOAs, in the same units as TOAs.
    """
    template = kwargs["template"] if "template" in kwargs else None
    mjdref = kwargs["mjdref"] if "mjdref" in kwargs else None
    nbin = kwargs["nbin"] if "nbin" in kwargs else 16
    pepoch = kwargs["pepoch"] if "pepoch" in kwargs else None
    timfile = kwargs["timfile"] if "timfile" in kwargs else "out.tim"
    gti = kwargs["gti"] if "gti" in kwargs else None
    label = kwargs["label"] if "label" in kwargs else None
    quick = kwargs["quick"] if "quick" in kwargs else False
    ephem = kwargs["ephem"] if "ephem" in kwargs else "DE421"

    pepoch = assign_value_if_none(pepoch, events[0])
    gti = np.asarray(assign_value_if_none(gti, [[events[0], events[-1]]]))
    # run exposure correction only if there are less than 1000 pulsations
    # in the interval
    length = gti.max() - gti.min()
    expocorr = folding_length < (1000 / frequency_derivatives[0])

    if template is not None:
        additional_phase = np.argmax(template) / template.size
    else:
        _, profile, _ = fold_events(
            copy.deepcopy(events),
            *frequency_derivatives,
            ref_time=pepoch,
            gti=copy.deepcopy(gti),
            expocorr=expocorr,
            nbin=nbin,
        )
        template, additional_phase = create_default_template(profile)
    print(template.size)

    min_phase_err = 1 / template.size
    fit_base = False
    if np.any(template < 0):
        fit_base = True

    starts = np.arange(gti[0, 0], gti[-1, 1], folding_length)

    freqs = np.zeros_like(starts) + frequency_derivatives[0]

    factorial = 1
    for i_f, f in enumerate(frequency_derivatives[1:]):
        factorial *= i_f + 1
        freqs += 1 / factorial * (starts - pepoch) ** (i_f + 1) * f

    phase_starts = pulse_phase((starts - pepoch), *frequency_derivatives, to_1=True)

    # Make each start happen at phase 0 or 1!
    starts -= phase_starts / freqs
    stops = starts + folding_length
    startidxs = np.searchsorted(events, starts)
    stopidxs = np.searchsorted(events, stops)

    toas = []
    toa_errs = []
    phs = []
    phs_errs = []
    for start, stop, startidx, stopidx, local_f in show_progress(
        zip(starts, stops, startidxs, stopidxs, freqs)
    ):
        # good = (events >= start) & (events < stop)
        events_tofold = events[startidx:stopidx]
        if len(events_tofold) < nbin:
            continue
        gti_tofold = copy.deepcopy(gti[(gti[:, 0] < stop) & (gti[:, 1] > start)])
        gti_tofold[0, 0] = start
        gti_tofold[-1, 1] = stop

        fder = copy.deepcopy(list(frequency_derivatives))
        fder[0] = local_f
        phase, profile, profile_err = fold_events(
            events_tofold,
            *fder,
            ref_time=start,
            gti=gti_tofold,
            expocorr=expocorr,
            nbin=nbin,
        )

        from .ml_timing import ml_pulsefit

        pars, errs = ml_pulsefit(
            profile, template, calculate_errors=True, fit_base=fit_base
        )

        if np.any(np.isnan(pars)) or pars[0] == 0.0 or np.any(np.isnan(errs)):
            warnings.warn(
                f"Invalid TOA in interval {start}-{stop} (idxs {startidx}:{stopidx}): {pars}, {errs}"
            )
            continue

        ph, phe = pars[1], errs[1]

        phe = max(min_phase_err, phe)

        toa = (ph + additional_phase) / frequency_derivatives[0] + start
        toaerr = phe / frequency_derivatives[0]
        toas.append(toa)
        toa_errs.append(toaerr)
        phs.append(ph)
        phs_errs.append(phe)

    if np.size(toas) == 0:
        return None, None

    toas, toa_errs = np.array(toas), np.array(toa_errs)
    phs, phs_errs = np.array(phs), np.array(phs_errs)

    factor = np.std(phs) / np.mean(phs_errs)

    if phs.size > 15:
        log.info(
            "Correcting TOA errors for the real scatter. Don't trust them literally"
        )

        # print(phs, phs_errs, factor)
        toa_errs = toa_errs * factor

    if mjdref is not None:
        toas = toas / 86400 + mjdref
        toa_errs = toa_errs * 1e6
        if HAS_PINT:
            label = assign_value_if_none(label, "hendrics")
            toa_list = _load_and_prepare_TOAs(toas, errs_us=toa_errs, ephem=ephem)
            # workaround until PR #368 is accepted in pint
            toa_list.table["clkcorr"] = 0
            toa_list.write_TOA_file(timfile, name=label, format="Tempo2")

        log.info("TOA(MJD)  TOAerr(us)")
    else:
        log.info("TOA(MET)  TOAerr(us)")

    for t, e in zip(toas, toa_errs):
        log.info(f"{t}, {e}")

    return toas, toa_errs


def _check_odd(n):
    return n // 2 * 2 + 1


def dbl_cos_fit_func(p, x):
    # the frequency is fixed
    """
    A double sinus (fundamental + 1st harmonic) used as a fit function
    """
    startidx = 0
    base = 0
    if len(p) % 2 != 0:
        base = p[0]
        startidx = 1
    first_harm = p[startidx] * np.cos(2 * np.pi * x + 2 * np.pi * p[startidx + 1])
    second_harm = p[startidx + 2] * np.cos(
        4.0 * np.pi * x + 4 * np.pi * p[startidx + 3]
    )
    return base + first_harm + second_harm


def std_fold_fit_func(p, x):
    """Chooses the fit function used in the fit."""

    return dbl_cos_fit_func(p, x)


def std_residuals(p, x, y):
    """The residual function used in the fit."""
    return std_fold_fit_func(p, x) - y


def adjust_amp_phase(pars):
    """Give the phases in the interval between 0 and 1.
    The calculation is based on the amplitude and phase given as input

    pars[0] is the initial amplitude; pars[1] is the initial phase
    If amplitude is negative, it makes it positive and changes the phase
    accordingly

    Examples
    --------
    >>> assert np.allclose(adjust_amp_phase([-0.5, 0.2]), [0.5, 0.7])
    >>> assert np.allclose(adjust_amp_phase([0.5, -1.2]), [0.5, 0.8])
    >>> assert np.allclose(adjust_amp_phase([0.5, 1.2]), [0.5, 0.2])
    """
    if pars[0] < 0:
        pars[0] = -pars[0]
        pars[1] += 0.5

    pars[1] = pars[1] - np.floor(pars[1])
    return pars


def fit_profile_with_sinusoids(
    profile, profile_err, debug=False, nperiods=1, baseline=False
):
    """
    Fit a folded profile with the std_fold_fit_func.

    Tries a number of different initial values for the fit, and returns the
    result of the best chi^2 fit

    Parameters
    ----------
    profile : array of floats
        The folded profile
    profile_err : array of floats
        the error on the folded profile elements

    Other parameters
    ----------------
    debug : bool, optional
        print debug info
    nperiods : int, optional
        number of periods in the folded profile. Default 1.

    Returns
    -------
    fit_pars : array-like
        the best-fit parameters
    success : bool
        whether the fit succeeded or not
    chisq : float
        the best chi^2
    """
    x = np.arange(0, len(profile) * nperiods, nperiods) / float(len(profile))
    guess_pars = [
        max(profile) - np.mean(profile),
        x[np.argmax(profile[: len(profile) // nperiods])],
        0,
        0.25,
    ]
    startidx = 0
    if baseline:
        guess_pars = [np.mean(profile)] + guess_pars
        if debug:
            log.debug(guess_pars)
        startidx = 1
    chisq_save = 1e32
    fit_pars_save = guess_pars
    success_save = -1
    if debug:
        import matplotlib.pyplot as plt

        fig = plt.figure("Debug profile")
        plt.title("Debug profile")
        plt.errorbar(x, profile, drawstyle="steps-mid")
        plt.plot(x, std_fold_fit_func(guess_pars, x), "r--")

    for phase in np.arange(0.0, 1.0, 0.1):
        guess_pars[3 + startidx] = phase
        if debug:
            log.debug(guess_pars)
            plt.plot(x, std_fold_fit_func(guess_pars, x), "r--")
        fit_pars, success = optimize.leastsq(
            std_residuals, guess_pars[:], args=(x, profile)
        )
        if debug:
            plt.plot(x, std_fold_fit_func(fit_pars, x), "g--")
        fit_pars[startidx : startidx + 2] = adjust_amp_phase(
            fit_pars[startidx : startidx + 2]
        )
        fit_pars[startidx + 2 : startidx + 4] = adjust_amp_phase(
            fit_pars[startidx + 2 : startidx + 4]
        )
        chisq = np.sum(
            (profile - std_fold_fit_func(fit_pars, x)) ** 2 / profile_err**2
        ) / (len(profile) - (startidx + 4))
        if debug:
            plt.plot(x, std_fold_fit_func(fit_pars, x), "b--")
        if chisq < chisq_save:
            chisq_save = chisq
            fit_pars_save = fit_pars[:]
            success_save = success

    if debug:
        plt.savefig("debug_fit_profile.png")
        plt.close(fig)
    return fit_pars_save, success_save, chisq_save


def fit_profile(
    profile,
    profile_err,
    debug=False,
    nperiods=1,
    phaseref="default",
    baseline=False,
):
    return fit_profile_with_sinusoids(
        profile,
        profile_err,
        debug=debug,
        nperiods=nperiods,
        baseline=baseline,
    )


def run_folding(
    file,
    freq,
    fdot=0,
    fddot=0,
    nbin=16,
    nebin=16,
    tref=None,
    test=False,
    emin=None,
    emax=None,
    norm="to1",
    smooth_window=None,
    deorbit_par=None,
    pepoch=None,
    out_file_root=None,
    colormap="cubehelix",
    **opts,
):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt

    file_label = ""
    ev = load_events(file)
    if deorbit_par is not None:
        ev = deorbit_events(ev, deorbit_par)

    plot_energy = True
    ev, elabel = filter_energy(ev, emin, emax)
    times, energy = ev.time, ev.energy
    if emin is None:
        emin = np.min(energy)
    if emax is None:
        emax = np.max(energy)

    if elabel == "":
        plot_energy = False

    if tref is not None and pepoch is not None:
        raise ValueError("Only specify one between tref and pepoch")
    elif pepoch is not None:
        tref = (pepoch - ev.mjdref) * 86400
    elif tref is None:
        tref = times[0]

    phases = pulse_phase(times - tref, freq, fdot, fddot, to_1=True)

    binx = np.linspace(0, 1, nbin + 1)
    if plot_energy:
        biny = np.percentile(energy, np.linspace(0, 100, nebin + 1))
        biny[0] = emin
        biny[-1] = emax

    profile, _ = np.histogram(phases, bins=binx)
    if smooth_window is None:
        smooth_window = np.min([len(profile), 10])
        smooth_window = _check_odd(smooth_window)

    smoothed_profile = savgol_filter(
        profile, window_length=smooth_window, polyorder=3, mode="wrap"
    )

    profile = np.concatenate((profile, profile))
    smooth = np.concatenate((smoothed_profile, smoothed_profile))

    if plot_energy:
        histen, _ = np.histogram(energy, bins=biny)

        hist2d, _, _ = np.histogram2d(
            phases.astype(np.float64), energy, bins=(binx, biny)
        )

    binx = np.concatenate((binx[:-1], binx + 1))
    meanbins = (binx[:-1] + binx[1:]) / 2

    if plot_energy:
        hist2d = np.vstack((hist2d, hist2d))
        hist2d_save = np.copy(hist2d)
        X, Y = np.meshgrid(binx, biny)
        hist2d = normalize_dyn_profile(hist2d.T, norm).T
        if norm is not None and norm != "":
            file_label = f"_{norm}"

    if out_file_root is None:
        out_file_root = hen_root(file)
    out_file_root = out_file_root + file_label

    plt.figure(figsize=(8, 8))
    plt.clf()
    if plot_energy:
        gs = GridSpec(2, 2, height_ratios=(1.5, 3))
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[1, 0], sharex=ax0)
        ax2 = plt.subplot(gs[1, 1], sharex=ax0)
        ax3 = plt.subplot(gs[0, 1])

    else:
        ax0 = plt.subplot()

    # Plot pulse profile
    max = np.max(smooth)
    min = np.min(smooth)
    ax0.plot(meanbins, profile, drawstyle="steps-mid", color="white", zorder=2)
    ax0.plot(
        meanbins,
        smooth,
        drawstyle="steps-mid",
        label="Smooth profile " "(P.F. = {:.1f}%)".format(100 * (max - min) / max),
        color="k",
        zorder=3,
    )
    err_low, err_high = poisson_conf_interval(
        smooth, interval="frequentist-confidence", sigma=3
    )

    try:
        ax0.fill_between(
            meanbins,
            err_low,
            err_high,
            color="grey",
            zorder=1,
            alpha=0.5,
            label="3-sigma confidence",
            step="mid",
        )
    except AttributeError:
        # MPL < 2
        ax0.fill_between(
            meanbins,
            err_low,
            err_high,
            color="grey",
            zorder=1,
            alpha=0.5,
            label="3-sigma confidence",
        )

    ax0.axhline(max, lw=1, color="k")
    ax0.axhline(min, lw=1, color="k")

    mean = np.mean(profile)
    ax0.fill_between(meanbins, mean - np.sqrt(mean), mean + np.sqrt(mean), alpha=0.5)
    ax0.axhline(mean, ls="--")
    ax0.legend()
    ax0.set_ylim([0, None])
    ax0.set_ylabel("Counts")
    ax0.set_xlabel("Phase")

    if plot_energy:
        ax1.pcolormesh(X, Y, hist2d.T, cmap=colormap)
        ax1.semilogy()

        ax1.set_xlabel("Phase")
        ax1.set_ylabel(elabel)
        ax1.set_xlim([0, 2])

        pfs = []
        errs = []
        meannrgs = (biny[:-1] + biny[1:]) / 2
        for i, prof in enumerate(hist2d_save.T):
            smooth = savgol_filter(
                prof, window_length=smooth_window, polyorder=3, mode="wrap"
            )
            mean = np.mean(smooth)
            shift = 3 * np.sqrt(mean)
            max = np.max(smooth)
            min = np.min(smooth)
            pf = 100 * (max - min) / max
            ax2.plot(
                meanbins,
                prof - mean + i * shift,
                drawstyle="steps-mid",
                alpha=0.5,
                color="k",
            )
            ax2.plot(
                meanbins,
                smooth - mean + i * shift,
                label="{}={:.2f}-{:.2f}".format(elabel, biny[i], biny[i + 1]),
            )
            std = np.std(prof - smooth)
            pfs.append(pf)
            errs.append(100 * std / max)
        ax2.set_xlabel("Phase")
        ax2.set_ylabel("Counts (shifted arbitrarily)")

        if len(meannrgs) < 6:
            ax2.legend()
        ax2.set_xlim([0, 2])

        ax3.errorbar(meannrgs, pfs, fmt="o", yerr=errs, xerr=(biny[1:] - biny[:-1]) / 2)
        from astropy.table import Table

        pf_results = Table(
            data=[meannrgs, (biny[1:] - biny[:-1]) / 2, pfs, errs],
            names=["E", "Ee", "pf", "pfe"],
        )
        pf_results.write(out_file_root + ".csv", overwrite=True)
        ax3.semilogx()
        # labels = [float(item.get_text()) for item in ax3.get_xticklabels() if item.get_text()!='']
        # ax3.set_xticklabels([f"{label:g}" for label in labels])
        ax3.set_xlabel("Energy")
        ax3.set_ylabel("Pulsed fraction")

    plt.tight_layout()
    plt.savefig(out_file_root + ".png")
    if not test:  # pragma:no cover
        plt.show()


def main_fold(args=None):
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Plot a folded profile"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Input event file", type=str)
    parser.add_argument(
        "-f",
        "--freq",
        type=float,
        required=False,
        help="Initial frequency to fold",
        default=None,
    )
    parser.add_argument(
        "--fdot", type=float, required=False, help="Initial fdot", default=0
    )
    parser.add_argument(
        "--fddot", type=float, required=False, help="Initial fddot", default=0
    )
    parser.add_argument(
        "--tref",
        type=float,
        required=False,
        help="Reference time (same unit as time array)",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--nbin",
        default=16,
        type=int,
        help="Number of phase bins (X axis) of the profile",
    )
    parser.add_argument(
        "--nebin",
        default=16,
        type=int,
        help="Number of energy bins (Y axis) of the profile",
    )
    parser.add_argument(
        "--emin",
        default=None,
        type=float,
        help="Minimum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--emax",
        default=None,
        type=float,
        help="Maximum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--out-file-root",
        default=None,
        help="Root of the output files (plots and csv tables)",
    )

    _add_default_args(
        parser,
        [
            "pepoch",
            "dynprofnorm",
            "colormap",
            "deorbit",
            "loglevel",
            "debug",
            "test",
        ],
    )

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    with log.log_to_file("HENfold.log"):
        frequency = args.freq
        fdot = args.fdot
        fddot = args.fddot

        run_folding(
            args.file,
            freq=frequency,
            fdot=fdot,
            fddot=fddot,
            nbin=args.nbin,
            nebin=args.nebin,
            tref=args.tref,
            test=args.test,
            emin=args.emin,
            emax=args.emax,
            norm=args.norm,
            deorbit_par=args.deorbit_par,
            pepoch=args.pepoch,
            out_file_root=args.out_file_root,
            colormap=args.colormap,
        )


def main_deorbit(args=None):
    import argparse
    from .base import hen_root, _add_default_args
    from .io import HEN_FILE_EXTENSION, load_events, save_events

    description = "Deorbit the event arrival times"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="Input event file", type=str, nargs="+")

    _add_default_args(parser, ["deorbit", "loglevel", "debug"])
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    with log.log_to_file("HENdeorbit.log"):
        for fname in args.files:
            log.info(f"Deorbiting events from {fname}")
            events = load_events(fname)
            events = deorbit_events(events, parameter_file=args.deorbit_par)
            outfile = hen_root(fname) + "_deorb" + HEN_FILE_EXTENSION

            save_events(events, outfile)
            log.info(f"Saved to {outfile}")
