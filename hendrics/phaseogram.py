"""Interactive phaseogram."""

import copy
import argparse
import warnings
from abc import abstractmethod

from scipy.interpolate import interp1d
import numpy as np
from astropy import log
from astropy.logger import AstropyUserWarning
from astropy.stats import poisson_conf_interval
from stingray.pulse.search import phaseogram
from stingray.utils import assign_value_if_none
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

from .base import normalize_dyn_profile
from .io import load_events, load_folding
from .fold import filter_energy
from .fold import get_TOAs_from_events
from .fold import create_default_template
from .base import hen_root, deorbit_events, get_model
from .efsearch import h_test
from .ml_timing import normalized_template, ml_pulsefit


DEFAULT_COLORMAP = "cubehelix"


def get_H_label(phas, prof):
    good = phas < 1
    nmax = max(len(prof[good]) // 3, 1)
    prof_h, prof_m = h_test(prof[good], nmax=nmax)
    # z2_detlev = z2_n_detection_level(n=2)
    h_label = f"H test = {prof_h:.1f} (M={prof_m})"
    return h_label


def sec_to_mjd(time, mjdref):
    return time / 86400 + mjdref


class SliderOnSteroids(Slider):
    def __init__(self, *args, **kwargs):
        self.hardvalmin = None
        self.hardvalmax = None
        if "hardvalmin" in kwargs:
            self.hardvalmin = kwargs["hardvalmin"]
            kwargs.pop("hardvalmin")
        if "hardvalmax" in kwargs:
            self.hardvalmax = kwargs["hardvalmax"]
            kwargs.pop("hardvalmax")
        Slider.__init__(self, *args, **kwargs)


def normalized_phaseogram(norm, *args, **kwargs):
    phas, phases, times, additional_info = phaseogram(*args, **kwargs)
    phas = normalize_dyn_profile(phas.T, norm).T
    return phas, phases, times, additional_info


class BasePhaseogram(object):
    def __init__(
        self,
        ev_times,
        freq,
        nph=128,
        nt=128,
        test=False,
        fdot=0,
        fddot=0,
        mjdref=None,
        pepoch=None,
        gti=None,
        label="phaseogram",
        norm=None,
        position=None,
        object=None,
        plot_only=False,
        time_corr=None,
        model=None,
        colormap=DEFAULT_COLORMAP,
        **kwargs,
    ):
        """Init BasePhaseogram class.

        Parameters
        ----------
        ev_times : array-like
            Event times
        freq : float
            Frequency of pulsation

        Other parameters
        ----------------
        nph : int
            Number of phase bins in the profile
        nt : int
            Number of time bins in the profile
        pepoch : float, default None
            Epoch of timing solution, in the same units as ev_times
        mjdref : float, default None
            Reference MJD
        fdot : float
            First frequency derivative
        fddot : float
            Second frequency derivative
        label : str
            Label for windows
        norm : str
            Normalization
        position : `astropy.Skycoord` object
            Position of the pulsar
        object : str
            Name of the pulsar
        **kwargs : keyword args
            additional arguments to pass to `self._construct_widgets`
        """
        self.model = model
        self.colormap = colormap
        self.fdot = fdot
        self.fddot = fddot
        self.nt = nt
        self.nph = nph
        if mjdref is None:
            warnings.warn("MJDREF not set. All MJD values will be incorrect.")
        self.mjdref = assign_value_if_none(mjdref, 0.0)
        self.ev_times = np.asarray(ev_times)
        self.gti = gti
        if gti is None:
            self.gti = np.array([[ev_times[0], ev_times[-1]]])
        self.label = label
        self.test = test

        self.pepoch = assign_value_if_none(pepoch, ev_times[0])

        self.time_corr = assign_value_if_none(time_corr, np.zeros_like(ev_times))
        self.freq = freq
        self.norm = norm
        self.position = position
        self.object = object
        self.timing_model_string = ""

        self.time_corr_fun = interp1d(
            self.ev_times,
            self.time_corr,
            bounds_error=False,
            fill_value="extrapolate",
        )
        self.time_corr_mjd_fun = interp1d(
            self.ev_times / 86400 + self.mjdref,
            self.time_corr / 86400,
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.fig = plt.figure(label, figsize=(6, 8))
        plt.clf()
        gs = GridSpec(3, 1, height_ratios=[2, 3, 0.2])
        plt.subplots_adjust(left=0.1, bottom=0.30, top=0.95, right=0.95)
        ax = plt.subplot(gs[1])
        self.profax = plt.subplot(gs[0], sharex=ax)
        self.profax.set_xticks([])
        colorbax = plt.subplot(gs[2])

        corrected_times = self.ev_times - self._delay_fun(self.ev_times)
        (
            self.unnorm_phaseogr,
            phases,
            times,
            additional_info,
        ) = normalized_phaseogram(
            None,
            corrected_times,
            freq,
            return_plot=True,
            nph=nph,
            nt=nt,
            fdot=fdot,
            fddot=fddot,
            plot=False,
            pepoch=self.pepoch,
        )

        self.phaseogr, phases, times, additional_info = normalized_phaseogram(
            self.norm,
            corrected_times,
            freq,
            return_plot=True,
            nph=nph,
            nt=nt,
            fdot=fdot,
            fddot=fddot,
            plot=False,
            pepoch=self.pepoch,
        )

        self.phases, self.times = phases, times
        vmin = None

        self.pcolor = ax.pcolormesh(
            phases, times, self.phaseogr.T, cmap=self.colormap, vmin=vmin
        )
        self.colorbar = plt.colorbar(
            self.pcolor, cax=colorbax, orientation="horizontal"
        )
        ax.set_xlabel("Phase")

        def s2d(x):
            return (x - self.pepoch) / 86400

        def d2s(x):
            return (x - self.mjdref) * 86400

        secax = ax.secondary_yaxis("left", functions=(s2d, d2s))
        secax.set_ylabel(f"d from MJD {self.pepoch / 86400 + self.mjdref}")

        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_yticks([])
        # plt.setp(ax.get_yticks(), visible=False)

        # ax.set_ylabel('MJD')
        # plt.colorbar()
        self.lines = []
        self.line_phases = np.arange(-2, 3, 0.5)
        for ph0 in self.line_phases:
            (newline,) = ax.plot(
                np.zeros_like(times) + ph0, times, zorder=10, lw=2, color="w"
            )
            self.lines.append(newline)

        ax.set_xlim([0, 2])

        axcolor = "#ff8888"
        self.slider_axes = []

        def newax_fn(*args, **kwargs):
            ax = self.fig.add_axes(*args, facecolor=axcolor)
            return ax

        self.slider_axes.append(newax_fn([0.15, 0.1, 0.75, 0.03], facecolor=axcolor))
        self.slider_axes.append(newax_fn([0.15, 0.15, 0.75, 0.03], facecolor=axcolor))
        self.slider_axes.append(newax_fn([0.15, 0.2, 0.75, 0.03], facecolor=axcolor))

        self._construct_widgets(**kwargs)

        self.closeax = self.fig.add_axes([0.15, 0.020, 0.15, 0.04])
        self.button_close = Button(
            self.closeax, "Quit", color=axcolor, hovercolor="0.8"
        )

        self.recalcax = self.fig.add_axes([0.3, 0.020, 0.15, 0.04])
        self.button_recalc = Button(
            self.recalcax, "Recalculate", color=axcolor, hovercolor="0.975"
        )

        self.resetax = self.fig.add_axes([0.45, 0.020, 0.15, 0.04])
        self.button_reset = Button(
            self.resetax, "Reset", color=axcolor, hovercolor="0.975"
        )

        self.zoominax = self.fig.add_axes([0.6, 0.020, 0.1, 0.04])
        self.button_zoomin = Button(
            self.zoominax, "+Zoom", color=axcolor, hovercolor="0.975"
        )

        self.zoomoutax = self.fig.add_axes([0.7, 0.020, 0.1, 0.04])
        self.button_zoomout = Button(
            self.zoomoutax, "-Zoom", color=axcolor, hovercolor="0.975"
        )

        self.toaax = self.fig.add_axes([0.8, 0.020, 0.1, 0.04])
        self.button_toa = Button(self.toaax, "TOA", color=axcolor, hovercolor="0.975")

        self.button_reset.on_clicked(self.reset)
        self.button_zoomin.on_clicked(self.zoom_in)
        self.button_zoomout.on_clicked(self.zoom_out)
        self.button_toa.on_clicked(self.toa)
        self.button_recalc.on_clicked(self.recalculate)
        self.button_close.on_clicked(self.quit)
        # self.profax = plt.axes([0.25, 0.75, 0.5, 0.2])

        prof = np.sum(np.nan_to_num(self.unnorm_phaseogr), axis=1)
        nbin = len(prof)
        phas = np.linspace(0, 2, nbin + 1)[:-1]
        (self.profile_fixed,) = self.profax.plot(
            phas, prof, drawstyle="steps-post", color="grey"
        )
        (self.profile,) = self.profax.plot(
            phas, prof, drawstyle="steps-post", color="k"
        )
        mean = np.mean(prof)
        low, high = poisson_conf_interval(
            mean, interval="frequentist-confidence", sigma=2
        )
        self.profax.fill_between(phas, low, high, alpha=0.5)
        z2_label = get_H_label(phas, prof)
        self.proftext = self.profax.text(
            0.1, 0.8, z2_label, transform=self.profax.transAxes
        )
        if not test and not plot_only:
            plt.show()
        if plot_only:
            plt.savefig(self.label + "_{:.10f}Hz.png".format(self.freq))

    @abstractmethod
    def _construct_widgets(self, **kwargs):  # pragma: no cover
        pass

    @abstractmethod
    def update(self, val):  # pragma: no cover
        pass

    @abstractmethod
    def recalculate(self, event):  # pragma: no cover
        pass

    def toa(self, event):  # pragma: no cover
        warnings.warn(
            "This function was not implemented for this Phaseogram. "
            "Try the basic one.",
            AstropyUserWarning,
        )

    def reset(self, event):
        for s in self.sliders:
            s.reset()
        self.pcolor.set_array(self.phaseogr.T.ravel())
        self._set_lines(False)
        prof = np.sum(np.nan_to_num(self.unnorm_phaseogr), axis=1)
        ph = np.linspace(0, 2, len(prof) + 1)[:-1]
        self.profile.set_ydata(prof)
        self.proftext.set_text(get_H_label(ph, prof))

    def zoom_in(self, event):
        for s in self.sliders:
            valinit = s.val
            valrange = s.valmax - s.valmin
            valmin = valinit - valrange / 4
            if s.hardvalmin is not None:
                valmin = max(s.hardvalmin, valmin)
            valmax = valinit + valrange / 4
            if s.hardvalmax is not None:
                valmax = min(s.hardvalmax, valmax)
            label = s.label.get_text()
            ax = s.ax
            ax.clear()

            s.__init__(
                ax,
                label,
                valmin=valmin,
                valmax=valmax,
                valinit=valinit,
                hardvalmax=s.hardvalmax,
                hardvalmin=s.hardvalmin,
            )
            ax.text(
                0,
                0,
                str(valmin),
                transform=ax.transAxes,
                horizontalalignment="left",
                color="white",
            )
            ax.text(
                1,
                0,
                str(valmax),
                transform=ax.transAxes,
                horizontalalignment="right",
                color="white",
            )
            s.on_changed(self.update)

    def zoom_out(self, event):
        for s in self.sliders:
            valinit = s.val
            valrange = s.valmax - s.valmin
            valmin = valinit - valrange
            if s.hardvalmin is not None:
                valmin = max(s.hardvalmin, valmin)
            valmax = valinit + valrange
            if s.hardvalmax is not None:
                valmax = min(s.hardvalmax, valmax)
            label = s.label.get_text()
            ax = s.ax
            ax.clear()

            s.__init__(
                ax,
                label,
                valmin=valmin,
                valmax=valmax,
                valinit=valinit,
                hardvalmax=s.hardvalmax,
                hardvalmin=s.hardvalmin,
            )
            ax.text(
                0,
                0,
                str(valmin),
                transform=ax.transAxes,
                horizontalalignment="left",
                color="white",
            )
            ax.text(
                1,
                0,
                str(valmax),
                transform=ax.transAxes,
                horizontalalignment="right",
                color="white",
            )
            s.on_changed(self.update)

    @abstractmethod
    def quit(self, event):  # pragma: no cover
        pass

    @abstractmethod
    def get_values(self):  # pragma: no cover
        pass

    @abstractmethod
    def _line_delay_fun(self, times):  # pragma: no cover
        pass

    @abstractmethod
    def _delay_fun(self, times):  # pragma: no cover
        """This is the delay function _without_ frequency derivatives."""
        pass

    @abstractmethod
    def _read_sliders(self):  # pragma: no cover
        pass

    def _set_lines(self, apply_delay=True):
        if apply_delay:
            func = self._line_delay_fun
        else:

            def func(x):
                return np.zeros_like(self.times)

        for i, ph0 in enumerate(self.line_phases):
            linephase = ph0 + func(self.times) - func(self.times[0])
            self.lines[i].set_xdata(linephase)

    def update_pint_model(self):
        self.model.PEPOCH.value = self.pepoch / 86400 + self.mjdref
        self.model.F0.value = self.freq
        if hasattr(self.model, "F1"):
            self.model.F1.value = self.fdot
        else:
            warnings.warn("Parameter F1 not in parfile. It will not be updated")
        if hasattr(self.model, "F2"):
            self.model.F2.value = self.fddot
        else:
            warnings.warn("Parameter F2 not in parfile. It will not be updated")
        start, stop = self.gti.min(), self.gti.max()
        self.model.START.value = start / 86400 + self.mjdref
        self.model.FINISH.value = stop / 86400 + self.mjdref
        if hasattr(self, "orbital_period") and self.orbital_period is not None:
            self.model.PB.value = self.orbital_period / 86400
            self.model.A1.value = self.asini
            tasc = self.t0 / 86400 + self.mjdref
            if hasattr(self.model, "T0"):
                self.model.T0.value = tasc
            if hasattr(self.model, "TASC"):
                self.model.TASC.value = tasc
        return self.model.as_parfile()

    def get_timing_model_string(self):
        if hasattr(self, "model") and self.model is not None:
            return self.update_pint_model()

        tm_string = ""

        if self.mjdref is not None:
            tm_string += "PEPOCH         {}\n".format(self.pepoch / 86400 + self.mjdref)
        tm_string += "PSRJ           {}\n".format(self.object)
        if self.position is not None:
            tm_string += "RAJ            {}\n".format(
                self.position.ra.to_string("hour", sep=":")
            )
            tm_string += "DECJ           {}\n".format(
                self.position.dec.to_string(sep=":")
            )

        tm_string += "F0             {}\n".format(self.freq)
        tm_string += "F1             {}\n".format(self.fdot)
        tm_string += "F2             {}\n".format(self.fddot)

        if hasattr(self, "orbital_period") and self.orbital_period is not None:
            tm_string += "BINARY BT\n"
            tm_string += "PB             {}\n".format(self.orbital_period / 86400)
            tm_string += "A1             {}\n".format(self.asini)
            if self.mjdref is not None:
                tm_string += "T0             {}\n".format(self.t0 / 86400 + self.mjdref)
            tm_string += "T0(MET)        {}\n".format(self.t0)
            tm_string += "PB(s)          {}\n".format(self.orbital_period)

        tm_string += "# PEPOCH(MET)  {}\n".format(self.pepoch)
        start, stop = self.gti.min(), self.gti.max()
        tm_string += "START         {}\n".format(start / 86400 + self.mjdref)
        tm_string += "FINISH        {}\n".format(stop / 86400 + self.mjdref)
        return tm_string


class InteractivePhaseogram(BasePhaseogram):
    def _construct_widgets(self):
        self.df = 0
        self.dfdot = 0
        self.dfddot = 0

        tseg = np.median(np.diff(self.times))
        tobs = tseg * self.nt
        delta_df_start = 4 / tobs
        self.df_order_of_mag = int(np.log10(delta_df_start))
        delta_df = delta_df_start / 10**self.df_order_of_mag

        delta_dfdot_start = 8 / tobs**2
        self.dfdot_order_of_mag = int(np.log10(delta_dfdot_start))
        delta_dfdot = delta_dfdot_start / 10**self.dfdot_order_of_mag

        delta_dfddot_start = 16 / tobs**3
        self.dfddot_order_of_mag = int(np.log10(delta_dfddot_start))
        delta_dfddot = delta_dfddot_start / 10**self.dfddot_order_of_mag

        freq_str = r"$\Delta$ F0" "x$10^{" + f"{self.df_order_of_mag}" + r"}$"
        fdot_str = r"$\Delta$ F1" r"x$10^{" + f"{self.dfdot_order_of_mag}" + r"}$"
        fddot_str = r"$\Delta$ F2" r"x$10^{" + f"{self.dfddot_order_of_mag}" + r"}$"

        self.sfreq = SliderOnSteroids(
            self.slider_axes[0],
            freq_str,
            -delta_df,
            delta_df,
            valinit=self.df,
            hardvalmin=0,
        )
        self.sfdot = SliderOnSteroids(
            self.slider_axes[1],
            fdot_str,
            -delta_dfdot,
            delta_dfdot,
            valinit=self.dfdot,
        )

        self.sfddot = SliderOnSteroids(
            self.slider_axes[2],
            fddot_str,
            -delta_dfddot,
            delta_dfddot,
            valinit=self.dfddot,
        )

        self.sfreq.on_changed(self.update)
        self.sfdot.on_changed(self.update)
        self.sfddot.on_changed(self.update)
        self.sliders = [self.sfreq, self.sfdot, self.sfddot]

    def update(self, val):
        self._set_lines()
        self.fig.canvas.draw_idle()

    def _read_sliders(self):
        fddot = self.sfddot.val * 10**self.dfddot_order_of_mag
        fdot = self.sfdot.val * 10**self.dfdot_order_of_mag
        freq = self.sfreq.val * 10**self.df_order_of_mag
        return freq, fdot, fddot

    def _line_delay_fun(self, times):
        freq, fdot, fddot = self._read_sliders()
        return (
            (times - self.pepoch).astype(np.float64) * freq
            + 0.5 * (times - self.pepoch) ** 2 * fdot
            + 1 / 6 * (times - self.pepoch) ** 3 * fddot
        )

    def _delay_fun(self, times):
        """This is the delay function _without_ frequency derivatives."""
        return 0

    def recalculate(self, event):
        dfreq, dfdot, dfddot = self._read_sliders()
        pepoch = self.pepoch

        self.fddot = self.fddot - dfddot
        self.fdot = self.fdot - dfdot
        self.freq = self.freq - dfreq

        self.unnorm_phaseogr, _, _, _ = normalized_phaseogram(
            None,
            self.ev_times,
            self.freq,
            fdot=self.fdot,
            plot=False,
            nph=self.nph,
            nt=self.nt,
            pepoch=pepoch,
            fddot=self.fddot,
        )
        self.phaseogr, _, _, _ = normalized_phaseogram(
            self.norm,
            self.ev_times,
            self.freq,
            fdot=self.fdot,
            plot=False,
            nph=self.nph,
            nt=self.nt,
            pepoch=pepoch,
            fddot=self.fddot,
        )

        self.reset(1)

        self.fig.canvas.draw()
        self.timing_model_string = self.get_timing_model_string()
        print("------------------------")
        print(self.timing_model_string)
        print("------------------------")

    def toa(self, event):
        dfreq, dfdot, dfddot = self._read_sliders()
        freqs = [self.freq - dfreq, self.fdot - dfdot, self.fddot - dfddot]
        folding_length = np.median(np.diff(self.times))
        nbin = self.nph

        corrected_times = self.ev_times - self._delay_fun(self.ev_times)
        raw_times = corrected_times - self.pepoch
        raw_phases = (
            raw_times * self.freq
            + 0.5 * raw_times**2 * self.fdot
            + 1 / 6 * raw_times**2 * self.fddot
        )
        raw_phases -= np.floor(raw_phases)

        template_raw, _ = np.histogram(raw_phases, bins=self.nph * 4)

        template, additional = create_default_template(template_raw)

        template = normalized_template(template, tomax=True, subtract_min=False)

        pars, errs = ml_pulsefit(template_raw, template, calculate_errors=True)
        ph, phe = pars[1], errs[1]
        if ph is None:
            warnings.warn("The pulse profile is not adequate for TOA fitting.")
            return
        toa = (ph + additional) / freqs[0] + self.times[0]
        toaerr = phe / freqs[0]
        full_toa = toa / 86400 + self.mjdref
        full_toaerr = toaerr * 1e6
        print(full_toa, full_toaerr)

        # template = np.roll(template, -np.argmax(template)) / self.nt
        ephem = "DE421"
        if self.model is not None and hasattr(self.model, "EPHEM"):
            ephem = self.model.EPHEM.value

        toa, toaerr = get_TOAs_from_events(
            self.ev_times,
            folding_length,
            *freqs,
            gti=self.gti,
            template=template,
            mjdref=self.mjdref,
            nbin=self.nph,
            pepoch=self.pepoch,
            timfile=self.label + ".tim",
            label=self.label[:10],
            quick=self.test,
            position=None,
            ephem=ephem,
        )
        if toa is None:
            warnings.warn("No valid TOAs found")
            return
        toa_corr = toa + self.time_corr_mjd_fun(toa)
        full_toa_corr = full_toa + self.time_corr_mjd_fun(full_toa)

        corr_string = ""

        if np.any(toa_corr != toa):
            corr_string = "_corr"

        with open(self.label + corr_string + ".tim", "w") as fobj:
            print("FORMAT 1", file=fobj)
            for t, te in zip(toa_corr, toaerr):
                print(self.label, 0, t, te, "@", file=fobj)

        if hasattr(self, "model"):
            if hasattr(self.model, "TZRMJD"):
                self.model.TZRMJD.value = full_toa_corr
                self.model.TZRSITE.value = "@"

        self.timing_model_string = self.get_timing_model_string()

        with open(self.label + ".par", "w") as fobj:
            print(self.timing_model_string, file=fobj)

    def quit(self, event):
        plt.close("all")

    def get_values(self):
        return self.freq, self.fdot, self.fddot


class BinaryPhaseogram(BasePhaseogram):
    def __init__(self, *args, **kwargs):
        """Init BinaryPhaseogram class.

        Parameters
        ----------
        ev_times : array-like
            Event times
        freq : float
            Frequency of pulsation

        Other parameters
        ----------------
        orbital_period : float
            orbital period in seconds
        asini : float
            projected semi-major axis in light-sec
        T0 : float
            passage through the ascending node, in seconds.
        **kwargs : keyword args
            additional arguments to pass to `BasePhaseogram`
        """
        self.orbital_period = None
        self.asini = 0
        self.t0 = None

        try:
            self.orbital_period = kwargs.pop("orbital_period")
            self.asini = kwargs.pop("asini")
            self.t0 = kwargs.pop("t0")
        except KeyError:
            raise RuntimeError("Please specify all binary parameters")

        BasePhaseogram.__init__(self, *args, **kwargs)

    def _construct_widgets(self):
        self.dperiod = 0
        self.dasini = 0
        self.dt0 = 0

        tseg = np.median(np.diff(self.times))
        tobs = tseg * self.nt

        delta_period = tobs * 5
        delta_asini = 5 / self.freq
        delta_t0 = delta_period

        self.speriod = SliderOnSteroids(
            self.slider_axes[0],
            "PB (s)",
            np.max([0, self.orbital_period - delta_period]),
            self.orbital_period + delta_period,
            valinit=self.orbital_period,
            hardvalmin=0,
        )

        self.sasini = SliderOnSteroids(
            self.slider_axes[1],
            "A1 (l-s)",
            0,
            self.asini + delta_asini,
            valinit=self.asini,
            hardvalmin=0,
        )

        self.st0 = SliderOnSteroids(
            self.slider_axes[2],
            "T0 (MET)",
            self.t0 - delta_t0,
            self.t0 + delta_t0,
            valinit=self.t0,
            hardvalmin=0,
        )

        self.speriod.on_changed(self.update)
        self.sasini.on_changed(self.update)
        self.st0.on_changed(self.update)
        self.sliders = [self.speriod, self.sasini, self.st0]

    def update(self, val):
        self._set_lines()
        self.fig.canvas.draw_idle()

    def _read_sliders(self):
        return self.speriod.val, self.sasini.val, self.st0.val

    def _line_delay_fun(self, times):
        orbital_period, asini, t0 = self._read_sliders()

        new_values = asini * np.sin(2 * np.pi * (times - t0) / orbital_period)
        old_values = self.asini * np.sin(
            2 * np.pi * (times - self.t0) / (self.orbital_period)
        )
        return (new_values - old_values) * self.freq

    def _delay_fun(self, times):
        if self.t0 is None:
            self.t0 = self.pepoch
        if self.orbital_period is None:
            self.orbital_period = self.ev_times[-1] - self.ev_times[0]

        return self.asini * np.sin(2 * np.pi * (times - self.t0) / self.orbital_period)

    def recalculate(self, event):
        self.orbital_period, self.asini, self.t0 = self._read_sliders()

        corrected_times = self.ev_times - self._delay_fun(self.ev_times)

        self.unnorm_phaseogr, _, _, _ = normalized_phaseogram(
            None,
            corrected_times,
            self.freq,
            fdot=self.fdot,
            plot=False,
            nph=self.nph,
            nt=self.nt,
            pepoch=self.pepoch,
            fddot=self.fddot,
        )

        self.phaseogr, _, _, _ = normalized_phaseogram(
            self.norm,
            corrected_times,
            self.freq,
            fdot=self.fdot,
            plot=False,
            nph=self.nph,
            nt=self.nt,
            pepoch=self.pepoch,
            fddot=self.fddot,
        )

        self.reset(event)
        self._set_lines(False)
        self.pcolor.set_array(self.phaseogr.T.ravel())
        self.sasini.valinit = self.asini
        self.speriod.valinit = self.orbital_period
        self.st0.valinit = self.t0
        self.st0.valmin = self.t0 - self.orbital_period
        self.st0.valmax = self.t0 + self.orbital_period
        self.fig.canvas.draw()

        self.timing_model_string = self.get_timing_model_string()
        print("------------------------")
        print(self.timing_model_string)
        print("------------------------")

    def quit(self, event):
        plt.close(self.fig)

    def get_values(self):
        return self.orbital_period, self.asini, self.t0


def run_interactive_phaseogram(
    event_file,
    freq,
    fdot=0,
    fddot=0,
    nbin=64,
    nt=32,
    binary=False,
    test=False,
    binary_parameters=[None, 0, None],
    pepoch=None,
    norm=None,
    plot_only=False,
    deorbit_par=None,
    emin=None,
    emax=None,
    colormap=DEFAULT_COLORMAP,
):
    from astropy.io.fits import Header
    from astropy.coordinates import SkyCoord

    events = load_events(event_file)
    if emin is not None or emax is not None:
        events, elabel = filter_energy(events, emin, emax)

    position = name = None

    if hasattr(events, "header") and events.header is not None:
        header = Header.fromstring(events.header)

        try:
            position = SkyCoord(
                header["RA_OBJ"],
                header["DEC_OBJ"],
                unit="deg",
                frame=header["RADECSYS"].lower(),
            )
        except (KeyError, AttributeError, ValueError):
            position = None

        if "OBJECT" in header:
            name = header["OBJECT"]

    pepoch_mjd = pepoch
    if pepoch is None:
        pepoch = events.gti[0, 0]
        # pepoch_mjd = pepoch / 86400 + events.mjdref
    else:
        pepoch = (pepoch_mjd - events.mjdref) * 86400

    model = None
    if deorbit_par is not None:
        model = get_model(deorbit_par)

    if binary:
        if binary_parameters[0] is None and model is not None:
            log.info("Reading binary parameters from parameter file")
            pb = model.PB.value * 86400
            a1 = model.A1.value
            key = "T0" if hasattr(model, "T0") else "TASC"
            t0 = (getattr(model, key).value - events.mjdref) * 86400
            binary_parameters = [pb, a1, t0]

        ip = BinaryPhaseogram(
            events.time,
            freq,
            nph=nbin,
            nt=nt,
            fdot=fdot,
            test=test,
            fddot=fddot,
            pepoch=pepoch,
            orbital_period=binary_parameters[0],
            asini=binary_parameters[1],
            t0=binary_parameters[2],
            mjdref=events.mjdref,
            gti=events.gti,
            label=hen_root(event_file),
            norm=norm,
            object=name,
            position=position,
            plot_only=plot_only,
            model=model,
            colormap=colormap,
        )
    else:
        events_save = copy.deepcopy(events)
        model = None
        if deorbit_par is not None:
            events = deorbit_events(events, deorbit_par)
            model = get_model(deorbit_par)

        ip = InteractivePhaseogram(
            events.time,
            freq,
            nph=nbin,
            nt=nt,
            fdot=fdot,
            test=test,
            fddot=fddot,
            pepoch=pepoch,
            mjdref=events.mjdref,
            gti=events.gti,
            label=hen_root(event_file),
            norm=norm,
            object=name,
            position=position,
            plot_only=plot_only,
            time_corr=events_save.time - events.time,
            model=model,
            colormap=colormap,
        )

    return ip


def main_phaseogram(args=None):
    description = "Plot an interactive phaseogram"
    from .base import _add_default_args, check_negative_numbers_in_args

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
        "--periodogram",
        type=str,
        required=False,
        help="Periodogram file",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--nbin",
        default=128,
        type=int,
        help="Number of phase bins (X axis) of the profile",
    )
    parser.add_argument(
        "--ntimes",
        default=64,
        type=int,
        help="Number of time bins (Y axis) of the phaseogram",
    )
    parser.add_argument(
        "--binary",
        help="Interact on binary parameters " "instead of frequency derivatives",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--binary-parameters",
        help="Initial values for binary parameters",
        default=[None, 0, None],
        nargs=3,
        type=float,
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
        "--plot-only",
        help="Only plot the phaseogram",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--get-toa",
        help="Only calculate TOAs",
        default=False,
        action="store_true",
    )
    _add_default_args(
        parser,
        [
            "pepoch",
            "dynprofnorm",
            "colormap",
            "deorbit",
            "test",
            "loglevel",
            "debug",
        ],
    )

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    with log.log_to_file("HENphaseogram.log"):
        if args.periodogram is None and args.freq is None:
            raise ValueError(
                "One of -f or --periodogram arguments MUST be " "specified"
            )
        elif args.periodogram is not None:
            periodogram = load_folding(args.periodogram)
            frequency = float(periodogram.peaks[0])
            fdot = 0
            fddot = 0
        else:
            frequency = args.freq
            fdot = args.fdot
            fddot = args.fddot

        ip = run_interactive_phaseogram(
            args.file,
            freq=frequency,
            fdot=fdot,
            fddot=fddot,
            nbin=args.nbin,
            nt=args.ntimes,
            test=args.test or args.get_toa,
            binary=args.binary,
            binary_parameters=args.binary_parameters,
            pepoch=args.pepoch,
            norm=args.norm,
            plot_only=args.plot_only,
            deorbit_par=args.deorbit_par,
            emin=args.emin,
            emax=args.emax,
            colormap=args.colormap,
        )
    if args.get_toa:
        ip.toa(1)

    plt.close(ip.fig)
