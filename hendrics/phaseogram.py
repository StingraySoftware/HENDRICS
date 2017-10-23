"""Interactive phaseogram."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import load_events, load_folding
from stingray.pulse.search import phaseogram
from stingray.utils import assign_value_if_none

import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import six
from abc import ABCMeta, abstractmethod
from matplotlib.widgets import Slider, Button


@six.add_metaclass(ABCMeta)
class BasePhaseogram(object):
    def __init__(self, ev_times, freq, nph=128, nt=128, test=False,
                 pepoch=None, fdot=0, fddot=0, **kwargs):

        self.fdot = fdot
        self.fddot = fddot
        self.nt = nt
        self.nph = nph

        self.pepoch = assign_value_if_none(pepoch, ev_times[0])
        self.ev_times = ev_times
        self.freq = freq

        self.fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.30)

        self.phaseogr, phases, times, additional_info = \
            phaseogram(ev_times, freq, return_plot=True, nph=nph, nt=nt,
                       fdot=fdot, fddot=fddot, plot=False, pepoch=pepoch)
        self.phases, self.times = phases, times

        self.pcolor = plt.pcolormesh(phases, times, self.phaseogr.T,
                                     cmap='magma')
        self.lines = []
        self.line_phases = np.arange(-2, 3, 0.5)
        for ph0 in self.line_phases:
            newline, = plt.plot(np.zeros_like(times) + ph0, times, zorder=10,
                                lw=2, color='w')
            self.lines.append(newline)

        plt.xlabel('Phase')
        plt.ylabel('Time')
        plt.colorbar()

        axcolor = 'lightgoldenrodyellow'
        self.axfreq = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
        self.axfdot = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
        self.axfddot = plt.axes([0.25, 0.2, 0.5, 0.03], facecolor=axcolor)

        self._construct_widgets(**kwargs)

        self.recalcax = plt.axes([0.4, 0.020, 0.2, 0.04])
        self.button_recalc = Button(self.recalcax, 'Recalculate',
                                    color=axcolor,
                                    hovercolor='0.975')

        self.closeax = plt.axes([0.2, 0.020, 0.2, 0.04])
        self.button_close = Button(self.closeax, 'Quit', color=axcolor,
                                   hovercolor='0.8')

        self.resetax = plt.axes([0.6, 0.020, 0.2, 0.04])
        self.button = Button(self.resetax, 'Reset', color=axcolor,
                             hovercolor='0.975')

        self.button.on_clicked(self.reset)
        self.button_recalc.on_clicked(self.recalculate)
        self.button_close.on_clicked(self.quit)

        if not test:
            plt.show()

    @abstractmethod
    def _construct_widgets(self, **kwargs):
        pass

    @abstractmethod
    def update(self, val):
        pass

    @abstractmethod
    def recalculate(self, event):
        pass

    def reset(self, event):
        for s in self.sliders:
            s.reset()
        self.pcolor.set_array(self.phaseogr.T.ravel())
        for i, ph0 in enumerate(self.line_phases):
            self.lines[i].set_xdata(ph0)

    @abstractmethod
    def quit(self):
        pass

    @abstractmethod
    def get_values(self):
        pass


class InteractivePhaseogram(BasePhaseogram):

    def _construct_widgets(self):
        self.df = 0
        self.dfdot = 0
        self.dfddot = 0

        tseg = np.median(np.diff(self.times))
        tobs = tseg * self.nt
        delta_df_start = 4 / tobs
        self.df_order_of_mag = np.int(np.log10(delta_df_start))
        delta_df = delta_df_start / 10 ** self.df_order_of_mag

        delta_dfdot_start = 8 / tobs ** 2
        self.dfdot_order_of_mag = np.int(np.log10(delta_dfdot_start))
        delta_dfdot = delta_dfdot_start / 10 ** self.dfdot_order_of_mag

        delta_dfddot_start = 16 / tobs ** 3
        self.dfddot_order_of_mag = np.int(np.log10(delta_dfddot_start))
        delta_dfddot = delta_dfddot_start / 10 ** self.dfddot_order_of_mag

        self.sfreq = Slider(self.axfreq,
                            'Delta freq x$10^{}$'.format(
                                self.df_order_of_mag),
                            -delta_df, delta_df, valinit=self.df)
        self.sfdot = Slider(self.axfdot, 'Delta fdot x$10^{}$'.format(
            self.dfdot_order_of_mag),
                            -delta_dfdot, delta_dfdot, valinit=self.dfdot)

        self.sfddot = Slider(self.axfddot, 'Delta fddot x$10^{}$'.format(
            self.dfddot_order_of_mag),
                             -delta_dfddot, delta_dfddot,
                             valinit=self.dfddot)

        self.sfreq.on_changed(self.update)
        self.sfdot.on_changed(self.update)
        self.sfddot.on_changed(self.update)
        self.sliders = [self.sfreq, self.sfdot, self.sfddot]


    def update(self, val):
        fddot = self.sfddot.val * 10 ** self.dfddot_order_of_mag
        fdot = self.sfdot.val * 10 ** self.dfdot_order_of_mag
        freq = self.sfreq.val * 10 ** self.df_order_of_mag
        pepoch = self.pepoch
        delay_fun = lambda times: (times - pepoch).astype(np.float64) * freq + \
                                   0.5 * (times - pepoch) ** 2 * fdot + \
                                   1/6 * (times - pepoch) ** 3 * fddot
        self.l1.set_xdata(0.5 + delay_fun(self.times))
        self.l2.set_xdata(1 + delay_fun(self.times))
        self.l3.set_xdata(1.5 + delay_fun(self.times))

        self.fig.canvas.draw_idle()

    def recalculate(self, event):
        dfddot = self.sfddot.val * 10 ** self.dfddot_order_of_mag
        dfdot = self.sfdot.val * 10 ** self.dfdot_order_of_mag
        dfreq = self.sfreq.val * 10 ** self.df_order_of_mag
        pepoch = self.pepoch

        self.fddot = self.fddot - dfddot
        self.fdot = self.fdot - dfdot
        self.freq = self.freq - dfreq

        self.phaseogr, _, _, _ = \
            phaseogram(self.ev_times, self.freq, fdot=self.fdot, plot=False,
                       nph=self.nph, nt=self.nt, pepoch=pepoch,
                       fddot=self.fddot)

        self.l1.set_xdata(0.5)
        self.l2.set_xdata(1)
        self.l3.set_xdata(1.5)

        self.sfreq.reset()
        self.sfdot.reset()
        self.sfddot.reset()
        # self.spepoch.reset()

        self.pcolor.set_array(self.phaseogr.T.ravel())

        self.fig.canvas.draw()
        print("------------------------")
        print("PEPOCH    {} + MJDREF".format(self.pepoch / 86400))
        print("F0        {}".format(self.freq))
        print("F1        {}".format(self.fdot))
        print("F2        {}".format(self.fddot))
        print("------------------------")

    def quit(self, event):
        plt.close(self.fig)

    def get_values(self):
        return self.freq, self.fdot, self.fddot


def run_interactive_phaseogram(event_file, freq, fdot=0, fddot=0, nbin=64,
                               nt=32, test=False):
    events = load_events(event_file)

    ip = InteractivePhaseogram(events.time, freq, nph=nbin, nt=nt, fdot=fdot,
                               test=test, fddot=fddot, pepoch=events.gti[0, 0])

    return ip


def main_phaseogram(args=None):
    description = ('Plot an interactive phaseogram')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Input event file", type=str)
    parser.add_argument("-f", "--freq", type=float, required=False,
                        help="Initial frequency to fold", default=None)
    parser.add_argument("--fdot", type=float, required=False,
                        help="Initial fdot", default=0)
    parser.add_argument("--fddot", type=float, required=False,
                        help="Initial fddot", default=0)
    parser.add_argument("--periodogram", type=str, required=False,
                        help="Periodogram file", default=None)
    parser.add_argument('-n', "--nbin", default=128, type=int,
                        help="Number of phase bins (X axis) of the profile")
    parser.add_argument("--ntimes", default=64, type=int,
                        help="Number of time bins (Y axis) of the phaseogram")
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--test",
                        help="Just a test. Destroys the window immediately",
                        default=False, action='store_true')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENefsearch.log', level=numeric_level,
                        filemode='w')

    if args.periodogram is None and args.freq is None:
        raise ValueError('One of -f or --periodogram arguments MUST be '
                         'specified')
    elif args.periodogram is not None:
        periodogram = load_folding(args.periodogram)
        frequency = float(periodogram.peaks[0])
        fdot = 0
    else:
        frequency = args.freq
        fdot = args.fdot

    ip = run_interactive_phaseogram(args.file, freq=frequency, fdot=fdot,
                                    nbin=args.nbin, nt=args.ntimes,
                                    test=args.test)
