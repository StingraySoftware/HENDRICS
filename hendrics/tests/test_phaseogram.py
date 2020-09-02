from stingray.lightcurve import Lightcurve
from stingray.events import EventList
import numpy as np
from hendrics.io import save_events, HEN_FILE_EXTENSION, load_folding
from hendrics.efsearch import main_zsearch
from hendrics.phaseogram import main_phaseogram, run_interactive_phaseogram
from hendrics.base import hen_root
from hendrics.fold import HAS_PINT
from hendrics.plot import plot_folding
import os
import pytest
import subprocess as sp
from astropy.tests.helper import remote_data


class TestPhaseogram:
    def setup_class(cls):
        cls.pulse_frequency = 1 / 0.101
        cls.tstart = 0
        cls.tend = 25.25
        cls.tseg = cls.tend - cls.tstart
        cls.dt = 0.00606
        cls.times = np.arange(cls.tstart, cls.tend, cls.dt) + cls.dt / 2
        cls.counts = 100 + 20 * np.cos(
            2 * np.pi * cls.times * cls.pulse_frequency
        )
        lc = Lightcurve(
            cls.times, cls.counts, gti=[[cls.tstart, cls.tend]], dt=cls.dt
        )
        events = EventList()
        events.simulate_times(lc)
        events.mjdref = 57000.0
        cls.event_times = events.time
        cls.dum = "events" + HEN_FILE_EXTENSION
        save_events(events, cls.dum)

        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        fits_file = os.path.join(cls.datadir, "monol_testA.evt")
        command = "HENreadevents {0}".format(fits_file)
        sp.check_call(command.split())

        cls.real_event_file = os.path.join(
            cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )

    def test_zsearch(self):
        evfile = self.dum
        main_zsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "64",
                "--fit-candidates",
                "--fit-frequency",
                str(self.pulse_frequency),
            ]
        )
        outfile = "events_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(
            efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25
        )
        # Defaults to 2 harmonics
        assert efperiod.N == 2

    def test_phaseogram_input_periodogram(self):
        evfile = self.dum
        main_phaseogram(
            [
                evfile,
                "--periodogram",
                "events_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION,
                "--test",
            ]
        )

    @pytest.mark.parametrize(
        "norm", ["to1", "mediansub", "mediannorm", "meansub", "meannorm"]
    )
    def test_phaseogram_input_norm(self, norm):
        evfile = self.dum
        main_phaseogram(
            [
                evfile,
                "--periodogram",
                "events_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION,
                "--test",
                "--norm",
                norm,
            ]
        )

    def test_phaseogram_input_norm_invalid(self):
        evfile = self.dum
        with pytest.warns(UserWarning) as record:
            main_phaseogram(
                [
                    evfile,
                    "--periodogram",
                    "events_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION,
                    "--test",
                    "--norm",
                    "arsdfajl",
                ]
            )
            assert np.any(
                [
                    "Profile normalization arsdfajl" in r.message.args[0]
                    for r in record
                ]
            )

    def test_phaseogram_input_f(self):
        evfile = self.dum
        main_phaseogram([evfile, "-f", "9.9", "--test", "--pepoch", "57000"])

    def test_phaseogram_input_real_data(self):
        evfile = self.real_event_file
        main_phaseogram([evfile, "-f", "9.9", "--test", "--pepoch", "57000"])

    @remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_phaseogram_input_f_change(self):
        evfile = self.dum
        ip = run_interactive_phaseogram(evfile, 9.9, test=True, nbin=16, nt=8)
        ip.update(1)
        ip.recalculate(1)
        ip.toa(1)
        ip.reset(1)
        ip.fdot = 2
        f, fdot, fddot = ip.get_values()
        assert fdot == 2
        assert f == 9.9
        par = hen_root(evfile) + ".par"
        assert os.path.exists(par)

    @remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_phaseogram_deorbit(self):
        evfile = self.dum

        par = hen_root(evfile) + ".par"
        with open(par, "a") as fobj:
            print("BINARY BT", file=fobj)
            print("PB  1e20", file=fobj)
            print("A1  0", file=fobj)
            print("T0  56000", file=fobj)
            print("EPHEM  DE200", file=fobj)
            print("RAJ  00:55:01", file=fobj)
            print("DECJ 12:00:40.2", file=fobj)

        ip = run_interactive_phaseogram(
            evfile, 9.9, test=True, nbin=16, nt=8, deorbit_par=par
        )
        ip.update(1)
        ip.recalculate(1)
        ip.toa(1)
        ip.reset(1)
        ip.fdot = 2
        f, fdot, fddot = ip.get_values()
        assert fdot == 2
        assert f == 9.9

    def test_phaseogram_raises(self):
        evfile = self.dum
        with pytest.raises(ValueError):
            main_phaseogram([evfile, "--test"])

    def test_phaseogram_input_periodogram_binary(self):
        evfile = self.dum
        main_phaseogram(
            [
                evfile,
                "--binary",
                "--periodogram",
                "events_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION,
                "--test",
                "--pepoch",
                "57000",
            ]
        )

    def test_phaseogram_input_f_binary(self):
        evfile = self.dum
        main_phaseogram(
            [
                evfile,
                "--binary",
                "-f",
                "9.9",
                "--test",
                "--binary-parameters",
                "10000",
                "0",
                "0",
            ]
        )

    def test_phaseogram_input_f_change_binary(self):
        evfile = self.dum
        ip = run_interactive_phaseogram(evfile, 9.9, test=True, binary=True)
        ip.update(1)
        ip.recalculate(1)
        ip.reset(1)
        ip.zoom_in(1)
        ip.zoom_out(1)
        with pytest.warns(UserWarning) as record:
            ip.toa(1)
        assert np.any(
            [
                "This function was not implemented" in r.message.args[0]
                for r in record
            ]
        )
        ip.orbital_period = 2
        orbital_period, fdot, fddot = ip.get_values()
        assert orbital_period == 2

    def test_phaseogram_raises_binary(self):
        evfile = self.dum
        with pytest.raises(ValueError):
            main_phaseogram([evfile, "--binary", "--test"])

    @classmethod
    def teardown_class(cls):
        os.unlink(cls.dum)
