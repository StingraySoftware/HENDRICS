import warnings
from astropy.io.fits import Header
from stingray.lightcurve import Lightcurve
from stingray.events import EventList
import numpy as np
from hendrics.io import save_events, HEN_FILE_EXTENSION, load_folding
from hendrics.efsearch import main_zsearch
from hendrics.phaseogram import main_phaseogram, run_interactive_phaseogram
from hendrics.phaseogram import InteractivePhaseogram, BinaryPhaseogram
from hendrics.base import hen_root
from hendrics.fold import HAS_PINT
from hendrics.plot import plot_folding
import os
import pytest
import subprocess as sp
from . import cleanup_test_dir


def create_parfile(parfile, withfX=False, withbt=False, withell1=False):
    with open(parfile, "w") as fobj:
        print("F0  9.9", file=fobj)
        if withfX:
            print("F1  1e-14", file=fobj)
            print("F2  1e-22", file=fobj)

        withorbit = withbt or withell1
        print("PEPOCH 57000", file=fobj)
        if withell1:
            print("BINARY ELL1", file=fobj)
        elif withbt:
            print("BINARY BT", file=fobj)

        if withorbit:
            print("PB  1e20", file=fobj)
            print("A1  0", file=fobj)
        if withell1:
            print("TASC  56000", file=fobj)
            print("EPS1  0", file=fobj)
            print("EPS2  0", file=fobj)
        elif withbt:
            print("T0  56000", file=fobj)

        print("EPHEM  DE200", file=fobj)
        print("RAJ  00:55:01", file=fobj)
        print("DECJ 12:00:40.2", file=fobj)


class TestPhaseogram:
    def setup_class(cls):
        cls.pulse_frequency = 1 / 0.101
        cls.tstart = 0
        cls.tend = 25.25
        cls.tseg = cls.tend - cls.tstart
        cls.dt = 0.00606
        cls.times = np.arange(cls.tstart, cls.tend, cls.dt) + cls.dt / 2
        cls.counts = 100 + 20 * np.cos(2 * np.pi * cls.times * cls.pulse_frequency)
        lc = Lightcurve(cls.times, cls.counts, gti=[[cls.tstart, cls.tend]], dt=cls.dt)
        events = EventList()
        events.simulate_times(lc)
        events.mjdref = 57000.0
        cls.event_times = events.time
        cls.dum = "events" + HEN_FILE_EXTENSION
        cls.dum_nohead = "events_nohead" + HEN_FILE_EXTENSION
        cls.dum_info = "events_info" + HEN_FILE_EXTENSION
        save_events(events, cls.dum_nohead)

        header = Header()
        events.header = header.tostring()

        save_events(events, cls.dum)
        header = Header()
        header["OBJECT"] = "BUBU"
        header["RADECSYS"] = "FK5"
        header["RA_OBJ"] = 0.4
        header["DEC_OBJ"] = 30.4
        events.header = header.tostring()
        save_events(events, cls.dum_info)

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
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25)
        # Defaults to 2 harmonics
        assert efperiod.N == 2

    @pytest.mark.parametrize("label", ("", "_info", "_nohead"))
    def test_phaseogram_input_periodogram(self, label):
        evfile = getattr(self, "dum" + label)
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

    def test_phaseogram_all_defaults(self):
        times = [0, 3.0, 4.0]
        with pytest.warns(UserWarning, match="MJDREF not set."):
            _ = InteractivePhaseogram(times, 1.0, test=True)

    def test_binary_all_no_orb(self):
        times = [0, 3.0, 4.0]
        with pytest.raises(RuntimeError, match="Please specify all binary parameters"):
            # Missing t0
            _ = BinaryPhaseogram(times, 1.0, orbital_period=3.0, asini=3.0)

    def test_phaseogram_input_norm_invalid(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="Profile normalization arsdfajl"):
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

    def test_phaseogram_input_f(self):
        evfile = self.dum
        main_phaseogram([evfile, "-f", "9.9", "--test", "--pepoch", "57000"])

    def test_phaseogram_input_real_data(self):
        evfile = self.real_event_file
        main_phaseogram([evfile, "-f", "9.9", "--test", "--pepoch", "57000"])

    @pytest.mark.remote_data
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

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    @pytest.mark.parametrize("withfX", [True, False])
    def test_phaseogram_deorbit(self, withfX):
        evfile = self.dum

        par = hen_root(evfile) + ".par"

        withell1 = withfX
        withbt = not withfX
        create_parfile(par, withfX=withfX, withell1=withell1, withbt=withbt)

        ip = run_interactive_phaseogram(
            evfile, 9.9, test=True, nbin=16, nt=8, deorbit_par=par
        )
        ip.update(1)
        with warnings.catch_warnings(record=True) as ws:
            ip.recalculate(1)
            if not withfX:
                assert np.any(["Parameter F1" in str(w.message) for w in ws])
                assert np.any(["Parameter F2" in str(w.message) for w in ws])
        ip.toa(1)

        ip.reset(1)
        ip.fdot = 2
        f, fdot, fddot = ip.get_values()
        assert fdot == 2
        assert f == 9.9
        os.unlink(par)

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
        with pytest.warns(UserWarning, match="This function was not implemented"):
            ip.toa(1)

        ip.orbital_period = 2
        orbital_period, fdot, fddot = ip.get_values()
        assert orbital_period == 2

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    @pytest.mark.parametrize("use_ell1", [True, False])
    def test_phaseogram_input_f_change_binary_deorbit(self, use_ell1):
        evfile = self.dum
        par = "orbit.par"
        create_parfile(par, withell1=use_ell1, withbt=not use_ell1)
        ip = run_interactive_phaseogram(
            evfile, 9.9, test=True, binary=True, deorbit_par=par
        )
        ip.update(1)
        with warnings.catch_warnings(record=True) as ws:
            ip.recalculate(1)
            assert np.any(["Parameter F1" in str(w.message) for w in ws])
            assert np.any(["Parameter F2" in str(w.message) for w in ws])

        ip.reset(1)
        ip.zoom_in(1)
        ip.zoom_out(1)
        with pytest.warns(UserWarning, match="This function was not implemented"):
            ip.toa(1)

        ip.orbital_period = 2
        orbital_period, fdot, fddot = ip.get_values()
        assert orbital_period == 2
        os.unlink(par)

    def test_phaseogram_raises_binary(self):
        evfile = self.dum
        with pytest.raises(ValueError):
            main_phaseogram([evfile, "--binary", "--test"])

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(".")
