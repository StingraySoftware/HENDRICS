import os
import copy
import glob
from collections.abc import Iterable

import pytest
from stingray.lightcurve import Lightcurve
from stingray.events import EventList
import numpy as np
from hendrics.io import (
    save_events,
    HEN_FILE_EXTENSION,
    load_folding,
    load_events,
    get_file_type,
    save_lcurve,
)
from hendrics.efsearch import main_efsearch, main_zsearch
from hendrics.efsearch import main_accelsearch, main_z2vspf
from hendrics.efsearch import decide_binary_parameters, folding_orbital_search
from hendrics.fold import main_fold, main_deorbit, std_fold_fit_func
from hendrics.fold import fit_profile_with_sinusoids, get_TOAs_from_events
from hendrics.plot import plot_folding
from hendrics.tests import _dummy_par
from hendrics.base import hen_root
from . import cleanup_test_dir

try:
    import pandas as pd

    HAS_PD = True
except ImportError:
    HAS_PD = False

from hendrics.fold import HAS_PINT
from hendrics.efsearch import HAS_IMAGEIO


class TestEFsearch:
    def setup_class(cls):
        cls.pulse_period = 0.101
        cls.pulse_frequency = 1 / cls.pulse_period
        cls.tstart = 0
        # cls.toa = 0.2 * cls.pulse_period
        cls.tend = 25.25
        cls.tseg = cls.tend - cls.tstart
        cls.dt = 0.00606
        cls.times = np.arange(cls.tstart, cls.tend, cls.dt) + cls.dt / 2
        cls.counts = 400 + 100 * np.cos(2 * np.pi * cls.times * cls.pulse_frequency)
        cls.mjdref = 56000

        lc = Lightcurve(cls.times, cls.counts, gti=[[cls.tstart, cls.tend]], dt=cls.dt)
        cls.lcfile = "lcurve" + HEN_FILE_EXTENSION
        save_lcurve(lc, cls.lcfile)
        events = EventList()
        events.mjdref = cls.mjdref
        events.simulate_times(lc)
        events.mission = "nusboh"
        cls.event_times = events.time
        cls.dum_noe = "events_noe" + HEN_FILE_EXTENSION
        save_events(events, cls.dum_noe)
        events.pi = np.random.uniform(0, 1000, len(events.time))
        cls.dum_pi = "events_pi" + HEN_FILE_EXTENSION
        save_events(events, cls.dum_pi)
        events.energy = np.random.uniform(3, 79, len(events.time))
        cls.dum = "events" + HEN_FILE_EXTENSION
        cls.dum_scramble = "events_scramble" + HEN_FILE_EXTENSION
        save_events(events, cls.dum)
        events_scramble = copy.deepcopy(events)
        events_scramble.time = np.sort(
            np.random.uniform(cls.tstart, cls.tend, events.time.size)
        )
        save_events(events_scramble, cls.dum_scramble)
        cls.par = "bububububu.par"
        _dummy_par(cls.par)
        events = EventList(time=np.sort(np.random.uniform(0, 100, 200)))
        cls.empty = "empty_ev" + HEN_FILE_EXTENSION
        save_events(events, cls.empty)

    def test_get_TOAs(self):
        events = load_events(self.dum)
        nbin = 32
        toas, toaerrs = get_TOAs_from_events(
            events.time,
            self.tseg,
            self.pulse_frequency,
            gti=events.gti,
            nbin=nbin,
            mjdref=events.mjdref,
            template=None,
        )

        assert toas is not None, toaerrs is not None
        possible_toas = events.mjdref + np.arange(2) * self.pulse_period / 86400
        closest = possible_toas[np.argmin(np.abs(possible_toas - toas[0]))]

        delta_toa_s = (toas[0] - closest) * 86400
        toa_err_s = max(toaerrs[0] / 1e6, 1 / nbin / self.pulse_frequency)

        assert np.abs(delta_toa_s) < toa_err_s * 4

    def test_get_TOAs_template(self):
        nbin = 32
        phases = np.arange(0, 1, 1 / nbin)
        template = np.cos(2 * np.pi * phases)
        events = load_events(self.dum)
        toas, toaerrs = get_TOAs_from_events(
            events.time,
            self.tseg,
            self.pulse_frequency,
            gti=events.gti,
            mjdref=events.mjdref,
            template=template,
            nbin=nbin,
        )
        assert toas is not None, toaerrs is not None

        possible_toas = events.mjdref + np.arange(2) * self.pulse_period / 86400
        closest = possible_toas[np.argmin(np.abs(possible_toas - toas[0]))]

        assert (toas[0] - closest) < toaerrs[0] / 86400000000

    def test_fit_profile_with_sinusoids(self):
        nbin = 32
        phases = np.arange(0, 1, 1 / nbin)
        prof_smooth = np.cos(2 * np.pi * phases) + 0.5 * np.cos(
            4 * np.pi * (phases + 0.5)
        )
        prof_smooth = (prof_smooth + 5) * 64
        prof = np.random.poisson(prof_smooth)
        baseline = np.mean(prof)
        proferr = np.sqrt(baseline)
        fit_pars_save, success_save, chisq_save = fit_profile_with_sinusoids(
            prof, proferr, debug=True, baseline=True
        )
        assert np.allclose(
            std_fold_fit_func(fit_pars_save, phases),
            prof_smooth,
            atol=3 * proferr,
        )

    def test_fold(self):
        evfile = self.dum
        evfile_noe = self.dum_noe
        evfile_pi = self.dum_pi

        main_fold(
            [
                evfile,
                "-f",
                str(self.pulse_frequency),
                "-n",
                "64",
                "--test",
                "--norm",
                "ratios",
            ]
        )
        outfile = hen_root(evfile) + "_ratios.png"
        assert os.path.exists(outfile)
        os.unlink(outfile)
        main_fold(
            [
                evfile_noe,
                "-f",
                str(self.pulse_frequency),
                "-n",
                "64",
                "--test",
                "--norm",
                "blablabla",
            ]
        )
        outfile = hen_root(evfile_noe) + ".png"
        assert os.path.exists(outfile)
        os.unlink(outfile)
        main_fold(
            [
                evfile_pi,
                "-f",
                str(self.pulse_frequency),
                "-n",
                "64",
                "--test",
                "--norm",
                "to1",
            ]
        )
        outfile = hen_root(evfile_pi) + "_to1.png"
        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_fold_invalid(self):
        evfile = self.dum

        with pytest.raises(ValueError, match="Only specify one between "):
            main_fold(
                [
                    evfile,
                    "-f",
                    str(self.pulse_frequency),
                    "-n",
                    "64",
                    "--test",
                    "--norm",
                    "ratios",
                    "--pepoch",
                    str(self.mjdref),
                    "--tref",
                    "0",
                ]
            )

    def test_efsearch(self):
        evfile = self.dum
        main_efsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "64",
                "--emin",
                "3",
                "--emax",
                "79",
                "--fit-candidates",
            ]
        )
        outfile = "events_EF_3-79keV_9.85-9.95Hz" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        ftype, efperiod = get_file_type(outfile)
        assert ftype == "folding"
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25)
        os.unlink(outfile)

    def test_efsearch_bad_freq(self):
        evfile = self.dum_scramble
        with pytest.warns(UserWarning, match="No peaks detected"):
            main_efsearch(
                [
                    evfile,
                    "-f",
                    "100",
                    "-F",
                    "100.01",
                    "-n",
                    "64",
                    "--emin",
                    "3",
                    "--emax",
                    "79",
                    "--fit-candidates",
                    "--conflevel",
                    "99.9999999999",
                ]
            )

    def test_efsearch_from_lc(self):
        evfile = self.lcfile

        main_efsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "64",
                "--fit-candidates",
            ]
        )
        outfile = "lcurve_EF_9.85-9.95Hz" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25)

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
                "--emin",
                "3",
                "--emax",
                "79",
                "--fit-candidates",
                "--fit-frequency",
                str(self.pulse_frequency),
                "--dynstep",
                "5",
            ]
        )
        outfile = "events_Z22_3-79keV_9.85-9.95Hz" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25)
        # Defaults to 2 harmonics
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_from_lc(self):
        evfile = self.lcfile

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
                "--dynstep",
                "5",
            ]
        )
        outfile = "lcurve_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25)
        # Defaults to 2 harmonics
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_fdots(self):
        evfile = self.dum
        main_zsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "32",
                "--fdotmin",
                " -0.01",
                "--fdotmax",
                "0.01",
                "--fit-candidates",
                "--fit-frequency",
                str(self.pulse_frequency),
            ]
        )
        outfile = "events_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True, output_data_file="bla.qdp")
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25)
        # Defaults to 2 harmonics
        assert len(efperiod.fdots) > 1
        assert efperiod.N == 2
        os.unlink(outfile)

    @pytest.mark.skipif("not HAS_IMAGEIO")
    def test_transient(self):
        evfile = self.dum
        main_zsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "32",
                "--fdotmin",
                " -0.1",
                "--fdotmax",
                "0.1",
                "--transient",
                "--n-transient-intervals",
                "16",
            ]
        )
        outfile = "events_Z22_9.85-9.95Hz_transient.gif"
        assert os.path.exists(outfile)
        os.unlink(outfile)

    @pytest.mark.skipif("HAS_IMAGEIO")
    def test_transient_warn_if_no_imageio(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="imageio needed"):
            main_zsearch(
                [
                    evfile,
                    "-f",
                    "9.85",
                    "-F",
                    "9.95",
                    "-n",
                    "64",
                    "--fdotmin",
                    " -0.1",
                    "--fdotmax",
                    "0.1",
                    "--transient",
                ]
            )

    def test_zsearch_print_upperlim(self):
        evfile = self.empty

        outfile = main_zsearch([evfile, "-f", "4", "-F", "6", "-N", "1", "--fast"])[0]
        plot_folding([outfile], ylog=True, output_data_file="bla.qdp")

        # assert "Upper limit for sinusoids:" in caplog.text
        os.unlink(outfile)

    def test_zsearch_fast(self):
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
                "--fast",
                "--find-candidates",
                "--oversample",
                "4",
            ]
        )
        outfile = "events_Z22_9.85-9.95Hz_fast" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True, output_data_file="bla.qdp")
        efperiod = load_folding(outfile)

        assert len(efperiod.fdots) > 1
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_fast_nofdot(self):
        evfile = self.dum
        outfiles = main_zsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "64",
                "--fast",
                "--fdotmin",
                0,
                "--fdotmax",
                "0",
            ]
        )
        outfile = outfiles[0]
        assert os.path.exists(outfile)
        efperiod = load_folding(outfile)

        assert not isinstance(efperiod.fdots, Iterable) or len(efperiod.fdots) <= 1
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_fast_nbin_small_warns(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="The number of bins is too small"):
            _ = main_zsearch(
                [
                    evfile,
                    "-f",
                    "9.85",
                    "-F",
                    "9.95",
                    "-n",
                    "2",
                    "--fast",
                ]
            )

    def test_zsearch_fdots_fast(self):
        evfile = self.dum
        outfiles = main_zsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "64",
                "--fast",
                "--mean-fdot",
                "1e-10",
            ]
        )

        outfile = outfiles[0]
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True, output_data_file="bla.qdp")
        efperiod = load_folding(outfile)

        assert len(efperiod.fdots) > 1
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_fddots_fast(self):
        evfile = self.dum
        outfiles = main_zsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "64",
                "--fast",
                "--mean-fdot",
                "0",
                "--mean-fddot",
                "1e-13",
            ]
        )

        outfile = outfiles[0]
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True, output_data_file="bla.qdp")
        efperiod = load_folding(outfile)

        assert len(efperiod.fdots) > 1
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_fdots_ffa(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="Folding Algorithm functionality"):
            main_zsearch(
                [
                    evfile,
                    "-f",
                    "9.89",
                    "-F",
                    "9.92",
                    "-n",
                    "32",
                    "--ffa",
                    "--find-candidates",
                ]
            )

        outfile = "events_Z22_9.89-9.92Hz_ffa" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True, output_data_file="bla_ffa.qdp")
        efperiod = load_folding(outfile)

        assert efperiod.N == 2
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency, atol=1 / 25.25)
        os.unlink(outfile)

    def test_fold_fast_fails(self):
        evfile = self.dum

        with pytest.raises(
            ValueError, match="The fast option is only available for z "
        ):
            main_efsearch([evfile, "-f", "9.85", "-F", "9.95", "-n", "64", "--fast"])

    def test_zsearch_fdots_fast_transient(self):
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
                "--fast",
                "--transient",
                "--n-transient-intervals",
                "16",
            ]
        )

    @pytest.mark.skipif("not HAS_PD")
    def test_orbital(self):
        import pandas as pd

        events = load_events(self.dum)
        csv_file = decide_binary_parameters(
            137430,
            [0.03, 0.035],
            [2.0 * 86400, 2.5 * 86400],
            [0.0, 1.0],
            fdot_range=[0, 5e-10],
            reset=False,
            NMAX=10,
        )
        table = pd.read_csv(csv_file)
        assert len(table) == 10
        folding_orbital_search(events, csv_file, chunksize=10, outfile="out.csv")
        table = pd.read_csv("out.csv")
        assert len(table) == 10
        assert np.all(table["done"])
        os.unlink(csv_file)
        os.unlink("out.csv")

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_efsearch_deorbit(self):
        evfile = self.dum

        ip = main_zsearch(
            [
                evfile,
                "-f",
                "9.85",
                "-F",
                "9.95",
                "-n",
                "64",
                "--deorbit-par",
                self.par,
            ]
        )

        outfile = "events_Z22_9.85-9.95Hz" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_fold_deorbit(self):
        evfile = self.dum

        main_fold(
            [
                evfile,
                "-f",
                str(self.pulse_frequency),
                "-n",
                "64",
                "--test",
                "--norm",
                "ratios",
                "--deorbit-par",
                self.par,
                "--pepoch",
                str(self.mjdref),
            ]
        )
        outfile = hen_root(evfile) + "_ratios.png"
        assert os.path.exists(outfile)
        os.unlink(outfile)

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_deorbit(self):
        evfile = self.dum
        main_deorbit([evfile, "--deorbit-par", self.par])
        outfile = hen_root(evfile) + "_deorb" + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_efsearch_deorbit_invalid(self):
        evfile = self.dum
        with pytest.raises(FileNotFoundError, match="Parameter file"):
            ip = main_efsearch(
                [
                    evfile,
                    "-f",
                    "9.85",
                    "-F",
                    "9.95",
                    "-n",
                    "64",
                    "--deorbit-par",
                    "nonexistent.par",
                ]
            )

    def test_accelsearch(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="The accelsearch functionality is "):
            outfile = main_accelsearch(
                [
                    evfile,
                    "--fmin",
                    "1",
                    "--fmax",
                    "10",
                    "--zmax",
                    "1",
                    "--delta-z",
                    "0.5",
                ]
            )

        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_accelsearch_nodetections(self):
        evfile = self.dum_scramble
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            outfile = main_accelsearch(
                [evfile, "--fmin", "1", "--fmax", "1.1", "--zmax", "1"]
            )

        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_accelsearch_detrend(self):
        evfile = self.dum_scramble
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            outfile = main_accelsearch(
                [
                    evfile,
                    "--fmin",
                    "1",
                    "--fmax",
                    "1.1",
                    "--zmax",
                    "1",
                    "--detrend",
                    "20",
                ]
            )

        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_accelsearch_rednoise(self):
        evfile = self.dum_scramble
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            outfile = main_accelsearch(
                [
                    evfile,
                    "--fmin",
                    "1",
                    "--fmax",
                    "1.1",
                    "--zmax",
                    "1",
                    "--red-noise-filter",
                ]
            )

        assert os.path.exists(outfile)
        os.unlink(outfile)

    @pytest.mark.skipif("not HAS_PINT")
    def test_accelsearch_deorbit(self):
        evfile = self.dum_scramble
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            outfile = main_accelsearch(
                [
                    evfile,
                    "--fmin",
                    "1",
                    "--fmax",
                    "1.1",
                    "--zmax",
                    "1",
                    "--deorbit-par",
                    self.par,
                ]
            )

        assert os.path.exists(outfile)
        os.unlink(outfile)

    @pytest.mark.skipif("HAS_PINT")
    def test_accelsearch_deorbit_fails_no_pint(self):
        evfile = self.dum_scramble
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            with pytest.raises(ImportError, match="PINT"):
                main_accelsearch(
                    [
                        evfile,
                        "--fmin",
                        "1",
                        "--fmax",
                        "1.1",
                        "--zmax",
                        "1",
                        "--deorbit-par",
                        self.par,
                    ]
                )

    def test_accelsearch_energy_and_freq_filt(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            outfile = main_accelsearch(
                [
                    evfile,
                    "--emin",
                    "3",
                    "--emax",
                    "80",
                    "--fmin",
                    "0.1",
                    "--fmax",
                    "1",
                    "--zmax",
                    "5",
                ]
            )

        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_accelsearch_pad(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            outfile = main_accelsearch([evfile, "--pad-to-double", "--zmax", "1"])

        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_accelsearch_interbin(self):
        evfile = self.dum
        with pytest.warns(UserWarning, match="The accelsearch functionality"):
            outfile = main_accelsearch([evfile, "--interbin", "--zmax", "1"])

        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_z2vspf(self):
        evfile = self.dum
        ip = main_z2vspf([evfile, "--show-z-values", "30", "--ntrial", "10"])

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(".")
