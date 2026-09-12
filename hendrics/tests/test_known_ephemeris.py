# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the targeted-search trials correction."""

import os
import shutil
import tempfile

import numpy as np
import pytest

from hendrics.base import HAS_PINT
from hendrics.known_ephemeris import (
    effective_ntrial,
    ephemeris_from_parfile,
    extrapolate_ephemeris,
    prior_corrected_p_value,
)


class TestExtrapolateEphemeris:
    def test_no_derivatives_is_a_no_op(self):
        f, fd, fdd = extrapolate_ephemeris(1.5, pepoch=50000, target_epoch=55000)
        assert np.isclose(f, 1.5)
        assert np.isclose(fd, 0)
        assert np.isclose(fdd, 0)

    def test_linear_spindown(self):
        # Losing exactly 1e-5 Hz over ten days
        fdot = -1e-5 / 864000
        f, fd, _ = extrapolate_ephemeris(1.0, fdot=fdot, pepoch=50000, target_epoch=50010)
        assert np.isclose(f, 1.0 - 1e-5)
        assert np.isclose(fd, fdot)

    def test_round_trip_with_fddot(self):
        f0, fdot0, fddot0 = 0.728, -4.45e-11, 3e-22
        f1, fdot1, fddot1 = extrapolate_ephemeris(
            f0, fdot=fdot0, fddot=fddot0, pepoch=56682, target_epoch=50000
        )
        back = extrapolate_ephemeris(f1, fdot=fdot1, fddot=fddot1, pepoch=50000, target_epoch=56682)
        assert np.isclose(back[0], f0)
        assert np.isclose(back[1], fdot0)

    def test_two_epoch_example(self):
        """The worked example from the documentation."""
        # 0.728 Hz at MJD 56682 and 0.711 Hz at MJD 61100
        fdot = (0.711 - 0.728) / ((61100 - 56682) * 86400)
        f, _, _ = extrapolate_ephemeris(0.728, fdot=fdot, pepoch=56682, target_epoch=61100)
        assert np.isclose(f, 0.711)

    def test_missing_epoch_raises(self):
        with pytest.raises(ValueError, match="Both pepoch and target_epoch"):
            extrapolate_ephemeris(1.0, pepoch=50000)


class TestEffectiveNtrial:
    # A frequency-only search and an f/fdot search, as seen by the formula
    ONE_D = dict(f_step=1e-5, n_grid=1000, ntrial_blind=200, search_fdot=False)
    TWO_D = dict(f_step=1e-5, fdot_step=1e-10, n_grid=10000, ntrial_blind=1000, search_fdot=True)

    def test_on_the_prediction_is_one_trial(self):
        assert np.isclose(effective_ntrial(0.0, **self.ONE_D), 1)
        assert np.isclose(effective_ntrial(0.0, 0.0, **self.TWO_D), 1)

    def test_frequency_only_counts_an_interval(self):
        # 50 grid steps away: the interval spans 100 of the 1000 grid cells,
        # so a tenth of the 200 blind trials
        assert np.isclose(effective_ntrial(50e-5, **self.ONE_D), 20)

    def test_fdot_search_counts_a_disc(self):
        # 10 grid steps away in frequency, right on the predicted fdot
        expected = 1000 * np.pi * 100 / 10000
        assert np.isclose(effective_ntrial(10e-5, 0.0, **self.TWO_D), expected)

    def test_fdot_offset_counts_too(self):
        # The same distance, but along the fdot axis
        expected = 1000 * np.pi * 100 / 10000
        assert np.isclose(effective_ntrial(0.0, 10e-10, **self.TWO_D), expected)

    def test_offsets_add_in_quadrature(self):
        one_axis = effective_ntrial(10e-5, 0.0, **self.TWO_D)
        both = effective_ntrial(10e-5, 10e-10, **self.TWO_D)
        assert np.isclose(both, 2 * one_axis)

    def test_sign_does_not_matter(self):
        assert np.isclose(
            effective_ntrial(1e-4, **self.ONE_D), effective_ntrial(-1e-4, **self.ONE_D)
        )

    def test_monotonic_with_offset(self):
        ntrial = effective_ntrial(np.arange(0, 50) * 1e-5, **self.ONE_D)
        assert np.all(np.diff(ntrial) >= 0)

    def test_capped_at_the_blind_search(self):
        # Very far away: we pay exactly what a blind search would have cost
        assert np.isclose(effective_ntrial(1.0, **self.ONE_D), 200)
        assert np.isclose(effective_ntrial(1.0, 1.0, **self.TWO_D), 1000)

    def test_never_below_one(self):
        assert effective_ntrial(1e-12, **self.ONE_D) == 1.0

    def test_covering_the_whole_grid_costs_the_blind_count(self):
        """Half the grid away in a 1-D search, we have paid the full price."""
        assert np.isclose(effective_ntrial(500e-5, **self.ONE_D), 200)

    def test_works_on_arrays(self):
        offsets = np.array([0.0, 50e-5, 1.0])
        ntrial = effective_ntrial(offsets, **self.ONE_D)
        assert ntrial.shape == offsets.shape
        assert np.allclose(ntrial, [1, 20, 200])

    def test_missing_arguments_raise(self):
        with pytest.raises(ValueError, match="f_step"):
            effective_ntrial(1e-4, n_grid=10, ntrial_blind=10)
        with pytest.raises(ValueError, match="n_grid and ntrial_blind"):
            effective_ntrial(1e-4, f_step=1e-5)
        with pytest.raises(ValueError, match="fdot_step"):
            effective_ntrial(1e-4, f_step=1e-5, n_grid=10, ntrial_blind=10, search_fdot=True)


class TestPriorCorrectedPValue:
    def test_single_trial_is_unchanged(self):
        assert np.isclose(prior_corrected_p_value(0.01, 1), 0.01)

    def test_union_bound_for_small_probabilities(self):
        assert np.isclose(prior_corrected_p_value(1e-8, 100), 1e-6, rtol=1e-4)

    def test_bounded_by_one(self):
        assert prior_corrected_p_value(0.5, 1e6) <= 1

    def test_monotonic_in_ntrial(self):
        ntrial = np.arange(1, 100)
        p = prior_corrected_p_value(1e-4, ntrial)
        assert np.all(np.diff(p) >= 0)

    def test_accurate_for_tiny_probabilities(self):
        """The naive 1 - (1 - p)**n would lose all precision here."""
        p = prior_corrected_p_value(1e-17, 10)
        assert np.isclose(p, 1e-16, rtol=1e-6)

    def test_fractional_ntrial_interpolates(self):
        low = prior_corrected_p_value(1e-4, 2)
        mid = prior_corrected_p_value(1e-4, 2.5)
        high = prior_corrected_p_value(1e-4, 3)
        assert low < mid < high


@pytest.mark.skipif("not HAS_PINT")
class TestEphemerisFromParfile:
    def test_read_back(self, tmp_path):
        parfile = tmp_path / "test.par"
        parfile.write_text(
            "PSR              TEST\n"
            "F0               0.728\n"
            "F1               -4.45e-11\n"
            "PEPOCH           56682\n"
            "EPHEM            DE421\n"
            "UNITS            TDB\n"
        )
        freq, fdot, fddot, pepoch = ephemeris_from_parfile(str(parfile))
        assert np.isclose(freq, 0.728)
        assert np.isclose(fdot, -4.45e-11)
        assert np.isclose(fddot, 0)
        assert np.isclose(pepoch, 56682)


class TestCalibration:
    """The correction must not produce more false alarms than it promises.

    A fast version of the Monte Carlo used to validate the formula: search
    pure noise, put the prior at the centre of the band (independently of the
    data, as the method requires), and check that the corrected p-value of the
    best candidate inside a window is distributed as advertised.
    """

    T = 200.0
    N_EVENTS = 200
    FMIN, FMAX = 9.0, 9.2
    OVERSAMPLE = 16
    NHARM = 2
    N_REAL = 400
    # The window covers a fifth of the band, centred on the prior
    WINDOW_FRACTION = 0.2

    @classmethod
    def setup_class(cls):
        from stingray.pulse.search import z_n_search
        from stingray.stats import z2_n_probability

        f_step = 1 / cls.T / cls.OVERSAMPLE
        freqs = np.arange(cls.FMIN, cls.FMAX, f_step)
        f_mid = (cls.FMIN + cls.FMAX) / 2
        half_width = (cls.FMAX - cls.FMIN) * cls.WINDOW_FRACTION / 2
        inside = np.abs(freqs - f_mid) <= half_width

        rng = np.random.default_rng(20250912)
        p_blind = np.zeros(cls.N_REAL)
        p_window = np.zeros(cls.N_REAL)
        for i in range(cls.N_REAL):
            times = np.sort(rng.uniform(0, cls.T, cls.N_EVENTS))
            _, stats = z_n_search(times, freqs, nbin=32, nharm=cls.NHARM)
            p_single = z2_n_probability(stats, n=cls.NHARM)
            p_blind[i] = p_single.min()
            p_window[i] = p_single[inside].min()

        cls.p_blind = p_blind
        cls.p_window = p_window
        cls.f_step = f_step
        cls.n_grid = freqs.size
        cls.half_width = half_width

    @staticmethod
    def _trials_from_median(p):
        """Trial count implied by the median of a set of best p-values."""
        return np.log(0.5) / np.log1p(-np.median(p))

    def test_blind_trials_exceed_naive_counting(self):
        """A sanity check on the simulation, and on why we take a ratio.

        The independent trial count of a folding search is several times
        larger than the band width divided by 1/T, which is exactly why
        ``effective_ntrial`` works with ratios instead of resolutions.
        """
        naive = (self.FMAX - self.FMIN) * self.T
        assert self._trials_from_median(self.p_blind) > 2 * naive

    def test_corrected_p_value_is_calibrated(self):
        ntrial_blind = self._trials_from_median(self.p_blind)
        n_eff = effective_ntrial(
            self.half_width,
            f_step=self.f_step,
            n_grid=self.n_grid,
            ntrial_blind=ntrial_blind,
            search_fdot=False,
        )
        # The window is a fifth of the band, so it should cost a fifth of the
        # trials
        assert np.isclose(n_eff / ntrial_blind, self.WINDOW_FRACTION, rtol=1e-6)

        p_corr = prior_corrected_p_value(self.p_window, n_eff)
        for alpha in (0.5, 0.2):
            rate = np.mean(p_corr < alpha) / alpha
            # Generous bounds: this must catch a wrong formula, not police
            # Monte Carlo noise
            assert 0.6 < rate < 1.5, f"false alarm rate {rate:.2f} at alpha={alpha}"

    def test_targeted_search_beats_the_blind_one(self):
        """The whole point: the same peak is more significant with a prior."""
        ntrial_blind = self._trials_from_median(self.p_blind)
        n_eff = effective_ntrial(
            self.half_width,
            f_step=self.f_step,
            n_grid=self.n_grid,
            ntrial_blind=ntrial_blind,
            search_fdot=False,
        )
        assert n_eff < ntrial_blind
        blind = prior_corrected_p_value(self.p_window, ntrial_blind)
        targeted = prior_corrected_p_value(self.p_window, n_eff)
        assert np.all(targeted <= blind)


class TestTargetedZSearch:
    """End-to-end: HENzsearch given a previously known spin solution."""

    T = 1000.0
    MJDREF = 56000
    FTRUE = 9.37
    # A solution measured 1000 days earlier, spinning down steadily
    KNOWN_PEPOCH = 55000
    KNOWN_FDOT = -1e-9

    @classmethod
    def setup_class(cls):
        from stingray.events import EventList

        from hendrics.io import HEN_FILE_EXTENSION, save_events

        cls.datadir = tempfile.mkdtemp()
        cls.cwd = os.getcwd()
        os.chdir(cls.datadir)

        rng = np.random.default_rng(5)
        n = 40000
        times = np.sort(rng.uniform(0, cls.T, n))
        # A weak sinusoidal pulsation on a bright constant background: too
        # weak to stand out in a blind search over the whole band
        keep = rng.uniform(0, 1, n) < 0.5 * (1 + 0.045 * np.cos(2 * np.pi * cls.FTRUE * times))
        events = EventList(time=times[keep], gti=np.array([[0, cls.T]]), mjdref=cls.MJDREF)
        events.instr = "test"
        cls.fname = "ev" + HEN_FILE_EXTENSION
        save_events(events, cls.fname)

        # The search refers everything to the middle of the observation
        pepoch_search = cls.MJDREF + (cls.T / 2) / 86400
        dt = (pepoch_search - cls.KNOWN_PEPOCH) * 86400
        cls.known_freq = cls.FTRUE - cls.KNOWN_FDOT * dt

        cls.common = [
            cls.fname,
            "-f",
            "9.0",
            "-F",
            "10.0",
            "-n",
            "16",
            "--fast",
            "--oversample",
            "16",
            "-N",
            "2",
        ]

    @classmethod
    def teardown_class(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.datadir, ignore_errors=True)

    @staticmethod
    def _candidates(outfiles):
        from hendrics.efsearch import analyze_qffa_results

        _, table = analyze_qffa_results(outfiles[0])
        return table

    def test_blind_search_misses_the_signal(self):
        """The reference point: without a prior this pulsation is not found."""
        from hendrics.efsearch import main_zsearch

        table = self._candidates(main_zsearch(self.common))
        # Every candidate is an upper limit, i.e. nothing was detected
        assert np.all(np.isnan(table["pulse_amp"]))
        # ...and the tallest peak is a noise peak, far from the true frequency
        best = table[np.argmax(table["power"])]
        assert np.abs(best["f"] - self.FTRUE) > 0.01

    def test_targeted_search_finds_it(self):
        """The same data, with the ephemeris extrapolated from 1000 days back."""
        from hendrics.efsearch import main_zsearch

        table = self._candidates(
            main_zsearch(
                self.common
                + [
                    "--known-freq",
                    str(self.known_freq),
                    "--known-fdot",
                    str(self.KNOWN_FDOT),
                    "--known-pepoch",
                    str(self.KNOWN_PEPOCH),
                ]
            )
        )
        assert "p_value" in table.colnames

        detected = table[~np.isnan(table["pulse_amp"])]
        assert len(detected) > 0, "the targeted search should detect the pulsation"

        best = detected[np.argmin(detected["p_value"])]
        # It is the real signal, sitting essentially on the prediction
        assert np.isclose(best["f"], self.FTRUE, atol=1e-3)
        assert np.abs(best["f_offset"]) < 1e-3
        # Right on the prediction, so it costs almost nothing in trials
        assert best["ntrial_eff"] < 10
        assert best["p_value"] < 1e-3

    def test_a_wrong_prior_does_not_invent_a_detection(self):
        """A prior far from the truth must not manufacture significance."""
        from hendrics.efsearch import main_zsearch

        table = self._candidates(
            main_zsearch(self.common + ["--known-freq", "9.8", "--known-pepoch", str(self.MJDREF)])
        )
        # Whatever it picks, the offset is paid for in trials
        assert np.all(table["ntrial_eff"] >= 1)
        for row in table:
            if np.abs(row["f_offset"]) > 0.1:
                assert row["ntrial_eff"] > 100

    def test_pepoch_is_required(self):
        from hendrics.efsearch import main_zsearch

        with pytest.raises(ValueError, match="known-pepoch"):
            main_zsearch(self.common + ["--known-freq", str(self.known_freq)])

    @pytest.mark.skipif("not HAS_PINT")
    def test_par_file_gives_the_same_answer(self):
        from hendrics.efsearch import main_zsearch

        parfile = "known.par"
        with open(parfile, "w") as fobj:
            print("PSR              TEST", file=fobj)
            print(f"F0               {self.known_freq}", file=fobj)
            print(f"F1               {self.KNOWN_FDOT}", file=fobj)
            print(f"PEPOCH           {self.KNOWN_PEPOCH}", file=fobj)
            print("EPHEM            DE421", file=fobj)
            print("UNITS            TDB", file=fobj)

        table = self._candidates(main_zsearch(self.common + ["--known-par", parfile]))
        detected = table[~np.isnan(table["pulse_amp"])]
        assert len(detected) > 0
        best = detected[np.argmin(detected["p_value"])]
        assert np.isclose(best["f"], self.FTRUE, atol=1e-3)


class TestTargetedAccelSearch:
    """End-to-end: HENaccelsearch given a previously known spin solution."""

    T = 1000.0
    MJDREF = 56000
    FTRUE = 9.37
    KNOWN_PEPOCH = 55000
    KNOWN_FDOT = -1e-9

    @classmethod
    def setup_class(cls):
        from stingray.events import EventList

        from hendrics.io import HEN_FILE_EXTENSION, save_events

        cls.datadir = tempfile.mkdtemp()
        cls.cwd = os.getcwd()
        os.chdir(cls.datadir)

        rng = np.random.default_rng(7)
        n = 60000
        times = np.sort(rng.uniform(0, cls.T, n))
        keep = rng.uniform(0, 1, n) < 0.5 * (1 + 0.06 * np.cos(2 * np.pi * cls.FTRUE * times))
        events = EventList(time=times[keep], gti=np.array([[0, cls.T]]), mjdref=cls.MJDREF)
        events.instr = "test"
        cls.fname = "ev" + HEN_FILE_EXTENSION
        save_events(events, cls.fname)

        # ``accelsearch`` refers its candidates to the start of the observation
        dt = (cls.MJDREF - cls.KNOWN_PEPOCH) * 86400
        cls.known_freq = cls.FTRUE - cls.KNOWN_FDOT * dt
        cls.prior_args = [
            "--known-freq",
            str(cls.known_freq),
            "--known-fdot",
            str(cls.KNOWN_FDOT),
            "--known-pepoch",
            str(cls.KNOWN_PEPOCH),
        ]

    @classmethod
    def teardown_class(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.datadir, ignore_errors=True)

    def _run(self, outfile, band=("9.3", "9.45"), extra=()):
        from astropy.table import Table
        from hendrics.efsearch import main_accelsearch

        out = main_accelsearch(
            [
                self.fname,
                "--fmin",
                band[0],
                "--fmax",
                band[1],
                "--zmax",
                "10",
                "--outfile",
                outfile,
            ]
            + list(extra)
        )
        return Table.read(out, format="ascii")

    def test_targeted_search_reports_the_correction(self):
        table = self._run("targeted.csv", extra=self.prior_args)
        for name in ("f_offset", "fdot_offset", "ntrial_eff", "p_value"):
            assert name in table.colnames

        best = table[np.argmin(table["p_value"])]
        assert np.isclose(best["frequency"], self.FTRUE, atol=2e-3)
        # It lands on the prediction, so it costs a single trial
        assert np.isclose(best["ntrial_eff"], 1.0)
        assert best["p_value"] < 1e-6

    def test_only_surviving_candidates_are_written(self):
        """The corrected p-value is applied before the file is written."""
        table = self._run("filtered.csv", extra=self.prior_args)
        assert len(table) > 0
        assert np.all(table["p_value"] < 0.068)
        # Everything far from the prior has been charged for the distance
        far = table[np.abs(table["f_offset"]) > 0.02]
        assert np.all(far["ntrial_eff"] > 1)

    def test_blind_search_is_untouched(self):
        """Without a prior, the output keeps its original columns."""
        table = self._run("blind.csv")
        assert "p_value" not in table.colnames
        assert "ntrial_eff" not in table.colnames

    def test_wide_band_warns_that_it_buys_little(self):
        with pytest.warns(UserWarning, match="too wide for the known ephemeris"):
            self._run("wide.csv", band=("1.0", "50.0"), extra=self.prior_args)
