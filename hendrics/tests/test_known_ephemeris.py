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
    extrapolate_ephemeris_uncertainty,
    prior_corrected_p_value,
    qffa_calibrated_ntrial,
    uncertainty_ntrial,
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


class TestExtrapolateEphemerisUncertainty:
    TEN_DAYS = 864000.0

    def test_frequency_error_alone_is_unchanged(self):
        f_err, fdot_err = extrapolate_ephemeris_uncertainty(1e-6, pepoch=50000, target_epoch=55000)
        assert np.isclose(f_err, 1e-6)
        assert fdot_err == 0

    def test_fdot_error_grows_linearly(self):
        f_err, fdot_err = extrapolate_ephemeris_uncertainty(
            0.0, fdot_err=1e-12, pepoch=50000, target_epoch=50010
        )
        assert np.isclose(f_err, 1e-12 * self.TEN_DAYS)
        assert np.isclose(fdot_err, 1e-12)

    def test_fddot_error_grows_quadratically(self):
        f_err, fdot_err = extrapolate_ephemeris_uncertainty(
            0.0, fddot_err=1e-20, pepoch=50000, target_epoch=50010
        )
        assert np.isclose(f_err, 0.5 * 1e-20 * self.TEN_DAYS**2)
        assert np.isclose(fdot_err, 1e-20 * self.TEN_DAYS)

    def test_terms_add_in_quadrature(self):
        f_err, _ = extrapolate_ephemeris_uncertainty(
            3e-6, fdot_err=4e-6 / self.TEN_DAYS, pepoch=50000, target_epoch=50010
        )
        assert np.isclose(f_err, 5e-6)

    def test_backwards_in_time_is_the_same(self):
        kw = dict(fdot_err=1e-12, fddot_err=1e-20, pepoch=50000)
        forward = extrapolate_ephemeris_uncertainty(1e-6, target_epoch=50010, **kw)
        backward = extrapolate_ephemeris_uncertainty(1e-6, target_epoch=49990, **kw)
        assert np.allclose(forward, backward)

    def test_missing_epoch_raises(self):
        with pytest.raises(ValueError, match="Both pepoch and target_epoch"):
            extrapolate_ephemeris_uncertainty(1e-6, target_epoch=50000)


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

    def test_floor_applies_near_the_prior(self):
        # On the prediction the rank costs 1 trial, the floor raises it
        assert np.isclose(effective_ntrial(0.0, ntrial_min=12, **self.ONE_D), 12)
        assert np.isclose(effective_ntrial(0.0, 0.0, ntrial_min=12, **self.TWO_D), 12)

    def test_floor_is_irrelevant_far_away(self):
        # 50 grid steps away the rank already costs 20 trials
        assert np.isclose(effective_ntrial(50e-5, ntrial_min=12, **self.ONE_D), 20)

    def test_floor_never_exceeds_the_blind_search(self):
        assert np.isclose(effective_ntrial(0.0, ntrial_min=1e6, **self.ONE_D), 200)

    def test_floor_works_on_arrays(self):
        offsets = np.array([0.0, 50e-5, 1.0])
        ntrial = effective_ntrial(offsets, ntrial_min=12, **self.ONE_D)
        assert np.allclose(ntrial, [12, 20, 200])


class TestUncertaintyNtrial:
    """Trials charged to every cell inside the uncertainty region of the prior."""

    ONE_D = TestEffectiveNtrial.ONE_D
    TWO_D = TestEffectiveNtrial.TWO_D

    def test_no_uncertainty_is_one_trial(self):
        assert np.isclose(uncertainty_ntrial(0.0, **self.ONE_D), 1)
        assert np.isclose(uncertainty_ntrial(0.0, 0.0, **self.TWO_D), 1)

    def test_frequency_only_counts_the_interval(self):
        # +-3 sigma = +-30 grid steps: 60 of the 1000 cells, i.e. 12 of 200 trials
        assert np.isclose(uncertainty_ntrial(10e-5, **self.ONE_D), 12)

    def test_continuous_with_the_rank_at_the_edge(self):
        """A candidate on the edge of the region costs the same either way."""
        f_err = 10e-5
        assert np.isclose(
            uncertainty_ntrial(f_err, **self.ONE_D), effective_ntrial(3 * f_err, **self.ONE_D)
        )
        fdot_err = 10e-10
        assert np.isclose(
            uncertainty_ntrial(f_err, fdot_err, **self.TWO_D),
            effective_ntrial(3 * f_err, 0.0, **self.TWO_D),
        )

    def test_fdot_search_counts_an_ellipse(self):
        # Semi-axes of 6 frequency cells and 12 fdot cells
        expected = 1000 * np.pi * 6 * 12 / 10000
        assert np.isclose(uncertainty_ntrial(2e-5, 4e-10, **self.TWO_D), expected)

    def test_semi_axes_are_at_least_half_a_cell(self):
        # A perfectly known fdot still spans the cell the prediction falls in
        expected = 1000 * np.pi * 30 * 0.5 / 10000
        assert np.isclose(uncertainty_ntrial(10e-5, 0.0, **self.TWO_D), expected)

    def test_nsigma_scales_the_region(self):
        one = uncertainty_ntrial(10e-5, nsigma=1, **self.ONE_D)
        three = uncertainty_ntrial(10e-5, nsigma=3, **self.ONE_D)
        assert np.isclose(three, 3 * one)

    def test_capped_at_the_blind_search(self):
        assert np.isclose(uncertainty_ntrial(1.0, **self.ONE_D), 200)
        assert np.isclose(uncertainty_ntrial(1.0, 1.0, **self.TWO_D), 1000)

    def test_missing_arguments_raise(self):
        with pytest.raises(ValueError, match="f_step"):
            uncertainty_ntrial(1e-4, n_grid=10, ntrial_blind=10, search_fdot=False)
        with pytest.raises(ValueError, match="n_grid and ntrial_blind"):
            uncertainty_ntrial(1e-4, f_step=1e-5, search_fdot=False)
        with pytest.raises(ValueError, match="fdot_step"):
            uncertainty_ntrial(
                1e-4, 1e-10, f_step=1e-5, n_grid=10, ntrial_blind=10, search_fdot=True
            )


class TestQffaCalibratedNtrial:
    """The Monte Carlo calibration of the trials of ``HENzsearch --fast``."""

    def test_tabulated_frequency_search(self):
        ntrial = qffa_calibrated_ntrial(1000, nharm=2, oversample=4, search_fdot=False)
        assert np.isclose(ntrial, 4700)

    def test_tabulated_fdot_search(self):
        ntrial = qffa_calibrated_ntrial(1000, nharm=2, oversample=4, search_fdot=True)
        assert np.isclose(ntrial, 11000)

    def test_frequency_and_fdot_tables_are_different(self):
        kw = dict(nharm=1, oversample=2)
        assert not np.isclose(
            qffa_calibrated_ntrial(1000, search_fdot=False, **kw),
            qffa_calibrated_ntrial(1000, search_fdot=True, **kw),
        )

    @pytest.mark.parametrize("search_fdot", [False, True])
    def test_between_oversamples_uses_the_next_larger(self, search_fdot):
        kw = dict(nharm=2, search_fdot=search_fdot)
        assert np.isclose(
            qffa_calibrated_ntrial(1000, oversample=3, **kw),
            qffa_calibrated_ntrial(1000, oversample=4, **kw),
        )

    @pytest.mark.parametrize("search_fdot", [False, True])
    def test_between_harmonics_uses_the_next_larger(self, search_fdot):
        kw = dict(oversample=4, search_fdot=search_fdot)
        assert np.isclose(
            qffa_calibrated_ntrial(1000, nharm=3, **kw),
            qffa_calibrated_ntrial(1000, nharm=4, **kw),
        )

    def test_oversample_below_one_uses_the_first_column(self):
        kw = dict(nharm=2, search_fdot=False)
        assert np.isclose(
            qffa_calibrated_ntrial(1000, oversample=0.5, **kw),
            qffa_calibrated_ntrial(1000, oversample=1, **kw),
        )

    @pytest.mark.parametrize(
        "kw, largest",
        [
            (dict(nharm=2, oversample=16, search_fdot=False), dict(nharm=2, oversample=8)),
            (dict(nharm=2, oversample=8, search_fdot=True), dict(nharm=2, oversample=4)),
            (dict(nharm=8, oversample=4, search_fdot=False), dict(nharm=4, oversample=4)),
        ],
    )
    def test_beyond_the_table_warns_and_uses_the_largest(self, kw, largest):
        with pytest.warns(UserWarning, match="not covered by the calibration"):
            ntrial = qffa_calibrated_ntrial(1000, **kw)
        expected = qffa_calibrated_ntrial(1000, search_fdot=kw["search_fdot"], **largest)
        assert np.isclose(ntrial, expected)

    @pytest.mark.parametrize("search_fdot", [False, True])
    def test_monotonic_in_harmonics_and_oversample(self, search_fdot):
        oversamples = (1, 2, 4, 8) if not search_fdot else (1, 2, 4)
        table = np.array(
            [
                [
                    qffa_calibrated_ntrial(1000, nharm=n, oversample=o, search_fdot=search_fdot)
                    for o in oversamples
                ]
                for n in (1, 2, 4)
            ]
        )
        assert np.all(np.diff(table, axis=0) >= 0)
        assert np.all(np.diff(table, axis=1) >= 0)

    def test_never_below_one(self):
        assert qffa_calibrated_ntrial(1, nharm=1, oversample=1, search_fdot=True) >= 1


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
    def test_read_uncertainties(self, tmp_path):
        parfile = tmp_path / "errors.par"
        with open(parfile, "w") as fobj:
            print("PSR              TEST", file=fobj)
            print("F0               0.728 1 2e-9", file=fobj)
            print("F1               -4.45e-11 1 3e-18", file=fobj)
            print("PEPOCH           56682", file=fobj)
            print("EPHEM            DE421", file=fobj)
            print("UNITS            TDB", file=fobj)

        freq, fdot, fddot, pepoch, errors = ephemeris_from_parfile(parfile, return_errors=True)
        assert np.isclose(freq, 0.728)
        assert np.isclose(pepoch, 56682)
        f_err, fdot_err, fddot_err = errors
        assert np.isclose(f_err, 2e-9)
        assert np.isclose(fdot_err, 3e-18)
        # F2 is not in the file, so it has no uncertainty either
        assert fddot_err == 0

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
            "4",
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

    def test_every_search_reports_all_significances(self):
        """Single-trial, naive and calibrated probabilities, with or without a prior."""
        from hendrics.efsearch import main_zsearch

        table = self._candidates(main_zsearch(self.common))
        for name in ("p_1trial", "p_ntrial", "p_ntrial_adj"):
            assert name in table.colnames
        assert np.all(table["p_ntrial"] >= table["p_1trial"])
        assert np.all(table["p_ntrial_adj"] >= table["p_ntrial"])

        # The calibrated count multiplies the naive one, for a Z^2_2 search of
        # frequency and fdot with 4 points per resolution element
        meta = table.meta
        assert np.isclose(
            meta["ntrial"],
            qffa_calibrated_ntrial(meta["ntrial_naive"], nharm=2, oversample=4, search_fdot=True),
        )
        assert np.allclose(
            table["p_ntrial_adj"], prior_corrected_p_value(table["p_1trial"], meta["ntrial"])
        )

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

    def test_uncertainty_sets_a_floor_on_the_trials(self):
        """With an uncertain solution, the on-prior candidate pays for the region."""
        from hendrics.efsearch import main_zsearch
        from hendrics.io import load_folding

        f_err = 1e-2
        outfiles = main_zsearch(
            self.common
            + [
                "--known-freq",
                str(self.known_freq),
                "--known-fdot",
                str(self.KNOWN_FDOT),
                "--known-pepoch",
                str(self.KNOWN_PEPOCH),
                "--known-freq-err",
                str(f_err),
            ]
        )
        ef = load_folding(outfiles[0])
        # With no fdot uncertainty, the frequency one is not changed by the
        # extrapolation
        assert np.isclose(ef.known_freq_err, f_err)
        assert ef.known_fdot_err == 0

        expected = uncertainty_ntrial(
            f_err,
            0.0,
            f_step=np.median(np.diff(ef.freq[0, :])),
            fdot_step=np.median(np.diff(ef.fdots[:, 0])),
            n_grid=ef.stat.size,
            # The calibrated count of the blind search over the same plane
            ntrial_blind=qffa_calibrated_ntrial(
                int(ef.stat.size / ef.oversample**2),
                nharm=2,
                oversample=ef.oversample,
                search_fdot=True,
            ),
            search_fdot=True,
        )
        # A meaningful floor, not the single trial of a precise prior
        assert expected > 5

        table = self._candidates(outfiles)
        assert np.all(table["ntrial_eff"] >= expected * (1 - 1e-6))
        # Paying for the whole region may well push this weak signal below the
        # detection threshold: that is the point. It is still the most
        # significant candidate, though.
        best = table[np.argmin(table["p_value"])]
        assert np.isclose(best["f"], self.FTRUE, atol=1e-3)
        # Sitting on the prediction, it pays exactly for the uncertainty region
        assert np.isclose(best["ntrial_eff"], expected)


class TestKnownEphemerisAt:
    """The known solution, and its uncertainty, at the epoch of the search."""

    TEN_DAYS = 864000.0

    @staticmethod
    def _args(**kwargs):
        from argparse import Namespace

        defaults = dict(
            known_par=None,
            known_freq=None,
            known_fdot=0.0,
            known_fddot=0.0,
            known_pepoch=None,
            known_freq_err=None,
            known_fdot_err=None,
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    @staticmethod
    def _write_par(path):
        with open(path, "w") as fobj:
            print("PSR              TEST", file=fobj)
            print("F0               1.0 1 2e-9", file=fobj)
            print("F1               -1e-12 1 3e-18", file=fobj)
            print("PEPOCH           50000", file=fobj)
            print("EPHEM            DE421", file=fobj)
            print("UNITS            TDB", file=fobj)
        return str(path)

    def test_blind_search(self):
        from hendrics.efsearch import _known_ephemeris_at

        freq, fdot, f_err, fdot_err = _known_ephemeris_at(self._args(), 50000)
        assert np.isnan(freq)
        assert np.isnan(fdot)
        assert f_err == 0
        assert fdot_err == 0

    def test_no_uncertainty_given(self):
        from hendrics.efsearch import _known_ephemeris_at

        args = self._args(known_freq=1.0, known_pepoch=50000)
        _, _, f_err, fdot_err = _known_ephemeris_at(args, 50010)
        assert f_err == 0
        assert fdot_err == 0

    def test_command_line_errors_are_propagated(self):
        from hendrics.efsearch import _known_ephemeris_at

        args = self._args(
            known_freq=1.0, known_pepoch=50000, known_freq_err=1e-6, known_fdot_err=1e-12
        )
        _, _, f_err, fdot_err = _known_ephemeris_at(args, 50010)
        assert np.isclose(f_err, np.hypot(1e-6, 1e-12 * self.TEN_DAYS))
        assert np.isclose(fdot_err, 1e-12)

    @pytest.mark.skipif("not HAS_PINT")
    def test_par_file_errors_are_used(self, tmp_path):
        from hendrics.efsearch import _known_ephemeris_at

        args = self._args(known_par=self._write_par(tmp_path / "err.par"))
        _, _, f_err, fdot_err = _known_ephemeris_at(args, 50010)
        assert np.isclose(f_err, np.hypot(2e-9, 3e-18 * self.TEN_DAYS))
        assert np.isclose(fdot_err, 3e-18)

    @pytest.mark.skipif("not HAS_PINT")
    def test_command_line_overrides_the_par_file(self, tmp_path):
        """Formal timing errors are often too optimistic: let the user widen them."""
        from hendrics.efsearch import _known_ephemeris_at

        args = self._args(known_par=self._write_par(tmp_path / "err.par"), known_freq_err=1e-3)
        _, _, f_err, fdot_err = _known_ephemeris_at(args, 50010)
        # The frequency error is replaced, the fdot one still comes from the file
        assert np.isclose(f_err, np.hypot(1e-3, 3e-18 * self.TEN_DAYS))
        assert np.isclose(fdot_err, 3e-18)


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

    def test_uncertainty_sets_a_floor_on_the_trials(self):
        f_err, fdot_err = 1e-2, 2e-6
        table = self._run(
            "uncertain.csv",
            extra=self.prior_args
            + ["--known-freq-err", str(f_err), "--known-fdot-err", str(fdot_err)],
        )
        # The epochs are 1000 days apart: propagate the fdot error
        dt = (self.MJDREF - self.KNOWN_PEPOCH) * 86400
        f_err_now = np.hypot(f_err, fdot_err * dt)
        length = table["length"][0]
        n_freq = int(table["ntrial"][0])
        # --zmax 10, default --delta-z 1
        n_z = np.arange(-10, 10, 1).size
        expected = uncertainty_ntrial(
            f_err_now,
            fdot_err,
            f_step=1 / length,
            fdot_step=1 / length**2,
            n_grid=n_freq * n_z,
            ntrial_blind=n_freq,
            search_fdot=True,
        )
        assert expected > 5

        assert np.all(table["ntrial_eff"] >= expected * (1 - 1e-6))
        best = table[np.argmin(table["p_value"])]
        assert np.isclose(best["frequency"], self.FTRUE, atol=2e-3)
        assert np.isclose(best["ntrial_eff"], expected)
