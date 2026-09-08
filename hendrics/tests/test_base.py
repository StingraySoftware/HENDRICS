import os

import numpy as np
import pytest
from stingray.events import EventList

from hendrics.base import (
    HAS_PINT,
    deorbit_events,
    hist3d_numba_seq,
    hist3d_numba_seq_weight,
    histnd_numba_seq,
    normalize_dyn_profile,
)
from hendrics.tests import _dummy_par


class TestNormalize:
    @classmethod
    def setup_class(cls):
        cls.mean = 20
        cls.std = 2
        cls.hist = [np.random.normal(cls.mean, cls.std, 100_000) for i in range(4)]

    @pytest.mark.parametrize("kind", ["mean", "median", ""])
    def test_normalize_norm(self, kind):
        norm = kind + "norm"
        nhist = normalize_dyn_profile(self.hist, norm)
        assert np.allclose(nhist.mean(axis=1), 0, atol=0.01)
        assert np.allclose(nhist.std(axis=1), self.std / self.mean, atol=0.01)

    @pytest.mark.parametrize("kind", ["mean", "median", ""])
    def test_normalize_std(self, kind):
        norm = kind + "std"
        nhist = normalize_dyn_profile(self.hist, norm)
        assert np.allclose(nhist.mean(axis=1), 0, atol=0.01)
        assert np.allclose(nhist.std(axis=1), 1, atol=0.01)

    @pytest.mark.parametrize("kind", ["mean", "median", ""])
    def test_normalize_norm_smooth(self, kind):
        norm = kind + "norm" + "_smooth"
        nhist = normalize_dyn_profile(self.hist, norm)
        assert np.allclose(nhist.mean(axis=1), 0, atol=0.01)
        # Smoothing reduces the standard deviation
        assert np.all(nhist.std(axis=1) < self.std / self.mean)

    @pytest.mark.parametrize("kind", ["mean", "median", ""])
    def test_normalize_to1(self, kind):
        norm = kind + "to1"
        nhist = normalize_dyn_profile(self.hist, norm)
        assert np.allclose(nhist.min(axis=1), 0, atol=0.01)
        assert np.allclose(nhist.max(axis=1), 1, atol=0.01)

    @pytest.mark.parametrize("kind", ["mean", "median", ""])
    def test_normalize_ratios(self, kind):
        norm = kind + "ratios"
        nhist = normalize_dyn_profile(self.hist, norm)
        assert np.allclose(nhist.mean(axis=1), 1, atol=0.01)


def test_deorbit_badpar():
    ev = np.asarray(1)
    with pytest.warns(UserWarning, match="No parameter file specified"):
        ev_deor = deorbit_events(ev, None)
    assert ev_deor == ev


def test_deorbit_non_existing_par():
    ev = np.asarray(1)
    with pytest.raises(FileNotFoundError, match="Parameter file warjladsfjqpeifjsdk.par"):
        deorbit_events(ev, "warjladsfjqpeifjsdk.par")


@pytest.mark.remote_data
@pytest.mark.skipif("not HAS_PINT")
def test_deorbit_bad_mjdref():
    from hendrics.base import deorbit_events

    ev = EventList(np.arange(100), gti=np.asarray([[0, 2]]))
    ev.mjdref = 2
    par = _dummy_par("bububu.par")
    with pytest.raises(ValueError, match="MJDREF is very low .<01-01-1950., "):
        deorbit_events(ev, par)
    os.remove("bububu.par")


@pytest.mark.remote_data
@pytest.mark.skipif("not HAS_PINT")
def test_deorbit_inverse():
    from hendrics.base import deorbit_events

    ev = EventList(
        np.sort(np.random.uniform(0, 1000, 10)),
        gti=np.asarray([[0, 1000]]),
        mjdref=55000,
    )
    par = _dummy_par("bububu.par", pb=1.0, a1=30)
    ev2 = deorbit_events(ev, par)
    ev3 = deorbit_events(ev, par, invert=True)
    assert np.allclose(ev.time - ev2.time, -(ev.time - ev3.time), atol=1e-6)
    os.remove("bububu.par")


@pytest.mark.remote_data
@pytest.mark.skipif("not HAS_PINT")
def test_deorbit_run():
    from hendrics.base import deorbit_events

    ev = EventList(np.arange(0, 210000, 1000), gti=np.asarray([[0.0, 210000]]))

    ev.mjdref = 56000.0
    ev.ephem = "de200"
    par = _dummy_par("bububu.par")
    _ = deorbit_events(ev, par)

    os.remove("bububu.par")


class TestHistograms:
    """Regression tests for the numba histogram kernels.

    Because the kernels are compiled in ``nopython`` mode there is no bounds
    checking at runtime, so a missing guard on one of the axes writes to
    arbitrary memory instead of raising.
    """

    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(20260908)
        # Deliberately generate data that spill out of ``ranges`` on every axis
        cls.x = rng.uniform(-1.0, 2.0, 1000)
        cls.y = rng.uniform(1.0, 4.0, 1000)
        cls.z = rng.uniform(3.0, 6.0, 1000)
        cls.weights = rng.uniform(0.0, 1.0, 1000)
        cls.bins = (5, 6, 7)
        cls.ranges = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]

    def test_hist3d_out_of_range(self):
        H, _ = np.histogramdd(
            (self.x, self.y, self.z), bins=self.bins, range=[tuple(r) for r in self.ranges]
        )
        Hn = hist3d_numba_seq((self.x, self.y, self.z), bins=self.bins, ranges=self.ranges)
        assert np.all(H == Hn)

    def test_hist3d_weight_out_of_range(self):
        H, _ = np.histogramdd(
            (self.x, self.y, self.z),
            bins=self.bins,
            range=[tuple(r) for r in self.ranges],
            weights=self.weights,
        )
        Hn = hist3d_numba_seq_weight(
            (self.x, self.y, self.z), self.weights, bins=self.bins, ranges=self.ranges
        )
        assert np.allclose(H, Hn)

    def test_histnd_out_of_range(self):
        H, _ = np.histogramdd(
            (self.x, self.y, self.z), bins=self.bins, range=[tuple(r) for r in self.ranges]
        )
        Hn = histnd_numba_seq(
            np.array([self.x, self.y, self.z]),
            bins=np.array(self.bins),
            ranges=np.array(self.ranges),
        )
        assert np.all(H == Hn)
