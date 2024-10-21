import numpy as np
from stingray.events import EventList
from stingray.lightcurve import Lightcurve

from hendrics.efsearch import fit
from hendrics.ffa import ffa_search


# @pytest.mark.skipif('not HAS_NUMBA')
def test_ffa():
    period = 0.01
    pmin = 0.0095
    pmax = 0.0105
    dt = 10 ** int(np.log10(period)) / 256
    length = 1
    times = np.arange(0, length, dt)

    flux = 10 + np.cos(2 * np.pi * times / period)

    lc_cont = Lightcurve(times, flux, err_dist="gauss")

    ev = EventList()
    ev.simulate_times(lc_cont)
    lc = Lightcurve.make_lightcurve(ev.time, dt=dt, tstart=0, tseg=length)

    per, st = ffa_search(lc.counts, dt, pmin, pmax)
    #  fit_sinc wants frequencies, not periods
    model = fit(1 / per[::-1], st[::-1], 1 / period, obs_length=10)
    assert np.isclose(1 / model.mean, period, atol=1e-6)


# @pytest.mark.skipif('not HAS_NUMBA')
def test_ffa_large_intv():
    period = 0.01
    pmin = 0.002789345
    pmax = 0.0105
    dt = 10 ** int(np.log10(0.002789345)) / 20
    length = 5
    times = np.arange(0, length, dt)

    flux = 10 + np.cos(2 * np.pi * times / period)

    lc_cont = Lightcurve(times, flux, err_dist="gauss")

    ev = EventList()
    ev.simulate_times(lc_cont)
    lc = Lightcurve.make_lightcurve(ev.time, dt=dt, tstart=0, tseg=length)

    per, st = ffa_search(lc.counts, dt, pmin, pmax)
    #  fit_sinc wants frequencies, not periods
    model = fit(1 / per[::-1], st[::-1], 1 / period, obs_length=10)
    assert np.isclose(1 / model.mean, period, atol=1e-6)


# @pytest.mark.skipif('not HAS_NUMBA')


def test_ffa_vs_folding_search():
    import time

    from stingray.events import EventList
    from stingray.lightcurve import Lightcurve

    from hendrics.efsearch import folding_search

    period = 0.01
    pmin = 0.0095
    pmax = 0.0105
    dt = 10 ** int(np.log10(period)) / 256
    length = 1
    times = np.arange(0, length, dt)

    flux = 5 + 1 * np.cos(2 * np.pi * times / period)

    lc_cont = Lightcurve(times, flux, err_dist="gauss")

    ev = EventList()
    ev.simulate_times(lc_cont)
    lc = Lightcurve.make_lightcurve(ev.time, dt=dt, tstart=0, tseg=length)

    t0 = time.time()
    per, st = ffa_search(lc.counts, dt, pmin, pmax)
    t1 = time.time()
    print(f"FFA completed in {t1 - t0:.1e} s")
    t1 = time.time()
    freqs, stats, _, _ = folding_search(ev, 1 / pmax, 1 / pmin, oversample=3, nbin=128)
    t2 = time.time()
    print(f"Standard search completed in {t2 - t1:.1e} s")

    comparable_stats = np.array([st[idx] for idx in np.searchsorted(per, 1 / freqs)])
    assert (comparable_stats - stats + 127).std() < 127
