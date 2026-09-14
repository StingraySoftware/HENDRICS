# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Monte Carlo calibration of the trials factor of HENzsearch and HENaccelsearch.

The effective number of independent trials of a search is measured on pure
noise. For each realization we record the smallest single-trial p-value over
the whole search plane. The number of trials ``N_eff(alpha)`` is the one that
makes ``1 - (1 - p_alpha)**N_eff = alpha``, where ``p_alpha`` is the
alpha-quantile of those minima. It is compared with the "naive" count the code
uses today and with the number of Fourier resolution elements of the search.

Experiments
-----------
qffa
    ``HENzsearch --fast`` (``search_with_qffa``), frequency only.
qffa_tail
    One reference ``qffa`` configuration, to be run with many realizations for
    the 0.1% false alarm level.
qffa_fdot
    The same, searching the frequency derivative as well.
folding
    ``HENzsearch`` without ``--fast`` (``z_n_search`` on a regular grid).
accel
    ``HENaccelsearch`` (the core of stingray's ``accelsearch``, without its
    internal threshold, so that the maximum is always seen).
sensitivity
    Recovered over true power of an injected signal, for the same grids.
targeted
    Calibration of the full targeted procedure: the argmin over the plane of
    the prior-corrected p-value, with several floors on the trials.
plot
    Make the figures from the saved tables.

Example::

    python notebooks/trials_calibration.py qffa --nreal 10000 --outdir calib
"""

import os

# Many single-threaded workers are much faster than a few multi-threaded ones
# on these small arrays
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import warnings
from functools import cache
from multiprocessing import Pool

import numpy as np

from astropy.table import Table

T = 1000.0
N_EVENTS = 5000
# HENzsearch defaults to 128, but that is ten times slower and the grid does
# not depend on it; a dedicated configuration checks that the trials don't
# either
NBIN = 32
ALPHAS = (0.5, 0.1, 0.01, 0.001)
F_CENTER = 9.5


def noise_events(rng, length=T, n_events=N_EVENTS):
    """Arrival times of a Poisson process."""
    return np.sort(rng.uniform(0, length, rng.poisson(n_events)))


def pulsed_events(rng, freq, amp, length=T, n_events=N_EVENTS):
    """Arrival times of a sinusoidally modulated Poisson process."""
    times = noise_events(rng, length, 2 * n_events)
    keep = rng.uniform(0, 1, times.size) < 0.5 * (1 + amp * np.cos(2 * np.pi * freq * times))
    return times[keep]


# ---------------------------------------------------------------------------
# The searches, exactly as the command line tools run them
# ---------------------------------------------------------------------------


def run_qffa(times, fmin, fmax, nharm, oversample, search_fdot, nbin=NBIN):
    from hendrics.efsearch import search_with_qffa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = search_with_qffa(
            times,
            fmin,
            fmax,
            nbin=max(nbin, 8 * nharm),
            n=nharm,
            oversample=oversample,
            search_fdot=search_fdot,
            silent=True,
        )
    if search_fdot:
        freqs, fdots, stats = res[:3]
    else:
        freqs, stats = res[:2]
        fdots = np.zeros_like(freqs)
    return np.asarray(freqs), np.asarray(fdots), np.asarray(stats)


def run_folding(times, fmin, fmax, nharm, oversample, nbin=NBIN):
    from stingray.pulse.search import z_n_search

    freqs = np.arange(fmin, fmax, 1 / oversample / T)
    _, stats = z_n_search(times, freqs, nharm=nharm, nbin=nbin)
    return freqs, np.zeros_like(freqs), np.asarray(stats)


@cache
def _responses(zmax, delta_z):
    from stingray.pulse.accelsearch import _create_responses

    # HENaccelsearch with zmax=0 would search no z at all; here it means the
    # plain FFT
    range_z = np.arange(-zmax, zmax, delta_z) if zmax > 0 else np.array([0.0])
    return range_z, _create_responses(range_z)


def accel_light_curve(times, fmax):
    """Bin the events as HENaccelsearch does (power-of-two number of bins)."""
    dt = 0.5 / (fmax * 5)
    nbins = int(2 ** np.ceil(np.log2(T / dt)))
    dt = T / nbins
    counts = np.histogram(times, bins=nbins, range=[0, T])[0].astype(float)
    return counts, dt


def run_accel(times, fmin, fmax, zmax, delta_z, interbin):
    """Leahy powers of the accelerated search plane, shape (n_z, n_freq)."""
    from scipy.fft import fft, fftfreq
    from scipy.signal import fftconvolve
    from stingray.pulse.accelsearch import interbin_fft

    counts, dt = accel_light_curve(times, fmax)
    spectr = fft(counts) * np.sqrt(2 / counts.sum())
    freq = fftfreq(counts.size, dt)
    band = np.flatnonzero((freq >= fmin) & (freq < fmax))

    range_z, responses = _responses(zmax, delta_z)
    # Convolving a padded slice gives exactly the same inner bins as
    # stingray's "same"-mode convolution of the whole spectrum, much faster
    pad = max(np.size(r) for r in responses) + 2
    chunk = spectr[band[0] - pad : band[-1] + 1 + pad]

    freqs_out = freq[band]
    powers = []
    for response in responses:
        if np.size(response) == 1:
            accel = chunk
        else:
            accel = fftconvolve(chunk, response, mode="same")
        accel = accel[pad : pad + band.size]
        if interbin:
            rf, accel = interbin_fft(np.arange(band.size), accel)
            freqs_out = freq[band[0]] + rf / T
        powers.append((accel * accel.conj()).real)
    powers = np.asarray(powers)
    fdots = np.broadcast_to((range_z / T**2)[:, None], powers.shape)
    freqs = np.broadcast_to(np.asarray(freqs_out)[None, :], powers.shape)
    return freqs, fdots, powers, band.size, range_z


def single_trial_p(stats, kind, nharm, freqs=None, fdots=None, interbin=False):
    from stingray.stats import z2_n_probability

    if kind == "accel":
        from hendrics.known_ephemeris import accel_single_trial_probability

        # Regular bins are chi^2 with 2 d.o.f.; the in-between bins of
        # interbinning have a stretched distribution, depending on z
        return accel_single_trial_probability(stats, freqs, fdots, T, interbin=interbin)
    return z2_n_probability(stats, n=nharm)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def configs(experiment):
    if experiment == "qffa":
        # oversample is the number of grid points per 1/T. The trials are
        # calibrated per resolution element, so a narrow band is enough; two
        # wider ones check that the count scales with it
        out = [
            dict(kind="qffa", nharm=n, oversample=o, search_fdot=False, band=0.25)
            for n, o in itertools.product((1, 2, 4), (1, 2, 4, 8))
        ]
        out += [
            dict(kind="qffa", nharm=2, oversample=4, search_fdot=False, band=b) for b in (1.0, 4.0)
        ]
        # ...and is it the same with the HENzsearch default of 128 bins?
        out += [dict(kind="qffa", nharm=2, oversample=4, search_fdot=False, band=0.25, nbin=128)]
    elif experiment == "qffa_tail":
        # A single reference configuration, run with many more realizations to
        # reach the 0.1% false alarm level
        out = [dict(kind="qffa", nharm=2, oversample=4, search_fdot=False, band=0.25)]
    elif experiment == "qffa_fdot":
        # What the fast search is meant for
        out = [
            dict(kind="qffa", nharm=n, oversample=o, search_fdot=True, band=0.05)
            for n, o in itertools.product((1, 2, 4), (1, 2, 4))
        ]
        # Does the count scale with the frequency band?
        out += [dict(kind="qffa", nharm=2, oversample=4, search_fdot=True, band=0.2)]
    elif experiment == "folding":
        out = [
            dict(kind="folding", nharm=n, oversample=o, search_fdot=False, band=0.25)
            for n, o in itertools.product((1, 2), (1, 2, 4, 8))
        ]
    elif experiment == "accel":
        # zmax=100 is the HENaccelsearch default
        out = [
            dict(kind="accel", nharm=1, zmax=z, delta_z=d, interbin=i, band=1.0)
            for z, d, i in itertools.product((0, 10, 100), (1.0, 0.5), (False, True))
            if not (z == 0 and d == 0.5)
        ]
    elif experiment == "sensitivity":
        out = (
            [
                dict(kind="qffa", nharm=2, oversample=o, search_fdot=False, band=0.02)
                for o in (1, 2, 4, 8)
            ]
            + [
                dict(kind="folding", nharm=2, oversample=o, search_fdot=False, band=0.02)
                for o in (1, 2, 4, 8)
            ]
            + [
                dict(kind="accel", nharm=1, zmax=0, delta_z=1.0, interbin=i, band=0.02)
                for i in (False, True)
            ]
        )
    elif experiment == "targeted":
        out = [
            # HENzsearch --fast searches fdot by default
            dict(kind="qffa", nharm=2, oversample=4, search_fdot=True, band=0.05),
            dict(kind="accel", nharm=1, zmax=10, delta_z=1.0, interbin=False, band=1.0),
        ]
    else:
        raise ValueError(f"Unknown experiment {experiment}")
    return out


def label(cfg):
    return "_".join(f"{k}{v}" for k, v in cfg.items())


def run_search(cfg, times):
    fmin, fmax = F_CENTER - cfg["band"] / 2, F_CENTER + cfg["band"] / 2
    if cfg["kind"] == "qffa":
        freqs, fdots, stats = run_qffa(
            times,
            fmin,
            fmax,
            cfg["nharm"],
            cfg["oversample"],
            cfg["search_fdot"],
            nbin=cfg.get("nbin", NBIN),
        )
        naive = stats.size / cfg["oversample"]
        n_z = 1
    elif cfg["kind"] == "folding":
        freqs, fdots, stats = run_folding(
            times, fmin, fmax, cfg["nharm"], cfg["oversample"], nbin=cfg.get("nbin", NBIN)
        )
        naive = stats.size / cfg["oversample"]
        n_z = 1
    else:
        freqs, fdots, stats, n_freq, range_z = run_accel(
            times, fmin, fmax, cfg["zmax"], cfg["delta_z"], cfg["interbin"]
        )
        # What stingray reports as ``ntrial``
        naive = n_freq
        n_z = range_z.size
    return freqs, fdots, stats, naive, n_z


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _warm_up():
    """Compile the numba functions once per worker, before timing anything."""
    rng = np.random.default_rng(0)
    times = noise_events(rng, length=100.0, n_events=500)
    for search_fdot in (False, True):
        run_qffa(times, 9.0, 9.05, 1, 2, search_fdot)
    run_folding(times, 9.0, 9.001, 1, 1)


def noise_worker(job):
    cfg, seed = job
    rng = np.random.default_rng(seed)
    freqs, fdots, stats, naive, n_z = run_search(cfg, noise_events(rng))
    p = single_trial_p(
        stats, cfg["kind"], cfg["nharm"], freqs, fdots, interbin=cfg.get("interbin", False)
    )
    out = dict(p_min=float(p.min()), naive=float(naive), n_cells=int(stats.size))
    out["fdot_span"] = float(np.ptp(fdots))
    out["n_fdot"] = int(np.unique(fdots).size)
    if cfg["kind"] == "accel":
        # The plain FFT row, to separate the z trials from the frequency ones
        z0 = np.argmin(np.abs(fdots[:, 0]))
        out["p_min_z0"] = float(p[z0].min())
    return out


def sensitivity_worker(job):
    from stingray.pulse.search import z_n_search

    cfg, seed = job
    rng = np.random.default_rng(seed)
    nharm = cfg["nharm"]
    # Anywhere within 8 resolution elements of the centre, so that every
    # position relative to any of these grids is sampled
    f_true = F_CENTER + rng.uniform(0, 8) / T
    # Z^2 of about 60 at the true frequency
    amp = np.sqrt(2 * 58 / N_EVENTS)
    times = pulsed_events(rng, f_true, amp)
    freqs, _, stats, _, _ = run_search(cfg, times)
    # Coarse grids (e.g. the fast search at oversample 1) can be wider than
    # 2/T: always include the nearest grid points
    grid_step = np.median(np.diff(np.unique(np.ravel(freqs))))
    window = np.abs(freqs - f_true) < 2 / T + grid_step
    # The reference is the unbinned statistic at the exact frequency (for a
    # single harmonic, this is also the Leahy power of the unbinned signal)
    _, true_stat = z_n_search(times, [f_true], nharm=nharm, nbin=1024)
    return dict(recovered=float(stats[window].max()), true=float(true_stat[0]))


def targeted_worker(job):
    from hendrics.known_ephemeris import best_candidate_ntrial, effective_ntrial

    cfg, seed, blind_scales, floors = job
    rng = np.random.default_rng(seed)
    freqs, fdots, stats, naive, _ = run_search(cfg, noise_events(rng))
    p = single_trial_p(
        stats, cfg["kind"], cfg["nharm"], freqs, fdots, interbin=cfg.get("interbin", False)
    )
    # -n * log1p(-p) is monotonic in the corrected p-value, and cheap
    log1m = -np.log1p(-np.clip(p, 0, 1 - 1e-16))

    search_fdot = stats.ndim > 1 and stats.shape[0] > 1
    if stats.ndim > 1:
        f_step = np.median(np.diff(freqs[0]))
        fdot_step = np.median(np.diff(fdots[:, 0])) if search_fdot else None
    else:
        f_step = np.median(np.diff(freqs))
        fdot_step = None

    out = dict(p_min=float(p.min()), naive=float(naive), n_cells=int(stats.size))
    for scale in blind_scales:
        ntrial_blind = naive * scale
        # The prior sits at the centre of the band, fixed before the data
        rank = effective_ntrial(
            freqs - F_CENTER,
            fdots if search_fdot else 0.0,
            f_step=f_step,
            fdot_step=fdot_step,
            n_grid=stats.size,
            ntrial_blind=ntrial_blind,
            search_fdot=search_fdot,
        )
        for floor in floors:
            n_eff = np.clip(rank, floor, ntrial_blind)
            best = np.argmin(n_eff * log1m)
            out[f"pcorr_s{scale}_f{floor}"] = float(-np.expm1(-(n_eff * log1m).flat[best]))
            # The same, charging the price of picking the best candidate
            n_best = best_candidate_ntrial(n_eff, ntrial_min=floor, ntrial_blind=ntrial_blind)
            out[f"pbest_s{scale}_f{floor}"] = float(-np.expm1(-np.min(n_best * log1m)))
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def n_eff_from_minima(p_min, alpha):
    """Number of independent trials implied by the alpha-quantile of p_min."""
    p_alpha = np.quantile(p_min, alpha)
    return np.log1p(-alpha) / np.log1p(-p_alpha)


def summarize_noise(cfg, rows, nboot=300, seed=0):
    rng = np.random.default_rng(seed)
    p_min = np.array([r["p_min"] for r in rows])
    n_res = cfg["band"] * T
    out = dict(cfg)
    out.update(nreal=p_min.size, naive=rows[0]["naive"], n_cells=rows[0]["n_cells"])
    out["n_res"] = n_res
    out["fdot_span_T2"] = rows[0]["fdot_span"] * T**2
    # Two candidate resolution elements in fdot: 1/T^2, and 4/T^2, which gives
    # the same phase error at the edges of the observation as 1/T in frequency
    # when phases are measured from its middle
    n_fdot = rows[0].get("n_fdot", 1)
    fdot_range_T2 = out["fdot_span_T2"] * n_fdot / (n_fdot - 1) if n_fdot > 1 else 0.0
    res_units = {"res": n_res}
    if n_fdot > 1:
        res_units["res_fdot1"] = n_res * fdot_range_T2
        res_units["res_fdot4"] = n_res * fdot_range_T2 / 4
    for unit, value in res_units.items():
        out[f"n_{unit}"] = value
    columns = [("p_min", p_min)]
    if "p_min_z0" in rows[0]:
        columns.append(("p_min_z0", np.array([r["p_min_z0"] for r in rows])))
    for name, values in columns:
        suffix = "" if name == "p_min" else "_z0"
        for alpha in ALPHAS:
            if alpha * values.size < 5:
                continue
            est = n_eff_from_minima(values, alpha)
            boot = [n_eff_from_minima(rng.choice(values, values.size), alpha) for _ in range(nboot)]
            lo, hi = np.percentile(boot, [16, 84])
            out[f"neff{suffix}_a{alpha}"] = est
            out[f"neff{suffix}_a{alpha}_lo"] = lo
            out[f"neff{suffix}_a{alpha}_hi"] = hi
            out[f"ratio_naive{suffix}_a{alpha}"] = est / out["naive"]
            for unit, value in res_units.items():
                out[f"ratio_{unit}{suffix}_a{alpha}"] = est / value
    return out


def summarize_sensitivity(cfg, rows):
    ratio = np.array([r["recovered"] / r["true"] for r in rows])
    out = dict(cfg)
    out.update(nreal=ratio.size, mean_ratio=ratio.mean(), min_ratio=np.percentile(ratio, 5))
    out["mean_ratio_err"] = ratio.std() / np.sqrt(ratio.size)
    return out


def summarize_targeted(cfg, rows, blind_scales, floors):
    out = []
    p_min = np.array([r["p_min"] for r in rows])
    n_blind_true = n_eff_from_minima(p_min, 0.5)
    for scale, floor in itertools.product(blind_scales, floors):
        pc = np.array([r[f"pcorr_s{scale}_f{floor}"] for r in rows])
        ntrial_blind = rows[0]["naive"] * scale
        row = dict(cfg)
        row.update(
            nreal=pc.size,
            blind_scale=scale,
            ntrial_blind=ntrial_blind,
            ntrial_blind_mc=n_blind_true,
            floor=floor,
            predicted=1 + np.log(max(ntrial_blind / floor, 1)),
        )
        for alpha in (0.1, 0.01, 0.001):
            if alpha * pc.size < 5:
                continue
            row[f"inflation_a{alpha}"] = np.mean(pc < alpha) / alpha
            if f"pbest_s{scale}_f{floor}" in rows[0]:
                pb = np.array([r[f"pbest_s{scale}_f{floor}"] for r in rows])
                row[f"inflation_best_a{alpha}"] = np.mean(pb < alpha) / alpha
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _save(rows, fname):
    table = Table(rows=rows)
    table.write(fname, overwrite=True)
    print(table)
    return table


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "experiment",
        choices=[
            "qffa",
            "qffa_tail",
            "qffa_fdot",
            "folding",
            "accel",
            "sensitivity",
            "targeted",
            "plot",
        ],
    )
    parser.add_argument("--nreal", type=int, default=10000)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="trials_calibration")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--blind-scales",
        type=float,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Multiples of the naive blind count to try (targeted only)",
    )
    parser.add_argument(
        "--floors",
        type=float,
        nargs="+",
        default=[1, 10, 100],
        help="Floors on the trials charged near the prior (targeted only)",
    )
    args = parser.parse_args(args)
    os.makedirs(args.outdir, exist_ok=True)

    if args.experiment == "plot":
        return plot(args.outdir)

    rows_out = []
    # A single pool for the whole run: starting a worker means importing
    # HENDRICS and compiling its numba functions, which takes far longer than
    # a realization
    with Pool(args.nproc, initializer=_warm_up) as pool:
        for i_cfg, cfg in enumerate(configs(args.experiment)):
            seeds = np.random.SeedSequence([args.seed, i_cfg]).spawn(args.nreal)
            if args.experiment == "sensitivity":
                worker, jobs = sensitivity_worker, [(cfg, s) for s in seeds]
            elif args.experiment == "targeted":
                worker = targeted_worker
                jobs = [(cfg, s, tuple(args.blind_scales), tuple(args.floors)) for s in seeds]
            else:
                worker, jobs = noise_worker, [(cfg, s) for s in seeds]

            print(f"Running {label(cfg)} ({args.nreal} realizations)", flush=True)
            chunksize = max(1, min(16, args.nreal // (4 * args.nproc)))
            rows = list(pool.imap_unordered(worker, jobs, chunksize=chunksize))

            Table(rows=rows).write(
                os.path.join(args.outdir, f"raw_{args.experiment}_{label(cfg)}.ecsv"),
                overwrite=True,
            )
            if args.experiment == "sensitivity":
                rows_out.append(summarize_sensitivity(cfg, rows))
            elif args.experiment == "targeted":
                rows_out.extend(summarize_targeted(cfg, rows, args.blind_scales, args.floors))
            else:
                rows_out.append(summarize_noise(cfg, rows))

    # Mixed configurations have different keys: fill the gaps
    keys = list(dict.fromkeys(k for row in rows_out for k in row))
    rows_out = [{k: row.get(k, np.nan) for k in keys} for row in rows_out]
    for row in rows_out:
        for k, v in row.items():
            if isinstance(v, bool):
                row[k] = int(v)
    return _save(rows_out, os.path.join(args.outdir, f"summary_{args.experiment}.ecsv"))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot(outdir):
    """Make the figures of the technical documentation from the summary tables."""
    import matplotlib.pyplot as plt

    from hendrics.known_ephemeris import (
        _ACCEL_TRIALS_PER_CELL,
        _QFFA_OVERSAMPLES,
        _QFFA_TRIALS_PER_ELEMENT,
        interbin_stretch,
    )

    plt.rcParams.update(
        {"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 7}
    )

    def _read(name):
        fname = os.path.join(outdir, f"summary_{name}.ecsv")
        return Table.read(fname) if os.path.exists(fname) else None

    def _errorbar(ax, x, table, col, **kwargs):
        lo = table[f"neff_{col}_lo"] / table[f"neff_{col}"] * table[f"ratio_{col}"]
        hi = table[f"neff_{col}_hi"] / table[f"neff_{col}"] * table[f"ratio_{col}"]
        y = table[f"ratio_{col}"]
        ax.errorbar(x, y, yerr=[y - lo, hi - y], ms=3, capsize=1.5, **kwargs)

    qffa, qffa_fdot, accel = _read("qffa"), _read("qffa_fdot"), _read("accel")
    if qffa is not None and qffa_fdot is not None and accel is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        ax = axes[0]
        colors = {1: "C0", 2: "C1", 4: "C2"}
        # The oversampling scan of the frequency-only search (32 phase bins)
        scan = qffa[(qffa["band"] == 0.25) & ~(qffa["nbin"] > 0)]
        for i_n, nharm in enumerate((1, 2, 4)):
            color = colors[nharm]
            sub = scan[scan["nharm"] == nharm]
            sub.sort("oversample")
            # Resolution elements are grid points / oversample
            sub["ratio_os_a0.01"] = sub["ratio_naive_a0.01"]
            for c in ("lo", "hi"):
                sub[f"neff_os_a0.01_{c}"] = sub[f"neff_a0.01_{c}"]
            sub["neff_os_a0.01"] = sub["neff_a0.01"]
            _errorbar(
                ax,
                sub["oversample"] * 0.97,
                sub,
                "os_a0.01",
                color=color,
                marker="o",
                ls="-",
                label=rf"$Z^2_{nharm}$, frequency",
            )
            sub = qffa_fdot[(qffa_fdot["nharm"] == nharm) & (qffa_fdot["band"] == 0.05)]
            sub.sort("oversample")
            sub["ratio_f4_a0.01"] = sub["ratio_res_fdot4_a0.01"]
            for c in ("", "_lo", "_hi"):
                sub[f"neff_f4_a0.01{c}"] = sub[f"neff_a0.01{c}"]
            _errorbar(
                ax,
                sub["oversample"] * 1.03,
                sub,
                "f4_a0.01",
                color=color,
                marker="s",
                ls="--",
                label=rf"$Z^2_{nharm}$, frequency and $\dot\nu$",
            )
            for search_fdot, marker in ((False, "_"), (True, "x")):
                oversamples = _QFFA_OVERSAMPLES[search_fdot]
                ax.scatter(
                    oversamples,
                    _QFFA_TRIALS_PER_ELEMENT[search_fdot][i_n, : len(oversamples)],
                    marker=marker,
                    color=color,
                    s=25,
                    zorder=3,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("--oversample (grid points per resolution element)")
        ax.set_ylabel("Independent trials per resolution element")
        ax.set_title("HENzsearch --fast, 1% false alarm probability", fontsize=7)
        ax.set_ylim(0.5, 100)
        ax.legend(ncol=2, loc="upper left")

        ax = axes[1]
        rows = list(accel)
        labels = []
        for i, row in enumerate(rows):
            n_z = max(np.arange(-row["zmax"], row["zmax"], row["delta_z"]).size, 1)
            n_z = n_z if row["zmax"] > 0 else 1
            norm = row["naive"] * n_z
            for alpha, color, dx in ((0.1, "C0", -0.12), (0.01, "C3", 0.12)):
                y = row[f"neff_a{alpha}"] / norm
                lo, hi = row[f"neff_a{alpha}_lo"] / norm, row[f"neff_a{alpha}_hi"] / norm
                ax.errorbar(
                    i + dx,
                    y,
                    yerr=[[y - lo], [hi - y]],
                    marker="o",
                    ms=3,
                    capsize=1.5,
                    color=color,
                    label=rf"$\alpha$={alpha}" if i == 0 else None,
                )
            independent, fine, coarse = _ACCEL_TRIALS_PER_CELL[bool(row["interbin"])]
            if row["zmax"] == 0:
                tab = independent
            else:
                tab = fine if row["delta_z"] <= 0.5 else coarse
            ax.scatter(
                i,
                tab,
                marker="_",
                s=150,
                color="k",
                zorder=3,
                label="tabulated" if i == 0 else None,
            )
            labels.append(
                f"zmax {row['zmax']:g}, dz {row['delta_z']:g}"
                + (", interbin" if row["interbin"] else "")
            )
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Independent trials per frequency bin per z row")
        ax.set_title("HENaccelsearch", fontsize=7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "trials_calibration.png"), dpi=200)
        plt.close(fig)

    sens = _read("sensitivity")
    if sens is not None:
        fig, ax = plt.subplots(figsize=(3.5, 2.65))
        for kind, marker in (("qffa", "o"), ("folding", "s")):
            sub = sens[sens["kind"] == kind]
            ax.errorbar(
                sub["oversample"],
                sub["mean_ratio"],
                yerr=sub["mean_ratio_err"],
                marker=marker,
                ms=3,
                label="--fast" if kind == "qffa" else "folding",
            )
        for row in sens[sens["kind"] == "accel"]:
            ax.axhline(
                row["mean_ratio"],
                ls="--" if row["interbin"] else ":",
                color="grey",
                label="HENaccelsearch" + (" --interbin" if row["interbin"] else ""),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("--oversample")
        ax.set_ylabel("Recovered / true power")
        ax.set_title(r"HENzsearch, $Z^2_2$", fontsize=7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "sensitivity_vs_oversample.png"), dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    z = np.concatenate([np.arange(0, 2, 0.05), np.arange(2, 100.5, 0.5)])
    ax.plot(z, interbin_stretch(z), color="C0")
    integer = np.arange(0, 101)
    ax.scatter(integer, interbin_stretch(integer), s=4, color="C1", zorder=3, label="integer z")
    ax.axhline(np.pi**2 / 8, color="grey", ls=":", label=r"$\pi^2/8$")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("z")
    ax.set_ylabel("Noise stretch s(z) of in-between bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "interbin_stretch.png"), dpi=200)
    plt.close(fig)

    targeted = _read("targeted")
    if targeted is not None:
        from hendrics.known_ephemeris import accel_calibrated_ntrial, qffa_calibrated_ntrial

        # The rows with the blind count that the tools really charge
        calibrated = {
            "qffa": qffa_calibrated_ntrial(8192 / 4**2, nharm=2, oversample=4, search_fdot=True),
            "accel": accel_calibrated_ntrial(1000, zmax=10, delta_z=1, interbin=False),
        }
        names = {"qffa": "HENzsearch --fast", "accel": "HENaccelsearch"}
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.65), sharey=True)
        for ax, alpha in zip(axes, (0.1, 0.01)):
            for kind, color in (("qffa", "C0"), ("accel", "C3")):
                sub = targeted[
                    (targeted["kind"] == kind)
                    & np.isclose(targeted["ntrial_blind"], calibrated[kind])
                ]
                sub.sort("floor")
                ax.plot(
                    sub["floor"],
                    sub[f"inflation_a{alpha}"],
                    color=color,
                    ls=":",
                    marker="o",
                    ms=3,
                    label=f"{names[kind]}, p_value",
                )
                ax.plot(
                    sub["floor"],
                    sub[f"inflation_best_a{alpha}"],
                    color=color,
                    ls="-",
                    marker="s",
                    ms=3,
                    label=f"{names[kind]}, p_value_best",
                )
            ax.axhline(1, color="grey", lw=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Floor on the trials (ntrial_min)")
            ax.set_title(rf"$\alpha$ = {alpha}", fontsize=7)
        axes[0].set_ylabel("False alarm rate / nominal")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "targeted_false_alarms.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
