# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Targeted searches around a known (or extrapolated) spin ephemeris.

When a pulsar has a previously measured spin solution, a blind search over a
wide frequency band is needlessly expensive in terms of statistical
significance: the number of independent trials -- and hence the detection
threshold -- is set by the whole search band, even though we only really care
about a small region around the expected frequency.

This module implements an *a priori ordering* correction.  All the cells of
the search plane are ranked, before looking at the data, by their distance
from the extrapolated ephemeris.  A candidate landing at rank ``k`` is then
charged ``k`` trials instead of the full blind-search count, because the
statement "I would have accepted a peak anywhere at rank <= k" is a perfectly
well defined test that was fixed in advance.  This is the same union-bound
argument that justifies the blind trials factor, applied to a subset of cells
chosen a priori.

The correction degrades gracefully: a candidate sitting exactly on the
prediction costs a single trial, one slightly off costs a handful, and one far
away costs the full blind-search number (the result is always capped there, so
using a prior can never make a candidate look *less* significant than it would
in a blind search).

.. warning::
    The ordering must be genuinely fixed before looking at the data.  Choosing
    the "known" ephemeris after inspecting the periodogram invalidates the
    correction entirely.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "effective_ntrial",
    "ephemeris_from_parfile",
    "extrapolate_ephemeris",
    "prior_corrected_p_value",
]


def extrapolate_ephemeris(freq, fdot=0.0, fddot=0.0, pepoch=None, target_epoch=None):
    """Extrapolate a spin solution to a different epoch.

    Both epochs are MJDs; the Taylor expansion is evaluated in seconds.

    Parameters
    ----------
    freq : float
        Spin frequency at ``pepoch``, in Hz.

    Other Parameters
    ----------------
    fdot : float, default 0
        First frequency derivative at ``pepoch``, in Hz/s.
    fddot : float, default 0
        Second frequency derivative at ``pepoch``, in Hz/s^2.
    pepoch : float
        Reference epoch of the input solution, in MJD.
    target_epoch : float
        Epoch the solution should be extrapolated to, in MJD.

    Returns
    -------
    freq : float
        Extrapolated spin frequency, in Hz.
    fdot : float
        Extrapolated first frequency derivative, in Hz/s.
    fddot : float
        Second frequency derivative (unchanged), in Hz/s^2.

    Examples
    --------
    >>> # A pulsar spinning down by 1e-5 Hz over ten days
    >>> f, fd, fdd = extrapolate_ephemeris(
    ...     1.0, fdot=-1e-5 / 864000, pepoch=50000, target_epoch=50010)
    >>> assert np.isclose(f, 1.0 - 1e-5)
    >>> # Going back to the original epoch gives the original solution
    >>> f2, fd2, _ = extrapolate_ephemeris(
    ...     f, fdot=fd, pepoch=50010, target_epoch=50000)
    >>> assert np.isclose(f2, 1.0)
    """
    if pepoch is None or target_epoch is None:
        raise ValueError("Both pepoch and target_epoch are needed to extrapolate an ephemeris")

    dt = (target_epoch - pepoch) * 86400.0

    new_freq = freq + fdot * dt + 0.5 * fddot * dt**2
    new_fdot = fdot + fddot * dt

    return new_freq, new_fdot, fddot


def ephemeris_from_parfile(parfile):
    """Read F0, F1, F2 and PEPOCH from a TEMPO2/PINT parameter file.

    Parameters
    ----------
    parfile : str
        Path to a parameter file in TEMPO2/PINT format.

    Returns
    -------
    freq : float
        Spin frequency at ``pepoch``, in Hz.
    fdot : float
        First frequency derivative, in Hz/s.
    fddot : float
        Second frequency derivative, in Hz/s^2.
    pepoch : float
        Reference epoch, in MJD.
    """
    from .base import get_model

    if get_model is None:
        raise ImportError("PINT is needed to read parameter files")

    model = get_model(parfile)

    def _value(name, default=0.0):
        if not hasattr(model, name) or getattr(model, name).value is None:
            return default
        return float(getattr(model, name).value)

    freq = _value("F0", None)
    if freq is None:
        raise ValueError(f"No spin frequency (F0) found in {parfile}")

    return freq, _value("F1"), _value("F2"), _value("PEPOCH", None)


def effective_ntrial(
    delta_f,
    delta_fdot=0.0,
    f_step=None,
    fdot_step=None,
    n_grid=None,
    ntrial_blind=None,
    search_fdot=True,
):
    """Number of trials to charge a candidate offset from a known ephemeris.

    The search cells are ranked, before looking at the data, by their distance
    from the prior.  A candidate is charged the number of cells at least as
    close to the prior as itself: the length of an interval in a
    frequency-only search, the area of a disc when a frequency derivative is
    searched as well.

    That count is expressed as a *fraction of the blind-search trial count*,
    rather than derived from the frequency resolution.  The distinction
    matters.  The number of independent trials of a folding search is not
    simply the band width divided by ``1/T``: the statistic is a smooth
    function of frequency, so the effective trial count of its maximum depends
    on the threshold it is evaluated at, and Monte Carlo simulations show the
    naive ``1/T`` counting to be low by a factor of several.  Taking a ratio
    sidesteps the question: the result is a faithful rescaling of whatever
    blind normalization the search already uses.  Simulations show it to be
    accurate to better than a factor 1.5 in trials -- about 0.1 sigma -- over
    four decades in significance.

    Parameters
    ----------
    delta_f : float or array of floats
        Offset between the candidate and the expected frequency, in Hz. Both
        signs are accepted; only the absolute value matters.

    Other Parameters
    ----------------
    delta_fdot : float or array of floats, default 0
        Offset between the candidate and the expected frequency derivative,
        in Hz/s. Ignored if ``search_fdot`` is False.
    f_step : float
        Step of the frequency grid, in Hz.
    fdot_step : float
        Step of the frequency derivative grid, in Hz/s. Only needed if
        ``search_fdot`` is True.
    n_grid : int
        Total number of points in the search grid.
    ntrial_blind : int
        Number of independent trials of the equivalent blind search. The
        result is capped here, since a prior can never cost more trials than
        searching the whole band.
    search_fdot : bool, default True
        Whether the search covers frequency derivatives as well.

    Returns
    -------
    ntrial : float or array of floats
        Effective number of trials, between 1 and ``ntrial_blind``.

    Examples
    --------
    >>> # A frequency-only search: 1000 grid points, 200 independent trials
    >>> kw = dict(f_step=1e-5, n_grid=1000, ntrial_blind=200,
    ...           search_fdot=False)
    >>> # Right on the prediction: a single trial
    >>> assert np.isclose(effective_ntrial(0, **kw), 1)
    >>> # A tenth of the way across the band: a tenth of the blind trials
    >>> assert np.isclose(effective_ntrial(50 * 1e-5, **kw), 20)
    >>> # Far away: exactly what the blind search would have cost
    >>> assert np.isclose(effective_ntrial(1, **kw), 200)
    """
    if f_step is None:
        raise ValueError("The frequency grid step f_step is needed")
    if n_grid is None or ntrial_blind is None:
        raise ValueError("Both n_grid and ntrial_blind are needed")

    rho_sq = (np.asarray(delta_f, dtype=float) / f_step) ** 2

    if search_fdot:
        if fdot_step is None:
            raise ValueError("The fdot grid step fdot_step is needed")
        rho_sq = rho_sq + (np.asarray(delta_fdot, dtype=float) / fdot_step) ** 2
        # Cells inside a disc of radius rho, one cell per unit area
        n_closer = np.pi * rho_sq
    else:
        # Cells inside an interval of half-width rho
        n_closer = 2 * np.sqrt(rho_sq)

    ntrial = ntrial_blind * n_closer / n_grid

    return np.clip(ntrial, 1.0, float(ntrial_blind))


def prior_corrected_p_value(p_single, ntrial):
    """Correct a single-trial p-value for a (possibly fractional) trials factor.

    Parameters
    ----------
    p_single : float or array of floats
        Single-trial probability of the candidate.
    ntrial : float or array of floats
        Effective number of trials, as returned by :func:`effective_ntrial`.
        Fractional values are allowed.

    Returns
    -------
    p : float or array of floats
        Probability that at least one of the ``ntrial`` trials produced a
        fluctuation at least as large.

    Examples
    --------
    >>> assert np.isclose(prior_corrected_p_value(0.01, 1), 0.01)
    >>> # Close to the union bound for small probabilities
    >>> assert np.isclose(prior_corrected_p_value(1e-6, 10), 1e-5, rtol=1e-3)
    >>> # Never exceeds 1
    >>> assert prior_corrected_p_value(0.5, 1e6) <= 1
    """
    p_single = np.asarray(p_single, dtype=float)
    ntrial = np.asarray(ntrial, dtype=float)

    # -expm1(n * log1p(-p)) is 1 - (1 - p)**n, accurate for tiny p
    return -np.expm1(ntrial * np.log1p(-p_single))
