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

import warnings

import numpy as np

__all__ = [
    "effective_ntrial",
    "ephemeris_from_parfile",
    "extrapolate_ephemeris",
    "extrapolate_ephemeris_uncertainty",
    "prior_corrected_p_value",
    "qffa_calibrated_ntrial",
    "uncertainty_ntrial",
]

# Effective number of trials per resolution element of ``search_with_qffa``
# (1/T in frequency, times 4/T^2 in frequency derivative when that is searched
# too), measured on pure Poisson noise with ``notebooks/trials_calibration.py``.
# Each value is the largest estimate at 10% and 1% false alarm probability over
# all the runs, rounded up to 0.1 and made non-decreasing in both axes.
# Rows: number of harmonics; columns: grid points per resolution element.
_QFFA_NHARMS = (1, 2, 4)
_QFFA_OVERSAMPLES = {False: (1, 2, 4, 8), True: (1, 2, 4)}
_QFFA_TRIALS_PER_ELEMENT = {
    False: np.array(
        [
            [1.1, 1.9, 3.0, 3.1],
            [1.1, 2.1, 4.7, 8.0],
            [1.1, 2.1, 5.0, 8.0],
        ]
    ),
    True: np.array(
        [
            [0.8, 3.7, 6.1],
            [0.9, 3.7, 11.0],
            [1.0, 4.3, 13.9],
        ]
    ),
}


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


def extrapolate_ephemeris_uncertainty(
    f_err, fdot_err=0.0, fddot_err=0.0, pepoch=None, target_epoch=None
):
    """Propagate the uncertainties of a spin solution to a different epoch.

    The contributions of each parameter are added in quadrature, ignoring the
    covariances between them.

    .. note::
        Formal timing uncertainties are usually far smaller than the effect of
        timing noise, glitches or unmodelled derivatives over a long
        extrapolation. Treat the result as a lower limit.

    Parameters
    ----------
    f_err : float
        Uncertainty on the spin frequency at ``pepoch``, in Hz.

    Other Parameters
    ----------------
    fdot_err : float, default 0
        Uncertainty on the first frequency derivative, in Hz/s.
    fddot_err : float, default 0
        Uncertainty on the second frequency derivative, in Hz/s^2.
    pepoch : float
        Reference epoch of the input solution, in MJD.
    target_epoch : float
        Epoch the solution should be extrapolated to, in MJD.

    Returns
    -------
    f_err : float
        Uncertainty on the extrapolated frequency, in Hz.
    fdot_err : float
        Uncertainty on the extrapolated first frequency derivative, in Hz/s.

    Examples
    --------
    >>> # An fdot known to 1e-12 Hz/s, ten days later
    >>> f_err, fdot_err = extrapolate_ephemeris_uncertainty(
    ...     0, fdot_err=1e-12, pepoch=50000, target_epoch=50010)
    >>> assert np.isclose(f_err, 1e-12 * 864000)
    >>> assert np.isclose(fdot_err, 1e-12)
    """
    if pepoch is None or target_epoch is None:
        raise ValueError("Both pepoch and target_epoch are needed to extrapolate an ephemeris")

    dt = (target_epoch - pepoch) * 86400.0

    new_f_err = np.sqrt(f_err**2 + (fdot_err * dt) ** 2 + (0.5 * fddot_err * dt**2) ** 2)
    new_fdot_err = np.sqrt(fdot_err**2 + (fddot_err * dt) ** 2)

    return new_f_err, new_fdot_err


def ephemeris_from_parfile(parfile, return_errors=False):
    """Read F0, F1, F2 and PEPOCH from a TEMPO2/PINT parameter file.

    Parameters
    ----------
    parfile : str
        Path to a parameter file in TEMPO2/PINT format.

    Other Parameters
    ----------------
    return_errors : bool, default False
        Also return the uncertainties on F0, F1 and F2. Parameters without an
        uncertainty (or absent from the file) are given an uncertainty of 0.

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
    errors : tuple of floats
        Uncertainties on ``freq``, ``fdot`` and ``fddot``. Only returned if
        ``return_errors`` is True.
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

    result = (freq, _value("F1"), _value("F2"), _value("PEPOCH", None))
    if not return_errors:
        return result

    def _error(name):
        if not hasattr(model, name) or getattr(model, name).uncertainty_value is None:
            return 0.0
        return float(getattr(model, name).uncertainty_value)

    return result + ((_error("F0"), _error("F1"), _error("F2")),)


def effective_ntrial(
    delta_f,
    delta_fdot=0.0,
    f_step=None,
    fdot_step=None,
    n_grid=None,
    ntrial_blind=None,
    search_fdot=True,
    ntrial_min=1.0,
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
    ntrial_min : float, default 1
        Minimum number of trials charged to any cell, typically the output of
        :func:`uncertainty_ntrial` when the known ephemeris has an uncertainty.
        Raising the charge of some cells can only lower the false alarm rate,
        so the correction stays valid.

    Returns
    -------
    ntrial : float or array of floats
        Effective number of trials, between ``ntrial_min`` (or 1, if larger)
        and ``ntrial_blind``.

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

    ntrial_min = min(max(float(ntrial_min), 1.0), float(ntrial_blind))

    return np.clip(ntrial, ntrial_min, float(ntrial_blind))


def uncertainty_ntrial(
    f_err,
    fdot_err=0.0,
    *,
    f_step=None,
    fdot_step=None,
    n_grid=None,
    ntrial_blind=None,
    search_fdot=True,
    nsigma=3.0,
):
    """Number of trials charged to every cell inside the uncertainty region.

    When the known ephemeris has an uncertainty, all the cells within
    ``nsigma`` standard deviations of it are, a priori, equally good places for
    the pulsation to be. Ranking them by their distance from the central value
    would be arbitrary, and would charge a noise peak that happens to land near
    the centre far too little. Instead, all of them are charged the same number
    of trials: the number of cells in the region, rescaled to the blind-search
    count exactly as :func:`effective_ntrial` does. Pass the result as
    ``ntrial_min`` to :func:`effective_ntrial`.

    The region is an interval in a frequency-only search, and an ellipse when a
    frequency derivative is searched too. Each semi-axis is at least half a
    grid cell, since even a perfectly known value spans the cell it falls in. A
    candidate on the edge of a circular region costs the same whether it is
    charged by the region or by its rank.

    Parameters
    ----------
    f_err : float
        Uncertainty (one standard deviation) on the expected frequency, in Hz.

    Other Parameters
    ----------------
    fdot_err : float, default 0
        Uncertainty (one standard deviation) on the expected frequency
        derivative, in Hz/s. Ignored if ``search_fdot`` is False.
    f_step : float
        Step of the frequency grid, in Hz.
    fdot_step : float
        Step of the frequency derivative grid, in Hz/s. Only needed if
        ``search_fdot`` is True.
    n_grid : int
        Total number of points in the search grid.
    ntrial_blind : int
        Number of independent trials of the equivalent blind search.
    search_fdot : bool, default True
        Whether the search covers frequency derivatives as well.
    nsigma : float, default 3
        Half-width of the region, in standard deviations.

    Returns
    -------
    ntrial : float
        Number of trials charged inside the region, between 1 and
        ``ntrial_blind``.

    Examples
    --------
    >>> kw = dict(f_step=1e-5, n_grid=1000, ntrial_blind=200,
    ...           search_fdot=False)
    >>> # +-3 sigma covers 60 of the 1000 grid cells: 6% of the blind trials
    >>> assert np.isclose(uncertainty_ntrial(1e-4, **kw), 12)
    >>> # A candidate on the edge of the region costs the same either way
    >>> assert np.isclose(effective_ntrial(3e-4, **kw), 12)
    """
    if f_step is None:
        raise ValueError("The frequency grid step f_step is needed")
    if n_grid is None or ntrial_blind is None:
        raise ValueError("Both n_grid and ntrial_blind are needed")

    half_f = nsigma * np.abs(np.asarray(f_err, dtype=float)) / f_step

    if search_fdot:
        if fdot_step is None:
            raise ValueError("The fdot grid step fdot_step is needed")
        half_fdot = nsigma * np.abs(np.asarray(fdot_err, dtype=float)) / fdot_step
        n_cells = np.pi * np.maximum(half_f, 0.5) * np.maximum(half_fdot, 0.5)
    else:
        n_cells = 2 * half_f

    return np.clip(ntrial_blind * n_cells / n_grid, 1.0, float(ntrial_blind))


def _next_tabulated(value, tabulated, name):
    """Index of the first tabulated value not smaller than ``value``.

    Values beyond the table get the last index, with a warning.
    """
    # A tiny tolerance, so that e.g. 4.0000000001 still maps to 4
    idx = int(np.searchsorted(tabulated, value * (1 - 1e-9)))
    if idx >= len(tabulated):
        warnings.warn(
            f"{name}={value:g} is not covered by the calibration of the number of "
            f"trials (largest tabulated value: {tabulated[-1]}). Using the largest "
            "one: the number of trials, and hence the significances, may be "
            "underestimated."
        )
        idx = len(tabulated) - 1
    return idx


def qffa_calibrated_ntrial(naive_ntrial, *, nharm, oversample, search_fdot):
    """Calibrated number of independent trials of a fast (QFFA) folding search.

    The grid of ``search_with_qffa`` is oversampled, so neighbouring points are
    correlated, but not so much that they count as a single trial: Monte Carlo
    simulations of pure noise show that the maximum of an oversampled
    :math:`Z^2_N` plane behaves as the maximum of several independent trials
    per resolution element. This function multiplies the naive count of
    resolution elements by the number of trials per element measured in those
    simulations (see ``notebooks/trials_calibration.py`` and the technical
    details in the documentation).

    Numbers of harmonics and oversampling factors between the tabulated values
    use the next larger one, which gives more trials and is thus conservative.
    Beyond the table the largest value is used, with a warning.

    Parameters
    ----------
    naive_ntrial : float
        Number of resolution elements covered by the search: the number of
        grid points divided by ``oversample`` for each searched axis.

    Other Parameters
    ----------------
    nharm : int
        Number of harmonics of the :math:`Z^2_N` statistic.
    oversample : float
        Grid points per resolution element (1/T in frequency, 4/T^2 in
        frequency derivative).
    search_fdot : bool
        Whether the frequency derivative was searched too.

    Returns
    -------
    ntrial : float
        Calibrated number of trials, at least 1.

    Examples
    --------
    >>> # 1000 resolution elements, Z^2_2, four points per 1/T
    >>> ntrial = qffa_calibrated_ntrial(1000, nharm=2, oversample=4, search_fdot=False)
    >>> assert np.isclose(ntrial, 4700)
    >>> # Three points per 1/T are charged as four
    >>> ntrial = qffa_calibrated_ntrial(1000, nharm=2, oversample=3, search_fdot=False)
    >>> assert np.isclose(ntrial, 4700)
    """
    search_fdot = bool(search_fdot)
    i_nharm = _next_tabulated(nharm, _QFFA_NHARMS, "nharm")
    i_os = _next_tabulated(oversample, _QFFA_OVERSAMPLES[search_fdot], "oversample")
    per_element = _QFFA_TRIALS_PER_ELEMENT[search_fdot][i_nharm, i_os]
    return max(float(naive_ntrial) * per_element, 1.0)


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
