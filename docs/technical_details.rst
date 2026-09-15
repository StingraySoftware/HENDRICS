.. _technical-details:

Significance of pulsation searches
==================================

This page explains how ``HENzsearch`` and ``HENaccelsearch`` turn the peaks of
a periodogram into probabilities, and how the numbers they use were measured.
The recipes for using them are in the :ref:`pulsation-searches-tutorial` and
:ref:`targeted-searches-tutorial` tutorials.

Every candidate table reports three probabilities:

``p_1trial``
    The probability that pure noise gives a peak at least this high in a
    single, specified point of the search.
``p_ntrial``
    The same, corrected for the naive number of trials: one per resolution
    element of the search.
``p_ntrial_adj``
    The same, corrected for the *calibrated* number of trials described below.
    This is the one to use for a blind search.

The correction for :math:`N` trials is
:math:`p_N = 1 - (1 - p_1)^N`, which is about :math:`N p_1` for small
probabilities.

Measuring the number of trials
------------------------------

The points of a search grid are correlated. Two frequencies closer than the
resolution :math:`1/T` see almost the same noise, but not exactly the same, and
the maximum of the search picks up the fluctuations between them. The number of
*independent* trials is therefore not the number of grid points, nor the number
of resolution elements: it has to be measured.

We simulate many data sets of pure noise, run the search on each, and record
the smallest single-trial probability :math:`p_{\min}` of the whole search. If
the search behaved as :math:`N` independent trials, :math:`p_{\min}` would be
smaller than :math:`p` with probability :math:`1 - (1 - p)^N`. Calling
:math:`p_\alpha` the value that :math:`p_{\min}` falls below in a fraction
:math:`\alpha` of the simulations, the effective number of trials at false alarm
probability :math:`\alpha` is

.. math::

    N_{\rm eff}(\alpha) = \frac{\ln(1 - \alpha)}{\ln(1 - p_\alpha)}.

The simulations use observations of :math:`T = 1000` s with 5000 events of
Poisson noise, and the uncertainties come from bootstrap resampling of the
simulations. They are run by ``notebooks/trials_calibration.py``, for example::

    $ python notebooks/trials_calibration.py qffa --nreal 3000 --outdir calib
    $ python notebooks/trials_calibration.py plot --outdir calib

The summary tables used on this page are in ``docs/images/trials_calibration``.

For every configuration, HENDRICS uses the largest estimate at 10% and 1% false
alarm probability, rounded up to 0.1, and forced not to decrease with the
number of harmonics or the oversampling. Between tabulated values it uses the
next larger one. All these choices overestimate the number of trials, which
makes the significances conservative.

.. figure:: images/trials_calibration/trials_calibration.png
    :width: 100%

    Independent trials measured on pure noise at 1% false alarm probability.
    Left: ``HENzsearch --fast``, per resolution element, with the tabulated
    values as horizontal ticks (frequency only) and crosses (frequency and
    frequency derivative). Right: ``HENaccelsearch``, per frequency bin per z
    row, with the tabulated values as black ticks.

The fast folding search (``HENzsearch --fast``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The naive number of trials is the number of grid points divided by
``--oversample`` for each searched axis. The resolution elements are
:math:`1/T` in frequency and :math:`4/T^2` in frequency derivative (the latter
gives the same phase error at the edges of the observation as :math:`1/T` in
frequency, with phases measured from its middle). The calibrated number of
trials is the naive one times:

.. list-table:: Trials per resolution element, frequency only
    :header-rows: 1

    * - Harmonics
      - ``--oversample`` 1
      - 2
      - 4
      - 8
    * - 1
      - 1.1
      - 1.9
      - 3.0
      - 3.1
    * - 2
      - 1.1
      - 2.1
      - 4.7
      - 8.0
    * - 4
      - 1.1
      - 2.1
      - 5.0
      - 8.0

.. list-table:: Trials per resolution element, frequency and frequency derivative
    :header-rows: 1

    * - Harmonics
      - ``--oversample`` 1
      - 2
      - 4
    * - 1
      - 0.8
      - 3.7
      - 6.1
    * - 2
      - 0.9
      - 3.7
      - 11.0
    * - 4
      - 1.0
      - 4.3
      - 13.9

Grids beyond the table use its largest value, with a warning: their
significances may be overestimated. On the same grids, the slow folding search
(``HENzsearch`` without ``--fast``) gives the same numbers within the
uncertainties (``summary_folding.ecsv``).

The tables were measured on bands of 0.25 Hz (frequency only) and 0.05 Hz (with
the frequency derivative), with 32 phase bins and 3000 simulations each. Wider
bands (1 and 4 Hz) and 128 phase bins give the same numbers within the
uncertainties, or smaller ones. A run of 20000 simulations of the
:math:`Z^2_2` search with ``--oversample 4`` gives 3.4, 3.7 and 2.9 trials per
resolution element at 10%, 1% and 0.1% false alarm probability, below the
tabulated 4.7: the calibration holds in the tail.

Why oversample, then? Because a signal between two grid points loses power.
The figure below shows the power recovered by the search relative to the power
at the exact frequency, for a :math:`Z^2_2` signal: 80% without oversampling,
95% with ``--oversample 2``, and all of it from ``--oversample 4``. The price is
in trials; the default of ``2 * N`` points per resolution element for
:math:`Z^2_N` is a compromise between the two.

.. figure:: images/trials_calibration/sensitivity_vs_oversample.png
    :width: 50%

    Power recovered by the search, relative to the power at the true frequency,
    averaged over signal frequencies. The horizontal lines are the accelerated
    search, with and without interbinning.

The accelerated search (``HENaccelsearch``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stingray searches :math:`n_{\rm freq}` Fourier bins for each of the
:math:`n_z` rows of ``np.arange(-zmax, zmax, delta_z)``, where :math:`z` is the
drift in Fourier bins over the observation. The calibrated number of trials is
:math:`n_{\rm freq} \times \max(n_z, 1) \times c`, with :math:`c` from:

.. list-table:: Trials per frequency bin per z row
    :header-rows: 1

    * - Search
      - Without ``--interbin``
      - With ``--interbin``
    * - No z search (``zmax`` 0)
      - 1.0
      - 2.1
    * - ``delta_z`` 1
      - 0.9
      - 1.1
    * - ``delta_z`` 0.5
      - 0.7
      - 0.9

Steps between 0.5 and 1 are interpolated linearly, finer steps use the 0.5
value, and coarser steps count the rows as independent. For the default search
(``zmax`` 100, ``delta_z`` 1) this is 180 trials per frequency bin, which raises
the detection threshold by about 10 in Leahy power compared with the old count
of one trial per frequency bin. The numbers were measured with 10000
simulations, with ``zmax`` 0, 10 and 100.

``HENaccelsearch`` delegates the search to Stingray, which only returns the
candidates above a power threshold computed with :math:`n_{\rm freq}` trials.
HENDRICS asks instead for the threshold corresponding to the requested false
alarm probability with the calibrated number of trials (lowered, if needed, so
that it also catches the in-between bins of interbinning, see below), and then
keeps the candidates with ``p_ntrial_adj`` below the false alarm probability.
With a known ephemeris it asks for the threshold of a single trial, and keeps the
candidates by their corrected probability.

``--pad_to_double`` was not simulated: its significances are only approximate.

Interbinning
~~~~~~~~~~~~

Interbinning (``--interbin``) recovers part of the power lost between Fourier
bins by adding a bin between each pair,

.. math::

    A_{k+1/2} = \frac{\pi}{4} \left(A_{k+1} - A_k\right).

The regular bins keep the usual noise distribution of Leahy powers, a
:math:`\chi^2` with two degrees of freedom: :math:`p_1 = e^{-P/2}`. The
in-between bins do not. They are the difference of two neighbouring bins, which
after the acceleration correction are correlated. Their noise is still
exponential, but stretched by

.. math::

    s(z) = \frac{\pi^2}{8}\left(1 - {\rm Re}\,\rho_1(z)\right),
    \qquad
    \rho_1(z) = \frac{\sum_q w_{q+1} w_q^*}{\sum_q |w_q|^2},

where :math:`w` is the response used by Stingray to correct for the drift
:math:`z`, and :math:`\rho_1` the correlation between neighbouring bins that it
introduces. The single-trial probability of an in-between bin is
:math:`p_1 = e^{-P/(2 s)}`. For a plain Fourier search (:math:`z = 0`) the bins
are independent and :math:`s = \pi^2/8 \approx 1.23`: without this correction,
a noise power :math:`P` in an in-between bin looked as significant as a power
:math:`1.23\,P` really is.

HENDRICS recognizes in-between bins from their frequency (Stingray reports
:math:`r/T`, so :math:`r` is a half integer), and computes :math:`s(z)` from
Stingray's responses (``hendrics.known_ephemeris.interbin_stretch``). The tests
check, on simulated white noise passed through Stingray's own convolution and
interbinning, that the resulting probabilities are uniformly distributed for
:math:`z` = 0, 0.25, 0.5, 1, 5, 10 and 100.

.. figure:: images/trials_calibration/interbin_stretch.png
    :width: 50%

    Stretch of the noise distribution of the in-between bins as a function of
    z. ``HENaccelsearch`` uses integer z values with the default ``--delta-z``.

Targeted searches
-----------------

With a known ephemeris (see :ref:`targeted-searches-tutorial`), each candidate
is charged a number of trials that grows with its distance from the prediction.
The cells of the search are ranked, before looking at the data, by that
distance. The charge of a candidate, ``ntrial_eff``, is the calibrated number
of trials of the blind search times the fraction of the grid that is at least as
close to the prediction, with a floor ``ntrial_min`` covering the uncertainty
region of the ephemeris (three standard deviations), and never more than the
blind search. Its probability is ``p_value``.

Picking the best candidate
~~~~~~~~~~~~~~~~~~~~~~~~~~

``p_value`` is honest for a candidate chosen before looking at the data. But
the reported candidate is the *most significant* one after the correction, and
that choice has a price. Think of the false alarm probability :math:`\alpha` as
a budget: a cell charged :math:`n` trials uses up about :math:`\alpha / n` of
it, and the chance that any cell gives a false alarm is at most the sum over
all cells. For a blind search, :math:`N` cells charged :math:`N` trials each use
exactly :math:`\alpha`. Ranking the cells charges the :math:`r`-th closest cell
about :math:`r` trials, and the sum becomes
:math:`\alpha \sum_r 1/r \approx \alpha \left(1 + \ln(N / n_{\min})\right)`,
where the first :math:`n_{\min}` cells, inside the floor, contribute
:math:`\alpha` together.

``p_value_best`` pays for this, charging
:math:`\min\left(k \times {\tt ntrial\_eff},\ N\right)` trials with
:math:`k = 1 + \ln(N / n_{\min})`. :math:`k` is about 10 for a typical search
and no uncertainty on the ephemeris, and 1 when the uncertainty region covers
the whole search. ``HENzsearch`` uses ``p_value_best`` to decide whether the
candidate is detected, and ``HENaccelsearch`` sorts its candidates by it.

The table below shows the rate of false alarms measured on 10000 simulations of
pure noise, relative to the nominal false alarm probability (1 is exact, less
is conservative), with the calibrated number of trials of each tool:
2 harmonics, ``--oversample 4`` and a frequency derivative search for
``HENzsearch --fast`` (5632 trials), ``zmax`` 10 for ``HENaccelsearch`` (18000
trials).

.. list-table:: False alarms relative to nominal, at 10%, 1% and 0.1%
    :header-rows: 1

    * - Tool
      - Floor ``ntrial_min``
      - ``p_value``
      - ``p_value_best``
      - Blind search
    * - ``HENzsearch --fast``
      - 1
      - 3.5, 5.8, 7.0
      - 1.15, 1.33, 1.7
      - 0.79, 0.85, 1.2
    * - ``HENzsearch --fast``
      - 100
      - 2.2, 3.3, 3.7
      - 1.06, 1.24, 1.6
      - 0.79, 0.85, 1.2
    * - ``HENaccelsearch``
      - 1
      - 4.0, 5.6, 5.2
      - 1.16, 1.34, 1.7
      - 0.79, 1.0, 1.1
    * - ``HENaccelsearch``
      - 100
      - 2.3, 2.8, 2.7
      - 1.05, 1.22, 1.5
      - 0.79, 1.0, 1.1

The figures at 0.1% rest on about ten events each, and are uncertain by
:math:`\pm 0.4`.

``p_value_best`` removes most of the excess, but not all of it: both tools
still give false alarms about 1.1 times too often at 10%, and 1.2-1.3 times at
1%. The reason is
the promise that a candidate never costs more than in a blind search. Most
cells of the grid are far enough from the prediction that
:math:`k \times {\tt ntrial\_eff}` would exceed :math:`N`, so they are charged
exactly :math:`N`: together, they use up almost the whole budget, as a blind
search would. The cells near the prediction, charged less, add their share on
top. For the ``HENaccelsearch`` grid above, the sum of the charges allows up to
1.8 times the nominal rate of false alarms for independent cells (1.6 with a
floor of 100). Splitting the budget in advance between the cells near the
prediction and a blind search would remove this, at the price of charging
distant candidates several times more than a blind search. We chose to keep the
promise and document the residual excess.

.. figure:: images/trials_calibration/targeted_false_alarms.png
    :width: 100%

    False alarm rate of the targeted searches relative to nominal, at 10%
    (left) and 1% (right) false alarm probability, as a function of the floor
    on the trials, with and without the price of picking the best candidate.
