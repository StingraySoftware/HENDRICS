.. _targeted-searches-tutorial:

Targeted searches around a known ephemeris
------------------------------------------

.. Note ::

    For an introduction to pulsation searches in HENDRICS, read the
    :ref:`pulsation-searches-tutorial` tutorial first.

Sometimes we are not searching blindly. The source has a previously measured
spin solution, and we want to know whether *this* observation shows the
pulsation at *that* frequency. A blind search would be needlessly harsh: the
detection threshold is set by the number of independent trials across the whole
band, even though we only really care about a narrow region around the expected
frequency.

``HENzsearch`` and ``HENaccelsearch`` accept a known solution and charge each
candidate a number of trials that depends on how far it landed from it:

::

    $ HENzsearch events_ev.nc -f 0.7 -F 0.8 --fast \
        --known-freq 0.728 --known-fdot -4.45e-11 --known-pepoch 56682

The solution is extrapolated to the reference epoch of this observation before
anything is compared, so the epochs may be years apart. Equivalently, the
solution can come from a parameter file::

    $ HENzsearch events_ev.nc -f 0.7 -F 0.8 --fast --known-par pulsar.par

which reads ``F0``, ``F1``, ``F2`` and ``PEPOCH``.

A worked example
~~~~~~~~~~~~~~~~

Say the pulsar was at 0.728 Hz at MJD 56682 and at 0.711 Hz at MJD 61100, and
we are searching an observation taken at MJD 50000. The two measurements give
a spin-down rate of

.. math::

    \dot\nu = \frac{0.711 - 0.728}{(61100 - 56682) \times 86400}
            = -4.45 \times 10^{-11}\ \mathrm{Hz\,s^{-1}}

and extrapolating back to MJD 50000 predicts 0.7537 Hz. Passing
``--known-freq 0.728 --known-fdot -4.45e-11 --known-pepoch 56682`` lets
HENDRICS do that arithmetic and use the result.

The output gains four columns:

``f_offset``, ``fdot_offset``
    How far the candidate landed from the extrapolated solution.
``ntrial_eff``
    The number of trials it was charged.
``p_value``
    Its significance once those trials are paid for.

A candidate sitting on the prediction costs a single trial. One slightly off
costs a handful. One at the far edge of the band costs exactly what a blind
search would have cost, so using a known ephemeris can never make a candidate
look *less* significant than it would have otherwise.

This changes which candidate is reported. The most interesting peak is no
longer the tallest one but the most significant one *after* the correction: a
modest peak on the expected solution beats a taller one at the other end of the
band.

How the trials are counted
~~~~~~~~~~~~~~~~~~~~~~~~~~

The cells of the search plane are ranked, *before looking at the data*, by
their distance from the expected solution. A candidate at rank :math:`k` is
charged :math:`k` trials, because "I would have accepted a peak anywhere at
rank :math:`\leq k`" is a well defined test that was fixed in advance. This is
the same union-bound argument that justifies the blind trials factor, applied
to a subset of cells chosen a priori.

Counting the rank is geometry: the cells at least as close to the prior as the
candidate fill an interval in a frequency-only search, and a disc when a
frequency derivative is searched too.

That count is expressed as a *fraction of the blind-search trial count* rather
than derived from the frequency resolution. The distinction matters. The number
of independent trials of a folding search is not the band width divided by
:math:`1/T`: the statistic is a smooth function of frequency, so the effective
trial count of its maximum depends on the threshold at which it is evaluated.
Monte Carlo simulations of pure noise put the true count several times higher
than the naive figure, and the factor itself drifts with the threshold and with
the number of harmonics -- roughly 3, 5 and 10 times the naive count for
:math:`Z^2_1`, :math:`Z^2_2` and :math:`Z^2_4` respectively. Working with
ratios sidesteps the question entirely: the correction is a faithful rescaling
of whatever blind normalization the search already uses.

Simulations of pure noise show the resulting p-value to be calibrated to better
than a factor 1.5 in trials -- about 0.1 sigma -- over four decades in
significance, which is negligible next to the several orders of magnitude the
prior buys back.

Caveats
~~~~~~~

**The ephemeris must be known beforehand.** The whole argument rests on the
ordering being fixed before the data are examined. Choosing ``--known-freq``
after looking at the periodogram, to "explain" a peak that caught the eye,
invalidates the correction completely.

**The extrapolation does not have to be good.** This is the useful part: we
never need to know how accurate the extrapolated solution is. If it is off,
the candidate simply lands further away and is charged more trials. Timing
noise, an unmodelled second derivative or a missed glitch cost significance,
they do not bias the result.

**Narrow the band in** ``HENaccelsearch``. ``HENaccelsearch`` delegates the
search to Stingray, which thresholds candidates internally using the false
alarm probability of the whole blind search. HENDRICS loosens that threshold as
much as the interface allows when a known ephemeris is given, but the loosest
single-trial probability it can reach is roughly :math:`20/N`, where :math:`N`
is the number of frequency bins in the band. Over a wide band that is still
very strict, and weak peaks on the expected solution may never be reported at
all; HENDRICS warns when this happens. Setting ``--fmin`` and ``--fmax`` around
the expected frequency -- which is what a targeted search should do anyway --
restores the sensitivity. ``HENzsearch`` keeps the whole statistic plane and
does not have this limitation.

**Frequencies refer to different epochs in the two tools.**
``HENzsearch --fast`` refers its candidates to the middle of the observation,
``HENaccelsearch`` to its start. The extrapolation uses whichever epoch the
tool itself reports, so the two are self-consistent.
