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

Besides the probabilities of a blind search (``p_1trial``, ``p_ntrial`` and
``p_ntrial_adj``, see :ref:`technical-details`), the output gains five columns:

``f_offset``, ``fdot_offset``
    How far the candidate landed from the extrapolated solution.
``ntrial_eff``
    The number of trials it was charged.
``p_value``
    Its significance once those trials are paid for.
``p_value_best``
    Its significance once we also pay for having picked the most significant
    candidate of the search (see `How the trials are counted`_). This is the
    number to quote: ``HENzsearch`` uses it to decide whether the pulsation is
    detected, and ``HENaccelsearch`` sorts its candidates by it.

A candidate sitting on the prediction costs a single trial. One slightly off
costs a handful. One at the far edge of the band costs exactly what a blind
search would have cost, so using a known ephemeris never makes a candidate
look *less* significant than it would have otherwise. Picking the best
candidate multiplies these charges by a factor of about 10 (at most up to the
blind count), so a known ephemeris still buys several orders of magnitude in
significance, but not all of what ``p_value`` suggests.

This changes which candidate is reported. The most interesting peak is no
longer the tallest one but the most significant one *after* the correction: a
modest peak on the expected solution beats a taller one at the other end of the
band.

Uncertain solutions
~~~~~~~~~~~~~~~~~~~

An extrapolated solution is never exact, and ranking candidates by their
distance from it has a side effect: a noise peak that happens to fall right on
the prediction pays a single trial, however imprecise the prediction was. When
the uncertainty on the solution is known, give it::

    $ HENzsearch events_ev.nc -f 0.7 -F 0.8 --fast \
        --known-freq 0.728 --known-fdot -4.45e-11 --known-pepoch 56682 \
        --known-freq-err 1e-6 --known-fdot-err 1e-15

The uncertainties (one standard deviation) refer to the reference epoch of the
known solution, and are propagated to the epoch of the observation, ignoring the
covariances between parameters. With ``--known-par`` they are read from the
parameter file, and the command line options override them.

Every candidate within three standard deviations of the extrapolated solution is
then charged the same number of trials: what the whole region would cost.
Candidates outside the region are still charged by their distance, so the charge
is continuous at the edge of the region. Charging some candidates more can only
make false alarms rarer, so the argument of the next section still holds.

Formal timing uncertainties are usually much smaller than the effect of timing
noise over a long extrapolation. Be generous: an uncertainty that is too large
costs a little significance, while one that is too small lets noise close to the
prediction look more significant than it is.

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
than derived from the frequency resolution. The number of independent trials of
a search is not the number of resolution elements: neighbouring grid points are
correlated, but not completely. HENDRICS uses the number of trials measured with
Monte Carlo simulations of pure noise, and working with fractions makes the
targeted correction a faithful rescaling of it.

There is a catch. The reported candidate is not chosen in advance, but is the
most significant one *after* the correction, and every distance from the
prediction offers a new chance for a noise fluke to look significant. Picking
the best of them costs a factor :math:`k = 1 + \ln(N / n_{\min})` in trials,
where :math:`N` is the blind count and :math:`n_{\min}` the floor set by the
uncertainty of the solution: about 10 for a typical search with no
uncertainty, and less the more uncertain the solution. ``p_value_best``
includes it.

Simulations of pure noise show that ``p_value`` alone gives false alarms 2 to 6
times more often than nominal, and ``p_value_best`` removes most of this excess.
The details, including a residual excess of about 1.3 times at 1% in both
tools, are in :ref:`technical-details`.

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
they do not bias the result. The one thing to be honest about is the
uncertainty: without one, noise sitting right on an imprecise prediction is
charged a single trial (see `Uncertain solutions`_).

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
