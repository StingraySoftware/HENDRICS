.. _pulsation-searches-tutorial:

Pulsation searches
------------------

.. Note ::

    For a general introduction to HENDRICS's workflow, please read the
    :ref:`quicklook-tutorial` tutorial

We have a pulsar observation with, e.g., *NuSTAR* and we want to find pulsations on it.
The general procedure is looking for pulsations using a power density spectrum
(see :ref:`quicklook-tutorial`) or similar methods, and if we do find a promising
candidate frequency, investigate more with the Epoch Folding or the Z search.

Let's say we have found a peak in the power density spectrum at about 0.101
seconds, or 9.9 Hz, and we want to investigate more.

We start from the _event_ file. If we have run `HENreadevents` on the original
mission-specific event file, we have a HENRICS-format event file (ending with
`_ev.nc` or `_ev.p`), e.g.

::

    $ ls
    002A.evt 002A_ev.nc

Accelerated searches
~~~~~~~~~~~~~~~~~~~~~~~~~
HENDRICS now implements the accelerated search à la Ransom+2002: starting from a single
FFT, we convolve it with responses corresponding to different values of "acceleration",
or spin-up/down. It is an extremely fast technique to find binary pulsars, albeit less
sensitive than a focused Z search if the pulsations are known. The method
is experimental, and detections should be assessed through simulations. Use with care.

Example with a *NuSTAR* observation of a famous X-ray binary pulsar:

::

    $ HENaccelsearch mistery_psrA_nustar_fpma_ev.nc --fmin 0.01 --fmax 5
    WARNING: The accelsearch functionality is experimental. Use with care, and feel free to report any issues. [hendrics.efsearch]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 7416.52it/s]
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:20<00:00,  9.63it/s]

    Best candidates:

           time           frequency                fdot                power             pepoch
    ----------------- ------------------ ----------------------- ----------------- ------------------
    81916106.03609267 2.8520477603166845  -5.024586630979715e-08 8771.116170366908 56145.103845139965
    81916106.03609267  2.852015406199768  -4.815228854688893e-08 8194.657124664562 56145.103845139965
    81916106.03609267  2.852080114433601  -5.233944407270536e-08 6016.665557776852 56145.103845139965
    81916106.03609267  1.426007703099884 -2.4076144273444466e-08 5460.323423255257 56145.103845139965
    81916106.03609267 2.8520477603166845 -4.9199077428343043e-08 5329.925664622505 56145.103845139965
    81916106.03609267 1.4260400572168006  -2.616972203635268e-08 5142.912459419216 56145.103845139965
    81916106.03609267 2.8519830520828515  -4.605871078398072e-08 4846.570816537878 56145.103845139965
    81916106.03609267  2.852015406199768  -4.710549966543483e-08 4618.153190476991 56145.103845139965
    81916106.03609267  2.852080114433601  -5.338623295415947e-08  4228.74089043784 56145.103845139965
    81916106.03609267  2.852080114433601  -5.129265519125125e-08 3847.880659073288 56145.103845139965
    See all 24740 candidates in mistery_psrA_nustar_fpma_accelsearch.csv

See ``HENaccelsearch -h`` for more details.


Fast Z2n searches
~~~~~~~~~~~~~~~~~
HENDRICS implements a fast algorithm for pulsation searches
with the Z^2_n statistics.
Select this algorithm with the ``--fast`` option on the command line of `HENzsearch`.

::

    $ HENefsearch -f 9.85 -F 9.95 -n 64 --fast -N 3 mistery_psrA_nustar_fpma_ev.nc

Here, we are searching from 9.85 to 9.95 Hz, using 3 harmonics (so, ..math:`Z^2_3`
stats), pre-binning the pulse profile with 64 bins.

Instead of calculating the phase of all photons at each trial value of frequency and
derivative, we pre-bin the data in small chunks and shift the different chunks to the
amount required by different trial values.

|fast_zsearch.jpeg|

Each pre-folding leads to a large number of trial values to be evaluated. This only
works if we assume that the trial frequency is sufficiently close to the initial one
that no signal leaks into nearby bins inside the sub-profiles. This requires that we
choose a sufficiently large number of sub-profiles, and limit the total shift to
reasonable values to limit this leak.

Given the wanted range of frequencies to search, the program chooses automatically the
number of trial frequencies and fdots to derive from each given pre-folding, and when
to perform a new pre-folding.

At the moment, the trial fdots are chosen automatically and cannot be defined by the user.
The only actions the user can do are the selection of the mean fdot and the parameter
``--npfact`` that increases the number of trial values to obtain from a single central
frequency/fdot combination (npfact=2 means that the number of trial values will be
double for both the frequency and the fdot, so four times the trials in the end).

The results of this Z search can be plotted with `HENplot`. HENplot does more than just
plotting the results: it also creates contours around the best solution, and uses
the ..math:`\chi^2_n` statistics to estimate the error bars on frequency and frequency
derivative.

|zsearch_plot.jpeg|

.. |zsearch_plot.jpeg| image:: ../images/zsearch_plot.jpeg
.. |fast_zsearch.jpeg| image:: ../images/fast_zsearch.jpeg


Slow (and outdated) method
~~~~~~~~~~~~~~~~~~~~~~~~~~
To look for pulsations with the epoch folding around the candidate frequency
9.9 Hz, we can run `HENefsearch` as such:

::

    $ HENefsearch -f 9.85 -F 9.95 -n 64 --fit-candidates mistery_psrA_nustar_fpma_ev.nc

where the options `-f` and `-F` give the minimum and maximum frequencies to
search, `-n` the number of bins in the pulsed profile and with `--fit-candidates`
we specify to not limit the search to the epoch folding, but also look for
peaks and fit them to find their centroids.

The output of the search is in a file ending with `_EF.nc` or `_EF.p`, while
the best-fit model is recorded in pickle files

::

    $ ls
        002A.evt 002A_ev.nc 002A_EF.nc 002A_EF__mod0__.p

To use the Z search, we can use the `HENzsearch` script with very similar options:

::

    $ HENzsearch -f 9.85 -F 9.95 -N 2 --fit-candidates

where the `-N` option specifies the number of harmonics to use for the search.

The output of the search and the fit is recorded in similar files as Epoch folding

::

    $ ls
        002A.evt 002A_ev.nc 002A_Z2n.nc 002A_Z2n__mod0__.p

We can plot the results of this search with `HENplot`, as such:

::

    $ HENplot 002A_Z2n.nc

|zn_search.png|


.. |zn_search.png| image:: ../images/zn_search.png


Measuring frequency derivatives interactively
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HENphaseogram`` is an interactive phaseogram to adjust the values of the frequency and frequency derivatives of pulsars.

|phaseogram.jpeg|


.. |phaseogram.jpeg| image:: ../images/phaseogram.jpeg


.. raw:: html

    <div style="max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/irm_S5rlqL8" frameborder="0" allowfullscreen></iframe>
    </div>

