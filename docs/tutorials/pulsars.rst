Pulsation searches
------------------
We have a pulsar observation with, e.g., *XMM-Newton* and we want to find pulsations on it.
The general procedure is looking for pulsations using a power density spectrum (see Quicklook tutorial) or similar methods, and if we do find a promising candidate frequency, investigate more with the Epoch Folding or the Z search.

Let's say we have found a peak in the power density spectrum at about 0.101 seconds, or 9.9 Hz, and we want to investigate more.

We start from the _event_ file. If we have run `HENreadevents` on the original mission-specific event file, we have a HENRICS-format event file (ending with `_ev.nc` or `_ev.p`), e.g.

::

    $ ls
    002A.evt 002A_ev.nc

To look for pulsations with the epoch folding around the candidate frequency 9.9 Hz, we can run `HENefsearch` as such:

::

    $ HENefsearch -f 9.85 -F 9.95 -n 64 --fit-candidates

where the options `-f` and `-F` give the minimum and maximum frequencies to search, `-n` the number of bins in the pulsed profile and with `--fit-candidates` we specify to not limit the search to the epoch folding, but also look for peaks and fit them to find their centroids.

The output of the search is in a file ending with `_EF.nc` or `_EF.p`, while the best-fit model is recorded in pickle files

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