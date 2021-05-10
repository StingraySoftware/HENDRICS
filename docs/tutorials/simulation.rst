.. _data-simulation-tutorial:

Data simulation
---------------

.. Note ::

    For a general introduction to HENDRICS's workflow, please read the
    :ref:`quicklook-tutorial` tutorial

To simulate datasets, `HENDRICS` includes the `HENfake` script. It can simulate
event lists with a fixed count rate or from an input light curve. Also, it is able
to apply a dead time filter to the simulated event lists.

Basic operations
~~~~~~~~~~~~~~~~
To simulate a short observation (1025 s) at a given count rate (e.g., 150 ct/s),
it is sufficient to call `HENfake -c <countrate>`

::

    $ HENfake -c 150
    $ ls
    events.evt

To simulate an event list from an input light curve, use the `-l` (or `--lc`)
option. The light curve can be in FITS or HENDRICS native format (or one can use
HENlcurve for the conversion from text format):

::

    $ HENfake -l lightcurve.fits

To apply dead time to the generated events, use the `--deadtime` option. deadtime
can be supplied as a single number, meaning a constant dead time

::

    $ HENfake -l lightcurve.fits --deadtime 2.5e-3

or as two numbers (`mean`, `sigma`), meaning a Gaussian distribution of dead
times with the specified mean and sigma.

More advanced options are available using the functions in `hendrics.fake` and,
of course, the powerful API in `stingray.simulate`.
