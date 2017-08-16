Data simulation
---------------

To simulate datasets, `MaLTPyNT` includes the `HENfake` script. It can simulate
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
option. The light curve can be in FITS or MaLTPyNT native format (or one can use
HENlcurve for the conversion from text format):

::

    $ HENfake -l lightcurve.fits

To apply dead time to the generated events, use the `--deadtime` option. deadtime
can be supplied as a single number, meaning a constant dead time

::

    $ HENfake -l lightcurve.fits --deadtime 2.5e-3

or as two numbers (`mean`, `sigma`), meaning a Gaussian distribution of dead
times with the specified mean and sigma.

More advanced options are available using the functions in `maltpynt.fake`.
