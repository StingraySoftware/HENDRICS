.. _quicklook-tutorial:

Introductory concepts and example analysis
------------------------------------------

Preliminary info
~~~~~~~~~~~~~~~~

This tutorial assumes that you have previous knowledge of timing
techniques, so that I don't repeat concepts as Nyquist frequency, the
importance of choosing carefully the binning time and the FFT length,
and so on. If you are not familiar with these concepts, `this paper by
Michiel is a very good place to
start <https://pure.uva.nl/ws/files/2212461/47104_Fourier_techniques.pdf>`__.
In this tutorial we will show an example based on _NuSTAR_ data. For this
satellite, it is advisable to use the cospectrum (real part of the cross
spectrum) of the data from the two separated detectors instead of the
power spectrum of the full light curve, to work around the effect of
dead time. See our `timing paper <https://arxiv.org/abs/1409.3248>`__ for
details.

This software works in separated steps. One starts from cleaned event
files (such as those produced by tools like ``nupipeline`` and possibly
barycentered with ``barycorr`` or equivalent), and produces a cascade
of intermediate products until the final result. For example:

1. Read the **event list** and save it to an intermediate file containing
   event arrival times and PI channel information

2. (optional) Produce **calibrated event lists**, where PI values have been
   converted to energy

3. Use calibrated or uncalibrated event lists to produce **light curves**
   with a given bin time.
   Only if starting from a calibrated event list, the light curve can be
   obtained by specifying an energy range, otherwise only the PI channel
   filtering is avaiable.

4. (optional) **summed light curves** if we want to join events from
   multiple instruments, or just from different observing times

5. **power spectrum** and/or **cross spectrum** (hereafter the
   \`\`frequency spectra'')

6. **rebinning** of frequency spectra

7. finally, **lags** and **cospectrum**

8. (optional) frequency spectra in XSpec format

Most of these tools have help information that can be accessed by typing
the name of the command plus -h or --help:

::

    $ HENcalibrate -h
    usage: HENcalibrate [-h] [-r RMF] [-o] files [files ...]

    Calibrates clean event files by associating the correct energy to each PI
    channel. Uses either a specified rmf file or (for NuSTAR only) an rmf file
    from the CALDB

    positional arguments:
      files              List of files

    optional arguments:
      -h, --help         show this help message and exit
      -r RMF, --rmf RMF  rmf file used for calibration
      -o, --overwrite    Overwrite; default: no

Some scripts (e.g. ``HENreadevents``, ``HENlcurve``, ``HENfspec``) have a
``--nproc`` option, useful when one needs to treat multiple files at a
time. The load is divided among ``nproc`` processors, that work in
parallel cutting down considerably the execution time.

For I/O, HENDRICS looks if the ``netCDF4`` library is installed. If it's
found in the system, files will be saved in this format. Otherwise, the
native Python ``pickle`` format format will be used. This format is
*much* slower (It might take some minutes to load some files) and files
will be bigger, but this possibility ensures portability. If you don't
use netCDF4, you'll notice that file names will have the ``.p``
extension instead of the ``.nc`` below. The rest is the same.

Loading event lists
~~~~~~~~~~~~~~~~~~~

Starting from cleaned event files, we will first save them in
``HENDRICS`` format (a ``pickle`` or ``netcdf4`` file). For example, I'm starting
from two event lists called ``002A.evt`` and ``002B.evt``, containing
the cleaned event lists from a source observed with NuSTAR's ``FPMA``
and ``FPMB`` respectively.

::

    $ HENreadevents 002A.evt 002B.evt
    Opening 002A.evt
    Saving events and info to 002A_ev.nc
    Opening 002B.evt
    Saving events and info to 002B_ev.nc

This will create new files with a ``_ev.nc`` extension (``_ev.p`` if you
don't use netCDF4), containing the event times and the energy *channel*
(e.g. ``PI``) of each event.

For a few missions (_XMM_, _NuSTAR_, _NICER_), Stingray will automatically calculate the energy in keV corresponding to the energy channels, so that the step in ``HENcalibrate`` can be avoided (unless there is a specific reason not to trust the default calibration).

Calibrating event lists (deprecated, use with caution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``HENcalibrate``. The most secure way to do this is to specify an ``rmf`` file with the
``-r`` option. For _NuSTAR_ only, ``HENcalibrate`` will look into the ``CALDB``, if the
environment variable has been defined!

::

    $ HENcalibrate 002A_ev.nc 002B_ev.nc
    Loading file 002A_ev.nc...
    Done.
    ###############ATTENTION!!#####################

    Rmf not specified. Using default NuSTAR rmf.

    ###############################################
    Saving calibrated data to 002A_ev_calib.nc
    Loading file 002B_ev.nc...
    Done.
    ###############ATTENTION!!#####################

    Rmf not specified. Using default NuSTAR rmf.

    ###############################################
    Saving calibrated data to 002B_ev_calib.nc

This will create two new files with ``_ev_calib.nc`` suffix that will
contain energy information. Optionally, you can overwrite the original
event lists.


Producing light curves
~~~~~~~~~~~~~~~~~~~~~~

Choose carefully the binning
time (option ``-b``). Since what we are interested in is a power
spectrum, this binning time will limit our maximum frequency in the
power spectrum. We are here specifying 2^-8 =0.00390625 for binning time
(how to use the ``-b`` option is of course documented. Use ``-h`` FMI).
Since we have calibrated the event files, we can also choose an event
energy range, here between 3 and 30 keV. Another thing that is useful in
NuSTAR data is taking some time intervals out from the start and the end
of each GTI. This is mostly to eliminate an increase of background level
that often appears at GTI borders and produces very nasty power spectral
shapes. Here I filter 100 s from the start and 300 s from the end of
each GTI.

::

    $ HENlcurve 002A_ev.nc 002B_ev.nc -b -8 -e 3 30 --safe-interval 100 300
    Loading file 002A_ev.nc...
    Done.
    Saving light curve to 002A_E3-30_lc.nc
    Loading file 002B_ev.nc...
    Done.
    Saving light curve to 002B_E3-30_lc.nc

To check the light curve that was produced, use the ``HENplot`` program:

::

    $ HENplot 002A_E3-30_lc.nc

``HENlcurve`` also accepts light curves in FITS and text format. FITS light curves
should be produced by the ``lcurve`` FTOOL or similar, while the text light
curves should have
two columns: time from the NuSTAR MJDREF (55197.00076601852) and intensity in
counts/bin.
Use
::

    $ HENlcurve --fits-input lcurve.fits

or

::

    $ HENlcurve --txt-input lcurve.txt

respectively.

Joining, summing and "scrunching" light curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we want a single light curve from multiple ones, either summing
multiple instruments or multiple energy or time ranges, we can use
``HENscrunchlc``:

::

    $ HENscrunchlc 002A_E3-30_lc.nc 002B_E3-30_lc.nc -o 002scrunch_3-30_lc.nc
    Loading file 002A_E3-30_lc.nc...
    Done.
    Loading file 002B_E3-30_lc.nc...
    Done.
    Saving joined light curve to out_lc.nc
    Saving scrunched light curve to 002scrunch_3-30_lc.nc

This is only tested in \`\`safe'' situations (files are not too big and
have consistent time and energy ranges), so it might give inconsistent
results or crash in untested situations. Please report any problems!

Producing power spectra and cross power spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us just produce the cross power spectrum for now. To produce also
the power spectra corresponding to each light curve, substitute
``"CPDS"`` with ``"PDS,CPDS"``. I use Fractional r.m.s. normalization
here, the default would be Leahy et al. 1983 normalization.

::

    $ HENfspec 002A_E3-30_lc.nc 002B_E3-30_lc.nc -k CPDS -o cpds_002_3-30 --norm frac
    Beware! For cpds and derivatives, I assume that the files are
    ordered as follows: obs1_FPMA, obs1_FPMB, obs2_FPMA, obs2_FPMB...
    Loading file 002A_E3-30_lc.nc...
    Loading file 002B_E3-30_lc.nc...
    Saving CPDS to ./cpds_002_3-30_0.nc

Note that it is possible to directly event lists to ``HENfspec``, instead of the pre-calculated light curve. In this case, one needs to also specify the bin time, and the command line changes to

::

    $ HENfspec 002A_ev.nc 002B_ev.nc -k CPDS -o cpds_002 --norm frac -b -8

Rebinning the spectrum
~~~~~~~~~~~~~~~~~~~~~~

Now let's rebin the spectrum. If the rebin factor is an integer, it is
interpreted as a constant rebinning. Otherwise (only if >1), it is
interpreted as a geometric binning.

::

    $ HENrebin cpds_002_3-30_0.nc -r 0.03
    Saving cpds to cpds_002_3-30_0_rebin0.03.nc

Calculating the cospectrum and phase/time lags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The calculation of lags and their errors is implemented in ``HENlags``,
and needs to be tested properly. For the cospectrum, it is sufficient to
read the real part of the cross power spectrum as depicted in the
relevant function in ``plot.py`` (Use the source, Luke!).

Saving the spectra in a format readable to XSpec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To save the cospectrum in a format readable to XSpec it is sufficient to
give the command

::

    $ HEN2xspec cpds_002_3-30_0_rebin0.03.nc --flx2xsp

Open and fit in XSpec!
~~~~~~~~~~~~~~~~~~~~~~

::

    $ xspec
    XSPEC> data cpds.pha
    XSPEC> cpd /xw; setp ener; setp comm log y
    XSPEC> mo lore + lore + lore
    (...)
    XSPEC> fit
    XSPEC> pl eufspe delchi

etc. |screenshot.png|


.. |screenshot.png| image:: ../images/3911632225-screenshot.png
