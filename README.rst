|Build Status| |Build status| |Coverage Status| |Documentation Status|

HENDRICS - High ENergy Data Reduction Interface from the Command Shell
======================================================================

.. Note ::

    This repository contains an evolution of MaLTPyNT. This software is
    being heavily rewritten in order to use Stingray's classes and functions
    whenever possible. To use the original MaLTPyNT, please go to
    `matteobachetti/MaLTPyNT <https://github.com/matteobachetti/MaLTPyNT>`__.

Description
-----------

This set of command-line scripts based on
`Stingray <https://github.com/StingraySoftware/stingray>`__ is designed
to do correctly and fairly easily a **quick-look (spectral-)timing
analysis** of X-ray data, treating properly the gaps in the data due,
e.g., to occultation from the Earth or passages through the SAA.
Originally, its development as MaLTPyNT - Matteo's Libraries and Tools
in Python for NuSTAR Timing - was driven by the need of performing
aperiodic timing analysis on NuSTAR data, whose long dead time made it
difficult to treat power density spectra with the usual tools. By
exploiting the presence of two independent detectors, one could use the
**cospectrum** as a proxy for the power density spectrum (for an
explanation of why this is important, look at Bachetti et al., *ApJ*,
800, 109 -`arXiv:1409.3248 <http://arxiv.org/abs/1409.3248>`__).

Today, this set of command line scripts is much more complete and it is
capable of working with the data of many more satellites. Among the
features already implemented are power density and cross spectra, time
lags, pulsar searches with the Epoch folding and the Z\_n^2 statistics,
color-color and color-intensity diagrams. More is in preparation:
rms-energy, lag-energy, covariance-energy spectra, Lomb-Scargle
periodograms and in general all that is available in
`Stingray <https://github.com/StingraySoftware/stingray>`__. The
analysis done in HENDRICS will be compatible with the graphical user
interface `DAVE <https://github.com/StingraySoftware/dave>`__, so that
users will have the choice to analyze single datasets with an easy
interactive interface, and continue the analysis in batch mode with
HENDRICS. The periodograms produced by HENDRICS (like a power density
spectrum or a cospectrum), can be saved in a format compatible with
`Xspec <http://heasarc.gsfc.nasa.gov/xanadu/xspec/>`__ or
`Isis <http://space.mit.edu/home/mnowak/isis_vs_xspec/mod.html>`__, for
those who are familiar with those fitting packages. Despite its original
main focus on NuSTAR, the software can be used to make standard
aperiodic timing analysis on X-ray data from, in principle, any other
satellite (for sure XMM-Newton and RXTE).

The **documentation** can be found
`here <http://hendrics.readthedocs.io>`__.

A **tutorial** is also available
`here <http://hendrics.readthedocs.org/en/latest/tutorial.html>`__

Installation instructions
-------------------------

To install stable or beta releases:

::

    $ pip install hendrics

For development versions:

::

    $ git clone git@github.com/StingraySoftware/HENDRICS
    $ cd HENDRICS
    $ python setup.py install


Development guidelines
----------------------

Please follow the development workflow for
`the Astropy project<http://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`__.
In the hendrics/tests
directory, there is a test suite called ``test_fullrun.py``. These tests
use the actual command line scripts, and should always pass (albeit with
some adaptations). The other test suites, e.g. ``test_unit.py``, tests
the API.

.. |Build Status| image:: https://travis-ci.org/StingraySoftware/HENDRICS.svg?branch=master
   :target: https://travis-ci.org/StingraySoftware/HENDRICS
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/ifm0iydpu6gd7vwk/branch/master?svg=true
   :target: https://ci.appveyor.com/project/matteobachetti/hendrics/branch/master
.. |Coverage Status| image:: https://coveralls.io/repos/github/StingraySoftware/HENDRICS/badge.svg?branch=master&cache-control=no-cache
   :target: https://coveralls.io/github/StingraySoftware/HENDRICS?branch=master
.. |Documentation Status| image:: https://readthedocs.org/projects/hendrics/badge/?version=master
   :target: http://hendrics.readthedocs.io/en/master/?badge=master
