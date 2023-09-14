|Build Status| |Coverage Status| |Documentation Status|

HENDRICS - High ENergy Data Reduction Interface from the Command Shell
======================================================================

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
800, 109 -`arXiv:1409.3248 <https://arxiv.org/abs/1409.3248>`__).

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
``XSpec`` or ``ISIS``, for
those who are familiar with those fitting packages. Despite its original
main focus on NuSTAR, the software can be used to make standard
aperiodic timing analysis on X-ray data from, in principle, any other
satellite (for sure XMM-Newton and RXTE).

The **documentation** can be found
`here <https://hendrics.readthedocs.io>`__.

A **tutorial** is also available
`here <https://hendrics.readthedocs.io/en/main/tutorials/index.html>`__.

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


License and notes for the users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This software is released with a 3-clause BSD license. You can find
license information in the ``LICENSE.rst`` file.

**If you use this software in a publication**, please refer to its
Astrophysics Source Code Library identifier:

1. Bachetti, M. 2018, HENDRICS: High ENergy Data Reduction Interface from the Command Shell, record `ascl:1805.019 <https://ascl.net/1805.019>`__.

and please also cite `stingray <https://stingray.science/stingray/citing.html>`

In particular, **if you use the cospectrum**, please also refer to:

2. Bachetti et al. 2015, `ApJ <https://iopscience.iop.org/0004-637X/800/2/109/>`__ , **800**, 109.

If you have found a bug please report it by creating a
new issue on the `HENDRICS GitHub issue tracker. <https://github.com/StingraySoftware/HENDRICS/issues>`_

Development guidelines
----------------------

Please follow the development workflow for
`the Astropy project <https://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`__.
In the hendrics/tests
directory, there is a test suite called ``test_fullrun.py``. These tests
use the actual command line scripts, and should always pass (albeit with
some adaptations). The other test suites, e.g. ``test_unit.py``, tests
the API.

.. |Build Status| image:: https://github.com/StingraySoftware/HENDRICS/workflows/CI%20Tests/badge.svg
    :target: https://github.com/StingraySoftware/HENDRICS/actions/
.. |Coverage Status| image:: https://codecov.io/gh/StingraySoftware/HENDRICS/branch/main/graph/badge.svg
  :target: https://app.codecov.io/gh/StingraySoftware/HENDRICS
.. |Documentation Status| image:: https://readthedocs.org/projects/hendrics/badge/?version=main
   :target: https://hendrics.readthedocs.io/en/main/?badge=main
