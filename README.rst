MaLTPyNT - Matteo's Libraries and Tools in Python for NuSTAR Timing.
====================================================================

Master: |Build Status Master| |Coverage Status Master|

Devel: |Build Status Unstable| |Coverage Status Unstable|

This software is designed to do correctly and fairly easily a
**quick-look timing analysis** of NuSTAR data, treating properly orbital
gaps and exploiting the presence of two independent detectors by using
the **cospectrum** as a proxy for the power density spectrum (for an
explanation of why this is important, look at Bachetti et al., *ApJ*,
800, 109 -`arXiv:1409.3248 <http://arxiv.org/abs/1409.3248>`__). The
output of the analysis is a cospectrum, or a power density spectrum,
that can be fitted with
`Xspec <http://heasarc.gsfc.nasa.gov/xanadu/xspec/>`__ or
`Isis <http://space.mit.edu/home/mnowak/isis_vs_xspec/mod.html>`__.
Also, one can calculate in the same easy way **time lags** (still under
testing, help is welcome). Despite its main focus on NuSTAR, the
software can be used to make standard spectral analysis on X-ray data
from, in principle, any other satellite (for sure XMM-Newton and RXTE).

The documentation can be found
`here <http://maltpynt.readthedocs.org/en/latest/>`__

.. |Build Status Master| image:: https://travis-ci.org/matteobachetti/MaLTPyNT.svg?branch=master
   :target: https://travis-ci.org/matteobachetti/MaLTPyNT
.. |Coverage Status Master| image:: https://coveralls.io/repos/matteobachetti/MaLTPyNT/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/matteobachetti/MaLTPyNT?branch=master
.. |Build Status Unstable| image:: https://travis-ci.org/matteobachetti/MaLTPyNT.svg?branch=unstable
   :target: https://travis-ci.org/matteobachetti/MaLTPyNT
.. |Coverage Status Unstable| image:: https://coveralls.io/repos/matteobachetti/MaLTPyNT/badge.svg?branch=unstable&service=github
   :target: https://coveralls.io/github/matteobachetti/MaLTPyNT?branch=unstable
