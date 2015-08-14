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

MaLTPyNT vs FTOOLS (and together with FTOOLS)
---------------------------------------------

vs POWSPEC
~~~~~~~~~~

MaLTPyNT does a better job than POWSPEC from several points of view:

-  **Good time intervals** (GTIs) are completely avoided in the
   computation. No gaps dirtying up the power spectrum! (This is
   particularly important for NuSTAR, as orbital gaps are always present
   in typical observation timescales)

-  The number of bins used in the power spectrum (or the cospectrum)
   need not be a power of two! No padding needed.

Clarification about dead time treatment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaLTPyNT **does not supersede**
`nulccorr <https://heasarc.gsfc.nasa.gov/ftools/caldb/help/nulccorr.html>`__.
If one is only interested in frequencies below ~0.5 Hz, nulccorr treats
robustly various dead time components and its use is recommended. Light
curves produced by nulccorr can be converted to MaLTPyNT format using
``MPlcurve --fits-input <lcname>.fits``, and used for the subsequent
steps of the timing analysis.

License and notes for the users
-------------------------------

This software is released with a 3-clause BSD license. You can find
license information in the ``LICENSE.rst`` file.

**If you use this software in a publication**, please refer to its
Astrophysics Source Code Library identifier:

1. Bachetti, M. 2015, MaLTPyNT, Astrophysics Source Code Library, record
   `ascl:1502.021 <http://ascl.net/1502.021>`__.

In particular, **if you use the cospectrum**, please also refer to:

2. Bachetti et al. 2015,
   `ApJ <http://iopscience.iop.org/0004-637X/800/2/109/>`__ , **800**,
   109.

I listed a number of **open issues** in the
`Issues <https://bitbucket.org/mbachett/maltpynt/issues?status=new&status=open>`__
page. Feel free to **comment** on them and **propose more**. Please
choose carefully the category: bugs, enhancements, etc.

Acknowledgements
----------------

I would like to thank all the co-authors of `the NuSTAR timing
paper <http://arxiv.org/abs/1409.3248>`__ and the NuSTAR X-ray binaries
working group. This software would not exist without the interesting
discussions before and around that paper. In particular, I would like to
thank Ivan Zolotukhin, Francesca Fornasini, Erin Kara, Felix Fürst,
Poshak Gandhi, John Tomsick and Abdu Zoghbi for helping testing the code
and giving various suggestions on how to improve it. Last but not least,
I would like to thank Marco Buttu (by the way, `check out his book if
you speak
Italian <http://www.amazon.it/Programmare-con-Python-completa-DigitalLifeStyle-ebook/dp/B00L95VURC/ref=sr_1_1?s=books&ie=UTF8&qid=1424298092&sr=1-1>`__)
for his priceless pointers on Python coding and code management
techniques.

.. |Build Status Master| image:: https://travis-ci.org/matteobachetti/MaLTPyNT.svg?branch=master
   :target: https://travis-ci.org/matteobachetti/MaLTPyNT
.. |Coverage Status Master| image:: https://coveralls.io/repos/matteobachetti/MaLTPyNT/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/matteobachetti/MaLTPyNT?branch=master
.. |Build Status Unstable| image:: https://travis-ci.org/matteobachetti/MaLTPyNT.svg?branch=unstable
   :target: https://travis-ci.org/matteobachetti/MaLTPyNT
.. |Coverage Status Unstable| image:: https://coveralls.io/repos/matteobachetti/MaLTPyNT/badge.svg?branch=unstable&service=github
   :target: https://coveralls.io/github/matteobachetti/MaLTPyNT?branch=unstable
.
