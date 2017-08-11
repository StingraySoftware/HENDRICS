[![Build Status](https://travis-ci.org/StingraySoftware/MaLTPyNT_reboot.svg?branch=master)](https://travis-ci.org/StingraySoftware/MaLTPyNT_reboot)
[![Coverage Status](https://coveralls.io/repos/github/StingraySoftware/MaLTPyNT_reboot/badge.svg?branch=master&cache-control=no-cache)](https://coveralls.io/github/StingraySoftware/MaLTPyNT_reboot?branch=master)

# Notice
This repository contains a fork of MaLTPyNT. This software is being heavily rewritten in order to use Stingray's classes and functions whenever possible. To use the original MaLTPyNT, please go to [matteobachetti/MaLTPyNT](https://github.com/matteobachetti/MaLTPyNT).

# Competition

MaLTPyNT means “Matteo’s Libraries and Tools in Python for NuSTAR Timing”. This acronym has multiple problems, including the fact that it mentions my name, it kind of limits the scope of the code to a single X-ray mission, NuSTAR, and uses a pun about beer that just a thousand other people have already used (see, e.g., the pulsar timing software PINT).
Therefore, I would like to propose a new competition:

*GIVE A NAME TO STINGRAY’S SCRIPTS*

Be creative, the acronym should mention synonyms of timing/variability/spectral timing, compact objects/black holes/pulsars, X-rays/high energies. Mentions to music are highly appreciated (and appreciating music is basically using spectral timing, by the way).

Please send your ideas to matteo __at__ matteobachetti __dot__ it

# Development guidelines

The development will initially be towards substituting MaLTPyNT's internal application programming interface (API) with Stingray's, while maintaining the same Command Line Interface (CLI). Ideally, a user that only uses MaLTPyNT from the command line, without executing python explicitly, should not notice the difference.
In the maltpynt/tests directory, there is a test suite called `test_fullrun.py`. These tests use the actual command line scripts, and should always pass (albeit with some adaptations). The other test suite, `test_unit.py`, tests the API, and will need to be rewritten to account for the API changes.

# MaLTPyNT - Matteo's Libraries and Tools in Python for NuSTAR Timing.

This software is designed to do correctly and fairly easily a **quick-look timing analysis** of NuSTAR data, treating properly orbital gaps and exploiting the presence of two independent detectors by using the **cospectrum** as a proxy for the power density spectrum (for an explanation of why this is important, look at Bachetti et al., _ApJ_, 800, 109 -[arXiv:1409.3248](http://arxiv.org/abs/1409.3248)). The output of the analysis is a cospectrum, or a power density spectrum, that can be fitted with [Xspec](http://heasarc.gsfc.nasa.gov/xanadu/xspec/) or [Isis](http://space.mit.edu/home/mnowak/isis_vs_xspec/mod.html). Also, one can calculate in the same easy way **time lags** (still under testing, help is welcome).
Despite its main focus on NuSTAR, the software can be used to make standard aperiodic timing analysis on X-ray data from, in principle, any other satellite (for sure XMM-Newton and RXTE).

The **documentation** can be found [here](http://maltpynt.readthedocs.org).

A **tutorial** is also available [here](http://maltpynt.readthedocs.org/en/stable/tutorial.html)
