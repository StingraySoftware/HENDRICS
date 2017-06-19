# Notice
This repository contains a fork of MaLTPyNT. This software is being heavily rewritten in order to use Stingray's classes and functions whenever possible. To use the original MaLTPyNT, please go to [matteobachetti/MaLTPyNT](https://github.com/matteobachetti/MaLTPyNT).

# Competition

MaLTPyNT means “Matteo’s Libraries and Tools in Python for NuSTAR Timing”. This acronym has multiple problems, including the fact that it mentions my name, it kind of limits the scope of the code to a single X-ray mission, NuSTAR, and uses a pun about beer that just a thousand other people have already used (see, e.g., the pulsar timing software PINT).
Therefore, I would like to propose a new competition:

*GIVE A NAME TO STINGRAY’S SCRIPTS*

Be creative, the acronym should mention synonyms of timing/variability/spectral timing, compact objects/black holes/pulsars, X-rays/high energies. Mentions to music are highly appreciated (and appreciating music is basically using spectral timing, by the way).

Please send your ideas to matteo __at__ matteobachetti __dot__ it

# MaLTPyNT - Matteo's Libraries and Tools in Python for NuSTAR Timing.

| **Devel 2.0** | [![Build Status Master](https://travis-ci.org/matteobachetti/MaLTPyNT.svg?branch=master)](https://travis-ci.org/matteobachetti/MaLTPyNT) | [![Coverage Status Master](https://coveralls.io/repos/matteobachetti/MaLTPyNT/badge.svg?branch=master&service=github)](https://coveralls.io/github/matteobachetti/MaLTPyNT?branch=master) | [![Documentation Status Master](https://readthedocs.org/projects/maltpynt/badge/?version=latest)](https://readthedocs.org/projects/maltpynt/badge/?version=latest) | [![AppVeyor status](https://ci.appveyor.com/api/projects/status/op01lg1v9p4wrasv/branch/master?svg=true)](https://ci.appveyor.com/project/matteobachetti/maltpynt/branch/master) |
| ------------- | ----------- | ------------- | ----------- | ----------- |
| **Bugfix 1.0** |  [![Build Status 1.0](https://travis-ci.org/matteobachetti/MaLTPyNT.svg?branch=1.0_bugfix)](https://travis-ci.org/matteobachetti/MaLTPyNT) | [![Coverage Status 1.0](https://coveralls.io/repos/matteobachetti/MaLTPyNT/badge.svg?branch=1.0_bugfix&service=github)](https://coveralls.io/github/matteobachetti/MaLTPyNT?branch=1.0_bugfix) | [![Documentation Status](https://readthedocs.org/projects/maltpynt/badge/?version=1.0_bugfix)](https://readthedocs.org/projects/maltpynt/badge/?version=1.0_bugfix) |  |


This software is designed to do correctly and fairly easily a **quick-look timing analysis** of NuSTAR data, treating properly orbital gaps and exploiting the presence of two independent detectors by using the **cospectrum** as a proxy for the power density spectrum (for an explanation of why this is important, look at Bachetti et al., _ApJ_, 800, 109 -[arXiv:1409.3248](http://arxiv.org/abs/1409.3248)). The output of the analysis is a cospectrum, or a power density spectrum, that can be fitted with [Xspec](http://heasarc.gsfc.nasa.gov/xanadu/xspec/) or [Isis](http://space.mit.edu/home/mnowak/isis_vs_xspec/mod.html). Also, one can calculate in the same easy way **time lags** (still under testing, help is welcome).
Despite its main focus on NuSTAR, the software can be used to make standard aperiodic timing analysis on X-ray data from, in principle, any other satellite (for sure XMM-Newton and RXTE).

The **documentation** can be found [here](http://maltpynt.readthedocs.org).

A **tutorial** is also available [here](http://maltpynt.readthedocs.org/en/stable/tutorial.html)
