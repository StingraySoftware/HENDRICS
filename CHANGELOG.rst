HENDRICS 8.2
~~~~~~~~~~~~

+ Brings all bugfixes coming with `Stingray 2.2.4 <https://github.com/StingraySoftware/stingray/releases/tag/v2.2.4>`__
+ New option for ``HENreadevents``: ``--safe-interval``, allowing to decrease GTI length by fixed amounts at the start and the end (useful, e.g., when observations show artifacts entering and exiting from occultation or SAA)

Internal changes:

+ Improvements with docs creation, in particular to the creation of the cli.rst file.

HENDRICS 8.1
~~~~~~~~~~~~

+ Brings all bugfixes coming with `Stingray 2.2 <https://github.com/StingraySoftware/stingray/releases/tag/v2.2>`__
+ New infrastructure, based on `pyproject.toml` as recommended by PEP 621
+ More bug fixes:
    + Fix issue with invalid coordinates
    + Solved an issue with recent Numba versions
    + Solved an issue with the initial values of TOA fitting
    + Make analyze_qffa_results more flexible
    + Fix bug when simulating by count rate in HENfake

HENDRICS 8.0.0
~~~~~~~~~~~~~~
+ Compatible with `Stingray 2.0.0 <https://github.com/StingraySoftware/stingray/releases/tag/v2.0.0>`__, which introduced:

    + Lomb-Scargle periodograms and cross spectra
    + Power colors
    + Easy filling of small gaps in light curves with random data
    + Generic timeseries (complex data, multi-dimensional data)

+ ``HENaccelsearch`` now has additional options for detrending, denoising and deorbiting
+ An improved Maximum likelihood algorithm as FFTFIT substitute for TOA calculation
+ NASA's IXPE added to supported missions
+ Better support of Stingray's native file formats

HENDRICS 7.0
~~~~~~~~~~~~

+ Based on `Stingray 1.0 <https://github.com/StingraySoftware/stingray/releases/tag/v1.0>`__, bringing a huge bump in performance
+ Following Astropy, Numpy and Scipy, HENDRICS 7.0 is only compatible with Python >3.8
+ Accepts many more file formats for round-trip of Stingray objects, thanks to the new functionality of Stingray.
+ Energy-filtered periodograms
+ A wider range of normalizations available for both ``HENfold`` and ``HENphaseogram``, with more options (e.g. smoothing) and higher-contrast color map by default
+ Many fixes to mission-specific files
+ Better info returned by Z/EF searches, including pulse amplitude estimates
+ New upper limit functionality in Z/EF searches with no candidates
+ ``HENplot`` now estimates the error of frequency and frequency derivative searches returned by ``HENzsearch``  and ``HENefsearch`` with option ``--fast``
+ Add ability to split files at a given MJD


HENDRICS 6.0
~~~~~~~~~~~~

+ Much Improved mission support
+ Lots of performance improvements with large datasets
+ Improved simulation and upper limit determination for Z searches
+ Improved candidate searching in Z searches
+ Lots of documentation fixes

HENDRICS 5.0
~~~~~~~~~~~~

More improvements to pulsar functionalities:

+ The accelerated search from Ransom+2002 is now available, to search the f-fdot space through Fourier analysis. It is highly performant but still needs some work. Please consider it experimental.
+ A much faster folding algorithm (See Bachetti+2020, ApJ) is now available, allowing to reduce the computing time of Z searches by a factor ~10, while simultaneously searching a 2D space of frequency and fdot. Select with ``--fast`` option
+ The classic Fast Folding Algorithm (Staelin 1969) is also available, to allow for extra-fast searches at low frequencies. However, this does not allow for "accelerated" searches on fdot. Also experimental and probably worth of further optimization.

Developed as part of CICLOPS -- Citizen Computing Pulsar Search, a project supported by *POR FESR Sardegna 2014 – 2020 Asse 1 Azione 1.1.3* (code RICERCA_1C-181), call for proposal "Aiuti per Progetti di Ricerca e Sviluppo 2017" managed by Sardegna Ricerche.


HENDRICS 4.0
~~~~~~~~~~~~

Lots of improvements to pulsar functionalities;

.. Note ::

    Windows support for Python <3.6 was dropped. Most of the code will still work on old versions,
    but the difficulty of tracking down library versions to test in Appveyor forces me
    to drop the obsolescent versions of Python from testing on that architecture.

HENDRICS 3.0
~~~~~~~~~~~~

The API is now rewritten to use
`Stingray <https://github.com/StingraySoftware/stingray>`__ where possible.
All MPxxx scripts are renamed to HENxxx.

Functionality additions:

+ Epoch folding search
+ Z-squared search
+ Color-Color Diagrams and Hardness-Intensity Diagrams
+ Power spectral fitting

(MaLTPyNT) 2.0
~~~~~~~~~~~~~~
.. Note ::

    MaLTPyNT provisionally accepted as an
    `Astropy affiliated package <https://www.astropy.org/affiliated/index.html>`__


In preparation for the 2.0 release, the API has received some visible changes.
Names do not have the `mp_` prefix anymore, as they were very redundant; the
structure of the code base is now based on the AstroPy structure; tests have
been moved and the documentation improved.

`HENexposure` is a new livetime correction script on sub-second timescales for
NuSTAR. It will be able to replace `nulccorr`, and get results on shorter bin
times, in observations done with a specific observing mode, where the observer
has explicitly requested to telemeter all events (including rejected) and the
user has run `nupipeline` with the `CLEANCOLS = NO` option.
This tool is under testing.

`HENfake` is a new script to create fake observation files in FITS format, for
testing. New functions to create fake data will be added to `hendrics.fake`.
