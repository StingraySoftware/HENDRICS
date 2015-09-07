Command line interface
======================

MP2xspec
--------

::

    usage: MP2xspec [-h] [--loglevel LOGLEVEL] [--debug] [--flx2xsp]
                    files [files ...]

    Save a frequency spectrum in a qdp file that can be read by flx2xsp and
    produce a XSpec-compatible spectrumfile

    positional arguments:
      files                List of files

    optional arguments:
      -h, --help           show this help message and exit
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              use DEBUG logging level
      --flx2xsp            Also call flx2xsp at the end


MPcalibrate
-----------

::

    usage: MPcalibrate [-h] [-r RMF] [-o] [--loglevel LOGLEVEL] [--debug]
                       [--nproc NPROC]
                       files [files ...]

    Calibrate clean event files by associating the correct energy to each PI
    channel. Uses either a specified rmf file or (for NuSTAR only) an rmf file
    from the CALDB

    positional arguments:
      files                List of files

    optional arguments:
      -h, --help           show this help message and exit
      -r RMF, --rmf RMF    rmf file used for calibration
      -o, --overwrite      Overwrite; default: no
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              use DEBUG logging level
      --nproc NPROC        Number of processors to use


MPcreategti
-----------

::

    usage: MPcreategti [-h] [-f FILTER] [-c] [-o] [-a APPLY_GTI]
                       [--safe-interval SAFE_INTERVAL SAFE_INTERVAL]
                       [--loglevel LOGLEVEL] [--debug]
                       files [files ...]

    Create GTI files from a filter expression, or applies previously created GTIs
    to a file

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      -f FILTER, --filter FILTER
                            Filter expression, that has to be a valid Python
                            boolean operation on a data variable contained in the
                            files
      -c, --create-only     If specified, creates GTIs withouth applyingthem to
                            files (Default: False)
      -o, --overwrite       Overwrite original file (Default: False)
      -a APPLY_GTI, --apply-gti APPLY_GTI
                            Apply a GTI from this file to input files
      --safe-interval SAFE_INTERVAL SAFE_INTERVAL
                            Interval at start and stop of GTIs used for filtering
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


MPdumpdyn
---------

::

    usage: MPdumpdyn [-h] [--noplot] files [files ...]

    Dump dynamical (cross) power spectra

    positional arguments:
      files       List of files in any valid MaLTPyNT format for PDS or CPDS

    optional arguments:
      -h, --help  show this help message and exit
      --noplot    plot results


MPexposure
----------

::

    usage: MPexposure [-h] [-o OUTROOT] [--loglevel LOGLEVEL] [--debug] [--plot]
                      lcfile uffile

    Create exposure light curve based on unfiltered event files.

    positional arguments:
      lcfile                Light curve file (MaltPyNT format)
      uffile                Unfiltered event file (FITS)

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTROOT, --outroot OUTROOT
                            Root of output file names
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level
      --plot                Plot on window


MPfake
------

::

    usage: MPfake [-h] [-e EVENT_LIST] [-o OUTNAME] [-i INSTRUMENT]
                  [--tstart TSTART] [--tstop TSTOP] [--mjdref MJDREF]
                  [--loglevel LOGLEVEL] [--debug]

    Create an event file in FITS format from an event list, or simulating it. If
    input event list is not specified, generates the events randomly

    optional arguments:
      -h, --help            show this help message and exit
      -e EVENT_LIST, --event-list EVENT_LIST
                            File containint event list
      -o OUTNAME, --outname OUTNAME
                            Output file name
      -i INSTRUMENT, --instrument INSTRUMENT
                            Instrument name
      --tstart TSTART       Start time of the observation (s from MJDREF)
      --tstop TSTOP         End time of the observation (s from MJDREF)
      --mjdref MJDREF       Reference MJD
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


MPfspec
-------

::

    usage: MPfspec [-h] [-b BINTIME] [-r REBIN] [-f FFTLEN] [-k KIND]
                   [--norm NORM] [--noclobber] [-o OUTROOT] [--loglevel LOGLEVEL]
                   [--nproc NPROC] [--back BACK] [--debug] [--save-dyn]
                   files [files ...]

    Create frequency spectra (PDS, CPDS, cospectrum) starting from well-defined
    input ligthcurves

    positional arguments:
      files                 List of light curve files

    optional arguments:
      -h, --help            show this help message and exit
      -b BINTIME, --bintime BINTIME
                            Light curve bin time; if negative, interpreted as
                            negative power of 2. Default: 2^-10, or keep input lc
                            bin time (whatever is larger)
      -r REBIN, --rebin REBIN
                            (C)PDS rebinning to apply. Default: none
      -f FFTLEN, --fftlen FFTLEN
                            Length of FFTs. Default: 512 s
      -k KIND, --kind KIND  Spectra to calculate, as comma-separated list
                            (Accepted: PDS and CPDS; Default: "PDS,CPDS")
      --norm NORM           Normalization to use (Accepted: Leahy and rms;
                            Default: "Leahy")
      --noclobber           Do not overwrite existing files
      -o OUTROOT, --outroot OUTROOT
                            Root of output file names for CPDS only
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --nproc NPROC         Number of processors to use
      --back BACK           Estimated background (non-source) count rate
      --debug               use DEBUG logging level
      --save-dyn            save dynamical power spectrum


MPlags
------

::

    usage: MPlags [-h] [-o OUTROOT] [--loglevel LOGLEVEL] [--noclobber] [--debug]
                  files [files ...]

    Calculate time lags from the cross power spectrum and the power spectra of the
    two channels

    positional arguments:
      files                 Three files: the cross spectrum and the two power
                            spectra

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTROOT, --outroot OUTROOT
                            Root of output file names
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG;default:WARNING)
      --noclobber           Do not overwrite existing files
      --debug               use DEBUG logging level


MPlcurve
--------

::

    usage: MPlcurve [-h] [-b BINTIME]
                    [--safe-interval SAFE_INTERVAL SAFE_INTERVAL]
                    [--pi-interval PI_INTERVAL PI_INTERVAL]
                    [-e E_INTERVAL E_INTERVAL] [-s] [-j] [-g] [--minlen MINLEN]
                    [--ignore-gtis] [-d OUTDIR] [--loglevel LOGLEVEL]
                    [--nproc NPROC] [--debug] [--noclobber] [--fits-input]
                    [--txt-input]
                    files [files ...]

    Create lightcurves starting from event files. It is possible to specify energy
    or channel filtering options

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      -b BINTIME, --bintime BINTIME
                            Bin time; if negative, negative power of 2
      --safe-interval SAFE_INTERVAL SAFE_INTERVAL
                            Interval at start and stop of GTIs used for filtering
      --pi-interval PI_INTERVAL PI_INTERVAL
                            PI interval used for filtering
      -e E_INTERVAL E_INTERVAL, --e-interval E_INTERVAL E_INTERVAL
                            Energy interval used for filtering
      -s, --scrunch         Create scrunched light curve (single channel)
      -j, --join            Create joint light curve (multiple channels)
      -g, --gti-split       Split light curve by GTI
      --minlen MINLEN       Minimum length of acceptable GTIs (default:4)
      --ignore-gtis         Ignore GTIs
      -d OUTDIR, --outdir OUTDIR
                            Output directory
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --nproc NPROC         Number of processors to use
      --debug               use DEBUG logging level
      --noclobber           Do not overwrite existing files
      --fits-input          Input files are light curves in FITS format
      --txt-input           Input files are light curves in txt format


MPplot
------

::

    usage: MPplot [-h] [--noplot] files [files ...]

    Plot the content of MaLTPyNT light curves and frequency spectra

    positional arguments:
      files       List of files

    optional arguments:
      -h, --help  show this help message and exit
      --noplot    Only create images, do not plot


MPreadevents
------------

::

    usage: MPreadevents [-h] [--loglevel LOGLEVEL] [--nproc NPROC] [--noclobber]
                        [-g] [--debug]
                        files [files ...]

    Read a cleaned event files and saves the relevant information in a standard
    format

    positional arguments:
      files                List of files

    optional arguments:
      -h, --help           show this help message and exit
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --nproc NPROC        Number of processors to use
      --noclobber          Do not overwrite existing event files
      -g, --gti-split      Split event list by GTI
      --debug              use DEBUG logging level


MPreadfile
----------

::

    usage: MPreadfile [-h] files [files ...]

    Print the content of MaLTPyNT files

    positional arguments:
      files       List of files

    optional arguments:
      -h, --help  show this help message and exit


MPrebin
-------

::

    usage: MPrebin [-h] [-r REBIN] [--loglevel LOGLEVEL] [--debug]
                   files [files ...]

    Rebin light curves and frequency spectra.

    positional arguments:
      files                 List of light curve files

    optional arguments:
      -h, --help            show this help message and exit
      -r REBIN, --rebin REBIN
                            Rebinning to apply. Only if the quantity to rebin is a
                            (C)PDS, it is possible to specify a non-integer rebin
                            factor, in which case it is interpreted as a
                            geometrical binning factor
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


MPscrunchlc
-----------

::

    usage: MPscrunchlc [-h] [-o OUT] [--loglevel LOGLEVEL] [--debug]
                       files [files ...]

    Sum lightcurves from different instruments or energy ranges

    positional arguments:
      files                List of files

    optional arguments:
      -h, --help           show this help message and exit
      -o OUT, --out OUT    Output file
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              use DEBUG logging level


MPsumfspec
----------

::

    usage: MPsumfspec [-h] [-o OUTNAME] files [files ...]

    Sum (C)PDSs contained in different files

    positional arguments:
      files                 List of light curve files

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTNAME, --outname OUTNAME
                            Output file name for summed (C)PDS. Default:
                            tot_(c)pds.nc


