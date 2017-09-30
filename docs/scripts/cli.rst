Command line interface
======================

HEN2xspec
---------

::

    usage: HEN2xspec [-h] [--loglevel LOGLEVEL] [--debug] [--flx2xsp]
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


HENbaseline
-----------

::

    usage: HENbaseline [-h] [-o OUT] [--loglevel LOGLEVEL] [--debug]
                       [-p ASYMMETRY] [-l LAM]
                       files [files ...]

    Subtract a baseline from the lightcurve using the Asymmetric Least Squares
    algorithm. The two parameters p and lambda control the asymmetry and
    smoothness of the baseline. See below for details.

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      -o OUT, --out OUT     Output file
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level
      -p ASYMMETRY, --asymmetry ASYMMETRY
                            "asymmetry" parameter. Smaller values make the
                            baseline more "horizontal". Typically 0.001 < p < 0.1,
                            but not necessarily.
      -l LAM, --lam LAM     lambda, or "smoothness", parameter. Larger values make
                            the baseline stiffer. Typically 1e2 < lam < 1e9


HENcalibrate
------------

::

    usage: HENcalibrate [-h] [-r RMF] [-o] [--loglevel LOGLEVEL] [--debug]
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


HENcolors
---------

::

    usage: HENcolors [-h] -e ENERGIES ENERGIES ENERGIES ENERGIES [-b BINTIME]
                     [-o OUT] [--use-pi USE_PI] [--loglevel LOGLEVEL] [--debug]
                     files [files ...]

    Calculate color light curves

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      -e ENERGIES ENERGIES ENERGIES ENERGIES, --energies ENERGIES ENERGIES ENERGIES ENERGIES
                            The energy boundaries in keV used to calculate the
                            color. E.g. -e 2 3 4 6 means that the color will be
                            calculated as 4.-6./2.-3. keV. If --use-pi is
                            specified, these are interpreted as PI channels
      -b BINTIME, --bintime BINTIME
                            Bin time; if negative, negative power of 2
      -o OUT, --out OUT     Output file
      --use-pi USE_PI       Use the PI channel instead of energies
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


HENcreategti
------------

::

    usage: HENcreategti [-h] [-f FILTER] [-c] [--overwrite] [-a APPLY_GTI]
                        [-l MINIMUM_LENGTH]
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
      --overwrite           Overwrite original file (Default: False)
      -a APPLY_GTI, --apply-gti APPLY_GTI
                            Apply a GTI from this file to input files
      -l MINIMUM_LENGTH, --minimum-length MINIMUM_LENGTH
                            Minimum length of GTIs (below this length, they will
                            be discarded)
      --safe-interval SAFE_INTERVAL SAFE_INTERVAL
                            Interval at start and stop of GTIs used for filtering
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


HENdumpdyn
----------

::

    usage: HENdumpdyn [-h] [--noplot] files [files ...]

    Dump dynamical (cross) power spectra

    positional arguments:
      files       List of files in any valid HENDRICS format for PDS or CPDS

    optional arguments:
      -h, --help  show this help message and exit
      --noplot    plot results


HENefsearch
-----------

::

    usage: HENefsearch [-h] -f FMIN -F FMAX [--fdotmin FDOTMIN]
                       [--fdotmax FDOTMAX] [-n NBIN] [--segment-size SEGMENT_SIZE]
                       [--step STEP] [--oversample OVERSAMPLE] [--expocorr]
                       [--find-candidates] [--conflevel CONFLEVEL]
                       [--fit-candidates] [--curve CURVE]
                       [--fit-frequency FIT_FREQUENCY] [--debug]
                       [--loglevel LOGLEVEL] [-N N]
                       files [files ...]

    Search for pulsars using the epoch folding or the Z_n^2 algorithm

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      -f FMIN, --fmin FMIN  Minimum frequency to fold
      -F FMAX, --fmax FMAX  Maximum frequency to fold
      --fdotmin FDOTMIN     Minimum fdot to fold
      --fdotmax FDOTMAX     Maximum fdot to fold
      -n NBIN, --nbin NBIN  Number of phase bins of the profile
      --segment-size SEGMENT_SIZE
                            Size of the event list segment to use (default None,
                            implying the whole observation)
      --step STEP           Step size of the frequency axis. Defaults to
                            1/oversample/observ.length.
      --oversample OVERSAMPLE
                            Oversampling factor - frequency resolution improvement
                            w.r.t. the standard FFT's 1/observ.length.
      --expocorr            Correct for the exposure of the profile bins. This
                            method is *much* slower, but it is useful for very
                            slow pulsars, where data gaps due to occultation or
                            SAA passages can significantly alter the exposure of
                            different profile bins.
      --find-candidates     Find pulsation candidates using thresholding
      --conflevel CONFLEVEL
                            percent confidence level for thresholding [0-100).
      --fit-candidates      Fit the candidate peaks in the periodogram
      --curve CURVE         Kind of curve to use (sinc or Gaussian)
      --fit-frequency FIT_FREQUENCY
                            Force the candidate frequency to FIT_FREQUENCY
      --debug               use DEBUG logging level
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      -N N                  The number of harmonics to use in the search (the 'N'
                            in Z^2_N; only relevant to Z search!)


HENexcvar
---------

::

    usage: HENexcvar [-h] [-c CHUNK_LENGTH] [--fraction-step FRACTION_STEP]
                     [--norm NORM] [--loglevel LOGLEVEL] [--debug]
                     files [files ...]

    Calculate excess variance in light curve chunks

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      -c CHUNK_LENGTH, --chunk-length CHUNK_LENGTH
                            Length in seconds of the light curve chunks
      --fraction-step FRACTION_STEP
                            If the step is not a full chunk_length but less,this
                            indicates the ratio between step step and
                            `chunk_length`
      --norm NORM           Choose between fvar, excvar and norm_excvar
                            normalization, referring to Fvar, excess variance and
                            normalized excess variance respectively (see Vaughan
                            et al. 2003 for details).
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


HENexposure
-----------

::

    usage: HENexposure [-h] [-o OUTROOT] [--loglevel LOGLEVEL] [--debug] [--plot]
                       lcfile uffile

    Create exposure light curve based on unfiltered event files.

    positional arguments:
      lcfile                Light curve file (HENDRICS format)
      uffile                Unfiltered event file (FITS)

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTROOT, --outroot OUTROOT
                            Root of output file names
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level
      --plot                Plot on window


HENfake
-------

::

    usage: HENfake [-h] [-e EVENT_LIST] [-l LC] [-c CTRATE] [-o OUTNAME]
                   [-i INSTRUMENT] [-m MISSION] [--tstart TSTART] [--tstop TSTOP]
                   [--mjdref MJDREF] [--deadtime DEADTIME [DEADTIME ...]]
                   [--loglevel LOGLEVEL] [--debug]

    Create an event file in FITS format from an event list, or simulating it. If
    input event list is not specified, generates the events randomly

    optional arguments:
      -h, --help            show this help message and exit
      -e EVENT_LIST, --event-list EVENT_LIST
                            File containint event list
      -l LC, --lc LC        File containing light curve
      -c CTRATE, --ctrate CTRATE
                            Count rate for simulated events
      -o OUTNAME, --outname OUTNAME
                            Output file name
      -i INSTRUMENT, --instrument INSTRUMENT
                            Instrument name
      -m MISSION, --mission MISSION
                            Mission name
      --tstart TSTART       Start time of the observation (s from MJDREF)
      --tstop TSTOP         End time of the observation (s from MJDREF)
      --mjdref MJDREF       Reference MJD
      --deadtime DEADTIME [DEADTIME ...]
                            Dead time magnitude. Can be specified as a single
                            number, or two. In this last case, the second value is
                            used as sigma of the dead time distribution
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


HENfspec
--------

::

    usage: HENfspec [-h] [-b BINTIME] [-r REBIN] [-f FFTLEN] [-k KIND]
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
      --norm NORM           Normalization to use (Accepted: leahy and rms;
                            Default: "leahy")
      --noclobber           Do not overwrite existing files
      -o OUTROOT, --outroot OUTROOT
                            Root of output file names for CPDS only
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --nproc NPROC         Number of processors to use
      --back BACK           Estimated background (non-source) count rate
      --debug               use DEBUG logging level
      --save-dyn            save dynamical power spectrum


HENlags
-------

::

    usage: HENlags [-h] [--loglevel LOGLEVEL] [--debug] files [files ...]

    Read timelags from cross spectrum results and save them to a qdp file

    positional arguments:
      files                List of files

    optional arguments:
      -h, --help           show this help message and exit
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              use DEBUG logging level


HENlcurve
---------

::

    usage: HENlcurve [-h] [-b BINTIME]
                     [--safe-interval SAFE_INTERVAL SAFE_INTERVAL]
                     [--pi-interval PI_INTERVAL PI_INTERVAL]
                     [-e E_INTERVAL E_INTERVAL] [-s] [-j] [-g] [--minlen MINLEN]
                     [--ignore-gtis] [-d OUTDIR] [-o OUTFILE]
                     [--loglevel LOGLEVEL] [--nproc NPROC] [--debug] [--noclobber]
                     [--fits-input] [--txt-input]
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
      -o OUTFILE, --outfile OUTFILE
                            Output file name
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --nproc NPROC         Number of processors to use
      --debug               use DEBUG logging level
      --noclobber           Do not overwrite existing files
      --fits-input          Input files are light curves in FITS format
      --txt-input           Input files are light curves in txt format


HENmodel
--------

::

    usage: HENmodel [-h] [-m MODELFILE] [--fitmethod FITMETHOD]
                    [--frequency-interval FREQUENCY_INTERVAL [FREQUENCY_INTERVAL ...]]
                    [--loglevel LOGLEVEL] [--debug]
                    files [files ...]

    Fit frequency spectra (PDS, CPDS, cospectrum) with user-defined models

    positional arguments:
      files                 List of light curve files

    optional arguments:
      -h, --help            show this help message and exit
      -m MODELFILE, --modelfile MODELFILE
                            File containing an Astropy model with or without
                            constraints
      --fitmethod FITMETHOD
                            Any scipy-compatible fit method
      --frequency-interval FREQUENCY_INTERVAL [FREQUENCY_INTERVAL ...]
                            Select frequency interval(s) to fit. Must be an even
                            number of frequencies in Hz, like "--frequency-
                            interval 0 2" or "--frequency-interval 0 2 5 10",
                            meaning that the spectrum will be fitted between 0 and
                            2 Hz, or using the intervals 0-2 Hz and 5-10 Hz.
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level


HENphaseogram
-------------

::

    usage: HENphaseogram [-h] [-f FREQ] [--fdot FDOT] [--periodogram PERIODOGRAM]
                         [-n NBIN] [--ntimes NTIMES] [--debug] [--test]
                         [--loglevel LOGLEVEL]
                         file

    Plot an interactive phaseogram

    positional arguments:
      file                  Input event file

    optional arguments:
      -h, --help            show this help message and exit
      -f FREQ, --freq FREQ  Initial frequency to fold
      --fdot FDOT           Initial fdot
      --periodogram PERIODOGRAM
                            Periodogram file
      -n NBIN, --nbin NBIN  Number of phase bins (X axis) of the profile
      --ntimes NTIMES       Number of time bins (Y axis) of the phaseogram
      --debug               use DEBUG logging level
      --test                Just a test. Destroys the window immediately
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)


HENplot
-------

::

    usage: HENplot [-h] [--noplot] [--CCD] [--HID] [--figname FIGNAME]
                   [-o OUTFILE] [--xlog] [--ylog] [--xlin] [--ylin] [--fromstart]
                   [--axes AXES AXES]
                   files [files ...]

    Plot the content of HENDRICS light curves and frequency spectra

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      --noplot              Only create images, do not plot
      --CCD                 This is a color-color diagram. In this case, the list
                            of files is expected to be given as soft0.nc,
                            hard0.nc, soft1.nc, hard1.nc, ...
      --HID                 This is a hardness-intensity diagram. In this case,
                            the list of files is expected to be given as
                            color0.nc, intensity0.nc, color1.nc, intensity1.nc,
                            ...
      --figname FIGNAME     Figure name
      -o OUTFILE, --outfile OUTFILE
                            Output data file in QDP format
      --xlog                Use logarithmic X axis
      --ylog                Use logarithmic Y axis
      --xlin                Use linear X axis
      --ylin                Use linear Y axis
      --fromstart           Times are measured from the start of the observation
                            (only relevant for light curves)
      --axes AXES AXES      Plot two variables contained in the file


HENreadevents
-------------

::

    usage: HENreadevents [-h] [--loglevel LOGLEVEL] [--nproc NPROC] [--noclobber]
                         [-g] [--min-length MIN_LENGTH] [--gti-string GTI_STRING]
                         [--debug]
                         files [files ...]

    Read a cleaned event files and saves the relevant information in a standard
    format

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      --nproc NPROC         Number of processors to use
      --noclobber           Do not overwrite existing event files
      -g, --gti-split       Split event list by GTI
      --min-length MIN_LENGTH
                            Minimum length of GTIs to consider
      --gti-string GTI_STRING
                            GTI string
      --debug               use DEBUG logging level


HENreadfile
-----------

::

    usage: HENreadfile [-h] files [files ...]

    Print the content of HENDRICS files

    positional arguments:
      files       List of files

    optional arguments:
      -h, --help  show this help message and exit


HENrebin
--------

::

    usage: HENrebin [-h] [-r REBIN] [--loglevel LOGLEVEL] [--debug]
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


HENscrunchlc
------------

::

    usage: HENscrunchlc [-h] [-o OUT] [--loglevel LOGLEVEL] [--debug]
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


HENsumfspec
-----------

::

    usage: HENsumfspec [-h] [-o OUTNAME] files [files ...]

    Sum (C)PDSs contained in different files

    positional arguments:
      files                 List of light curve files

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTNAME, --outname OUTNAME
                            Output file name for summed (C)PDS. Default:
                            tot_(c)pds.nc


HENzsearch
----------

::

    usage: HENzsearch [-h] -f FMIN -F FMAX [--fdotmin FDOTMIN] [--fdotmax FDOTMAX]
                      [-n NBIN] [--segment-size SEGMENT_SIZE] [--step STEP]
                      [--oversample OVERSAMPLE] [--expocorr] [--find-candidates]
                      [--conflevel CONFLEVEL] [--fit-candidates] [--curve CURVE]
                      [--fit-frequency FIT_FREQUENCY] [--debug]
                      [--loglevel LOGLEVEL] [-N N]
                      files [files ...]

    Search for pulsars using the epoch folding or the Z_n^2 algorithm

    positional arguments:
      files                 List of files

    optional arguments:
      -h, --help            show this help message and exit
      -f FMIN, --fmin FMIN  Minimum frequency to fold
      -F FMAX, --fmax FMAX  Maximum frequency to fold
      --fdotmin FDOTMIN     Minimum fdot to fold
      --fdotmax FDOTMAX     Maximum fdot to fold
      -n NBIN, --nbin NBIN  Number of phase bins of the profile
      --segment-size SEGMENT_SIZE
                            Size of the event list segment to use (default None,
                            implying the whole observation)
      --step STEP           Step size of the frequency axis. Defaults to
                            1/oversample/observ.length.
      --oversample OVERSAMPLE
                            Oversampling factor - frequency resolution improvement
                            w.r.t. the standard FFT's 1/observ.length.
      --expocorr            Correct for the exposure of the profile bins. This
                            method is *much* slower, but it is useful for very
                            slow pulsars, where data gaps due to occultation or
                            SAA passages can significantly alter the exposure of
                            different profile bins.
      --find-candidates     Find pulsation candidates using thresholding
      --conflevel CONFLEVEL
                            percent confidence level for thresholding [0-100).
      --fit-candidates      Fit the candidate peaks in the periodogram
      --curve CURVE         Kind of curve to use (sinc or Gaussian)
      --fit-frequency FIT_FREQUENCY
                            Force the candidate frequency to FIT_FREQUENCY
      --debug               use DEBUG logging level
      --loglevel LOGLEVEL   use given logging level (one between INFO, WARNING,
                            ERROR, CRITICAL, DEBUG; default:WARNING)
      -N N                  The number of harmonics to use in the search (the 'N'
                            in Z^2_N; only relevant to Z search!)


