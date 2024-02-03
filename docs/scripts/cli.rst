Command line interface
======================

HEN2xspec
---------

::

    usage: HEN2xspec [-h] [--flx2xsp] [--loglevel LOGLEVEL] [--debug]
                     files [files ...]

    Save a frequency spectrum in a qdp file that can be read by flx2xsp and
    produce a XSpec-compatible spectrumfile

    positional arguments:
      files                List of files

    options:
      -h, --help           show this help message and exit
      --flx2xsp            Also call flx2xsp at the end
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              set DEBUG logging level


HENaccelsearch
--------------

::

    usage: HENaccelsearch [-h] [--outfile OUTFILE] [--emin EMIN] [--emax EMAX]
                          [--fmin FMIN] [--fmax FMAX] [--nproc NPROC]
                          [--zmax ZMAX] [--delta-z DELTA_Z] [--interbin]
                          [--pad-to-double] [--detrend DETREND]
                          [--deorbit-par DEORBIT_PAR] [--red-noise-filter]
                          [--loglevel LOGLEVEL] [--debug]
                          fname

    Run the accelerated search on pulsar data.

    positional arguments:
      fname                 Input file name

    options:
      -h, --help            show this help message and exit
      --outfile OUTFILE     Output file name
      --emin EMIN           Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX           Maximum energy (or PI if uncalibrated) to plot
      --fmin FMIN           Minimum frequency to search, in Hz
      --fmax FMAX           Maximum frequency to search, in Hz
      --nproc NPROC         Number of processors to use
      --zmax ZMAX           Maximum acceleration (in spectral bins)
      --delta-z DELTA_Z     Fdot step for search (1 is the default resolution)
      --interbin            Use interbinning
      --pad-to-double       Pad to the double of bins (sort-of interbinning)
      --detrend DETREND     Detrending timescale
      --deorbit-par DEORBIT_PAR
                            Parameter file in TEMPO2/PINT format
      --red-noise-filter    Correct FFT for red noise (use with caution)
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


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

    options:
      -h, --help            show this help message and exit
      -o OUT, --out OUT     Output file
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               use DEBUG logging level
      -p ASYMMETRY, --asymmetry ASYMMETRY
                            "asymmetry" parameter. Smaller values make the
                            baseline more "horizontal". Typically 0.001 < p <
                            0.1, but not necessarily.
      -l LAM, --lam LAM     lambda, or "smoothness", parameter. Larger values
                            make the baseline stiffer. Typically 1e2 < lam <
                            1e9


HENbinary
---------

::

    usage: HENbinary [-h] [-l MAX_LENGTH] [-b BINTIME]
                     [-e ENERGY_INTERVAL ENERGY_INTERVAL] [-p DEORBIT_PAR]
                     [--nproc NPROC] [--loglevel LOGLEVEL] [--debug]
                     files [files ...]

    Save light curves in a format readable to PRESTO

    positional arguments:
      files                 List of input light curves

    options:
      -h, --help            show this help message and exit
      -l MAX_LENGTH, --max-length MAX_LENGTH
                            Maximum length of light curves (split otherwise)
      -b BINTIME, --bintime BINTIME
                            Bin time
      -e ENERGY_INTERVAL ENERGY_INTERVAL, --energy-interval ENERGY_INTERVAL ENERGY_INTERVAL
                            Energy interval used for filtering
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      --nproc NPROC         Number of processors to use
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENcalibrate
------------

::

    usage: HENcalibrate [-h] [-r RMF] [--rough] [-o] [--nproc NPROC]
                        [--loglevel LOGLEVEL] [--debug]
                        files [files ...]

    Calibrate clean event files by associating the correct energy to each PI
    channel. Uses either a specified rmf file or (for NuSTAR only) an rmf file
    from the CALDB

    positional arguments:
      files                List of files

    options:
      -h, --help           show this help message and exit
      -r RMF, --rmf RMF    rmf file used for calibration. Not working with XMM
                           data
      --rough              Rough calibration, without rmf file (only for
                           NuSTAR, XMM, and NICER). Only for compatibility
                           purposes. This is done automatically by
                           HENreadevents
      -o, --overwrite      Overwrite; default: no
      --nproc NPROC        Number of processors to use
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              set DEBUG logging level


HENcolors
---------

::

    usage: HENcolors [-h] -e ENERGIES ENERGIES ENERGIES ENERGIES [-b BINTIME]
                     [--use-pi] [-o OUTFILE] [--loglevel LOGLEVEL] [--debug]
                     files [files ...]

    Calculate color light curves

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      -e ENERGIES ENERGIES ENERGIES ENERGIES, --energies ENERGIES ENERGIES ENERGIES ENERGIES
                            The energy boundaries in keV used to calculate the
                            color. E.g. -e 2 3 4 6 means that the color will
                            be calculated as 4.-6./2.-3. keV. If --use-pi is
                            specified, these are interpreted as PI channels
      -b BINTIME, --bintime BINTIME
                            Bin time
      --use-pi              Use the PI channel instead of energies
      -o OUTFILE, --outfile OUTFILE
                            Output file
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENcreategti
------------

::

    usage: HENcreategti [-h] [-f FILTER] [-c] [--overwrite] [-a APPLY_GTI]
                        [-l MINIMUM_LENGTH]
                        [--safe-interval SAFE_INTERVAL SAFE_INTERVAL]
                        [--loglevel LOGLEVEL] [--debug]
                        files [files ...]

    Create GTI files from a filter expression, or applies previously created
    GTIs to a file

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      -f FILTER, --filter FILTER
                            Filter expression, that has to be a valid Python
                            boolean operation on a data variable contained in
                            the files
      -c, --create-only     If specified, creates GTIs withouth applyingthem
                            to files (Default: False)
      --overwrite           Overwrite original file (Default: False)
      -a APPLY_GTI, --apply-gti APPLY_GTI
                            Apply a GTI from this file to input files
      -l MINIMUM_LENGTH, --minimum-length MINIMUM_LENGTH
                            Minimum length of GTIs (below this length, they
                            will be discarded)
      --safe-interval SAFE_INTERVAL SAFE_INTERVAL
                            Interval at start and stop of GTIs used for
                            filtering
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENdeorbit
----------

::

    usage: HENdeorbit [-h] [-p DEORBIT_PAR] [--loglevel LOGLEVEL] [--debug]
                      files [files ...]

    Deorbit the event arrival times

    positional arguments:
      files                 Input event file

    options:
      -h, --help            show this help message and exit
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENdumpdyn
----------

::

    usage: HENdumpdyn [-h] [--noplot] files [files ...]

    Dump dynamical (cross) power spectra. This script is being reimplemented.
    Please be patient :)

    positional arguments:
      files       List of files in any valid HENDRICS format for PDS or CPDS

    options:
      -h, --help  show this help message and exit
      --noplot    plot results


HENefsearch
-----------

::

    usage: HENefsearch [-h] -f FMIN -F FMAX [--emin EMIN] [--emax EMAX]
                       [--mean-fdot MEAN_FDOT] [--mean-fddot MEAN_FDDOT]
                       [--fdotmin FDOTMIN] [--fdotmax FDOTMAX]
                       [--dynstep DYNSTEP] [--npfact NPFACT]
                       [--n-transient-intervals N_TRANSIENT_INTERVALS]
                       [-n NBIN] [--segment-size SEGMENT_SIZE] [--step STEP]
                       [--oversample OVERSAMPLE] [--fast] [--ffa]
                       [--transient] [--expocorr] [--find-candidates]
                       [--conflevel CONFLEVEL] [--fit-candidates]
                       [--curve CURVE] [--fit-frequency FIT_FREQUENCY] [-N N]
                       [-p DEORBIT_PAR] [--loglevel LOGLEVEL] [--debug]
                       files [files ...]

    Search for pulsars using the epoch folding or the Z_n^2 algorithm

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      -f FMIN, --fmin FMIN  Minimum frequency to fold
      -F FMAX, --fmax FMAX  Maximum frequency to fold
      --emin EMIN           Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX           Maximum energy (or PI if uncalibrated) to plot
      --mean-fdot MEAN_FDOT
                            Mean fdot to fold (only useful when using --fast)
      --mean-fddot MEAN_FDDOT
                            Mean fddot to fold (only useful when using --fast)
      --fdotmin FDOTMIN     Minimum fdot to fold
      --fdotmax FDOTMAX     Maximum fdot to fold
      --dynstep DYNSTEP     Dynamical EF step
      --npfact NPFACT       Size of search parameter space
      --n-transient-intervals N_TRANSIENT_INTERVALS
                            Number of transient intervals to investigate
      -n NBIN, --nbin NBIN  Number of phase bins of the profile
      --segment-size SEGMENT_SIZE
                            Size of the event list segment to use (default
                            None, implying the whole observation)
      --step STEP           Step size of the frequency axis. Defaults to
                            1/oversample/observ.length.
      --oversample OVERSAMPLE
                            Oversampling factor - frequency resolution
                            improvement w.r.t. the standard FFT's
                            1/observ.length.
      --fast                Use a faster folding algorithm. It automatically
                            searches for the first spin derivative using an
                            optimized step.This option ignores expocorr,
                            fdotmin/max, segment-size, and step
      --ffa                 Use *the* Fast Folding Algorithm by Staelin+69. No
                            accelerated search allowed at the moment. Only
                            recommended to search for slow pulsars.
      --transient           Look for transient emission (produces an animated
                            GIF with the dynamic Z search)
      --expocorr            Correct for the exposure of the profile bins. This
                            method is *much* slower, but it is useful for very
                            slow pulsars, where data gaps due to occultation
                            or SAA passages can significantly alter the
                            exposure of different profile bins.
      --find-candidates     Find pulsation candidates using thresholding
      --conflevel CONFLEVEL
                            percent confidence level for thresholding [0-100).
      --fit-candidates      Fit the candidate peaks in the periodogram
      --curve CURVE         Kind of curve to use (sinc or Gaussian)
      --fit-frequency FIT_FREQUENCY
                            Force the candidate frequency to FIT_FREQUENCY
      -N N                  The number of harmonics to use in the search (the
                            'N' in Z^2_N; only relevant to Z search!)
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENexcvar
---------

::

    usage: HENexcvar [-h] [-c CHUNK_LENGTH] [--fraction-step FRACTION_STEP]
                     [--norm NORM] [--loglevel LOGLEVEL] [--debug]
                     files [files ...]

    Calculate excess variance in light curve chunks

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      -c CHUNK_LENGTH, --chunk-length CHUNK_LENGTH
                            Length in seconds of the light curve chunks
      --fraction-step FRACTION_STEP
                            If the step is not a full chunk_length but
                            less,this indicates the ratio between step step
                            and `chunk_length`
      --norm NORM           Choose between fvar, excvar and norm_excvar
                            normalization, referring to Fvar, excess variance,
                            and normalized excess variance respectively (see
                            Vaughan et al. 2003 for details).
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENexposure
-----------

::

    usage: HENexposure [-h] [-o OUTROOT] [--plot] [--loglevel LOGLEVEL]
                       [--debug]
                       lcfile uffile

    Create exposure light curve based on unfiltered event files.

    positional arguments:
      lcfile                Light curve file (HENDRICS format)
      uffile                Unfiltered event file (FITS)

    options:
      -h, --help            show this help message and exit
      -o OUTROOT, --outroot OUTROOT
                            Root of output file names
      --plot                Plot on window
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENfake
-------

::

    usage: HENfake [-h] [-e EVENT_LIST] [-l LC] [-c CTRATE] [-o OUTNAME]
                   [-i INSTRUMENT] [-m MISSION] [--tstart TSTART]
                   [--tstop TSTOP] [--mjdref MJDREF]
                   [--deadtime DEADTIME [DEADTIME ...]] [--loglevel LOGLEVEL]
                   [--debug]

    Create an event file in FITS format from an event list, or simulating it.
    If input event list is not specified, generates the events randomly

    options:
      -h, --help            show this help message and exit
      -e EVENT_LIST, --event-list EVENT_LIST
                            File containing event list
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
                            number, or two. In this last case, the second
                            value is used as sigma of the dead time
                            distribution
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENfiltevents
-------------

::

    usage: HENfiltevents [-h] [--emin EMIN] [--emax EMAX]
                         [--loglevel LOGLEVEL] [--debug] [--test]
                         files [files ...]

    Filter events

    positional arguments:
      files                Input event files

    options:
      -h, --help           show this help message and exit
      --emin EMIN          Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX          Maximum energy (or PI if uncalibrated) to plot
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              set DEBUG logging level
      --test               Only used for tests


HENfold
-------

::

    usage: HENfold [-h] [-f FREQ] [--fdot FDOT] [--fddot FDDOT] [--tref TREF]
                   [-n NBIN] [--nebin NEBIN] [--emin EMIN] [--emax EMAX]
                   [--out-file-root OUT_FILE_ROOT] [--pepoch PEPOCH]
                   [--norm NORM] [--colormap COLORMAP] [-p DEORBIT_PAR]
                   [--loglevel LOGLEVEL] [--debug] [--test]
                   file

    Plot a folded profile

    positional arguments:
      file                  Input event file

    options:
      -h, --help            show this help message and exit
      -f FREQ, --freq FREQ  Initial frequency to fold
      --fdot FDOT           Initial fdot
      --fddot FDDOT         Initial fddot
      --tref TREF           Reference time (same unit as time array)
      -n NBIN, --nbin NBIN  Number of phase bins (X axis) of the profile
      --nebin NEBIN         Number of energy bins (Y axis) of the profile
      --emin EMIN           Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX           Maximum energy (or PI if uncalibrated) to plot
      --out-file-root OUT_FILE_ROOT
                            Root of the output files (plots and csv tables)
      --pepoch PEPOCH       Reference epoch for timing parameters (MJD)
      --norm NORM           Normalization for the dynamical phase plot. Can
                            be: 'to1' (each profile normalized from 0 to 1);
                            'std' (subtract the mean and divide by the
                            standard deviation); 'sub' (just subtract the mean
                            of each profile); 'ratios' (divide by the average
                            profile, to highlight changes). Prepending
                            'median' to any of those options uses the median
                            in place of the mean. Appending '_smooth' smooths
                            the 2d array with a Gaussian filter. E.g.
                            mediansub_smooth subtracts the median and smooths
                            the imagedefault None
      --colormap COLORMAP   Change the color map of the image. Any matplotlib
                            colormap is valid
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level
      --test                Only used for tests


HENfspec
--------

::

    usage: HENfspec [-h] [-b BINTIME] [-r REBIN] [-f FFTLEN] [-k KIND]
                    [--norm NORM] [--noclobber] [-o OUTROOT] [--back BACK]
                    [--save-dyn] [--ignore-instr] [--ignore-gtis] [--save-all]
                    [--save-lcs] [--no-auxil] [--test] [--emin EMIN]
                    [--emax EMAX] [--loglevel LOGLEVEL] [--debug]
                    files [files ...]

    Create frequency spectra (PDS, CPDS, cospectrum) starting from well-
    defined input ligthcurves

    positional arguments:
      files                 List of light curve files

    options:
      -h, --help            show this help message and exit
      -b BINTIME, --bintime BINTIME
                            Light curve bin time; if negative, interpreted as
                            negative power of 2. Default: 2^-10, or keep input
                            lc bin time (whatever is larger)
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
      --back BACK           Estimated background (non-source) count rate
      --save-dyn            save dynamical power spectrum
      --ignore-instr        Ignore instrument names in channels
      --ignore-gtis         Ignore GTIs. USE AT YOUR OWN RISK
      --save-all            Save all information contained in spectra,
                            including light curves and dynamical spectra.
      --save-lcs            Save all information contained in spectra,
                            including light curves.
      --no-auxil            Do not save auxiliary spectra (e.g. pds1 and pds2
                            of cross spectrum)
      --test                Only to be used in testing
      --emin EMIN           Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX           Maximum energy (or PI if uncalibrated) to plot
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENjoinevents
-------------

::

    usage: HENjoinevents [-h] [-o OUTPUT] [--ignore-instr] files [files ...]

    Read a cleaned event files and saves the relevant information in a
    standard format

    positional arguments:
      files                 Files to join

    options:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            Name of output file
      --ignore-instr        Ignore instrument names in channels


HENlags
-------

::

    usage: HENlags [-h] [--loglevel LOGLEVEL] [--debug] files [files ...]

    Read timelags from cross spectrum results and save them to a qdp file

    positional arguments:
      files                List of files

    options:
      -h, --help           show this help message and exit
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              set DEBUG logging level


HENlcurve
---------

::

    usage: HENlcurve [-h] [-b BINTIME]
                     [--safe-interval SAFE_INTERVAL SAFE_INTERVAL]
                     [-e ENERGY_INTERVAL ENERGY_INTERVAL]
                     [--pi-interval PI_INTERVAL PI_INTERVAL] [-s] [-j] [-g]
                     [--minlen MINLEN] [--ignore-gtis] [-d OUTDIR]
                     [--noclobber] [--fits-input] [--txt-input]
                     [--weight-on WEIGHT_ON] [-p DEORBIT_PAR] [-o OUTFILE]
                     [--loglevel LOGLEVEL] [--debug] [--nproc NPROC]
                     files [files ...]

    Create lightcurves starting from event files. It is possible to specify
    energy or channel filtering options

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      -b BINTIME, --bintime BINTIME
                            Bin time; if negative, negative power of 2
      --safe-interval SAFE_INTERVAL SAFE_INTERVAL
                            Interval at start and stop of GTIs used for
                            filtering
      -e ENERGY_INTERVAL ENERGY_INTERVAL, --energy-interval ENERGY_INTERVAL ENERGY_INTERVAL
                            Energy interval used for filtering
      --pi-interval PI_INTERVAL PI_INTERVAL
                            PI interval used for filtering
      -s, --scrunch         Create scrunched light curve (single channel)
      -j, --join            Create joint light curve (multiple channels)
      -g, --gti-split       Split light curve by GTI
      --minlen MINLEN       Minimum length of acceptable GTIs (default:4)
      --ignore-gtis         Ignore GTIs
      -d OUTDIR, --outdir OUTDIR
                            Output directory
      --noclobber           Do not overwrite existing files
      --fits-input          Input files are light curves in FITS format
      --txt-input           Input files are light curves in txt format
      --weight-on WEIGHT_ON
                            Use a given attribute of the event list as weights
                            for the light curve
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      -o OUTFILE, --outfile OUTFILE
                            Output file
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level
      --nproc NPROC         Number of processors to use


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

    options:
      -h, --help            show this help message and exit
      -m MODELFILE, --modelfile MODELFILE
                            File containing an Astropy model with or without
                            constraints
      --fitmethod FITMETHOD
                            Any scipy-compatible fit method
      --frequency-interval FREQUENCY_INTERVAL [FREQUENCY_INTERVAL ...]
                            Select frequency interval(s) to fit. Must be an
                            even number of frequencies in Hz, like "--
                            frequency-interval 0 2" or "--frequency-interval 0
                            2 5 10", meaning that the spectrum will be fitted
                            between 0 and 2 Hz, or using the intervals 0-2 Hz
                            and 5-10 Hz.
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENphaseogram
-------------

::

    usage: HENphaseogram [-h] [-f FREQ] [--fdot FDOT] [--fddot FDDOT]
                         [--periodogram PERIODOGRAM] [-n NBIN]
                         [--ntimes NTIMES] [--binary]
                         [--binary-parameters BINARY_PARAMETERS BINARY_PARAMETERS BINARY_PARAMETERS]
                         [--emin EMIN] [--emax EMAX] [--plot-only] [--get-toa]
                         [--pepoch PEPOCH] [--norm NORM] [--colormap COLORMAP]
                         [-p DEORBIT_PAR] [--test] [--loglevel LOGLEVEL]
                         [--debug]
                         file

    Plot an interactive phaseogram

    positional arguments:
      file                  Input event file

    options:
      -h, --help            show this help message and exit
      -f FREQ, --freq FREQ  Initial frequency to fold
      --fdot FDOT           Initial fdot
      --fddot FDDOT         Initial fddot
      --periodogram PERIODOGRAM
                            Periodogram file
      -n NBIN, --nbin NBIN  Number of phase bins (X axis) of the profile
      --ntimes NTIMES       Number of time bins (Y axis) of the phaseogram
      --binary              Interact on binary parameters instead of frequency
                            derivatives
      --binary-parameters BINARY_PARAMETERS BINARY_PARAMETERS BINARY_PARAMETERS
                            Initial values for binary parameters
      --emin EMIN           Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX           Maximum energy (or PI if uncalibrated) to plot
      --plot-only           Only plot the phaseogram
      --get-toa             Only calculate TOAs
      --pepoch PEPOCH       Reference epoch for timing parameters (MJD)
      --norm NORM           Normalization for the dynamical phase plot. Can
                            be: 'to1' (each profile normalized from 0 to 1);
                            'std' (subtract the mean and divide by the
                            standard deviation); 'sub' (just subtract the mean
                            of each profile); 'ratios' (divide by the average
                            profile, to highlight changes). Prepending
                            'median' to any of those options uses the median
                            in place of the mean. Appending '_smooth' smooths
                            the 2d array with a Gaussian filter. E.g.
                            mediansub_smooth subtracts the median and smooths
                            the imagedefault None
      --colormap COLORMAP   Change the color map of the image. Any matplotlib
                            colormap is valid
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      --test                Only used for tests
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENphasetag
-----------

::

    usage: HENphasetag [-h] [--parfile PARFILE] [-f FREQS [FREQS ...]]
                       [-n NBIN] [--plot] [--tomax] [--test]
                       [--refTOA PULSE_REF_TIME] [--pepoch PEPOCH]
                       file

    positional arguments:
      file                  Event file

    options:
      -h, --help            show this help message and exit
      --parfile PARFILE     Parameter file
      -f FREQS [FREQS ...], --freqs FREQS [FREQS ...]
                            Frequency derivatives
      -n NBIN, --nbin NBIN  Nbin
      --plot                Plot profile
      --tomax               Refer phase to pulse max
      --test                Only for unit tests! Do not use
      --refTOA PULSE_REF_TIME
                            Reference TOA in MJD (overrides --tomax) for
                            reference pulse phase
      --pepoch PEPOCH       Reference time for timing solution


HENplot
-------

::

    usage: HENplot [-h] [--noplot] [--CCD] [--HID] [--figname FIGNAME]
                   [-o OUTFILE] [--xlog] [--ylog] [--xlin] [--ylin]
                   [--fromstart] [--axes AXES AXES]
                   files [files ...]

    Plot the content of HENDRICS light curves and frequency spectra

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      --noplot              Only create images, do not plot
      --CCD                 This is a color-color diagram. In this case, the
                            list of files is expected to be given as soft0.nc,
                            hard0.nc, soft1.nc, hard1.nc, ...
      --HID                 This is a hardness-intensity diagram. In this
                            case, the list of files is expected to be given as
                            color0.nc, intensity0.nc, color1.nc,
                            intensity1.nc, ...
      --figname FIGNAME     Figure name
      -o OUTFILE, --outfile OUTFILE
                            Output data file in QDP format
      --xlog                Use logarithmic X axis
      --ylog                Use logarithmic Y axis
      --xlin                Use linear X axis
      --ylin                Use linear Y axis
      --fromstart           Times are measured from the start of the
                            observation (only relevant for light curves)
      --axes AXES AXES      Plot two variables contained in the file


HENreadevents
-------------

::

    usage: HENreadevents [-h] [--noclobber] [-g] [--discard-calibration]
                         [-l LENGTH_SPLIT] [--min-length MIN_LENGTH]
                         [--gti-string GTI_STRING]
                         [--randomize-by RANDOMIZE_BY]
                         [--additional ADDITIONAL [ADDITIONAL ...]]
                         [-o OUTFILE] [--loglevel LOGLEVEL] [--debug]
                         [--nproc NPROC]
                         files [files ...]

    Read a cleaned event files and saves the relevant information in a
    standard format

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      --noclobber           Do not overwrite existing event files
      -g, --gti-split       Split event list by GTI
      --discard-calibration
                            Discard automatic calibration (if any)
      -l LENGTH_SPLIT, --length-split LENGTH_SPLIT
                            Split event list by length
      --min-length MIN_LENGTH
                            Minimum length of GTIs to consider
      --gti-string GTI_STRING
                            GTI string
      --randomize-by RANDOMIZE_BY
                            Randomize event arrival times by this amount (e.g.
                            it might be the 0.073-s frame time in XMM)
      --additional ADDITIONAL [ADDITIONAL ...]
                            Additional columns to be read from the FITS file
      -o OUTFILE, --outfile OUTFILE
                            Output file
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level
      --nproc NPROC         Number of processors to use


HENreadfile
-----------

::

    usage: HENreadfile [-h] [--print-header] files [files ...]

    Print the content of HENDRICS files

    positional arguments:
      files           List of files

    options:
      -h, --help      show this help message and exit
      --print-header  Print the full FITS header if present in the meta data.


HENrebin
--------

::

    usage: HENrebin [-h] [-r REBIN] [--loglevel LOGLEVEL] [--debug]
                    files [files ...]

    Rebin light curves and frequency spectra.

    positional arguments:
      files                 List of light curve files

    options:
      -h, --help            show this help message and exit
      -r REBIN, --rebin REBIN
                            Rebinning to apply. Only if the quantity to rebin
                            is a (C)PDS, it is possible to specify a non-
                            integer rebin factor, in which case it is
                            interpreted as a geometrical binning factor
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENscramble
-----------

::

    usage: HENscramble [-h] [--smooth-kind {smooth,flat,pulsed}]
                       [--deadtime DEADTIME] [--dt DT]
                       [--pulsed-fraction PULSED_FRACTION] [-f FREQUENCY]
                       [--outfile OUTFILE] [-p DEORBIT_PAR]
                       [-e ENERGY_INTERVAL ENERGY_INTERVAL]
                       [--loglevel LOGLEVEL] [--debug]
                       fname

    Scramble the events inside an event list, maintaining the same energies
    and GTIs

    positional arguments:
      fname                 File containing input event list

    options:
      -h, --help            show this help message and exit
      --smooth-kind {smooth,flat,pulsed}
                            Special testing value
      --deadtime DEADTIME   Dead time magnitude. Can be specified as a single
                            number, or two. In this last case, the second
                            value is used as sigma of the dead time
                            distribution
      --dt DT               Time resolution of smoothed light curve
      --pulsed-fraction PULSED_FRACTION
                            Pulsed fraction of simulated pulsations
      -f FREQUENCY, --frequency FREQUENCY
                            Pulsed fraction of simulated pulsations
      --outfile OUTFILE     Output file name
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      -e ENERGY_INTERVAL ENERGY_INTERVAL, --energy-interval ENERGY_INTERVAL ENERGY_INTERVAL
                            Energy interval used for filtering
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENscrunchlc
------------

::

    usage: HENscrunchlc [-h] [-o OUT] [--loglevel LOGLEVEL] [--debug]
                        files [files ...]

    Sum lightcurves from different instruments or energy ranges

    positional arguments:
      files                List of files

    options:
      -h, --help           show this help message and exit
      -o OUT, --out OUT    Output file
      --loglevel LOGLEVEL  use given logging level (one between INFO, WARNING,
                           ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug              use DEBUG logging level


HENsplitevents
--------------

::

    usage: HENsplitevents [-h] [-l LENGTH_SPLIT] [--overlap OVERLAP]
                          [--split-at-mjd SPLIT_AT_MJD]
                          fname

    Reads a cleaned event files and splits the file into overlapping multiple
    chunks of fixed length

    positional arguments:
      fname                 File 1

    options:
      -h, --help            show this help message and exit
      -l LENGTH_SPLIT, --length-split LENGTH_SPLIT
                            Split event list by GTI
      --overlap OVERLAP     Overlap factor. 0 for no overlap, 0.5 for half-
                            interval overlap, and so on.
      --split-at-mjd SPLIT_AT_MJD
                            Split at this MJD


HENsumfspec
-----------

::

    usage: HENsumfspec [-h] [-o OUTNAME] files [files ...]

    Sum (C)PDSs contained in different files

    positional arguments:
      files                 List of light curve files

    options:
      -h, --help            show this help message and exit
      -o OUTNAME, --outname OUTNAME
                            Output file name for summed (C)PDS. Default:
                            tot_(c)pds.p


HENvarenergy
------------

::

    usage: HENvarenergy [-h] [-f FREQ_INTERVAL FREQ_INTERVAL]
                        [--energy-values ENERGY_VALUES ENERGY_VALUES ENERGY_VALUES ENERGY_VALUES]
                        [--segment-size SEGMENT_SIZE]
                        [--ref-band REF_BAND REF_BAND] [--rms] [--covariance]
                        [--use-pi] [--cross-instr] [--lag] [--count]
                        [--label LABEL] [--norm NORM] [--format FORMAT]
                        [-b BINTIME] [--loglevel LOGLEVEL] [--debug]
                        files [files ...]

    Calculates variability-energy spectra

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      -f FREQ_INTERVAL FREQ_INTERVAL, --freq-interval FREQ_INTERVAL FREQ_INTERVAL
                            Frequence interval
      --energy-values ENERGY_VALUES ENERGY_VALUES ENERGY_VALUES ENERGY_VALUES
                            Choose Emin, Emax, number of intervals,interval
                            spacing, lin or log
      --segment-size SEGMENT_SIZE
                            Length of the light curve intervals to be averaged
      --ref-band REF_BAND REF_BAND
                            Reference band when relevant
      --rms                 Calculate rms
      --covariance          Calculate covariance spectrum
      --use-pi              Energy intervals are specified as PI channels
      --cross-instr         Use data files in pairs, for example with
                            thereference band from one and the subbands from
                            the other (useful in NuSTAR and multiple-detector
                            missions)
      --lag                 Calculate lag-energy
      --count               Calculate lag-energy
      --label LABEL         Additional label to be added to file names
      --norm NORM           When relevant, the normalization of the spectrum.
                            One of ['abs', 'frac', 'rms', 'leahy', 'none']
      --format FORMAT       Output format for the table. Can be ECSV, QDP, or
                            any other format accepted by astropy
      -b BINTIME, --bintime BINTIME
                            Bin time
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENz2vspf
---------

::

    usage: HENz2vspf [-h] [--ntrial NTRIAL] [--outfile OUTFILE]
                     [--show-z-values SHOW_Z_VALUES [SHOW_Z_VALUES ...]]
                     [--emin EMIN] [--emax EMAX] [-N N] [--loglevel LOGLEVEL]
                     [--debug]
                     fname

    Get Z2 vs pulsed fraction for a given observation. Takes the original
    event list, scrambles the event arrival time, adds a pulsation with random
    pulsed fraction, and takes the maximum value of Z2 in a small interval
    around the pulsation. Does this ntrial times, and plots.

    positional arguments:
      fname                 Input file name

    options:
      -h, --help            show this help message and exit
      --ntrial NTRIAL       Number of trial values for the pulsed fraction
      --outfile OUTFILE     Output table file name
      --show-z-values SHOW_Z_VALUES [SHOW_Z_VALUES ...]
                            Show these Z values in the plot
      --emin EMIN           Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX           Maximum energy (or PI if uncalibrated) to plot
      -N N                  The N in Z^2_N
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


HENzsearch
----------

::

    usage: HENzsearch [-h] -f FMIN -F FMAX [--emin EMIN] [--emax EMAX]
                      [--mean-fdot MEAN_FDOT] [--mean-fddot MEAN_FDDOT]
                      [--fdotmin FDOTMIN] [--fdotmax FDOTMAX]
                      [--dynstep DYNSTEP] [--npfact NPFACT]
                      [--n-transient-intervals N_TRANSIENT_INTERVALS]
                      [-n NBIN] [--segment-size SEGMENT_SIZE] [--step STEP]
                      [--oversample OVERSAMPLE] [--fast] [--ffa] [--transient]
                      [--expocorr] [--find-candidates] [--conflevel CONFLEVEL]
                      [--fit-candidates] [--curve CURVE]
                      [--fit-frequency FIT_FREQUENCY] [-N N] [-p DEORBIT_PAR]
                      [--loglevel LOGLEVEL] [--debug]
                      files [files ...]

    Search for pulsars using the epoch folding or the Z_n^2 algorithm

    positional arguments:
      files                 List of files

    options:
      -h, --help            show this help message and exit
      -f FMIN, --fmin FMIN  Minimum frequency to fold
      -F FMAX, --fmax FMAX  Maximum frequency to fold
      --emin EMIN           Minimum energy (or PI if uncalibrated) to plot
      --emax EMAX           Maximum energy (or PI if uncalibrated) to plot
      --mean-fdot MEAN_FDOT
                            Mean fdot to fold (only useful when using --fast)
      --mean-fddot MEAN_FDDOT
                            Mean fddot to fold (only useful when using --fast)
      --fdotmin FDOTMIN     Minimum fdot to fold
      --fdotmax FDOTMAX     Maximum fdot to fold
      --dynstep DYNSTEP     Dynamical EF step
      --npfact NPFACT       Size of search parameter space
      --n-transient-intervals N_TRANSIENT_INTERVALS
                            Number of transient intervals to investigate
      -n NBIN, --nbin NBIN  Number of phase bins of the profile
      --segment-size SEGMENT_SIZE
                            Size of the event list segment to use (default
                            None, implying the whole observation)
      --step STEP           Step size of the frequency axis. Defaults to
                            1/oversample/observ.length.
      --oversample OVERSAMPLE
                            Oversampling factor - frequency resolution
                            improvement w.r.t. the standard FFT's
                            1/observ.length.
      --fast                Use a faster folding algorithm. It automatically
                            searches for the first spin derivative using an
                            optimized step.This option ignores expocorr,
                            fdotmin/max, segment-size, and step
      --ffa                 Use *the* Fast Folding Algorithm by Staelin+69. No
                            accelerated search allowed at the moment. Only
                            recommended to search for slow pulsars.
      --transient           Look for transient emission (produces an animated
                            GIF with the dynamic Z search)
      --expocorr            Correct for the exposure of the profile bins. This
                            method is *much* slower, but it is useful for very
                            slow pulsars, where data gaps due to occultation
                            or SAA passages can significantly alter the
                            exposure of different profile bins.
      --find-candidates     Find pulsation candidates using thresholding
      --conflevel CONFLEVEL
                            percent confidence level for thresholding [0-100).
      --fit-candidates      Fit the candidate peaks in the periodogram
      --curve CURVE         Kind of curve to use (sinc or Gaussian)
      --fit-frequency FIT_FREQUENCY
                            Force the candidate frequency to FIT_FREQUENCY
      -N N                  The number of harmonics to use in the search (the
                            'N' in Z^2_N; only relevant to Z search!)
      -p DEORBIT_PAR, --deorbit-par DEORBIT_PAR
                            Deorbit data with this parameter file (requires
                            PINT installed)
      --loglevel LOGLEVEL   use given logging level (one between INFO,
                            WARNING, ERROR, CRITICAL, DEBUG; default:WARNING)
      --debug               set DEBUG logging level


