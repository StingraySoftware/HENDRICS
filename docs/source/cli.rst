
MP2xspec
********

Save a frequency spectrum in a qdp file that can be read by flx2xsp
and produce a XSpec-compatible spectrum file

usage: MP2xspec [-h] [--loglevel LOGLEVEL] [--debug] [--flx2xsp]
files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--debug``

   use DEBUG logging level

``--flx2xsp``

   Also call flx2xsp at the end


MPcalibrate
***********

Calibrate clean event files by associating the correct energy to each
PI channel. Uses either a specified rmf file or (for NuSTAR only) an
rmf file from the CALDB

usage: MPcalibrate [-h] [-r RMF] [-o] [--loglevel LOGLEVEL] [--debug]
[--nproc NPROC]                    files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit

``-r <rmf>, --rmf <rmf>``

   rmf file used for calibration

``-o, --overwrite``

   Overwrite; default: no

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--debug``

   use DEBUG logging level

``--nproc <nproc>``

   Number of processors to use


MPcreategti
***********

Create GTI files from a filter expression, or applies previously
created GTIs to a file

usage: MPcreategti [-h] [-f FILTER] [-c] [-o] [-a APPLY_GTI]
[--safe-interval SAFE_INTERVAL SAFE_INTERVAL]
[--loglevel LOGLEVEL] [--debug]                    files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit

``-f <filter>, --filter <filter>``

   Filter expression, that has to be a valid Python boolean operation
   on a data variable contained in the files

``-c, --create-only``

   If specified, creates GTIs withouth applyingthem to files (Default:
   False)

``-o, --overwrite``

   Overwrite original file (Default: False)

``-a <apply_gti>, --apply-gti <apply_gti>``

   Apply a GTI from this file to input files

``--safe-interval <safe_interval>``

   Interval at start and stop of GTIs used for filtering

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--debug``

   use DEBUG logging level


MPdumpdyn
*********

Dump dynamical (cross) power spectra

usage: MPdumpdyn [-h] [--noplot] files [files ...]

``files``

   List of (c)PDS files

``-h, --help``

   show this help message and exit

``--noplot``

   plot results


MPfspec
*******

Create frequency spectra (PDS, CPDS, cospectrum) starting from
well-defined input ligthcurves

usage: MPfspec [-h] [-b BINTIME] [-r REBIN] [-f FFTLEN] [-k KIND]
[--norm NORM] [--noclobber] [-o OUTROOT] [--loglevel LOGLEVEL]
[--nproc NPROC] [--back BACK] [--debug] [--save-dyn]
files [files ...]

``files``

   List of light curve files

``-h, --help``

   show this help message and exit

``-b <bintime>, --bintime <bintime>``

   Light curve bin time; if negative, interpreted as negative power of
   2. Default: 2^-10, or keep input lc bin time (whatever is larger)

``-r <rebin>, --rebin <rebin>``

   (C)PDS rebinning to apply. Default: none

``-f <fftlen>, --fftlen <fftlen>``

   Length of FFTs. Default: 512 s

``-k <kind>, --kind <kind>``

   Spectra to calculate, as comma-separated list (Accepted: PDS and
   CPDS; Default: "PDS,CPDS")

``--norm <norm>``

   Normalization to use (Accepted: Leahy and rms; Default: "Leahy")

``--noclobber``

   Do not overwrite existing files

``-o <outroot>, --outroot <outroot>``

   Root of output file names for CPDS only

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--nproc <nproc>``

   Number of processors to use

``--back <back>``

   Estimated background (non-source) count rate

``--debug``

   use DEBUG logging level

``--save-dyn``

   save dynamical power spectrum


MPlags
******

Calculate time lags from the cross power spectrum and the power
spectra of the two channels

usage: MPlags [-h] [-o OUTROOT] [--loglevel LOGLEVEL] [--noclobber]
[--debug]               files [files ...]

``files``

   Three files: the cross spectrum and the two power spectra

``-h, --help``

   show this help message and exit

``-o <outroot>, --outroot <outroot>``

   Root of output file names

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG;default:WARNING)

``--noclobber``

   Do not overwrite existing files

``--debug``

   use DEBUG logging level


MPlcurve
********

Create lightcurves starting from event files. It is possible to
specify energy or channel filtering options

usage: MPlcurve [-h] [-b BINTIME]                 [--safe-interval
SAFE_INTERVAL SAFE_INTERVAL]                 [--pi-interval
PI_INTERVAL PI_INTERVAL]                 [-e E_INTERVAL E_INTERVAL]
[-s] [-j] [-g] [--minlen MINLEN]                 [--ignore-gtis] [-d
OUTDIR] [--loglevel LOGLEVEL]                 [--nproc NPROC]
[--debug] [--noclobber] [--fits-input]                 [--txt-input]
files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit

``-b <bintime>, --bintime <bintime>``

   Bin time; if negative, negative power of 2

``--safe-interval <safe_interval>``

   Interval at start and stop of GTIs used for filtering

``--pi-interval <pi_interval>``

   PI interval used for filtering

``-e <e_interval>, --e-interval <e_interval>``

   Energy interval used for filtering

``-s, --scrunch``

   Create scrunched light curve (single channel)

``-j, --join``

   Create joint light curve (multiple channels)

``-g, --gti-split``

   Split light curve by GTI

``--minlen <minlen>``

   Minimum length of acceptable GTIs (default:4)

``--ignore-gtis``

   Ignore GTIs

``-d <outdir>, --outdir <outdir>``

   Output directory

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--nproc <nproc>``

   Number of processors to use

``--debug``

   use DEBUG logging level

``--noclobber``

   Do not overwrite existing files

``--fits-input``

   Input files are light curves in FITS format

``--txt-input``

   Input files are light curves in txt format


MPplot
******

Plot the content of MaLTPyNT light curves and frequency spectra

usage: MPplot [-h] files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit


MPreadevents
************

Read a cleaned event files and saves the relevant information in a
standard format

usage: MPreadevents [-h] [--loglevel LOGLEVEL] [--nproc NPROC]
[--noclobber]                     [-g] [--debug]
files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--nproc <nproc>``

   Number of processors to use

``--noclobber``

   Do not overwrite existing event files

``-g, --gti-split``

   Split event list by GTI

``--debug``

   use DEBUG logging level


MPreadfile
**********

Print the content of MaLTPyNT files

usage: MPreadfile [-h] files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit


MPrebin
*******

Rebin light curves and frequency spectra.

usage: MPrebin [-h] [-r REBIN] [--loglevel LOGLEVEL] [--debug]
files [files ...]

``files``

   List of light curve files

``-h, --help``

   show this help message and exit

``-r <rebin>, --rebin <rebin>``

   Rebinning to apply. Only if the quantity to rebin is a (C)PDS, it
   is possible to specify a non-integer rebin factor, in which case it
   is interpreted as a geometrical binning factor

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--debug``

   use DEBUG logging level


MPscrunchlc
***********

Sum lightcurves from different instruments or energy ranges

usage: MPscrunchlc [-h] [-o OUT] [--loglevel LOGLEVEL] [--debug]
files [files ...]

``files``

   List of files

``-h, --help``

   show this help message and exit

``-o <out>, --out <out>``

   Output file

``--loglevel <loglevel>``

   use given logging level (one between INFO, WARNING, ERROR,
   CRITICAL, DEBUG; default:WARNING)

``--debug``

   use DEBUG logging level


MPsumfspec
**********

Sum (C)PDSs contained in different files

usage: MPsumfspec [-h] [-o OUTNAME] files [files ...]

``files``

   List of light curve files

``-h, --help``

   show this help message and exit

``-o <outname>, --outname <outname>``

   Output file name for summed (C)PDS. Default: tot_(c)pds.nc
