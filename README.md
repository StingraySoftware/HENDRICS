# MaLTPyNT - Matteo's Libraries and Tools in Python for NuSTAR Timing.
** BEWARE! STILL UNDER TESTING. USE WITH CARE IN PRODUCTION!**

This software is mostly focused on doing correctly and fairly easily a **quick-look timing analysis** of NuSTAR data, treating properly orbital gaps and exploiting the presence of two independent detectors by using the **cospectrum** as a proxy for the power density spectrum (for an explanation of why this is important, look at Bachetti et al., _ApJ_, in press - [arXiv:1409.3248](http://arxiv.org/abs/1409.3248)). The output of the analysis is a cospectrum, or a power density spectrum, that can be fitted with [Xspec](http://heasarc.gsfc.nasa.gov/xanadu/xspec/) or [Isis](http://space.mit.edu/home/mnowak/isis_vs_xspec/mod.html). Also, one can calculate in the same easy way **time lags** (still under testing, help is welcome).
Despite its main focus on NuSTAR, the software can be used to make standard spectral analysis on X-ray data from, in principle, any other satellite (for sure XMM-Newton and RXTE). 

## MaLTPyNT vs FTOOLS (and together with FTOOLS)

### vs POWSPEC
MaLTPyNT does a better job than POWSPEC from several points of view:

- **Good time intervals** (GTIs) are completely avoided in the computation. No gaps dirtying up the power spectrum! (This is particularly important for NuSTAR, as orbital gaps are always present in typical observation timescales)

- The number of bins used in the power spectrum (or the cospectrum) need not be a power of two! No padding needed.

### Clarification about dead time treatment
MaLTPyNT **does not supersede [nulccorr](https://heasarc.gsfc.nasa.gov/ftools/caldb/help/nulccorr.html)**. If one is only interested in frequencies below ~0.5 Hz, nulccorr treats robustly various dead time components and its use is recommended. Light curves produced by nulccorr can be converted to MaLTPyNT format using `MPlcurve --fits-input <lcname>.fits`, and used for the subsequent steps of the timing analysis. 

## License and notes for the users
This software is released with a 3-clause BSD license. You can find license information in the `LICENSE.rst` file.

**If you use this software in a publication**, it would be great if you wrote something along these lines in the acknowledgements: "This work made use of the MaLTPyNT software for timing analysis". In particular **if you use the cospectrum**, please refer to Bachetti et al. 2015, _ApJ_, in press ([arXiv:1409.3248](http://arxiv.org/abs/1409.3248)).

I listed a number of **open issues** in the [Issues](https://bitbucket.org/mbachett/maltpynt/issues?status=new&status=open) page. Feel free to **comment** on them and **propose more**. Please choose carefully the category: bugs, enhancements, etc.

```
## Acknowledgements
First of all, I would like to thank all the co-authors of [the NuSTAR timing paper](http://arxiv.org/abs/1409.3248) and the NuSTAR X-ray binaries working group. This software would not exist without the interesting discussions before and around that paper.
In particular, I would like to thank Ivan Zolotukhin, Francesca Fornasini, Erin Kara, Poshak Gandhi, John Tomsick and Abdu Zoghbi for helping testing the code and giving various suggestions on how to improve it.
Last but not least, I would like to thank Marco Buttu (by the way, [check out his book if you speak Italian](http://www.amazon.it/Programmare-con-Python-completa-DigitalLifeStyle-ebook/dp/B00L95VURC/ref=sr_1_1?s=books&ie=UTF8&qid=1424298092&sr=1-1)) for his priceless pointers on Python coding and code management techniques.

## Installation
You'll need a recent python 2.6+ or 3.3+ installation, and the [Numpy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/), [Scipy](http://scipy.org/) and [Astropy](http://www.astropy.org/) libraries. You should also have a working [HEASoft](http://heasarc.nasa.gov/lheasoft/) installation to produce the cleaned event files and to use [XSpec](http://heasarc.nasa.gov/lheasoft/xanadu/xspec/index.html). 

An **optional but recommended** dependency is the [netCDF 4 library](http://www.unidata.ucar.edu/software/netcdf/) with its [python bindings](https://github.com/Unidata/netcdf4-python).

To install, download the distribution directory:
```
#!console
$ git clone git@bitbucket.org:mbachett/maltpynt.git
```
To use this command you will probably need to setup an SSH key for your account (in Manage Account, recommended!). Otherwise, you can use the command
```
#!console
$ git clone https://<yourusername>@bitbucket.org/mbachett/maltpynt.git
```
To update the software, just run
```
#!console
$ git pull
```
from the source directory (usually, the command gives troubleshooting information if this doesn't work the first time).

Enter  the distribution directory and run
```
#!console
$ python setup.py install
```
this will check for the existing dependencies and install the files in a proper way.
From that point on, executables will be somewhere in your PATH and python libraries will be available in python scripts with the usual
```
#!python

import maltpynt

## Tutorial
This is the same tutorial you can find in the Wiki, for convenience
### 0. Preliminary info
This tutorial assumes that you have previous knowledge of timing techniques, so that I don't repeat concepts as Nyquist frequency, the importance of choosing carefully the binning time and the FFT length, and so on. If you are not familiar with these concepts, [this paper by Michiel is a very good place to start](http://dare.uva.nl/document/2/47104). Why in the example below I use the cospectrum instead of the PDS, is written in our [timing paper](http://arxiv.org/abs/1409.3248).

This software has a modular structure. One starts from cleaned event files (such as those produced by tools like `nupipeline` and possibly barycentered with `barycorr` or equivalent), and produces a series of products with subsequent steps:

1. **event lists** containing event arrival times and PI channel information

2. (optional) **calibrated event lists**, where PI values have been converted to energy

3. **light curves**, choosing the energy band and the bin time

4. (optional) **summed light curves** if we want to join events from multiple instruments, or just from different observing times

5. **power spectrum** and/or **cross spectrum** (hereafter the ``frequency spectra'')

6. **rebinning** of frequency spectra 

7. finally, **lags** and **cospectrum**

8. (optional) frequency spectra in XSpec format

Most of these tools have help information that can be accessed by typing the name of the command plus -h or --help:
```
#!console

$ MPcalibrate -h
usage: MPcalibrate [-h] [-r RMF] [-o] files [files ...]

Calibrates clean event files by associating the correct energy to each PI
channel. Uses either a specified rmf file or (for NuSTAR only) an rmf file
from the CALDB

positional arguments:
  files              List of files

optional arguments:
  -h, --help         show this help message and exit
  -r RMF, --rmf RMF  rmf file used for calibration
  -o, --overwrite    Overwrite; default: no
```
For I/O, MaLTPyNT looks if the `netCDF4` library is installed. If it's found in the system, files will be saved in this format. Otherwise, the native Python `pickle` format format will be used. This format is _much_ slower (It might take some minutes to load some files) and files will be bigger, but this possibility ensures portability. If you don't use netCDF4, you'll notice that file names will have the `.p` extension instead of the `.nc` below. The rest is the same.

### 1. Loading event lists
Starting from cleaned event files, we will first save them in `MaLTPyNT` format (a pickle file basically). For example, I'm starting from two event lists called `002A.evt` and `002B.evt`, containing the cleaned event lists from a source observed with NuSTAR's `FPMA` and `FPMB` respectively.
```
#!console

$ MPreadevents 002A.evt 002B.evt
Opening 002A.evt
Saving events and info to 002A_ev.nc
Opening 002B.evt
Saving events and info to 002B_ev.nc
```
This will create new files with a `_ev.nc` extension (`_ev.p` if you don't use netCDF4), containing the event times and the energy _channel_ (`PI`) of each event

### 2. Calibrating event lists
Use `MPcalibrate`. You can either specify an `rmf` file with the `-r` option, or just let it look for it in the NuSTAR `CALDB` (the environment variable has to be defined!)
```
#!console

$ MPcalibrate 002A_ev.nc 002B_ev.nc
Loading file 002A_ev.nc...
Done.
###############ATTENTION!!#####################

Rmf not specified. Using default NuSTAR rmf.

###############################################
Saving calibrated data to 002A_ev_calib.nc
Loading file 002B_ev.nc...
Done.
###############ATTENTION!!#####################

Rmf not specified. Using default NuSTAR rmf.

###############################################
Saving calibrated data to 002B_ev_calib.nc
```
This will create two new files with `_ev_calib.nc` suffix that will contain energy information. Optionally, you can overwrite the original event lists.
### 3. Producing light curves
Choose carefully the binning time (option `-b`). Since what we are interested in is a power spectrum, this binning time will limit our maximum frequency in the power spectrum. We are here specifying 2^-8 =0.00390625 for binning time (how to use the `-b` option is of course documented. Use `-h` FMI).
Since we have calibrated the event files, we can also choose an event energy range, here between 3 and 30 keV.
Another thing that is useful in NuSTAR data is taking some time intervals out from the start and the end of each GTI. This is mostly to eliminate an increase of background level that often appears at GTI borders and produces very nasty power spectral shapes. Here I filter 100 s from the start and 300 s from the end of each GTI.
```
#!console

$ MPlcurve 002A_ev_calib.nc 002B_ev_calib.p -b -6 -e 3 30 --safe_interval 100 300
Loading file 002A_ev_calib.nc...
Done.
Saving light curve to 002A_E3-30_lc.nc
Loading file 002B_ev_calib.nc...
Done.
Saving light curve to 002B_E3-30_lc.nc
```
To check the light curve that was produced, use the `MPplot` program:
```
#!console

$ MPplot 002A_E3-30_lc.nc
```
### 4. Joining, summing and ``scrunching'' light curves
If we want a single light curve from multiple ones, either summing multiple instruments or multiple energy or time ranges, we can use `mp_scrunch_lc`:
```
#!console

$ MPscrunchlc 002A_E3-30_lc.nc 002B_E3-30_lc.nc -o 002scrunch_3-30_lc.nc
Loading file 002A_E3-30_lc.nc...
Done.
Loading file 002B_E3-30_lc.nc...
Done.
Saving joined light curve to out_lc.nc
Saving scrunched light curve to 002scrunch_3-30_lc.nc
```
This is only tested in ``safe'' situations (files are not too big and have consistent time and energy ranges), so it might give inconsistent results or crash in untested situations. Please report any problems!

### 5. Producing power spectra and cross power spectra
Let us just produce the cross power spectrum for now. To produce also the power spectra corresponding to each light curve, substitute `"CPDS"` with `"PDS,CPDS"`. I use rms normalization here, default would be Leahy normalization.
```
#!console

$ MPfspec 002A_E3-30_lc.nc 002B_E3-30_lc.nc -k CPDS -o cpds_002_3-30 --norm rms
Beware! For cpds and derivatives, I assume that the files are
ordered as follows: obs1_FPMA, obs1_FPMB, obs2_FPMA, obs2_FPMB...
Loading file 002A_E3-30_lc.nc...
Loading file 002B_E3-30_lc.nc...
Saving CPDS to ./cpds_002_3-30_0.nc
```

### 6. Rebinning the spectrum
Now let's rebin the spectrum. If the rebin factor is an integer, it is interpreted as a constant rebinning. Otherwise (only if >1), it is interpreted as a geometric binning.
```
#!console

$ MPrebin cpds_002_3-30_0.nc -r 1.03
Saving cpds to cpds_002_3-30_0_rebin1.03.nc
```

### 7. Calculating the cospectrum and phase/time lags
The calculation of lags and their errors is implemented in `MPlags`, and needs to be tested properly. 
For the cospectrum, it is sufficient to read the real part of the cross power spectrum as depicted in the relevant function in `mp_plot.py` ([Use the source, Luke!](http://adastraerrans.com/archivos/use-the-source-luke.png)).

### 8. Saving the spectra in a format readable to XSpec
To save the cospectrum in a format readable to XSpec it is sufficient to give the commands
```
#!console

$ MP2xspec cpds_002_3-30_0_rebin1.03.nc
Saving to cpds_002_3-30_0_rebin1.03_xsp.dat
$ flx2xsp cpds_002_3-30_0_rebin1.03_xsp.dat cpds.pha cpds.rsp
```

### 9. Open and fit in XSpec!
```
#!console
$ xspec
XSPEC> data cpds.pha
XSPEC> cpd /xw; setp ener; setp comm log y
XSPEC> mo lore + lore + lore
(...)
XSPEC> fit
XSPEC> pl eufspe delchi  
```
etc.
![screenshot.png](https://bitbucket.org/repo/XA95dR/images/3911632225-screenshot.png)

(NOTE: [I know, Mike, it's unfolded... but for a flat response it shouldn't matter, right?](http://space.mit.edu/home/mnowak/isis_vs_xspec/plots.html)  ;) )