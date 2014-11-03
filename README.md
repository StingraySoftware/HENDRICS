# MaLTPyNT
** BEWARE! STILL UNDER TESTING. DO NOT USE IN PRODUCTION!**

These tools are a stripped-down version of a huge and messy library of timing codes I've developed in the years. They contain what is needed for a quick look at the timing properties of an X-ray source.
This software is **heavily focused on NuSTAR**, and it's mostly for educational purposes. Everything can be generalized to any point, but here I tried to focus on what can be useful to people making their first steps on NuSTAR timing data. I also decided to use very few external libraries in order to make the installation easier, at the cost of losing some (well, a lot of) performance, in particular in I/O.

## Note for the users
I listed a number of **open issues** in the [Issues](https://bitbucket.org/mbachett/maltpynt/issues?status=new&status=open) page. Feel free to **comment** on them and **propose more**. Please choose carefully the category: bugs, enhancements, etc.

## Installation
You'll need a recent python 2.7 installation, and the [Numpy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/)  and [Astropy](http://www.astropy.org/) libraries. You should also have a working [HEASoft](http://heasarc.nasa.gov/lheasoft/) installation to produce the cleaned event files and to use [XSpec](http://heasarc.nasa.gov/lheasoft/xanadu/xspec/index.html).
Put the python codes in the directory where you are analyzing the data. That's all. Then, you can call them with a python interpreter, e.g.
```
$ python mp_read_events.py filename.evt
```
I will make these codes a python package that can be installed properly, eventually.

## Tutorial
### 0. Preliminary info
This software has a modular structure. One starts from cleaned event files (such as those produced by nupipeline), and produces a series of products with subsequent steps:

1. **event lists** containing event arrival times and PI channel information

2. (optional) **calibrated event lists**, where PI values have been converted to energy

3. **light curves**, choosing the energy band and the bin time

4. (optional) **summed light curves** if we want to join events from multiple instruments, or just from different observing times

5. **power spectrum** and/or **cross spectrum** (hereafter the ``frequency spectra'')

6. **rebinning** of frequency spectra 

7. finally, **lags** (still not implemented... ehm...) and **cospectrum**

8. (optional) frequency spectra in XSpec format

Most of these tools have help information that can be accessed by typing the name of the command plus -h or --help:
```
#!console

$ python mp_calibrate.py 002*_ev.p --help
usage: mp_calibrate.py [-h] [-r RMF] [-o] files [files ...]

positional arguments:
  files              List of files

optional arguments:
  -h, --help         show this help message and exit
  -r RMF, --rmf RMF  rmf file used for calibration
  -o, --overwrite    Overwrite; default: no
```
For I/O, I mostly use native Python `pickle` format. This is _very_ slow (It might take some minutes to load some files). I will eventually use some different file format like NetCDF as I do in my other codes. In this particular case, I valued portability over rapidity.

### 1. Loading event lists
Starting from cleaned event files, we will first save them in `MaLTPyNT` format (a pickle file basically). For example, I'm starting from two event lists called `002A.evt` and `002B.evt`, containing the cleaned event lists from a source observed with NuSTAR's `FPMA` and `FPMB` respectively.
```
#!console

$ python mp_read_events.py 002A.evt 002B.evt
Opening 002A.evt
Saving events and info to 002A_ev.p
Opening 002B.evt
Saving events and info to 002B_ev.p
```
This will create new files with a `_ev.p` extension, containing the event times and the energy _channel_ (`PI`) of each event

### 2. Calibrating event lists
Use `mp_calibrate`. You can either specify an `rmf` file with the `-r` option, or just let it look for it in the NuSTAR `CALDB` (the environment variable has to be defined!)
```
#!console

$ python mp_calibrate.py 002A_ev.p 002B_ev.p
Loading file 002A_ev.p...
Done.
###############ATTENTION!!#####################

Rmf not specified. Using default NuSTAR rmf.

###############################################
Saving calibrated data to 002A_ev_calib.p
Loading file 002B_ev.p...
Done.
###############ATTENTION!!#####################

Rmf not specified. Using default NuSTAR rmf.

###############################################
Saving calibrated data to 002B_ev_calib.p
```
This will create two new files with `_ev_calib.p` suffix that will contain energy information. Optionally, you can overwrite the original event lists.
### 3. Producing light curves
Choose carefully the binning time (option `-b`). Since what we are interested in is a power spectrum, this binning time will limit our maximum frequency in the power spectrum. We are here specifying 2^-8 =0.00390625 for binning time (how to use the `-b` option is of course documented. Use `-h` FMI).
Since we have calibrated the event files, we can also choose an event energy range, here between 3 and 30 keV.
Another thing that is useful in NuSTAR data is taking some time intervals out from the start and the end of each GTI. This is mostly to eliminate an increase of background level that often appears at GTI borders and produces very nasty power spectral shapes. Here I filter 100 s from the start and 300 s from the end of each GTI.
```
#!console

$ python mp_lcurve.py 002A_ev_calib.p 002B_ev_calib.p -b -6 -e 3 30 --safe_interval 100 300
Loading file 002A_ev_calib.p...
Done.
Saving light curve to 002A_3-30_lc.p
Loading file 002B_ev_calib.p...
Done.
Saving light curve to 002B_3-30_lc.p
```
To check the light curve that was produced, use the `test_lc` program in `tests/`:
```
#!console

$ python tests/test_lc.py 002A_3-30_lc.p
```
### 4. Joining, summing and ``scrunching'' light curves
If we want a single light curve from multiple ones, either summing multiple instruments or multiple energy or time ranges, we can use `mp_scrunch_lc`:
```
#!console

$ python mp_scrunch_lc.py 002A_3-30_lc.p 002B_3-30_lc.p -o 002scrunch_3-30_lc.p
Loading file 002A_3-30_lc.p...
Done.
Loading file 002B_3-30_lc.p...
Done.
Saving joined light curve to out_lc.p
Saving scrunched light curve to 002scrunch_3-30_lc.p
```
This is only tested in ``safe'' situations (files are not too big and have consistent time and energy ranges), so it might give inconsistent results or crash in untested situations. Please report any problems!

### 5. Producing power spectra and cross power spectra
Let us just produce the cross power spectrum for now. To produce also the power spectra corresponding to each light curve, substitute `"CPDS"` with `"PDS,CPDS"`. I use rms normalization here, default would be Leahy normalization.
```
#!console

$ python mp_fspec.py 002A_3-30_lc.p 002B_3-30_lc.p -k CPDS -o cpds_002_3-30 --norm rms
Beware! For cpds and derivatives, I assume that the files are
ordered as follows: obs1_FPMA, obs1_FPMB, obs2_FPMA, obs2_FPMB...
Loading file 002A_3-30_lc.p...
Loading file 002B_3-30_lc.p...
Saving CPDS to ./cpds_002_3-30_0.p
```

### 6. Rebinning the spectrum
Now let's rebin the spectrum. If the rebin factor is an integer, it is interpreted as a constant rebinning. Otherwise (only if >1), it is interpreted as a geometric binning.
```
#!console

$ python mp_rebin.py cpds_002_3-30_0.p -r 1.03
Saving cpds to cpds_002_3-30_0_rebin1.03.p
```

### 7. Calculating the cospectrum and phase/time lags
Lag error calculation still to be implemented (quite easy, will do soon).
For the cospectrum, it is sufficient to read the real part of the cross power spectrum as depicted in `tests/test_cpds.py`

### 8. Saving the spectra in a format readable to XSpec
To save the cospectrum in a format readable to XSpec it is sufficient to give the commands
```
#!console

$ python mp_save_as_xspec.py cpds_002_3-30_0_rebin1.03.p
Saving to cpds_002_3-30_0_rebin1.03_xsp.qdp
$ flx2xsp cpds_002_3-30_0_rebin1.03_xsp.qdp cpds.pha cpds.rsp
```

### 9. Open and fit in XSpec!
![screenshot.png](https://bitbucket.org/repo/XA95dR/images/3911632225-screenshot.png)