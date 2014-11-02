# MaLTPyNT
These tools are a stripped-down version of a huge and messy library of codes I've developed in the years to do timing with NuSTAR and other satellites. They contain what is needed for a quick look at the timing properties of an X-ray source.

## Installation
You'll need a recent python 2.7 installation, and the [Numpy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/)  and [Astropy](http://www.astropy.org/) libraries.
Put the python codes in the directory where you are analyzing the data. That's all. Then, you can call them with a python interpreter, e.g.
```
$ python mp_read_events.py filename.evt
```
Otherwise, you can put them in a directory on your PATH, `chmod +x` them and then just call them by name
```
$ mp_read_events.py filename.evt
```
I will hereafter use the first style, as it suits better the current evolutionary state of the code (easier to handle new dependencies that might be needed).

## Tutorial
This software has a modular structure. One starts from cleaned event files (such as those produced by nupipeline), and produces a series of products with subsequent steps:

1. **event lists** containing event arrival times and PI channel information

2. (optional) **calibrated event lists**, where PI values have been converted to energy

3. **light curves**, choosing the energy band and the bin time

4. (optional) **summed light curves** if we want to join events from multiple instruments, or just from different observing times

5. **power spectrum** and/or **cross spectrum**

6. finally, **lags** and **cospectrum**

7. (optional) frequency spectra in XSpec format

### 0. Preliminary info
Most of these tools have help information that can be accessed by typing the name of the command plus -h or --help:
```
$ python mp_calibrate.py 002[AB]_ev.p --help
usage: mp_calibrate.py [-h] [-r RMF] [-o] files [files ...]

positional arguments:
  files              List of files

optional arguments:
  -h, --help         show this help message and exit
  -r RMF, --rmf RMF  rmf file used for calibration
  -o, --overwrite    Overwrite; default: no
```

### 1. Loading event lists
Starting from cleaned event files, we will first save them in `MaLTPyNT` format (a pickle file basically). For example, I'm starting from two event lists called `002A.evt` and `002B.evt`, containing the cleaned event lists from a source observed with NuSTAR's `FPMA` and `FPMB` respectively.
```
$ python mp_read_events.py 002[AB].evt
Opening 002A.evt
Saving events and info to 002A_ev.p
Opening 002B.evt
Saving events and info to 002B_ev.p
```
This will create new files with a `_ev.p` extension, containing the event times and the energy _channel_ (`PI`) of each event

### 2. Calibrating event lists
Use `mp_calibrate`. You can either specify an `rmf` file with the `-r` option, or just let it look for it in the NuSTAR `CALDB` (the environment variable has to be defined!)
```
$ python mp_calibrate.py 002[AB]_ev.p
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
TBD

### 4. Joining, summing and ``scrunching'' light curves
TBD

### 5. Producing power spectra and cross power spectra
TBD

### 6. Calculating the cospectrum and phase/time lags
TBD

### 7. Saving the spectra in a format readable to XSpec
TBD