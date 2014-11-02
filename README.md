# MaLTPyNT #
These tools are a stripped-down version of a huge and messy library of codes I've developed in the years to do timing with NuSTAR and other satellites. They contain what is needed for a quick look at the timing properties of an X-ray source.

## Installation ##
You'll need a recent python 2.7 installation, and the [Numpy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/)  and [Astropy](http://www.astropy.org/) libraries.
Put the python codes in the directory where you are analyzing the data. That's all.

## Usage ##
This software has a modular structure. One starts from cleaned event files (such as those produced by nupipeline), and produces a series of products with subsequent steps:

1. **light curves**, choosing the energy band and the bin time

2. (optional) **summed light curves** if we want to join events from multiple instruments

3. **power spectrum** and/or **cross spectrum**

4. finally, **lags** and **cospectrum**

5. (optional) frequency spectra in XSpec format

### Loading event lists ###

Starting from the cleaned event files A.evt and B.evt, we will first save them in MaLTPyNT format (a pickle file basically)
```
python mp_read_events.py *.evt
```
This will create new files with a _ev.p extension

TBC