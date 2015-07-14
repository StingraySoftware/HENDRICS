#!/bin/bash
TRAVIS_PYTHON_VERSION=$1
conda install --yes python=$TRAVIS_PYTHON_VERSION libnetcdf hdf5 h5py netCDF4 scipy matplotlib astropy
