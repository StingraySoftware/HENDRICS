#!/bin/bash
TRAVIS_PYTHON_VERSION=$1
WITH_NETCDF=$2

if test ! -e /home/travis/miniconda/pkgs/; then
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -f
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda
    if test $WITH_NETCDF = "yes"; then
        conda install --yes python=$TRAVIS_PYTHON_VERSION libnetcdf hdf5 h5py netCDF4 scipy matplotlib astropy numpy
    else
        conda install --yes python=$TRAVIS_PYTHON_VERSION scipy matplotlib astropy numpy
    fi
fi
