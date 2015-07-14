#!/bin/bash
TRAVIS_PYTHON_VERSION=$1
conda install --yes python=$TRAVIS_PYTHON_VERSION scipy matplotlib astropy
