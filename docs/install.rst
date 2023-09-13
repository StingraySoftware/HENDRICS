Installation Instructions
=========================

Prerequisites
-------------

You'll need a recent python 3.8+ installation.
The main dependency is

1. `Stingray <http://github.com/stingraysoftware/stingray>`__,

which in turn depends on

2. `Numpy <http://www.numpy.org/>`__;

3. `Matplotlib <http://matplotlib.org/>`__;

4. `Scipy <http://scipy.org/>`__;

5. `Astropy <http://www.astropy.org/>`__

**Optional but recommended** dependencies are:

6. the `netCDF 4 library <http://www.unidata.ucar.edu/software/netcdf/>`__ with its
`python bindings <https://github.com/Unidata/netcdf4-python>`__;

7. `Numba <http://numba.pydata.org>`__;

8. `statsmodels <http://www.statsmodels.org/stable/index.html>`__

9. `emcee <https://emcee.readthedocs.io/en/stable/>`__

10. `pint <http://github.com/nanograv/pint/>`__

You should also
have a working `HEASoft <http://heasarc.nasa.gov/lheasoft/>`__
installation to produce the cleaned event files and to use
`XSpec <http://heasarc.nasa.gov/lheasoft/xanadu/xspec/index.html>`__.

Installing releases
-------------------
::

    $ pip install hendrics numba emcee statsmodels netcdf4 matplotlib stingray>=1.0b1


Installing the Development version
----------------------------------

Download
~~~~~~~~

Download the distribution directory:

::

    $ git clone git@github.com/StingraySoftware/HENDRICS

Or

::

    $ git clone https://github.com/StingraySoftware/HENDRICS

To update the software, just run

::

    $ git pull

from the source directory (usually, the command gives troubleshooting
information if this doesn't work the first time).

Installation
~~~~~~~~~~~~

Install the dependencies above; then, enter the distribution directory and run

::

    $ pip install .

this will check for the existing dependencies and install the files in a
proper way. From that point on, executables will be somewhere in your
PATH and python libraries will be available in python scripts with the
usual

.. code-block :: python

    import hendrics
