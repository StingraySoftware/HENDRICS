Installation Instructions
=========================

Prerequisites
-------------

You'll need a recent python 2.7+ or 3.4+ installation, and the
`Numpy <http://www.numpy.org/>`__,
`Matplotlib <http://matplotlib.org/>`__, `Scipy <http://scipy.org/>`__
and `Astropy <http://www.astropy.org/>`__ libraries. You should also
have a working `HEASoft <http://heasarc.nasa.gov/lheasoft/>`__
installation to produce the cleaned event files and to use
`XSpec <http://heasarc.nasa.gov/lheasoft/xanadu/xspec/index.html>`__.

An **optional but recommended** dependency is the `netCDF 4
library <http://www.unidata.ucar.edu/software/netcdf/>`__ with its
`python bindings <https://github.com/Unidata/netcdf4-python>`__.
An additional dependency that is now used only sparsely (if installed) but will
become important in future versions is `Numba <http://numba.pydata.org>`__.

Installing releases
-------------------

The usual:

::

    $ pip install hendrics


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

Enter the distribution directory and run

::

    $ python setup.py install

this will check for the existing dependencies and install the files in a
proper way. From that point on, executables will be somewhere in your
PATH and python libraries will be available in python scripts with the
usual

.. code-block :: python

    import hendrics
