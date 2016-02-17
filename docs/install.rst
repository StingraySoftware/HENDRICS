Installation Instructions
=========================

Prerequisites
-------------

You'll need a recent python 2.7+ or 3.3+ installation, and the
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

Quick Installation (release candidates+, recommended)
-----------------------------------------------------
Run

::

    $ pip install --pre maltpynt

and that's it!

Quick Installation (stable releases)
------------------------------------
Run

::

    $ pip install maltpynt

and that's it!

Installing the Development version
----------------------------------

Download
~~~~~~~~

Download the distribution directory:

::

    $ git clone git@bitbucket.org:mbachett/maltpynt.git

To use this command you will probably need to setup an SSH key for your
account (in Manage Account, recommended!). Otherwise, you can use the
command

::

    $ git clone https://<yourusername>@bitbucket.org/mbachett/maltpynt.git

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

    import maltpynt
