from __future__ import print_function
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import glob
import sys
import os

if 'test' in sys.argv:
    # If running tests, use the Agg backend. This avoids DISPLAY problems in
    # Travis CI
    import matplotlib
    matplotlib.use('Agg')


def generate_version_py(version):
    fname = os.path.join('maltpynt', 'version.py')
    versionstring = "version = '{}'\n".format(version)
    with open(fname, 'w') as fobj:
        fobj.write(versionstring)


PY2 = sys.version_info[0] == 2
PYX6 = sys.version_info[1] <= 6

install_requires = [
    'matplotlib',
    'scipy',
    'numpy',
    'astropy'
    ]

if PY2 and PYX6:
    install_requires += ['unittest2']

version = '1.0.7'

generate_version_py(version)

setup(name='maltpynt',
      version=version,
      description="Matteo's Library and Tools in Python for NuSTAR Timing",
      packages=['maltpynt'],
      package_data={'': ['README.md']},
      include_package_data=True,
      author='Matteo Bachetti',
      author_email="matteo@matteobachetti.it",
      license='3-clause BSD',
      url='https://bitbucket.org/mbachett/maltpynt',
      keywords='X-ray astronomy nustar rxte xmm timing cospectrum PDS',
      scripts=glob.glob('scripts/*'),
      platforms='all',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Topic :: Scientific/Engineering :: Astronomy'
          ],
      install_requires=install_requires,
      test_suite='tests')
