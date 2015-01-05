try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import glob

setup(name='maltpynt',
      version='beta',
      description="Matteo's Library and Tools in Python for NuSTAR Timing",
      packages=['maltpynt'],
      package_data={'': ['README.md']},
      include_package_data=True,
      author='Matteo Bachetti',
      author_email="matteo@matteobachetti.it",
      license='read README.md',
      url='',
      keywords='X-ray astronomy nustar rxte xmm timing cospectrum PDS',
      scripts=glob.glob('scripts/*'),
      platforms='all',
      classifiers=[
          'Intended Audience :: Science/Research, Education, Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Topic :: Scientific/Engineering :: Astronomy'
          ],
      install_requires=[
          'matplotlib',
          'scipy',
          'numpy',
          'astropy'
          ]
      )
