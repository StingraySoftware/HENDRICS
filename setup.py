from distutils.core import setup
import glob

setup(
    name = 'maltpynt',
    version = 'beta',
    description  = "Matteo's Library and Tools in Python for NuSTAR Timing",
    long_description = open('README.md').read(),
    packages = ['maltpynt'],
    author = 'Matteo Bachetti',
    author_email = "matteo@matteobachetti.it",
    license = 'MIT',
    url = '',
    keywords = 'X-ray astronomy nustar rxte xmm timing cospectrum PDS',
    scripts = glob.glob('scripts/*'),
    platforms = 'all',
    classifiers = [
        'Intended Audience :: Science/Research, Education, Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Astronomy'
   ],
)