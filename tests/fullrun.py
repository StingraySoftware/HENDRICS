# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import maltpynt as mp
import numpy as np
MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION
import logging
import os
import sys
import glob
import subprocess as sp

PY2 = sys.version_info[0] == 2
PYX6 = sys.version_info[1] == 6

if PY2 and PYX6:
    import unittest2 as unittest
else:
    import unittest

logging.basicConfig(filename='MP.log', level=logging.DEBUG, filemode='w')
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestCommandline(unittest.TestCase):

    """Test how command lines work.

    When command line is missing, revert
    to library calls (some overlap with utests.py).
    """

    def step00_scripts_are_installed0(self):
        """Test only once that command line scripts are installed."""
        command = 'MPreadevents -h'
        sp.check_call(command.split())

    def step01_load_events(self):
        """Test event file reading."""
        try:
            command = '{0} {1}'.format(
                os.path.join(datadir, 'monol_testA.evt'),
                os.path.join(datadir, 'monol_testB.evt'))
            mp.read_events.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2}); Path: {3}".format(
                'Loading event file ', type(e), e, sys.path))

    def step02_calibrate(self):
        """Test event file calibration."""
        try:
            command = '{0} {1} -r {2}'.format(
                os.path.join(datadir, 'monol_testA_ev' + MP_FILE_EXTENSION),
                os.path.join(datadir, 'monol_testB_ev' + MP_FILE_EXTENSION),
                os.path.join(datadir, 'test.rmf'))
            mp.calibrate.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'Calibrating event file', type(e), e))

    def step03a_lcurve(self):
        """Test light curve production."""
        try:
            command = '{0} {1} -e {2} {3} --safe-interval {4} {5}'.format(
                os.path.join(datadir, 'monol_testA_ev_calib' +
                             MP_FILE_EXTENSION),
                os.path.join(datadir, 'monol_testB_ev_calib' +
                             MP_FILE_EXTENSION),
                3, 50, 100, 300)
            print(command)
            mp.lcurve.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'Producing LC from event file', type(e), e))

    def step03b_fits_lcurve(self):
        """Test light curves from FITS."""
        try:
            lcurve_ftools_orig = os.path.join(datadir, 'lcurveA.fits')
            mp.lcurve.lcurve_from_events(
                os.path.join(datadir,
                             'monol_testA_ev') + MP_FILE_EXTENSION,
                outfile=os.path.join(datadir,
                                     'lcurve_lc'))
            mp.lcurve.lcurve_from_fits(
                lcurve_ftools_orig,
                outfile=os.path.join(datadir,
                                     'lcurve_ftools_lc'))
            lcurve_ftools = os.path.join(datadir,
                                         'lcurve_ftools_lc' +
                                         MP_FILE_EXTENSION)
            lcurve_mp = os.path.join(datadir,
                                     'lcurve_lc' +
                                     MP_FILE_EXTENSION)
            mp.io.load_lcurve(lcurve_mp)
            mp.io.load_lcurve(lcurve_ftools)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('LC fits', type(e), e))

    def step03c_txt_lcurve(self):
        """Test light curves from txt."""
        try:
            lcurve_mp = os.path.join(datadir,
                                     'lcurve_lc' +
                                     MP_FILE_EXTENSION)
            lcdata_mp = mp.io.load_lcurve(lcurve_mp)
            lc_mp = lcdata_mp['lc']
            time_mp = lcdata_mp['time']

            lcurve_txt_orig = os.path.join(datadir,
                                           'lcurve_txt_lc.txt')

            mp.io.save_as_ascii([time_mp, lc_mp], lcurve_txt_orig)

            lcurve_txt = os.path.join(datadir,
                                      'lcurve_txt_lc' +
                                      MP_FILE_EXTENSION)
            mp.lcurve.lcurve_from_txt(lcurve_txt_orig,
                                      outfile=lcurve_txt)

        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('LC txt', type(e), e))

    def step04a_pds(self):
        """Test PDS production."""
        try:
            command = '{0} {1} -f 128 --save-dyn -k PDS'.format(
                os.path.join(datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION)
            mp.fspec.main(command.split())
        except:
            raise(Exception('Production of PDSs failed'))

    def step04b_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)
        try:
            command = '{0} -f 128'.format(lcurve_ftools)
            mp.fspec.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'PDS LC FITS', type(e), e))

    def step04c_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(datadir,
                                  'lcurve_txt_lc' +
                                  MP_FILE_EXTENSION)
        try:
            command = '{0} -f 128'.format(lcurve_txt)
            mp.fspec.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('PDS LC txt', type(e), e))

    def step05_cpds(self):
        """Test CPDS production."""
        try:
            command = \
                '{0} {1} -f 128 --save-dyn -k CPDS -o {2}'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_lc') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_testB_E3-50_lc') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_test_E3-50'))
            mp.fspec.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('CPDS', type(e), e))

    def step06_lags(self):
        """Test Lag calculations."""
        try:
            command = '{0} {1} {2} -o {3}'.format(
                os.path.join(datadir, 'monol_test_E3-50_cpds') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_testA_E3-50_pds') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_testB_E3-50_pds') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_test'))
            mp.lags.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('Lags production',
                                                     type(e), e))

    def step07_rebinlc(self):
        """Test LC rebinning."""
        try:
            command = '{0} -r 2'.format(
                os.path.join(datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION)
            mp.rebin.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('LC rebin', type(e), e))

    def step08_rebinpds1(self):
        """Test PDS rebinning 1."""
        try:
            command = '{0} -r 2'.format(
                os.path.join(datadir, 'monol_testA_E3-50_pds') +
                MP_FILE_EXTENSION)
            mp.rebin.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('PDS rebin Test 1',
                                                     type(e), e))

    def step08a_rebinpds2(self):
        """Test PDS rebinning 2."""
        try:
            command = '{0} -r 1.03'.format(
                os.path.join(datadir, 'monol_testA_E3-50_pds') +
                MP_FILE_EXTENSION)
            mp.rebin.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('PDS rebin Test 2',
                                                     type(e), e))

    def step09_rebincpds(self):
        """Test CPDS rebinning."""
        try:
            command = '{0} -r 1.03'.format(
                os.path.join(datadir, 'monol_test_E3-50_cpds') +
                MP_FILE_EXTENSION)
            mp.rebin.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('CPDS rebin Test 1.03',
                                                     type(e), e))

    def step10_savexspec1(self):
        """Test save as Xspec 1."""
        try:
            command = '{0}'.format(
                os.path.join(datadir, 'monol_testA_E3-50_pds_rebin2') +
                MP_FILE_EXTENSION)
            mp.save_as_xspec.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('MP2xspec Test 1',
                                                     type(e), e))

    def step11_savexspec2(self):
        """Test save as Xspec 2."""
        try:
            command = '{0}'.format(
                os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') +
                MP_FILE_EXTENSION)
            mp.save_as_xspec.main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('MP2xspec Test 2',
                                                     type(e), e))

    def step12_joinlcs(self):
        """Test produce joined light curves."""
        try:
            mp.lcurve.join_lightcurves(
                [os.path.join(datadir, 'monol_testA_E3-50_lc') +
                 MP_FILE_EXTENSION,
                 os.path.join(datadir, 'monol_testB_E3-50_lc') +
                 MP_FILE_EXTENSION],
                os.path.join(datadir, 'monol_test_joinlc' +
                             MP_FILE_EXTENSION))
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('Join lcs', type(e),
                                                     e))

    def step13_scrunchlcs(self):
        """Test produce scrunched light curves."""
        try:
            command = '{0} {1} -o {2}'.format(
                os.path.join(datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_test_scrunchlc') +
                MP_FILE_EXTENSION)
            mp.lcurve.scrunch_main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('MPscrunchlc', type(e),
                                                     e))

    def step13_dumpdynpds(self):
        """Test dump dynamical PDSs."""
        try:
            command = '--noplot ' + \
                os.path.join(datadir,
                             'monol_testA_E3-50_pds_rebin1.03') + \
                MP_FILE_EXTENSION
            mp.fspec.dumpdyn_main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('MPdumpdyn <pds>',
                                                     type(e), e))

    def step14_dumpdyncpds(self):
        """Test produce scrunched light curves."""
        try:
            command = '--noplot ' + \
                os.path.join(datadir,
                             'monol_test_E3-50_cpds_rebin1.03') + \
                MP_FILE_EXTENSION
            mp.fspec.dumpdyn_main(command.split())
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('MPdumpdyn <cpds>',
                                                     type(e), e))

    def step15_create_gti1(self):
        """Test creating a GTI file"""

        fname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        command = "{0} -f lc>0 -c --debug".format(fname)
        print(command.split())
        mp.create_gti.main(command.split())

    def step15_create_gti2(self):
        """Test creating a GTI file"""

        fname = os.path.join(datadir, 'monol_testA_E3-50_gti') + \
            MP_FILE_EXTENSION
        lcfname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        command = "{0} -a {1} --debug".format(lcfname, fname)

        mp.create_gti.main(command.split())

    def step16_readfile1(self):
        """Test reading and dumping a MaLTPyNT file"""

        fname = os.path.join(datadir, 'monol_testA_E3-50_gti') + \
            MP_FILE_EXTENSION
        command = "{0}".format(fname)

        mp.io.main(command.split())

    def step16_readfile2(self):
        """Test reading and dumping a FITS file"""

        fitsname = os.path.join(datadir, 'monol_testA.evt')
        command = "{0}".format(fitsname)

        mp.io.main(command.split())

    def step17_exposure(self):
        """Test exposure calculations from unfiltered files"""

        lcname = os.path.join(datadir,
                              'monol_testA_E3-50_lc' + MP_FILE_EXTENSION)
        ufname = os.path.join(datadir, 'monol_testA_uf.evt')
        command = "{0} {1}".format(lcname, ufname)

        mp.exposure.main(command.split())

    def step18_plot(self):
        """Test plotting"""
        pname = os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            MP_FILE_EXTENSION
        cname = os.path.join(datadir, 'monol_test_E3-50_cpds_rebin1.03') + \
            MP_FILE_EXTENSION
        lname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        mp.plot.main(pname, cname, lname)

    def _all_steps(self):

        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        """Test a full run of the scripts (command lines)."""
        print('')
        for name, step in self._all_steps():
            try:
                print('- ', step.__doc__, '...', end=' (command line) ')
                step()
                print('OK')
            except Exception as e:
                self.fail("{0} failed ({1}: {2})".format(step, type(e), e))
                print('Failed')
        print('Cleaning up...')

        file_list = \
            glob.glob(os.path.join(datadir,
                                   '*monol_test*')
                      + MP_FILE_EXTENSION) + \
            glob.glob(os.path.join(datadir,
                                   '*lcurve*')
                      + MP_FILE_EXTENSION) + \
            glob.glob(os.path.join(datadir,
                                   '*lcurve*.txt')) + \
            glob.glob(os.path.join(datadir,
                                   '*.log')) + \
            glob.glob(os.path.join(datadir,
                                   '*monol_test*.dat')) + \
            glob.glob(os.path.join(datadir,
                                   '*monol_test*.txt'))
        for f in file_list:
            os.remove(f)


if __name__ == '__main__':
    unittest.main(verbosity=2)
