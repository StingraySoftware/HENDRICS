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

    def step01_load_events(self):
        """Test event file reading."""
        try:
            sp.check_call('MPreadevents {} {}'.format(
                os.path.join(datadir, 'monol_testA.evt'),
                os.path.join(datadir, 'monol_testB.evt')).split())
        except:
            raise(Exception('Loading event file failed'))

    def step02_calibrate(self):
        """Test event file calibration."""
        try:
            sp.check_call('MPcalibrate {} {} -r {}'.format(
                os.path.join(datadir, 'monol_testA_ev' + MP_FILE_EXTENSION),
                os.path.join(datadir, 'monol_testB_ev' + MP_FILE_EXTENSION),
                os.path.join(datadir, 'test.rmf')).split())
        except:
            raise(Exception('Calibrating event file failed'))

    def step03a_lcurve(self):
        """Test light curve production."""
        try:
            sp.check_call(
                'MPlcurve {} {} -e {} {} --safe-interval {} {}'.format(
                    os.path.join(datadir, 'monol_testA_ev_calib' +
                                 MP_FILE_EXTENSION),
                    os.path.join(datadir, 'monol_testB_ev_calib' +
                                 MP_FILE_EXTENSION),
                    3, 50, 100, 300).split())
        except:
            raise(Exception('Production of light curve failed'))

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
            lcdata_mp = mp.io.load_lcurve(lcurve_mp)
            lcdata_ftools = mp.io.load_lcurve(lcurve_ftools)

            lc_mp = lcdata_mp['lc']
            lenmp = len(lc_mp)
            lc_ftools = lcdata_ftools['lc']
            lenftools = len(lc_ftools)
            goodlen = min([lenftools, lenmp])
        except Exception as e:
            self.fail("{} failed ({}: {})".format('LC fits', type(e), e))

        assert np.all(np.abs(lc_mp[:goodlen] - lc_ftools[:goodlen]) <= 1e-3), \
            'Light curve data do not coincide between FITS and MP'

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
            lcdata_txt = mp.io.load_lcurve(lcurve_txt)

            lc_txt = lcdata_txt['lc']

        except Exception as e:
            self.fail("{} failed ({}: {})".format('LC txt', type(e), e))

        assert np.all(np.abs(lc_mp - lc_txt) <= 1e-3), \
            'Light curve data do not coincide between txt and MP'

    def step04a_pds(self):
        """Test PDS production."""
        try:
            sp.check_call(
                'MPfspec {} {} -f 128 --save-dyn -k PDS'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_lc') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_testB_E3-50_lc') +
                    MP_FILE_EXTENSION).split())
        except:
            raise(Exception('Production of PDSs failed'))

    def step04b_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)
        try:
            sp.check_call(
                'MPfspec {} -f 128'.format(lcurve_ftools).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('PDS LC FITS', type(e), e))

    def step04c_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(datadir,
                                  'lcurve_txt_lc' +
                                  MP_FILE_EXTENSION)
        try:
            sp.check_call(
                'MPfspec {} -f 128'.format(lcurve_txt).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('PDS LC txt', type(e), e))

    def step05_cpds(self):
        """Test CPDS production."""
        try:
            sp.check_call(
                'MPfspec {} {} -f 128 --save-dyn -k CPDS -o {}'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_lc') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_testB_E3-50_lc') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_test_E3-50')).split())
        except:
            raise(Exception('Production of CPDS failed'))

    def step06_lags(self):
        """Test Lag calculations."""
        try:
            sp.check_call(
                'MPlags {} {} {} -o {}'.format(
                    os.path.join(datadir, 'monol_test_E3-50_cpds') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_testA_E3-50_pds') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_testB_E3-50_pds') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_test')).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('Lags production',
                                                  type(e), e))

    def step07_rebinlc(self):
        """Test LC rebinning."""
        try:
            sp.check_call(
                'MPrebin {} -r 2'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_lc') +
                    MP_FILE_EXTENSION).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('LC rebin', type(e), e))

    def step08_rebinpds1(self):
        """Test PDS rebinning 1."""
        try:
            sp.check_call(
                'MPrebin {} -r 2'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_pds') +
                    MP_FILE_EXTENSION).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('PDS rebin Test 1', type(e),
                                                  e))

    def step08a_rebinpds2(self):
        """Test PDS rebinning 2."""
        try:
            sp.check_call(
                'MPrebin {} -r 1.03'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_pds') +
                    MP_FILE_EXTENSION).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('PDS rebin Test 2', type(e),
                                                  e))

    def step09_rebincpds(self):
        """Test CPDS rebinning."""
        try:
            sp.check_call(
                'MPrebin {} -r 1.03'.format(
                    os.path.join(datadir, 'monol_test_E3-50_cpds') +
                    MP_FILE_EXTENSION).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('CPDS rebin Test 1.03',
                                                  type(e), e))

    def step10_savexspec1(self):
        """Test save as Xspec 1."""
        try:
            sp.check_call(
                'MP2xspec {}'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_pds_rebin2') +
                    MP_FILE_EXTENSION).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('MP2xspec Test 1', type(e),
                                                  e))

    def step11_savexspec2(self):
        """Test save as Xspec 2."""
        try:
            sp.check_call(
                'MP2xspec {}'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') +
                    MP_FILE_EXTENSION).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('MP2xspec Test 2', type(e),
                                                  e))

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
            self.fail("{} failed ({}: {})".format('Join lcs', type(e),
                                                  e))

    def step13_scrunchlcs(self):
        """Test produce scrunched light curves."""
        try:
            sp.check_call(
                'MPscrunchlc {} {} -o {}'.format(
                    os.path.join(datadir, 'monol_testA_E3-50_lc') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_testB_E3-50_lc') +
                    MP_FILE_EXTENSION,
                    os.path.join(datadir, 'monol_test_scrunchlc') +
                    MP_FILE_EXTENSION).split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('MPscrunchlc', type(e),
                                                  e))

    def step13_dumpdynpds(self):
        """Test dump dynamical PDSs."""
        import subprocess as sp
        try:
            command = 'MPdumpdyn --noplot ' + \
                os.path.join(datadir,
                             'monol_testA_E3-50_pds_rebin1.03') + \
                MP_FILE_EXTENSION
            sp.check_output(command.split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('MPdumpdyn <pds>', type(e),
                                                  e))

    def step14_dumpdyncpds(self):
        """Test produce scrunched light curves."""
        import subprocess as sp
        try:
            command = 'MPdumpdyn --noplot ' + \
                os.path.join(datadir,
                             'monol_test_E3-50_cpds_rebin1.03') + \
                MP_FILE_EXTENSION
            sp.check_output(command.split())
        except Exception as e:
            self.fail("{} failed ({}: {})".format('MPdumpdyn <cpds>', type(e),
                                                  e))

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
                self.fail("{} failed ({}: {})".format(step, type(e), e))
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
