# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import maltpynt as mp
import logging
import os
import glob
import subprocess as sp
import numpy as np
from astropy.tests.helper import catch_warnings
import unittest

MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION

logging.basicConfig(filename='MP.log', level=logging.DEBUG, filemode='w')
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestFullRun(unittest.TestCase):
    """Test how command lines work.

    Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/python-unittest-testcase-execution-order

    When command line is missing, uses some function calls
    """  # NOQA

    def step00a_scripts_are_installed(self):
        """Test only once that command line scripts are installed correctly."""
        fits_file = os.path.join(datadir, 'monol_testA.evt')
        command = 'MPreadfile {0}'.format(fits_file)
        sp.check_call(command.split())

    def step01a_fake_file(self):
        """Test produce a fake event file."""
        fits_file = os.path.join(datadir, 'monol_test_fake.evt')
        mp.fake.main(['-o', fits_file, '--instrument', 'FPMB'])
        info = mp.io.print_fits_info(fits_file, hdu=1)
        assert info['Instrument'] == 'FPMB'

    def step01b_fake_file(self):
        """Test produce a fake event file from input light curve."""
        lcurve_in = os.path.join(datadir, 'lcurveA.fits')
        fits_file = os.path.join(datadir, 'monol_test_fake_lc.evt')
        mp.fake.main(['--lc', lcurve_in, '-o', fits_file])

    def step01c_fake_file(self):
        """Test produce a fake event file and apply deadtime."""
        fits_file = os.path.join(datadir, 'monol_test_fake_lc.evt')
        mp.fake.main(['--deadtime', '2.5e-3',
                      '--ctrate', '2000',
                      '-o', fits_file])

    def step02a_load_events(self):
        """Test event file reading."""
        command = '{0} {1} --nproc 2'.format(
            os.path.join(datadir, 'monol_testA.evt'),
            os.path.join(datadir, 'monol_testA_timezero.evt'),
            os.path.join(datadir, 'monol_test_fake.evt'))
        mp.read_events.main(command.split())

    def step02b_load_events(self):
        """Test event file reading."""
        command = '{0}'.format(
            os.path.join(datadir, 'monol_testB.evt'))
        mp.read_events.main(command.split())

    def step02c_load_events_split(self):
        """Test event file splitting."""
        command = \
            '{0} -g --min-length 0'.format(
                os.path.join(datadir, 'monol_testB.evt'))
        mp.read_events.main(command.split())

    def step02d_load_gtis(self):
        """Test loading of GTIs from FITS files."""
        fits_file = os.path.join(datadir, 'monol_testA.evt')
        mp.io.load_gtis(fits_file)

    def step02e_load_events_noclobber(self):
        """Test event file reading w. noclobber option."""
        with catch_warnings() as w:
            command = \
                '{0} --noclobber'.format(
                    os.path.join(datadir, 'monol_testB.evt'))
            mp.read_events.main(command.split())
        assert str(w[0].message).strip().endswith(
            "noclobber option used. Skipping"), \
            "Unexpected warning output"

    def step03a_calibrate(self):
        """Test event file calibration."""
        command = '{0} -r {1}'.format(
            os.path.join(datadir, 'monol_testA_ev' + MP_FILE_EXTENSION),
            os.path.join(datadir, 'test.rmf'))
        mp.calibrate.main(command.split())

    def step03b_calibrate(self):
        """Test event file calibration."""
        command = '{0} -r {1} --nproc 2'.format(
            os.path.join(datadir, 'monol_testB_ev' + MP_FILE_EXTENSION),
            os.path.join(datadir, 'test.rmf'))
        mp.calibrate.main(command.split())

    def step04a_lcurve(self):
        """Test light curve production."""
        command = ('{0} -e {1} {2} --safe-interval '
                   '{3} {4}  --nproc 2 -b 0.5').format(
            os.path.join(datadir, 'monol_testA_ev_calib' +
                         MP_FILE_EXTENSION),
            3, 50, 100, 300)
        mp.lcurve.main(command.split())

        command = ('{0} -e {1} {2} --safe-interval '
                   '{3} {4} -b 0.5').format(
            os.path.join(datadir, 'monol_testB_ev_calib' +
                         MP_FILE_EXTENSION),
            3, 50, 100, 300)
        mp.lcurve.main(command.split())

    def step04b_lcurve_split(self):
        """Test lc with gti-split option, and reading of split event file."""
        command = '{0} -g'.format(
            os.path.join(datadir, 'monol_testB_ev_0' +
                         MP_FILE_EXTENSION))
        mp.lcurve.main(command.split())

    def step04c_fits_lcurve0(self):
        """Test light curves from FITS."""
        lcurve_ftools_orig = os.path.join(datadir, 'lcurveA.fits')

        lcurve_ftools = os.path.join(datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)

        command = "{0} --outfile {1}".format(
            os.path.join(datadir,
                         'monol_testA_ev') + MP_FILE_EXTENSION,
            os.path.join(datadir,
                         'lcurve_lc'))
        mp.lcurve.main(command.split())

        command = "--fits-input {0} --outfile {1}".format(
            lcurve_ftools_orig,
            lcurve_ftools)
        mp.lcurve.main(command.split())

    def step04c_fits_lcurve1(self):
        """Test light curves from FITS."""
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

        diff = lc_mp[:goodlen] - lc_ftools[:goodlen]

        assert np.all(np.abs(diff) <= 1e-3), \
            'Light curve data do not coincide between FITS and MP'

    def step04d_txt_lcurve(self):
        """Test light curves from txt."""
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
        mp.lcurve.main(['--txt-input', lcurve_txt_orig,
                        '--outfile', lcurve_txt])
        lcdata_txt = mp.io.load_lcurve(lcurve_txt)

        lc_txt = lcdata_txt['lc']

        assert np.all(np.abs(lc_mp - lc_txt) <= 1e-3), \
            'Light curve data do not coincide between txt and MP'

    def step04e_joinlcs(self):
        """Test produce joined light curves."""
        mp.lcurve.join_lightcurves(
            [os.path.join(datadir, 'monol_testA_E3-50_lc') +
             MP_FILE_EXTENSION,
             os.path.join(datadir, 'monol_testB_E3-50_lc') +
             MP_FILE_EXTENSION],
            os.path.join(datadir, 'monol_test_joinlc' +
                         MP_FILE_EXTENSION))

    def step04f_scrunchlcs(self):
        """Test produce scrunched light curves."""
        command = '{0} {1} -o {2}'.format(
            os.path.join(datadir, 'monol_testA_E3-50_lc') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_testB_E3-50_lc') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_test_scrunchlc') +
            MP_FILE_EXTENSION)
        mp.lcurve.scrunch_main(command.split())

    def step04g_lcurve(self):
        """Test light curve error from uncalibrated file."""
        command = ('{0} -e {1} {2}').format(
            os.path.join(datadir, 'monol_testA_ev' +
                         MP_FILE_EXTENSION), 3, 50)

        with catch_warnings() as w:
            mp.lcurve.main(command.split())

        assert np.any([str(i.message).strip().endswith(
            "Did you run MPcalibrate?") for i in w]), \
            "Unexpected behavior in lcurve"

    def step04h_lcurve(self):
        """Test light curve using PI filtering."""
        command = ('{0} --pi-interval {1} {2}').format(
            os.path.join(datadir, 'monol_testA_ev' +
                         MP_FILE_EXTENSION), 10, 300)

        mp.lcurve.main(command.split())

    def step05a_pds(self):
        """Test PDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k PDS --norm rms  --nproc 2 '.format(
                os.path.join(datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION)
        mp.fspec.main(command.split())

    def step05b_pds(self):
        """Test PDS production."""
        command = \
            '{0} -f 128 --save-dyn -k PDS --norm rms'.format(
                os.path.join(datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION)
        mp.fspec.main(command.split())

    def step05c_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)
        command = '{0} -f 128'.format(lcurve_ftools)
        mp.fspec.main(command.split())

    def step05d_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(datadir,
                                  'lcurve_txt_lc' +
                                  MP_FILE_EXTENSION)
        command = '{0} -f 128'.format(lcurve_txt)
        mp.fspec.main(command.split())

    def step05e_cpds(self):
        """Test CPDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k CPDS --norm rms -o {2}'.format(
                os.path.join(datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_test_E3-50'))
        mp.fspec.main(command.split())

    def step05f_cpds(self):
        """Test CPDS production."""
        command = \
            ('{0} {1} -f 128 --save-dyn -k '
             'CPDS --norm rms -o {2} --nproc 2').format(
                os.path.join(datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(datadir, 'monol_test_E3-50'))
        mp.fspec.main(command.split())

    def step05g_dumpdynpds(self):
        """Test dump dynamical PDSs."""
        command = '--noplot ' + \
            os.path.join(datadir,
                         'monol_testA_E3-50_pds') + \
            MP_FILE_EXTENSION
        mp.fspec.dumpdyn_main(command.split())

    def step05h_sumpds(self):
        """Test the sum of pdss."""
        mp.sum_fspec.main([
            os.path.join(datadir,
                         'monol_testA_E3-50_pds') + MP_FILE_EXTENSION,
            os.path.join(datadir,
                         'monol_testB_E3-50_pds') + MP_FILE_EXTENSION,
            '-o', os.path.join(datadir,
                               'monol_test_sum' + MP_FILE_EXTENSION)])

    def step05i_dumpdyncpds(self):
        """Test dumping CPDS file."""
        command = '--noplot ' + \
            os.path.join(datadir,
                         'monol_test_E3-50_cpds') + \
            MP_FILE_EXTENSION
        mp.fspec.dumpdyn_main(command.split())

    def step06a_rebinlc(self):
        """Test LC rebinning."""
        command = '{0} -r 4'.format(
            os.path.join(datadir, 'monol_testA_E3-50_lc') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())

    def step06b_rebinpds(self):
        """Test PDS rebinning 1."""
        command = '{0} -r 2'.format(
            os.path.join(datadir, 'monol_testA_E3-50_pds') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())

    def step06c_rebinpds(self):
        """Test geometrical PDS rebinning."""
        command = '{0} {1} -r 1.03'.format(
            os.path.join(datadir, 'monol_testA_E3-50_pds') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_testB_E3-50_pds') +
            MP_FILE_EXTENSION
            )
        mp.rebin.main(command.split())

    def step06d_rebincpds(self):
        """Test CPDS rebinning."""
        command = '{0} -r 2'.format(
            os.path.join(datadir, 'monol_test_E3-50_cpds') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())

    def step06e_rebincpds(self):
        """Test CPDS geometrical rebinning."""
        command = '{0} -r 1.03'.format(
            os.path.join(datadir, 'monol_test_E3-50_cpds') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())

    def step06f_dumpdyncpds_reb(self):
        """Test dumping rebinned CPDS file."""
        command = '--noplot ' + \
            os.path.join(datadir,
                         'monol_test_E3-50_cpds_rebin1.03') + \
            MP_FILE_EXTENSION
        mp.fspec.dumpdyn_main(command.split())

    def step07a_lags(self):
        """Test Lag calculations."""
        command = '{0} {1} {2} -o {3}'.format(
            os.path.join(datadir, 'monol_test_E3-50_cpds') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_testA_E3-50_pds') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_testB_E3-50_pds') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_test'))
        mp.lags.main(command.split())

    def step07b_lags(self):
        """Test Lag calculations in rebinned data."""
        command = '{0} {1} {2} -o {3}'.format(
            os.path.join(datadir, 'monol_test_E3-50_cpds_rebin1.03') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_testB_E3-50_pds_rebin1.03') +
            MP_FILE_EXTENSION,
            os.path.join(datadir, 'monol_test_reb'))
        mp.lags.main(command.split())

    def step08a_savexspec(self):
        """Test save as Xspec 1."""
        command = '{0}'.format(
            os.path.join(datadir, 'monol_testA_E3-50_pds_rebin2') +
            MP_FILE_EXTENSION)
        mp.save_as_xspec.main(command.split())

    def step08b_savexspec(self):
        """Test save as Xspec 2."""
        command = '{0}'.format(
            os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') +
            MP_FILE_EXTENSION)
        mp.save_as_xspec.main(command.split())

    def step09a_create_gti(self):
        """Test creating a GTI file."""
        fname = os.path.join(datadir, 'monol_testA_E3-50_lc_rebin4') + \
            MP_FILE_EXTENSION
        command = "{0} -f lc>0 -c --debug".format(fname)
        mp.create_gti.main(command.split())

    def step09b_create_gti(self):
        """Test applying a GTI file."""
        fname = os.path.join(datadir, 'monol_testA_E3-50_rebin4_gti') + \
            MP_FILE_EXTENSION
        lcfname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        command = "{0} -a {1} --debug".format(lcfname, fname)
        mp.create_gti.main(command.split())

    def step09c_create_gti(self):
        """Test creating a GTI file and apply minimum length."""
        fname = os.path.join(datadir, 'monol_testA_E3-50_lc_rebin4') + \
            MP_FILE_EXTENSION
        command = "{0} -f lc>0 -c -l 10 --debug".format(fname)
        mp.create_gti.main(command.split())

    def step09d_create_gti(self):
        """Test applying a GTI file and apply minimum length."""
        fname = os.path.join(datadir, 'monol_testA_E3-50_rebin4_gti') + \
            MP_FILE_EXTENSION
        lcfname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        command = "{0} -a {1} -l 10 --debug".format(lcfname, fname)
        mp.create_gti.main(command.split())

    def step10a_readfile(self):
        """Test reading and dumping a MaLTPyNT file."""
        fname = os.path.join(datadir, 'monol_testA_E3-50_rebin4_gti') + \
            MP_FILE_EXTENSION
        command = "{0}".format(fname)

        mp.io.main(command.split())

    def step10b_readfile(self):
        """Test reading and dumping a FITS file."""
        fitsname = os.path.join(datadir, 'monol_testA.evt')
        command = "{0}".format(fitsname)

        mp.io.main(command.split())

    def step10c_save_as_qdp(self):
        """Test saving arrays in a qdp file."""
        arrays = [np.array([0, 1, 3]), np.array([1, 4, 5])]
        errors = [np.array([1, 1, 1]), np.array([[1, 0.5], [1, 0.5], [1, 1]])]
        mp.io.save_as_qdp(arrays, errors,
                          filename=os.path.join(datadir,
                                                "monol_test_qdp.txt"))

    def step10d_save_as_ascii(self):
        """Test saving arrays in a ascii file."""
        array = np.array([0, 1, 3])
        errors = np.array([1, 1, 1])
        mp.io.save_as_ascii(
            [array, errors],
            filename=os.path.join(datadir, "monol_test.txt"),
            colnames=["array", "err"])

    def step10e_get_file_type(self):
        """Test getting file type."""
        file_list = {'events': 'monol_testA_ev',
                     'lc': 'monol_testA_E3-50_lc',
                     'pds': 'monol_testA_E3-50_pds',
                     'GTI': 'monol_testA_E3-50_rebin4_gti',
                     'cpds': 'monol_test_E3-50_cpds',
                     'rebcpds': 'monol_test_E3-50_cpds_rebin1.03',
                     'rebpds': 'monol_testA_E3-50_pds_rebin1.03',
                     'lag': 'monol_test_lag'}
        for realtype in file_list.keys():
            fname = os.path.join(datadir,
                                 file_list[realtype] + MP_FILE_EXTENSION)
            ftype, _ = mp.io.get_file_type(fname)
            assert ftype == realtype, "File types do not match"

    def step11_exposure(self):
        """Test exposure calculations from unfiltered files."""
        lcname = os.path.join(datadir,
                              'monol_testA_E3-50_lc' + MP_FILE_EXTENSION)
        ufname = os.path.join(datadir, 'monol_testA_uf.evt')
        command = "{0} {1}".format(lcname, ufname)

        mp.exposure.main(command.split())

    def step12a_plot(self):
        """Test plotting with linear axes."""
        pname = os.path.join(datadir, 'monol_testA_E3-50_pds') + \
            MP_FILE_EXTENSION
        cname = os.path.join(datadir, 'monol_test_E3-50_cpds') + \
            MP_FILE_EXTENSION
        lname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        mp.plot.main([pname, cname, lname, '--noplot', '--xlin', '--ylin'])
        mp.plot.main([lname, '--noplot',
                      '--axes', 'time', 'lc', '--xlin', '--ylin'])

    def step12b_plot(self):
        """Test plotting with log axes."""
        pname = os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            MP_FILE_EXTENSION
        cname = os.path.join(datadir, 'monol_test_E3-50_cpds_rebin1.03') + \
            MP_FILE_EXTENSION
        mp.plot.main([pname, cname, '--noplot', '--xlog', '--ylog'])
        mp.plot.main([pname, '--noplot', '--axes', 'pds', 'epds',
                      '--xlin', '--ylin'])

    def step12c_plot(self):
        """Test plotting and saving figure."""
        pname = os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            MP_FILE_EXTENSION
        mp.plot.main([pname, '--noplot', '--figname',
                      os.path.join(datadir,
                                   'monol_testA_E3-50_pds_rebin1.03.png')])

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
                                   '*monol_test*') + MP_FILE_EXTENSION) + \
            glob.glob(os.path.join(datadir,
                                   '*lcurve*') + MP_FILE_EXTENSION) + \
            glob.glob(os.path.join(datadir,
                                   '*lcurve*.txt')) + \
            glob.glob(os.path.join(datadir,
                                   '*.log')) + \
            glob.glob(os.path.join(datadir,
                                   '*monol_test*.dat')) + \
            glob.glob(os.path.join(datadir,
                                   '*monol_test*.png')) + \
            glob.glob(os.path.join(datadir,
                                   '*monol_test*.txt')) + \
            glob.glob(os.path.join(datadir,
                                   'monol_test_fake*.evt'))
        for f in file_list:
            print("Removing " + f)
            os.remove(f)


if __name__ == '__main__':
    unittest.main(verbosity=2)
