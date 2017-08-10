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
from astropy.io import fits
import pytest
from stingray import Lightcurve
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION

logging.basicConfig(filename='MP.log', level=logging.DEBUG, filemode='w')


class TestFullRun(object):
    """Test how command lines work.

    Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/python-unittest-testcase-execution-order

    When command line is missing, uses some function calls
    """  # NOQA
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, 'data')

    def test_00a_scripts_are_installed(self):
        """Test only once that command line scripts are installed correctly."""
        fits_file = os.path.join(self.datadir, 'monol_testA.evt')
        command = 'MPreadfile {0}'.format(fits_file)
        sp.check_call(command.split())

    def test_00b_scripts_are_installed(self):
        """Test only once that command line scripts are installed correctly."""
        fits_file = os.path.join(self.datadir, 'monol_testA.evt')
        command = 'HENreadfile {0}'.format(fits_file)
        sp.check_call(command.split())

    def test_01a_fake_file(self):
        """Test produce a fake event file."""
        fits_file = os.path.join(self.datadir, 'monol_test_fake.evt')
        mp.fake.main(['-o', fits_file, '--instrument', 'FPMB'])
        info = mp.io.print_fits_info(fits_file, hdu=1)
        assert info['Instrument'] == 'FPMB'

    def test_01b_fake_file(self):
        """Test produce a fake event file from input light curve."""
        lcurve_in = os.path.join(self.datadir, 'lcurveA.fits')
        fits_file = os.path.join(self.datadir, 'monol_test_fake_lc.evt')
        mp.fake.main(['--lc', lcurve_in, '-o', fits_file])

    def test_01c_fake_file(self):
        """Test produce a fake event file and apply deadtime."""
        fits_file = os.path.join(self.datadir, 'monol_test_fake_lc.evt')
        mp.fake.main(['--deadtime', '2.5e-3',
                      '--ctrate', '2000',
                      '-o', fits_file])

    def test_01d_fake_file_xmm(self):
        """Test produce a fake event file and apply deadtime."""
        fits_file = os.path.join(self.datadir, 'monol_test_fake_lc_xmm.evt')
        mp.fake.main(['--deadtime', '1e-4', '-m', 'XMM', '-i', 'epn',
                      '--ctrate', '2000',
                      '-o', fits_file])
        hdu_list = fits.open(fits_file)
        hdunames = [hdu.name for hdu in hdu_list]
        assert 'STDGTI01' in hdunames
        assert 'STDGTI02' in hdunames
        assert 'STDGTI07' in hdunames

    def test_02a_load_events(self):
        """Test event file reading."""
        command = '{0} {1} --nproc 2'.format(
            os.path.join(self.datadir, 'monol_testA.evt'),
            os.path.join(self.datadir, 'monol_testA_timezero.evt'),
            os.path.join(self.datadir, 'monol_test_fake.evt'))
        mp.read_events.main(command.split())

    def test_02b_load_events(self):
        """Test event file reading."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_testB.evt'))
        mp.read_events.main(command.split())

    def test_02c_load_events_split(self):
        """Test event file splitting."""
        command = \
            '{0} -g --min-length 0'.format(
                os.path.join(self.datadir, 'monol_testB.evt'))
        mp.read_events.main(command.split())
        new_filename = os.path.join(self.datadir,
                                    'monol_testB_nustar_fpmb_gti0_ev' +
                                    MP_FILE_EXTENSION)
        assert os.path.exists(new_filename)

    def test_02d_load_gtis(self):
        """Test loading of GTIs from FITS files."""
        fits_file = os.path.join(self.datadir, 'monol_testA.evt')
        mp.io.load_gtis(fits_file)

    def test_02e_load_events_noclobber(self):
        """Test event file reading w. noclobber option."""
        with catch_warnings() as w:
            command = \
                '{0} --noclobber'.format(
                    os.path.join(self.datadir, 'monol_testB.evt'))
            mp.read_events.main(command.split())
        assert str(w[0].message).strip().endswith(
            "noclobber option used. Skipping"), \
            "Unexpected warning output"

    def test_02b_load_events_xmm(self):
        """Test event file reading."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_test_fake_lc_xmm.evt'))
        mp.read_events.main(command.split())

    def test_03a_calibrate(self):
        """Test event file calibration."""
        command = '{0} -r {1}'.format(
            os.path.join(self.datadir,
                         'monol_testA_nustar_fpma_ev' + MP_FILE_EXTENSION),
            os.path.join(self.datadir, 'test.rmf'))
        mp.calibrate.main(command.split())
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testA_nustar_fpma_ev_calib' +
                                           MP_FILE_EXTENSION))

    def test_03b_calibrate(self):
        """Test event file calibration."""
        command = '{0} -r {1} --nproc 2'.format(
            os.path.join(self.datadir,
                         'monol_testB_nustar_fpmb_ev' + MP_FILE_EXTENSION),
            os.path.join(self.datadir, 'test.rmf'))
        mp.calibrate.main(command.split())
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testB_nustar_fpmb_ev_calib' +
                                           MP_FILE_EXTENSION))
    def test_lcurve(self):
        """Test light curve production."""
        command = ('{0} -e {1} {2} --safe-interval '
                   '{3} {4}  --nproc 2 -b 0.5 -o {5}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev_calib' +
                         MP_FILE_EXTENSION),
            3, 50, 100, 300,
            os.path.join(self.datadir, 'monol_testA_E3-50_lc' +
                         MP_FILE_EXTENSION)
        )
        mp.lcurve.main(command.split())

        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testA_E3-50_lc' +
                                           MP_FILE_EXTENSION))

        command = ('{0} -e {1} {2} --safe-interval '
                   '{3} {4} -b 0.5 -o {5}').format(
            os.path.join(self.datadir, 'monol_testB_nustar_fpmb_ev_calib' +
                         MP_FILE_EXTENSION),
            3, 50, 100, 300,
            os.path.join(self.datadir, 'monol_testB_E3-50_lc' +
                         MP_FILE_EXTENSION))
        mp.lcurve.main(command.split())
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testB_E3-50_lc' +
                                           MP_FILE_EXTENSION))

    def test_lcurve_split(self):
        """Test lc with gti-split option, and reading of split event file."""
        command = '{0} -g'.format(
            os.path.join(self.datadir, 'monol_testB_nustar_fpmb_gti0_ev' +
                         MP_FILE_EXTENSION))
        mp.lcurve.main(command.split())

    def test_fits_lcurve0(self):
        """Test light curves from FITS."""
        lcurve_ftools_orig = os.path.join(self.datadir, 'lcurveA.fits')

        lcurve_ftools = os.path.join(self.datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)

        command = "{0} --outfile {1}".format(
            os.path.join(self.datadir,
                         'monol_testA_nustar_fpma_ev') + MP_FILE_EXTENSION,
            os.path.join(self.datadir,
                         'lcurve_lc'))
        mp.lcurve.main(command.split())
        print(glob.glob(os.path.join(self.datadir,
                              'lcurve_lc*')))
        assert os.path.exists(os.path.join(self.datadir,
                              'lcurve_lc') + MP_FILE_EXTENSION)

        command = "--fits-input {0} --outfile {1}".format(
            lcurve_ftools_orig,
            lcurve_ftools)
        mp.lcurve.main(command.split())

    def test_fits_lcurve1(self):
        """Test light curves from FITS."""
        lcurve_ftools = os.path.join(self.datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)

        lcurve_mp = os.path.join(self.datadir,
                                 'lcurve_lc' +
                                 MP_FILE_EXTENSION)

        lcdata_mp = mp.io.load_data(lcurve_mp)
        lcdata_ftools = mp.io.load_data(lcurve_ftools)

        lc_mp = lcdata_mp['counts']

        lenmp = len(lc_mp)
        lc_ftools = lcdata_ftools['counts']
        lenftools = len(lc_ftools)
        goodlen = min([lenftools, lenmp])

        diff = lc_mp[:goodlen] - lc_ftools[:goodlen]

        assert np.all(np.abs(diff) <= 1e-3), \
            'Light curve data do not coincide between FITS and MP'

    def test_txt_lcurve(self):
        """Test light curves from txt."""
        lcurve_mp = os.path.join(self.datadir,
                                 'lcurve_lc' +
                                 MP_FILE_EXTENSION)
        lcdata_mp = mp.io.load_data(lcurve_mp)
        lc_mp = lcdata_mp['counts']
        time_mp = lcdata_mp['time']

        lcurve_txt_orig = os.path.join(self.datadir,
                                       'lcurve_txt_lc.txt')

        mp.io.save_as_ascii([time_mp, lc_mp], lcurve_txt_orig)

        lcurve_txt = os.path.join(self.datadir,
                                  'lcurve_txt_lc' +
                                  MP_FILE_EXTENSION)
        mp.lcurve.main(['--txt-input', lcurve_txt_orig,
                        '--outfile', lcurve_txt])
        lcdata_txt = mp.io.load_data(lcurve_txt)

        lc_txt = lcdata_txt['counts']

        assert np.all(np.abs(lc_mp - lc_txt) <= 1e-3), \
            'Light curve data do not coincide between txt and MP'

    def test_joinlcs(self):
        """Test produce joined light curves."""
        mp.lcurve.join_lightcurves(
            [os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
             MP_FILE_EXTENSION,
             os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
             MP_FILE_EXTENSION],
            os.path.join(self.datadir, 'monol_test_joinlc' +
                         MP_FILE_EXTENSION))

    def test_scrunchlcs(self):
        """Test produce scrunched light curves."""
        command = '{0} {1} -o {2}'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
            MP_FILE_EXTENSION,
            os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
            MP_FILE_EXTENSION,
            os.path.join(self.datadir, 'monol_test_scrunchlc') +
            MP_FILE_EXTENSION)
        mp.lcurve.scrunch_main(command.split())

    def test_lcurve_error_uncalibrated(self):
        """Test light curve error from uncalibrated file."""
        command = ('{0} -e {1} {2}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev' +
                         MP_FILE_EXTENSION), 3, 50)

        with pytest.raises(ValueError) as excinfo:
            mp.lcurve.main(command.split())
        message = str(excinfo.value)
        assert str(message).strip().endswith("Did you run MPcalibrate?")

    def test_lcurve_pi_filtering(self):
        """Test light curve using PI filtering."""
        command = ('{0} --pi-interval {1} {2}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev' +
                         MP_FILE_EXTENSION), 10, 300)

        mp.lcurve.main(command.split())

    def test_colors_fail_uncalibrated(self):
        """Test light curve using PI filtering."""
        command = ('{0} -b 100 -e {1} {2} {2} {3}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev' +
                         MP_FILE_EXTENSION), 3, 5, 10)
        with pytest.raises(ValueError) as excinfo:
            mp.colors.main(command.split())

        assert "No energy information is present " in str(excinfo.value)

    def test_colors(self):
        """Test light curve using PI filtering."""
        # calculate colors
        command = ('{0} -b 100 -e {1} {2} {2} {3}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev_calib' +
                         MP_FILE_EXTENSION), 3, 5, 10)
        mp.colors.main(command.split())

        assert os.path.exists(
            os.path.join(self.datadir,
                         'monol_testA_nustar_fpma_E_10-5_over_5-3')
                              + MP_FILE_EXTENSION)


    def test_05a_pds(self):
        """Test PDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k PDS --norm frac --nproc 2 '.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION)
        mp.fspec.main(command.split())

        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testB_E3-50_pds')
                              + MP_FILE_EXTENSION)
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testA_E3-50_pds')
                              + MP_FILE_EXTENSION)

    def test_05c_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(self.datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)
        command = '{0} -f 128'.format(lcurve_ftools)
        mp.fspec.main(command.split())

    def test_05d_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(self.datadir,
                                  'lcurve_txt_lc' +
                                  MP_FILE_EXTENSION)
        command = '{0} -f 128'.format(lcurve_txt)
        mp.fspec.main(command.split())

    def test_05e_cpds(self):
        """Test CPDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k CPDS --norm frac -o {2}'.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))
        mp.fspec.main(command.split())

    def test_05f_cpds(self):
        """Test CPDS production."""
        command = \
            ('{0} {1} -f 128 --save-dyn -k '
             'CPDS --norm frac -o {2} --nproc 2').format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                MP_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))
        mp.fspec.main(command.split())

    # def test_05g_dumpdynpds(self):
    #     """Test dump dynamical PDSs."""
    #     command = '--noplot ' + \
    #         os.path.join(self.datadir,
    #                      'monol_testA_E3-50_pds') + \
    #         MP_FILE_EXTENSION
    #     mp.fspec.dumpdyn_main(command.split())

    def test_05h_sumpds(self):
        """Test the sum of pdss."""
        mp.sum_fspec.main([
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds') + MP_FILE_EXTENSION,
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds') + MP_FILE_EXTENSION,
            '-o', os.path.join(self.datadir,
                               'monol_test_sum' + MP_FILE_EXTENSION)])

    # def test_05i_dumpdyncpds(self):
    #     """Test dumping CPDS file."""
    #     command = '--noplot ' + \
    #         os.path.join(self.datadir,
    #                      'monol_test_E3-50_cpds') + \
    #         MP_FILE_EXTENSION
    #     mp.fspec.dumpdyn_main(command.split())

    def test_06a_rebinlc(self):
        """Test LC rebinning."""
        command = '{0} -r 4'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())

    def test_06b_rebinpds(self):
        """Test PDS rebinning 1."""
        command = '{0} -r 2'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin2' +
                                    MP_FILE_EXTENSION))

    def test_06c_rebinpds(self):
        """Test geometrical PDS rebinning."""
        command = '{0} {1} -r 1.03'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds') +
            MP_FILE_EXTENSION,
            os.path.join(self.datadir, 'monol_testB_E3-50_pds') +
            MP_FILE_EXTENSION
            )
        mp.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin1.03' +
                                    MP_FILE_EXTENSION))
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testB_E3-50_pds_rebin1.03' +
                                    MP_FILE_EXTENSION))

    def test_06d_rebincpds(self):
        """Test CPDS rebinning."""
        command = '{0} -r 2'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin2' +
                                    MP_FILE_EXTENSION))

    def test_06e_rebincpds(self):
        """Test CPDS geometrical rebinning."""
        command = '{0} -r 1.03'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds') +
            MP_FILE_EXTENSION)
        mp.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03' +
                                    MP_FILE_EXTENSION))

    def test_07a_fit_pds(self):
        modelstring = '''
from astropy.modeling import models
model = models.Const1D()
        '''
        modelfile = 'bubu__model__.py'
        print(modelstring, file=open(modelfile, 'w'))
        pdsfile1 = \
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds' + MP_FILE_EXTENSION)
        pdsfile2 = \
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds' + MP_FILE_EXTENSION)

        command = '{0} {1} -m {2}'.format(pdsfile1, pdsfile2, modelfile)
        mp.modeling.main(command.split())

        assert os.path.exists(
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds_bestfit.p'))
        assert os.path.exists(
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds_bestfit.p'))

        m, k, c = mp.io.load_model(
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds_bestfit.p'))
        assert hasattr(m, 'amplitude')
    # def test_06f_dumpdyncpds_reb(self):
    #     """Test dumping rebinned CPDS file."""
    #     command = '--noplot ' + \
    #         os.path.join(self.datadir,
    #                      'monol_test_E3-50_cpds_rebin1.03') + \
    #         MP_FILE_EXTENSION
    #     mp.fspec.dumpdyn_main(command.split())

    def test_08a_savexspec(self):
        """Test save as Xspec 1."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds_rebin2') +
            MP_FILE_EXTENSION)
        mp.save_as_xspec.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin2.pha'))

    def test_08b_savexspec(self):
        """Test save as Xspec 2."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds_rebin1.03') +
            MP_FILE_EXTENSION)
        mp.save_as_xspec.main(command.split())

        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03.pha'))
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03_lags.pha'))


    def test_09a_create_gti(self):
        """Test creating a GTI file."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_lc_rebin4') + \
            MP_FILE_EXTENSION
        command = "{0} -f counts>0 -c --debug".format(fname)
        mp.create_gti.main(command.split())

    def test_09b_create_gti(self):
        """Test applying a GTI file."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            MP_FILE_EXTENSION
        lcfname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        command = "{0} -a {1} --debug".format(lcfname, fname)
        mp.create_gti.main(command.split())

    def test_09c_create_gti(self):
        """Test creating a GTI file and apply minimum length."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_lc_rebin4') + \
            MP_FILE_EXTENSION
        command = "{0} -f counts>0 -c -l 10 --debug".format(fname)
        mp.create_gti.main(command.split())

    def test_09d_create_gti(self):
        """Test applying a GTI file and apply minimum length."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            MP_FILE_EXTENSION
        lcfname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        command = "{0} -a {1} -l 10 --debug".format(lcfname, fname)
        mp.create_gti.main(command.split())

    def test_10a_readfile(self):
        """Test reading and dumping a MaLTPyNT file."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            MP_FILE_EXTENSION
        command = "{0}".format(fname)

        mp.io.main(command.split())

    def test_10b_readfile(self):
        """Test reading and dumping a FITS file."""
        fitsname = os.path.join(self.datadir, 'monol_testA.evt')
        command = "{0}".format(fitsname)

        mp.io.main(command.split())

    def test_10c_save_as_qdp(self):
        """Test saving arrays in a qdp file."""
        arrays = [np.array([0, 1, 3]), np.array([1, 4, 5])]
        errors = [np.array([1, 1, 1]), np.array([[1, 0.5], [1, 0.5], [1, 1]])]
        mp.io.save_as_qdp(arrays, errors,
                          filename=os.path.join(self.datadir,
                                                "monol_test_qdp.txt"))

    def test_10d_save_as_ascii(self):
        """Test saving arrays in a ascii file."""
        array = np.array([0, 1, 3])
        errors = np.array([1, 1, 1])
        mp.io.save_as_ascii(
            [array, errors],
            filename=os.path.join(self.datadir, "monol_test.txt"),
            colnames=["array", "err"])

    def test_10e_get_file_type(self):
        """Test getting file type."""
        file_list = {'events': 'monol_testA_nustar_fpma_ev',
                     'lc': 'monol_testA_E3-50_lc',
                     'pds': 'monol_testA_E3-50_pds',
                     'gti': 'monol_testA_E3-50_rebin4_gti',
                     'cpds': 'monol_test_E3-50_cpds'}
        for realtype in file_list.keys():
            fname = os.path.join(self.datadir,
                                 file_list[realtype] + MP_FILE_EXTENSION)
            ftype, _ = mp.io.get_file_type(fname)
            assert ftype == realtype, "File types do not match"

    def test_11_exposure(self):
        """Test exposure calculations from unfiltered files."""
        lcname = os.path.join(self.datadir,
                              'monol_testA_E3-50_lc' + MP_FILE_EXTENSION)
        ufname = os.path.join(self.datadir, 'monol_testA_uf.evt')
        command = "{0} {1}".format(lcname, ufname)

        mp.exposure.main(command.split())
        fname = os.path.join(self.datadir,
                             'monol_testA_E3-50_lccorr'  + MP_FILE_EXTENSION)
        assert os.path.exists(fname)
        ftype, contents = mp.io.get_file_type(fname)

        assert isinstance(contents, Lightcurve)
        assert hasattr(contents, 'expo')

    def test_12a_plot(self):
        """Test plotting with linear axes."""
        pname = os.path.join(self.datadir, 'monol_testA_E3-50_pds') + \
            MP_FILE_EXTENSION
        cname = os.path.join(self.datadir, 'monol_test_E3-50_cpds') + \
            MP_FILE_EXTENSION
        lname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        mp.plot.main([pname, cname, lname, '--noplot', '--xlin', '--ylin'])
        mp.plot.main([lname, '--noplot',
                      '--axes', 'time', 'counts', '--xlin', '--ylin'])

    def test_12b_plot(self):
        """Test plotting with log axes."""
        pname = os.path.join(self.datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            MP_FILE_EXTENSION
        cname = os.path.join(self.datadir, 'monol_test_E3-50_cpds_rebin1.03') + \
            MP_FILE_EXTENSION
        mp.plot.main([pname, cname, '--noplot', '--xlog', '--ylog'])
        mp.plot.main([pname, '--noplot', '--axes', 'power', 'power_err',
                      '--xlin', '--ylin'])

    def test_12c_plot(self):
        """Test plotting and saving figure."""
        pname = os.path.join(self.datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            MP_FILE_EXTENSION
        mp.plot.main([pname, '--noplot', '--figname',
                      os.path.join(self.datadir,
                                   'monol_testA_E3-50_pds_rebin1.03.png')])

    def test_plot_color(self):
        """Test plotting with linear axes."""
        lname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            MP_FILE_EXTENSION
        cname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            MP_FILE_EXTENSION
        mp.plot.main([cname, lname, '--noplot', '--xlog', '--ylog', '--CCD'])

    def test_plot_hid(self):
        """Test plotting with linear axes."""
        # also produce a light curve with the same binning
        command = ('{0} -b 100 --e-interval {1} {2}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev_calib' +
                         MP_FILE_EXTENSION), 3, 10)

        mp.lcurve.main(command.split())
        lname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E3-10_lc') + \
            MP_FILE_EXTENSION
        os.path.exists(lname)
        cname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            MP_FILE_EXTENSION
        mp.plot.main([cname, lname, '--noplot', '--xlog', '--ylog', '--HID'])

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        file_list = \
            glob.glob(os.path.join(self.datadir,
                                   '*monol_test*') + MP_FILE_EXTENSION) + \
            glob.glob(os.path.join(self.datadir,
                                   '*lcurve*') + MP_FILE_EXTENSION) + \
            glob.glob(os.path.join(self.datadir,
                                   '*lcurve*.txt')) + \
            glob.glob(os.path.join(self.datadir,
                                   '*.log')) + \
            glob.glob(os.path.join(self.datadir,
                                   '*monol_test*.dat')) + \
            glob.glob(os.path.join(self.datadir,
                                   '*monol_test*.png')) + \
            glob.glob(os.path.join(self.datadir,
                                   '*monol_test*.txt')) + \
            glob.glob(os.path.join(self.datadir,
                                   'monol_test_fake*.evt')) + \
            glob.glob(os.path.join(self.datadir,
                                   'bubu*'))
        for f in file_list:
            print("Removing " + f)
            os.remove(f)
