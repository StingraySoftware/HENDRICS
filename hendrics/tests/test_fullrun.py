# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import hendrics as hen
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

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION

logging.basicConfig(filename='HEN.log', level=logging.DEBUG, filemode='w')


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
        cls.first_event_file = os.path.join(cls.datadir,
                                            'monol_testA_nustar_fpma_ev' +
                                            HEN_FILE_EXTENSION)

    def test_scripts_are_installed(self):
        """Test only once that command line scripts are installed correctly."""
        fits_file = os.path.join(self.datadir, 'monol_testA.evt')
        command = 'HENreadfile {0}'.format(fits_file)
        sp.check_call(command.split())

    def test_fake_file(self):
        """Test produce a fake event file."""
        fits_file = os.path.join(self.datadir, 'monol_test_fake.evt')
        hen.fake.main(['-o', fits_file, '--instrument', 'FPMB'])
        info = hen.io.print_fits_info(fits_file, hdu=1)
        assert info['Instrument'] == 'FPMB'

    def test_fake_file_from_input_lc(self):
        """Test produce a fake event file from input light curve."""
        lcurve_in = os.path.join(self.datadir, 'lcurveA.fits')
        fits_file = os.path.join(self.datadir, 'monol_test_fake_lc.evt')
        hen.fake.main(['--lc', lcurve_in, '-o', fits_file])

    def test_fake_file_with_deadtime(self):
        """Test produce a fake event file and apply deadtime."""
        fits_file = os.path.join(self.datadir, 'monol_test_fake_lc.evt')
        hen.fake.main(['--deadtime', '2.5e-3',
                      '--ctrate', '2000',
                      '-o', fits_file])

    def test_fake_file_xmm(self):
        """Test produce a fake event file and apply deadtime."""
        fits_file = os.path.join(self.datadir, 'monol_test_fake_lc_xmm.evt')
        hen.fake.main(['--deadtime', '1e-4', '-m', 'XMM', '-i', 'epn',
                      '--ctrate', '2000',
                      '-o', fits_file])
        hdu_list = fits.open(fits_file)
        hdunames = [hdu.name for hdu in hdu_list]
        assert 'STDGTI01' in hdunames
        assert 'STDGTI02' in hdunames
        assert 'STDGTI07' in hdunames

    def test_load_events(self):
        """Test event file reading."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_testA.evt'))
        hen.read_events.main(command.split())
        new_filename = self.first_event_file
        ev = hen.io.load_events(new_filename)
        assert hasattr(ev, 'header')
        assert hasattr(ev, 'gti')

    def test_load_events_with_2_cpus(self):
        """Test event file reading."""
        command = '{0} {1} --nproc 2'.format(
            os.path.join(self.datadir, 'monol_testB.evt'),
            os.path.join(self.datadir, 'monol_testA_timezero.evt'),
            os.path.join(self.datadir, 'monol_test_fake.evt'))
        hen.read_events.main(command.split())

    def test_load_events_split(self):
        """Test event file splitting."""
        command = \
            '{0} -g --min-length 0'.format(
                os.path.join(self.datadir, 'monol_testB.evt'))
        hen.read_events.main(command.split())
        new_filename = os.path.join(self.datadir,
                                    'monol_testB_nustar_fpmb_gti0_ev' +
                                    HEN_FILE_EXTENSION)
        assert os.path.exists(new_filename)

    def test_save_binary_events(self):
        f = self.first_event_file
        hen.binary.main_presto("{} -b 0.1 -e 3 59".format(f).split())

    def test_load_gtis(self):
        """Test loading of GTIs from FITS files."""
        fits_file = os.path.join(self.datadir, 'monol_testA.evt')
        hen.io.load_gtis(fits_file)

    def test_load_events_noclobber(self):
        """Test event file reading w. noclobber option."""
        with catch_warnings() as w:
            command = \
                '{0} --noclobber'.format(
                    os.path.join(self.datadir, 'monol_testB.evt'))
            hen.read_events.main(command.split())
        assert str(w[0].message).strip().endswith(
            "noclobber option used. Skipping"), \
            "Unexpected warning output"

    def test_load_events_xmm(self):
        """Test event file reading."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_test_fake_lc_xmm.evt'))
        hen.read_events.main(command.split())

    def test_calibrate(self):
        """Test event file calibration."""
        command = '{0} -r {1}'.format(
            os.path.join(self.datadir,
                         'monol_testA_nustar_fpma_ev' + HEN_FILE_EXTENSION),
            os.path.join(self.datadir, 'test.rmf'))
        hen.calibrate.main(command.split())
        new_filename = os.path.join(self.datadir,
                                    'monol_testA_nustar_fpma_ev_calib' +
                                    HEN_FILE_EXTENSION)
        assert os.path.exists(new_filename)
        ev = hen.io.load_events(new_filename)
        assert hasattr(ev, 'header')
        assert hasattr(ev, 'gti')
        gti_to_test = hen.io.load_events(self.first_event_file).gti
        assert np.allclose(gti_to_test, ev.gti)

    def test_calibrate_2_cpus(self):
        """Test event file calibration."""
        command = '{0} -r {1} --nproc 2'.format(
            os.path.join(self.datadir,
                         'monol_testB_nustar_fpmb_ev' + HEN_FILE_EXTENSION),
            os.path.join(self.datadir, 'test.rmf'))
        hen.calibrate.main(command.split())
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testB_nustar_fpmb_ev_calib' +
                                           HEN_FILE_EXTENSION))

    def test_save_varen_rms(self):
        fname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_ev_calib' +
                             HEN_FILE_EXTENSION)
        hen.varenergy.main([fname,"-f", "0", "100", "--energy-values",
                            "0.3", "12", "5", "lin", "--rms", "-b", "0.5",
                            "--segment-size", "128"])
        out = hen.base.hen_root(fname) + "_rms" + '.qdp'
        os.path.exists(out)

    def test_save_varen_lag(self):
        fname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_ev_calib' +
                             HEN_FILE_EXTENSION)
        hen.varenergy.main([fname,"-f", "0", "100", "--energy-values",
                            "0.3", "12", "5", "lin", "--lag", "-b", "0.5",
                            "--segment-size", "128"])
        out = hen.base.hen_root(fname) + "_lag" + '.qdp'
        os.path.exists(out)


    def test_lcurve(self):
        """Test light curve production."""
        command = ('{0} -e {1} {2} --safe-interval '
                   '{3} {4}  --nproc 2 -b 0.5 -o {5}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev_calib' +
                         HEN_FILE_EXTENSION),
            3, 50, 100, 300,
            os.path.join(self.datadir, 'monol_testA_E3-50_lc' +
                         HEN_FILE_EXTENSION)
        )
        hen.lcurve.main(command.split())

        new_filename = \
            os.path.join(os.path.join(self.datadir,
                                      'monol_testA_E3-50_lc' +
                                      HEN_FILE_EXTENSION))
        assert os.path.exists(new_filename)
        lc = hen.io.load_lcurve(new_filename)
        assert hasattr(lc, 'header')
        assert hasattr(lc, 'gti')
        gti_to_test = hen.io.load_events(self.first_event_file).gti
        assert np.allclose(gti_to_test, lc.gti)

    def test_save_binary_lc(self):
        f = \
            os.path.join(os.path.join(self.datadir,
                                      'monol_testA_E3-50_lc' +
                                      HEN_FILE_EXTENSION))
        hen.binary.main_presto("{}".format(f).split())

    def test_lcurve_B(self):
        command = ('{0} -e {1} {2} --safe-interval '
                   '{3} {4} -b 0.5 -o {5}').format(
            os.path.join(self.datadir, 'monol_testB_nustar_fpmb_ev_calib' +
                         HEN_FILE_EXTENSION),
            3, 50, 100, 300,
            os.path.join(self.datadir, 'monol_testB_E3-50_lc' +
                         HEN_FILE_EXTENSION))
        hen.lcurve.main(command.split())
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testB_E3-50_lc' +
                                           HEN_FILE_EXTENSION))

    def test_lcurve_from_split_event(self):
        """Test lc reading of split event file."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_testB_nustar_fpmb_gti0_ev' +
                         HEN_FILE_EXTENSION))
        hen.lcurve.main(command.split())
        new_filename = os.path.join(self.datadir,
                                    'monol_testB_nustar_fpmb_gti0_lc' +
                                    HEN_FILE_EXTENSION)
        assert os.path.exists(new_filename)
        lc = hen.io.load_lcurve(new_filename)
        gti_to_test = hen.io.load_events(self.first_event_file).gti[0]
        assert np.allclose(gti_to_test, lc.gti)


    def test_lcurve_split(self):
        """Test lc with gti-split option."""
        command = '{0} {1} -g'.format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev' +
                         HEN_FILE_EXTENSION),
            os.path.join(self.datadir, 'monol_testB_nustar_fpmb_ev' +
                         HEN_FILE_EXTENSION))
        hen.lcurve.main(command.split())
        new_filename = os.path.join(self.datadir,
                                    'monol_testA_nustar_fpma_gti0_lc' +
                                    HEN_FILE_EXTENSION)
        assert os.path.exists(new_filename)
        lc = hen.io.load_lcurve(new_filename)
        gti_to_test = hen.io.load_events(self.first_event_file).gti[0]
        assert np.allclose(gti_to_test, lc.gti)

    def test_fits_lcurve0(self):
        """Test light curves from FITS."""
        lcurve_ftools_orig = os.path.join(self.datadir, 'lcurveA.fits')

        lcurve_ftools = os.path.join(self.datadir,
                                     'lcurve_ftools_lc' +
                                     HEN_FILE_EXTENSION)

        command = "{0} --outfile {1}".format(
            os.path.join(self.datadir,
                         'monol_testA_nustar_fpma_ev') + HEN_FILE_EXTENSION,
            os.path.join(self.datadir,
                         'lcurve_lc'))
        hen.lcurve.main(command.split())
        assert os.path.exists(os.path.join(self.datadir,
                              'lcurve_lc') + HEN_FILE_EXTENSION)

        command = "--fits-input {0} --outfile {1}".format(
            lcurve_ftools_orig,
            lcurve_ftools)
        hen.lcurve.main(command.split())

    def test_fits_lcurve1(self):
        """Test light curves from FITS."""
        lcurve_ftools = os.path.join(self.datadir,
                                     'lcurve_ftools_lc' +
                                     HEN_FILE_EXTENSION)

        lcurve_mp = os.path.join(self.datadir,
                                 'lcurve_lc' +
                                 HEN_FILE_EXTENSION)

        lcdata_mp = hen.io.load_data(lcurve_mp)
        lcdata_ftools = hen.io.load_data(lcurve_ftools)

        lc_mp = lcdata_mp['counts']

        lenmp = len(lc_mp)
        lc_ftools = lcdata_ftools['counts']
        lenftools = len(lc_ftools)
        goodlen = min([lenftools, lenmp])

        diff = lc_mp[:goodlen] - lc_ftools[:goodlen]

        assert np.all(np.abs(diff) <= 1e-3), \
            'Light curve data do not coincide between FITS and HEN'

    def test_txt_lcurve(self):
        """Test light curves from txt."""
        lcurve_mp = os.path.join(self.datadir,
                                 'lcurve_lc' +
                                 HEN_FILE_EXTENSION)
        lcdata_mp = hen.io.load_data(lcurve_mp)
        lc_mp = lcdata_mp['counts']
        time_mp = lcdata_mp['time']

        lcurve_txt_orig = os.path.join(self.datadir,
                                       'lcurve_txt_lc.txt')

        hen.io.save_as_ascii([time_mp, lc_mp], lcurve_txt_orig)

        lcurve_txt = os.path.join(self.datadir,
                                  'lcurve_txt_lc' +
                                  HEN_FILE_EXTENSION)
        hen.lcurve.main(['--txt-input', lcurve_txt_orig,
                        '--outfile', lcurve_txt])
        lcdata_txt = hen.io.load_data(lcurve_txt)

        lc_txt = lcdata_txt['counts']

        assert np.all(np.abs(lc_mp - lc_txt) <= 1e-3), \
            'Light curve data do not coincide between txt and HEN'

    def test_joinlcs(self):
        """Test produce joined light curves."""
        new_filename = os.path.join(
            self.datadir, 'monol_test_joinlc' + HEN_FILE_EXTENSION)
        #because join_lightcurves separates by instrument
        new_actual_filename = os.path.join(
            self.datadir, 'FPMAmonol_test_joinlc' + HEN_FILE_EXTENSION)
        hen.lcurve.join_lightcurves(
            glob.glob(os.path.join(self.datadir,
                                   'monol_testA_nustar_fpma_gti[0-9]_lc*')) +
            glob.glob(os.path.join(self.datadir,
                                   'monol_testB_nustar_fpmb_gti[0-9]_lc*')),
            new_filename)

        lc = hen.io.load_lcurve(new_actual_filename)
        assert hasattr(lc, 'gti')
        gti_to_test = hen.io.load_events(self.first_event_file).gti
        assert np.allclose(gti_to_test, lc.gti)

    def test_scrunchlcs(self):
        """Test produce scrunched light curves."""
        a_in = os.path.join(self.datadir, 'monol_testA_E3-50_lc' + \
            HEN_FILE_EXTENSION)
        b_in = os.path.join(self.datadir, 'monol_testB_E3-50_lc' + \
            HEN_FILE_EXTENSION)
        out = os.path.join(self.datadir, 'monol_test_scrunchlc' + \
            HEN_FILE_EXTENSION)
        command = '{0} {1} -o {2}'.format(a_in, b_in, out)

        hen.lcurve.scrunch_main(command.split())
        a_lc = hen.io.load_lcurve(a_in)
        b_lc = hen.io.load_lcurve(b_in)
        out_lc = hen.io.load_lcurve(out)
        assert np.all(out_lc.counts == a_lc.counts + b_lc.counts)
        gti_to_test = hen.io.load_events(self.first_event_file).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def testbaselinelc(self):
        """Test produce scrunched light curves."""
        a_in = os.path.join(self.datadir, 'monol_testA_E3-50_lc' + \
            HEN_FILE_EXTENSION)
        out = os.path.join(self.datadir, 'monol_test_baselc')
        command = '{0} -o {1} -p 0.001 --lam 1e5'.format(a_in, out)

        hen.lcurve.baseline_main(command.split())
        out_lc = hen.io.load_lcurve(out + '_0' + HEN_FILE_EXTENSION)
        assert hasattr(out_lc, 'base')
        gti_to_test = hen.io.load_events(self.first_event_file).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def testbaselinelc_nooutroot(self):
        """Test produce scrunched light curves."""
        a_in = os.path.join(self.datadir, 'monol_testA_E3-50_lc' + \
            HEN_FILE_EXTENSION)
        command = '{0} -p 0.001 --lam 1e5'.format(a_in)

        hen.lcurve.baseline_main(command.split())
        out_lc = hen.io.load_lcurve(hen.base.hen_root(a_in) + '_lc_baseline' +
                                    HEN_FILE_EXTENSION)
        assert hasattr(out_lc, 'base')
        gti_to_test = hen.io.load_events(self.first_event_file).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def test_lcurve_error_uncalibrated(self):
        """Test light curve error from uncalibrated file."""
        command = ('{0} -e {1} {2}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev' +
                         HEN_FILE_EXTENSION), 3, 50)

        with pytest.raises(ValueError) as excinfo:
            hen.lcurve.main(command.split())
        message = str(excinfo.value)
        assert str(message).strip().endswith("Did you run HENcalibrate?")

    def test_lcurve_pi_filtering(self):
        """Test light curve using PI filtering."""
        command = ('{0} --pi-interval {1} {2}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev' +
                         HEN_FILE_EXTENSION), 10, 300)

        hen.lcurve.main(command.split())

    def test_colors_fail_uncalibrated(self):
        """Test light curve using PI filtering."""
        command = ('{0} -b 100 -e {1} {2} {2} {3}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev' +
                         HEN_FILE_EXTENSION), 3, 5, 10)
        with pytest.raises(ValueError) as excinfo:
            hen.colors.main(command.split())

        assert "No energy information is present " in str(excinfo.value)

    def test_colors(self):
        """Test light curve using PI filtering."""
        # calculate colors
        command = ('{0} -b 100 -e {1} {2} {2} {3}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev_calib' +
                         HEN_FILE_EXTENSION), 3, 5, 10)
        hen.colors.main(command.split())

        new_filename = \
            os.path.join(self.datadir,
                         'monol_testA_nustar_fpma_E_10-5_over_5-3' +
                         HEN_FILE_EXTENSION)
        assert os.path.exists(new_filename)
        out_lc = hen.io.load_lcurve(new_filename)
        gti_to_test = hen.io.load_events(self.first_event_file).gti
        assert np.allclose(gti_to_test, out_lc.gti)


    def test_pds(self):
        """Test PDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k PDS --norm frac --nproc 2 '.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                HEN_FILE_EXTENSION)
        hen.fspec.main(command.split())

        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testB_E3-50_pds')
                              + HEN_FILE_EXTENSION)
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testA_E3-50_pds')
                              + HEN_FILE_EXTENSION)

    def test_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(self.datadir,
                                     'lcurve_ftools_lc' +
                                     HEN_FILE_EXTENSION)
        command = '{0} -f 128'.format(lcurve_ftools)
        hen.fspec.main(command.split())

    def test_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(self.datadir,
                                  'lcurve_txt_lc' +
                                  HEN_FILE_EXTENSION)
        command = '{0} -f 128'.format(lcurve_txt)
        hen.fspec.main(command.split())

    def test_cpds(self):
        """Test CPDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k CPDS --norm frac -o {2}'.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))
        hen.fspec.main(command.split())

    def test_cpds_2cpus(self):
        """Test CPDS production."""
        command = \
            ('{0} {1} -f 128 --save-dyn -k '
             'CPDS --norm frac -o {2} --nproc 2').format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))
        hen.fspec.main(command.split())

    # def test_dumpdynpds(self):
    #     """Test dump dynamical PDSs."""
    #     command = '--noplot ' + \
    #         os.path.join(self.datadir,
    #                      'monol_testA_E3-50_pds') + \
    #         HEN_FILE_EXTENSION
    #     hen.fspec.dumpdyn_main(command.split())

    def test_sumpds(self):
        """Test the sum of pdss."""
        hen.sum_fspec.main([
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds') + HEN_FILE_EXTENSION,
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds') + HEN_FILE_EXTENSION,
            '-o', os.path.join(self.datadir,
                               'monol_test_sum' + HEN_FILE_EXTENSION)])

    # def test_dumpdyncpds(self):
    #     """Test dumping CPDS file."""
    #     command = '--noplot ' + \
    #         os.path.join(self.datadir,
    #                      'monol_test_E3-50_cpds') + \
    #         HEN_FILE_EXTENSION
    #     hen.fspec.dumpdyn_main(command.split())

    def test_rebinlc(self):
        """Test LC rebinning."""
        command = '{0} -r 4'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())

    def test_rebinpds(self):
        """Test PDS rebinning 1."""
        command = '{0} -r 2'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin2' +
                                    HEN_FILE_EXTENSION))

    def test_rebinpds_geom(self):
        """Test geometrical PDS rebinning."""
        command = '{0} {1} -r 1.03'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds') +
            HEN_FILE_EXTENSION,
            os.path.join(self.datadir, 'monol_testB_E3-50_pds') +
            HEN_FILE_EXTENSION
            )
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin1.03' +
                                    HEN_FILE_EXTENSION))
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testB_E3-50_pds_rebin1.03' +
                                    HEN_FILE_EXTENSION))

    def test_rebincpds(self):
        """Test CPDS rebinning."""
        command = '{0} -r 2'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin2' +
                                    HEN_FILE_EXTENSION))

    def test_rebincpds_geom(self):
        """Test CPDS geometrical rebinning."""
        command = '{0} -r 1.03'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03' +
                                    HEN_FILE_EXTENSION))

    def test_save_lags(self):
        fname = os.path.join(self.datadir,
                             'monol_test_E3-50_cpds_rebin2' +
                             HEN_FILE_EXTENSION)
        hen.timelags.main([fname])
        out = hen.base.hen_root(fname) + '_lags.qdp'
        os.path.exists(out)

    def test_save_fvar(self):
        fname = os.path.join(self.datadir,
                             'monol_testA_E3-50_lc' +
                             HEN_FILE_EXTENSION)
        hen.exvar.main([fname,"-c", "10", "--fraction-step", "0.6",
                        "--norm", "fvar"])
        out = hen.base.hen_root(fname) + "_fvar" + '.qdp'
        os.path.exists(out)

    def test_save_excvar(self):
        fname = os.path.join(self.datadir,
                             'monol_testA_E3-50_lc' +
                             HEN_FILE_EXTENSION)
        hen.exvar.main([fname])
        out = hen.base.hen_root(fname) + "_excvar" + '.qdp'
        os.path.exists(out)

    def test_save_excvar_norm(self):
        fname = os.path.join(self.datadir,
                             'monol_testA_E3-50_lc' +
                             HEN_FILE_EXTENSION)
        hen.exvar.main([fname, "--norm", "norm_excvar"])
        out = hen.base.hen_root(fname) + "_norm_excvar" + '.qdp'
        os.path.exists(out)

    def test_save_excvar_wrong_norm(self):
        fname = os.path.join(self.datadir,
                             'monol_testA_E3-50_lc' +
                             HEN_FILE_EXTENSION)
        with pytest.raises(ValueError) as excinfo:
            hen.exvar.main([fname, '--norm', 'cicciput'])
        assert 'Normalization must be fvar, ' in str(excinfo.value)

    def test_fit_pds(self):
        modelstring = '''
from astropy.modeling import models
model = models.Const1D()
        '''
        modelfile = 'bubu__model__.py'
        with open(modelfile, 'w') as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = \
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds' + HEN_FILE_EXTENSION)
        pdsfile2 = \
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds' + HEN_FILE_EXTENSION)

        command = '{0} {1} -m {2} --frequency-interval 0 10'.format(pdsfile1,
                                                                    pdsfile2,
                                                                    modelfile)
        hen.modeling.main_model(command.split())

        out0 = os.path.join(self.datadir, 'monol_testA_E3-50_pds_bestfit.p')
        out1 = os.path.join(self.datadir, 'monol_testB_E3-50_pds_bestfit.p')
        assert os.path.exists(out0)
        assert os.path.exists(out1)
        m, k, c = hen.io.load_model(
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds_bestfit.p'))
        assert hasattr(m, 'amplitude')
        os.unlink(out0)
        os.unlink(out1)

    def test_fit_pds_f_no_of_intervals_invalid(self):
        modelstring = '''
from astropy.modeling import models
model = models.Const1D()
        '''
        modelfile = 'bubu__model__.py'
        with open(modelfile, 'w') as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = \
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds' + HEN_FILE_EXTENSION)
        pdsfile2 = \
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds' + HEN_FILE_EXTENSION)

        command = '{0} {1} -m {2} --frequency-interval 0 1 9'.format(pdsfile1,
                                                                     pdsfile2,
                                                                     modelfile)
        with pytest.raises(ValueError) as excinfo:
            hen.modeling.main_model(command.split())

        assert "Invalid number of frequencies specified" in str(excinfo.value)

    # def test_dumpdyncpds_reb(self):
    #     """Test dumping rebinned CPDS file."""
    #     command = '--noplot ' + \
    #         os.path.join(self.datadir,
    #                      'monol_test_E3-50_cpds_rebin1.03') + \
    #         HEN_FILE_EXTENSION
    #     hen.fspec.dumpdyn_main(command.split())

    def test_savexspec(self):
        """Test save as Xspec 1."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds_rebin2') +
            HEN_FILE_EXTENSION)
        hen.save_as_xspec.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin2.pha'))

    def test_savexspec_geom(self):
        """Test save as Xspec 2."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds_rebin1.03') +
            HEN_FILE_EXTENSION)
        hen.save_as_xspec.main(command.split())

        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03.pha'))
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03_lags.pha'))

    def test_create_gti(self):
        """Test creating a GTI file."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_lc_rebin4') + \
            HEN_FILE_EXTENSION
        command = "{0} -f counts>0 -c --debug".format(fname)
        hen.create_gti.main(command.split())

    def test_apply_gti(self):
        """Test applying a GTI file."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            HEN_FILE_EXTENSION
        lcfname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            HEN_FILE_EXTENSION
        command = "{0} -a {1} --debug".format(lcfname, fname)
        hen.create_gti.main(command.split())

    def test_create_gti_and_minlen(self):
        """Test creating a GTI file and apply minimum length."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_lc_rebin4') + \
            HEN_FILE_EXTENSION
        command = "{0} -f counts>0 -c -l 10 --debug".format(fname)
        hen.create_gti.main(command.split())

    def test_create_gti_and_apply(self):
        """Test applying a GTI file and apply minimum length."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            HEN_FILE_EXTENSION
        lcfname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            HEN_FILE_EXTENSION
        command = "{0} -a {1} -l 10 --debug".format(lcfname, fname)
        hen.create_gti.main(command.split())

    def test_readfile(self):
        """Test reading and dumping a HENDRICS file."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            HEN_FILE_EXTENSION
        command = "{0}".format(fname)

        hen.io.main(command.split())

    def test_readfile_fits(self):
        """Test reading and dumping a FITS file."""
        fitsname = os.path.join(self.datadir, 'monol_testA.evt')
        command = "{0}".format(fitsname)

        hen.io.main(command.split())

    def test_save_as_qdp(self):
        """Test saving arrays in a qdp file."""
        arrays = [np.array([0, 1, 3]), np.array([1, 4, 5])]
        errors = [np.array([1, 1, 1]), np.array([[1, 0.5], [1, 0.5], [1, 1]])]
        hen.io.save_as_qdp(arrays, errors,
                          filename=os.path.join(self.datadir,
                                                "monol_test_qdp.txt"))
        hen.io.save_as_qdp(arrays, errors,
                          filename=os.path.join(self.datadir,
                                                "monol_test_qdp.txt"),
                          mode='a')

    def test_save_as_ascii(self):
        """Test saving arrays in a ascii file."""
        array = np.array([0, 1, 3])
        errors = np.array([1, 1, 1])
        hen.io.save_as_ascii(
            [array, errors],
            filename=os.path.join(self.datadir, "monol_test.txt"),
            colnames=["array", "err"])

    def test_get_file_type(self):
        """Test getting file type."""
        file_list = {'events': 'monol_testA_nustar_fpma_ev',
                     'lc': 'monol_testA_E3-50_lc',
                     'pds': 'monol_testA_E3-50_pds',
                     'gti': 'monol_testA_E3-50_rebin4_gti',
                     'cpds': 'monol_test_E3-50_cpds'}
        for realtype in file_list.keys():
            fname = os.path.join(self.datadir,
                                 file_list[realtype] + HEN_FILE_EXTENSION)
            ftype, _ = hen.io.get_file_type(fname)
            assert ftype == realtype, "File types do not match"

    def test_exposure(self):
        """Test exposure calculations from unfiltered files."""
        lcname = os.path.join(self.datadir,
                              'monol_testA_E3-50_lc' + HEN_FILE_EXTENSION)
        ufname = os.path.join(self.datadir, 'monol_testA_uf.evt')
        command = "{0} {1}".format(lcname, ufname)

        hen.exposure.main(command.split())
        fname = os.path.join(self.datadir,
                             'monol_testA_E3-50_lccorr'  + HEN_FILE_EXTENSION)
        assert os.path.exists(fname)
        ftype, contents = hen.io.get_file_type(fname)

        assert isinstance(contents, Lightcurve)
        assert hasattr(contents, 'expo')

    def test_plot_lin(self):
        """Test plotting with linear axes."""
        pname = os.path.join(self.datadir, 'monol_testA_E3-50_pds') + \
            HEN_FILE_EXTENSION
        cname = os.path.join(self.datadir, 'monol_test_E3-50_cpds') + \
            HEN_FILE_EXTENSION
        lname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            HEN_FILE_EXTENSION
        hen.plot.main([pname, cname, lname, '--noplot', '--xlin', '--ylin',
                      '-o', 'dummy.qdp'])
        hen.plot.main([lname, '--noplot',
                      '--axes', 'time', 'counts', '--xlin', '--ylin',
                      '-o', 'dummy.qdp'])

    def test_plot_lcurve_baseline(self):
        a_in = os.path.join(self.datadir, 'monol_testA_E3-50_lc' + \
            HEN_FILE_EXTENSION)
        base_file = hen.base.hen_root(a_in) + '_lc_baseline' + \
            HEN_FILE_EXTENSION
        hen.plot.main([base_file, '--noplot', '-o', 'dummy_base.qdp'])
        filedata = np.genfromtxt('dummy_base.qdp')

        assert filedata.shape[1] == 3

    def test_plot_log(self):
        """Test plotting with log axes."""
        pname = os.path.join(self.datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            HEN_FILE_EXTENSION
        cname = os.path.join(self.datadir, 'monol_test_E3-50_cpds_rebin1.03') + \
            HEN_FILE_EXTENSION
        hen.plot.main([pname, cname, '--noplot', '--xlog', '--ylog',
                      '-o', 'dummy.qdp'])
        hen.plot.main([pname, '--noplot', '--axes', 'power', 'power_err',
                      '--xlin', '--ylin',
                      '-o', 'dummy.qdp'])

    def test_plot_save_figure(self):
        """Test plotting and saving figure."""
        pname = os.path.join(self.datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            HEN_FILE_EXTENSION
        hen.plot.main([pname, '--noplot', '--figname',
                      os.path.join(self.datadir,
                                   'monol_testA_E3-50_pds_rebin1.03.png'),
                      '-o', 'dummy.qdp'])

    def test_plot_color(self):
        """Test plotting with linear axes."""
        lname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            HEN_FILE_EXTENSION
        cname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            HEN_FILE_EXTENSION
        hen.plot.main([cname, lname, '--noplot', '--xlog', '--ylog', '--CCD',
                      '-o', 'dummy.qdp'])

    def test_plot_hid(self):
        """Test plotting with linear axes."""
        # also produce a light curve with the same binning
        command = ('{0} -b 100 --e-interval {1} {2}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev_calib' +
                         HEN_FILE_EXTENSION), 3, 10)

        hen.lcurve.main(command.split())
        lname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E3-10_lc') + \
            HEN_FILE_EXTENSION
        os.path.exists(lname)
        cname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            HEN_FILE_EXTENSION
        hen.plot.main([cname, lname, '--noplot', '--xlog', '--ylog', '--HID',
                      '-o', 'dummy.qdp'])

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        file_list = \
            glob.glob(os.path.join(self.datadir,
                                   '*monol_test*') + HEN_FILE_EXTENSION) + \
            glob.glob(os.path.join(self.datadir,
                                   '*lcurve*') + HEN_FILE_EXTENSION) + \
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
