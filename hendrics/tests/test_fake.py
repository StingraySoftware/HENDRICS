# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import shutil
import os
import glob
import subprocess as sp

import numpy as np
from astropy import log
from astropy.io import fits
import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics import fake, calibrate, read_events, io

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION

log.setLevel('DEBUG')
# log.basicConfig(filename='HEN.log', level=log.DEBUG, filemode='w')


def test_filter_for_deadtime_nonpar():
    """Test dead time filter, non-paralyzable case."""
    events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
    filt_events = hen.fake.filter_for_deadtime(events, 0.11)
    expected = np.array([1, 2, 2.2, 3, 3.2])
    assert np.all(filt_events == expected), \
        "Wrong: {} vs {}".format(filt_events, expected)


def test_filter_for_deadtime_nonpar_bkg():
    """Test dead time filter, non-paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = \
        hen.fake.filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events,
                                     return_all=True)
    expected_ev = np.array([2, 2.2, 3, 3.2])
    expected_bk = np.array([1])
    assert np.all(filt_events == expected_ev), \
        "Wrong: {} vs {}".format(filt_events, expected_ev)
    assert np.all(info.bkg == expected_bk), \
        "Wrong: {} vs {}".format(info.bkg, expected_bk)


def test_filter_for_deadtime_par():
    """Test dead time filter, paralyzable case."""
    events = np.array([1, 1.1, 2, 2.2, 3, 3.1, 3.2])
    assert np.all(hen.fake.filter_for_deadtime(
        events, 0.11, paralyzable=True) == np.array([1, 2, 2.2, 3]))


def test_filter_for_deadtime_par_bkg():
    """Test dead time filter, paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = \
        hen.fake.filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events,
                                     paralyzable=True, return_all=True)
    expected_ev = np.array([2, 2.2, 3])
    expected_bk = np.array([1])
    assert np.all(filt_events == expected_ev), \
        "Wrong: {} vs {}".format(filt_events, expected_ev)
    assert np.all(info.bkg == expected_bk), \
        "Wrong: {} vs {}".format(info.bkg, expected_bk)


def test_deadtime_mask_par():
    """Test dead time filter, paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = \
        hen.fake.filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events,
                                     paralyzable=True, return_all=True)

    assert np.all(filt_events == events[info.mask])


def test_deadtime_conversion():
    """Test the functions for count rate conversion."""
    original_rate = np.arange(1, 1000, 10)
    deadtime = 2.5e-3
    rdet = hen.base.r_det(deadtime, original_rate)
    rin = hen.base.r_in(deadtime, rdet)
    np.testing.assert_almost_equal(rin, original_rate)


class TestFake(object):
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
        cls.par = _dummy_par("bubububu.par")

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

    def test_load_events_xmm(self):
        """Test event file reading."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_test_fake_lc_xmm.evt'))
        hen.read_events.main(command.split())

    def test_load_events_randomize(self):
        """Test event file reading."""
        # command = '{0} --randomize-by 0.073'.format(
        #     os.path.join(self.datadir, 'monol_testA.evt'))
        newfiles = hen.read_events.treat_event_file(
            os.path.join(self.datadir, 'monol_testA.evt'),
            randomize_by=0.073)
        clean_file = self.first_event_file
        ev_clean = hen.io.load_events(clean_file)
        ev = hen.io.load_events(newfiles[0])
        diff = ev.time - ev_clean.time
        assert np.all(np.abs(diff) <= 0.073 / 2)
        assert np.all(np.abs(diff) > 0.)

    def test_scramble_events(self):
        command = f'{self.first_event_file}'
        newfile = hen.fake.main_scramble(command.split())
        assert os.path.exists(newfile)

    def test_calibrate_xmm(self):
        """Test event file calibration."""
        xmm_file = glob.glob(
            os.path.join(self.datadir,
                         'monol_test_fake*_xmm*ev' + HEN_FILE_EXTENSION))[0]
        command = '{0} -r {1} --nproc 2'.format(
            xmm_file,
            os.path.join(self.datadir, 'test.rmf'))
        hen.calibrate.main(command.split())

    def test_calibrate_xmm_normf(self):
        """Test event file calibration."""
        xmm_file = glob.glob(
            os.path.join(self.datadir,
                         'monol_test_fake*_xmm*ev' + HEN_FILE_EXTENSION))[0]
        command = '{0} --rough --nproc 2'.format(xmm_file)
        hen.calibrate.main(command.split())

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        def find_file_pattern_in_dir(pattern, directory):
            return glob.glob(os.path.join(directory, pattern))

        patterns = [
            '*monol_test*' + HEN_FILE_EXTENSION,
            '*lcurve*' + HEN_FILE_EXTENSION,
            '*lcurve*.txt',
            '*.log',
            '*monol_test*.dat',
            '*monol_test*.png',
            '*monol_test*.txt',
            '*monol_test_fake*.evt',
            '*bubu*',
            '*.p',
            '*.qdp',
            '*.inf'
        ]

        file_list = []
        for pattern in patterns:
            file_list.extend(
                find_file_pattern_in_dir(pattern, self.datadir)
            )

        for f in file_list:
            if os.path.exists(f):
                print("Removing " + f)
                os.remove(f)

        patterns = ['*_pds*/', '*_cpds*/', '*_sum/']

        dir_list = []
        for pattern in patterns:
            dir_list.extend(
                find_file_pattern_in_dir(pattern, self.datadir)
            )
        for f in dir_list:
            if os.path.exists(f):
                shutil.rmtree(f)
