# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""First set of tests."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import maltpynt as mp
import numpy as np
MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION
import logging
import os
import sys
import glob

PY2 = sys.version_info[0] == 2
PYX6 = sys.version_info[1] == 6

if PY2 and PYX6:
    import unittest2 as unittest
    print("\nunittest2!!\n")
else:
    import unittest

logging.basicConfig(filename='MP.log', level=logging.DEBUG, filemode='w')
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


def _ratio(a, b):
    return np.abs(a - b) / np.abs(a + b)


class TestFullRun(unittest.TestCase):

    """Monolithic test case.

    Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/
    python-unittest-testcase-execution-order
    """

    def step00_print_info(self):
        """Test printing info about FITS file"""
        fits_file = os.path.join(datadir, 'monol_testA.evt')
        mp.io.print_fits_info(fits_file, hdu=1)

    def step01_load_events(self):
        """Test event file reading."""
        try:
            mp.read_events.treat_event_file(
                os.path.join(datadir, 'monol_testA.evt'))
            mp.read_events.treat_event_file(
                os.path.join(datadir, 'monol_testA_timezero.evt'))
            mp.read_events.treat_event_file(
                os.path.join(datadir, 'monol_testB.evt'))
            mp.read_events.treat_event_file(
                os.path.join(datadir, 'monol_testB.evt'),
                gti_split=True, noclobber=True, min_length=0)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'Loading event file', type(e), e))

    def step02_calibrate(self):
        """Test event file calibration."""
        try:
            mp.calibrate.calibrate(os.path.join(datadir, 'monol_testA_ev' +
                                                MP_FILE_EXTENSION),
                                   os.path.join(datadir,
                                                'monol_testA_ev_calib') +
                                   MP_FILE_EXTENSION,
                                   os.path.join(datadir,
                                   'test.rmf'))
            mp.calibrate.calibrate(os.path.join(datadir, 'monol_testB_ev' +
                                                MP_FILE_EXTENSION),
                                   os.path.join(datadir,
                                                'monol_testB_ev_calib') +
                                   MP_FILE_EXTENSION,
                                   os.path.join(datadir,
                                   'test.rmf'))
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'Calibrating event file', type(e), e))

    def step03a_lcurve(self):
        """Test light curve production."""
        try:
            mp.lcurve.lcurve_from_events(
                os.path.join(datadir,
                             'monol_testA_ev_calib') + MP_FILE_EXTENSION,
                e_interval=[3, 50],
                safe_interval=[100, 300])
            mp.lcurve.lcurve_from_events(
                os.path.join(datadir,
                             'monol_testB_ev_calib') + MP_FILE_EXTENSION,
                e_interval=[3, 50],
                safe_interval=[100, 300])
            mp.lcurve.lcurve_from_events(
                os.path.join(datadir,
                             'monol_testB_ev_0') + MP_FILE_EXTENSION)
            mp.lcurve.lcurve_from_events(
                os.path.join(datadir,
                             'monol_testB_ev_0') + MP_FILE_EXTENSION,
                gti_split=True)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'Production of light curve', type(e), e))

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

            diff = lc_mp[:goodlen] - lc_ftools[:goodlen]

        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('LC fits', type(e), e))

        assert np.all(np.abs(diff) <= 1e-3), \
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
            self.fail("{0} failed ({1}: {2})".format('LC txt', type(e), e))

        assert np.all(np.abs(lc_mp - lc_txt) <= 1e-3), \
            'Light curve data do not coincide between txt and MP'

    def step04a_pds(self):
        """Test PDS production."""
        try:
            mp.fspec.calc_pds(os.path.join(datadir,
                                           'monol_testA_E3-50_lc') +
                              MP_FILE_EXTENSION,
                              128, save_dyn=True, normalization='rms')
            mp.fspec.calc_pds(os.path.join(datadir,
                                           'monol_testB_E3-50_lc') +
                              MP_FILE_EXTENSION,
                              128, save_dyn=True, normalization='rms')
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'Production of PDS', type(e), e))

    def step04b_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(datadir,
                                     'lcurve_ftools_lc' +
                                     MP_FILE_EXTENSION)
        try:
            mp.fspec.calc_pds(lcurve_ftools, 128)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('PDS LC FITS',
                                                     type(e), e))

    def step04c_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(datadir,
                                  'lcurve_txt_lc' +
                                  MP_FILE_EXTENSION)
        try:
            mp.fspec.calc_pds(lcurve_txt, 128)

        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('PDS LC txt', type(e), e))

    def step05_cpds(self):
        """Test CPDS production."""
        try:
            mp.fspec.calc_cpds(os.path.join(datadir,
                                            'monol_testA_E3-50_lc') +
                               MP_FILE_EXTENSION,
                               os.path.join(datadir,
                                            'monol_testB_E3-50_lc') +
                               MP_FILE_EXTENSION,
                               128,
                               outname=os.path.join(datadir,
                                                    'monol_test_E3-50_cpds')
                               + MP_FILE_EXTENSION, save_dyn=True,
                               normalization='rms')
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('CPDS', type(e), e))

    def step05b_fspecs(self):
        """Test all frequency spectra production."""
        files = [os.path.join(datadir, 'monol_testA_E3-50_lc') +
                 MP_FILE_EXTENSION,
                 os.path.join(datadir, 'monol_testB_E3-50_lc') +
                 MP_FILE_EXTENSION]
        outroot = os.path.join(datadir, 'monol_test_E3-50_fspecs')
        try:
            mp.fspec.calc_fspec(files, 128, outroot=outroot)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format(
                'Production of frequency spectra', type(e), e))

    def step06_lags(self):
        """Test Lag calculations."""
        try:
            mp.lags.lags_from_spectra(
                os.path.join(datadir,
                             'monol_test_E3-50_cpds') + MP_FILE_EXTENSION,
                os.path.join(datadir,
                             'monol_testA_E3-50_pds') + MP_FILE_EXTENSION,
                os.path.join(datadir,
                             'monol_testB_E3-50_pds') + MP_FILE_EXTENSION,
                outroot=os.path.join(datadir,
                                     'monol_test_lags' + MP_FILE_EXTENSION))
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('Lags production',
                                                     type(e), e))

    def step07_rebinlc(self):
        """Test LC rebinning."""
        try:
            mp.rebin.rebin_file(os.path.join(datadir,
                                             'monol_testA_E3-50_lc') +
                                MP_FILE_EXTENSION,
                                2)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('LC rebin', type(e), e))

    def step08_rebinpds1(self):
        """Test PDS rebinning 1."""
        try:
            mp.rebin.rebin_file(os.path.join(datadir,
                                             'monol_testA_E3-50_pds') +
                                MP_FILE_EXTENSION,
                                2)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('PDS rebin Test 1',
                                                     type(e), e))

    def step08a_rebinpds2(self):
        """Test PDS rebinning 2."""
        try:
            mp.rebin.rebin_file(os.path.join(datadir,
                                             'monol_testA_E3-50_pds') +
                                MP_FILE_EXTENSION, 1.03)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('PDS rebin Test 2',
                                                     type(e), e))

    def step09_rebincpds(self):
        """Test CPDS rebinning."""
        try:
            mp.rebin.rebin_file(os.path.join(datadir,
                                             'monol_test_E3-50_cpds') +
                                MP_FILE_EXTENSION, 2)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('CPDS rebin Test 1.03',
                                                     type(e), e))

    def step09a_rebincpds2(self):
        """Test CPDS rebinning."""
        try:
            mp.rebin.rebin_file(os.path.join(datadir,
                                             'monol_test_E3-50_cpds') +
                                MP_FILE_EXTENSION, 1.03)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('CPDS rebin Test 1.03',
                                                     type(e), e))

    def step10_savexspec1(self):
        """Test save as Xspec 1."""
        try:
            mp.save_as_xspec.save_as_xspec(
                os.path.join(datadir, 'monol_testA_E3-50_pds_rebin2')
                + MP_FILE_EXTENSION)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('MP2xspec Test 1',
                                                     type(e), e))

    def step11_savexspec2(self):
        """Test save as Xspec 2."""
        try:
            mp.save_as_xspec.save_as_xspec(
                os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03')
                + MP_FILE_EXTENSION)
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
            self.fail("{0} failed ({1}: {2})".format('Join lcs', type(e), e))

    def step13_scrunchlcs(self):
        """Test produce scrunched light curves."""
        try:
            mp.lcurve.scrunch_lightcurves(
                [os.path.join(datadir, 'monol_testA_E3-50_lc') +
                 MP_FILE_EXTENSION,
                 os.path.join(datadir, 'monol_testB_E3-50_lc') +
                 MP_FILE_EXTENSION],
                os.path.join(datadir, 'monol_test_scrunchlc' +
                             MP_FILE_EXTENSION),
                save_joint=False)
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('MPscrunchlc', type(e),
                                                     e))

    def step14_sumpds(self):
        """Test the sum of pdss."""
        try:
            mp.sum_fspec.sum_fspec([
                os.path.join(datadir,
                             'monol_testA_E3-50_pds') + MP_FILE_EXTENSION,
                os.path.join(datadir,
                             'monol_testB_E3-50_pds') + MP_FILE_EXTENSION],
                outname=os.path.join(datadir,
                                     'monol_test_sum' + MP_FILE_EXTENSION))
        except Exception as e:
            self.fail("{0} failed ({1}: {2})".format('Lags production',
                                                     type(e), e))

    def step15_plotpds(self):
        """Test plotting a PDS"""
        fname = os.path.join(datadir, 'monol_testA_E3-50_pds_rebin1.03') + \
            MP_FILE_EXTENSION
        figname = os.path.join(datadir, 'monol_test_pds.png')
        mp.plot.plot_pds(fname, figname=figname)

    def step16_plotcpds(self):
        """Test plotting a cospectrum"""
        fname = os.path.join(datadir, 'monol_test_E3-50_cpds_rebin1.03') + \
            MP_FILE_EXTENSION
        figname = os.path.join(datadir, 'monol_test_cpds.png')
        mp.plot.plot_cospectrum(fname, figname=figname)

    def step17_plotlc(self):
        """Test plotting a light curve"""

        fname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        figname = os.path.join(datadir, 'monol_test_lc.png')
        mp.plot.plot_lc(fname, figname=figname)

    def step18_create_gti(self):
        """Test creating a GTI file"""

        fname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        mp.create_gti.create_gti(fname, filter_expr='lc>0',
                                 outfile=fname.replace('_lc', '_gti'))

    def step19_apply_gti(self):
        """Test applying GTIs to a light curve"""

        fname = os.path.join(datadir, 'monol_testA_E3-50_lc') + \
            MP_FILE_EXTENSION
        mp.create_gti.apply_gti(fname, [[80000100, 80000300]])

    def step20_load_gtis(self):
        """Test loading of GTIs from FITS files"""
        fits_file = os.path.join(datadir, 'monol_testA.evt')
        mp.read_events.load_gtis(fits_file)

    def step21_save_as_qdp(self):
        """Test saving arrays in a qdp file"""
        arrays = [np.array([0, 1, 3]), np.array([1, 4, 5])]
        errors = [np.array([1, 1, 1]), np.array([[1, 0.5], [1, 0.5], [1, 1]])]
        mp.io.save_as_qdp(arrays, errors,
                          filename=os.path.join(datadir,
                                                "monol_test_qdp.txt"))

    def _all_steps(self):

        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        """Test a full run of the codes on two event lists."""
        print('')
        for name, step in self._all_steps():
            try:
                print('- ', step.__doc__, '...', end=' ')
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
                                   '*monol_test*.txt')) + \
            glob.glob(os.path.join(datadir,
                                   '*monol_test*.png'))
        for f in file_list:
            os.remove(f)


class TestPDS(unittest.TestCase):
    """Test PDS statistics."""

    @classmethod
    def setUpClass(cls):
        print("Setting up.")
        print("This test is about the statistical properties of frequency "
              "spectra and it is based on random number generation. It might, "
              "randomly, fail. Always repeat the test if it does and only "
              "worry if it repeatedly fails.")
        import numpy.random as ra
        cls.length = 512000
        cls.tstart = 0
        cls.tstop = cls.tstart + cls.length
        cls.ctrate = 100
        cls.bintime = 1
        cls.nphot = ra.poisson(cls.length * cls.ctrate)

        events = ra.uniform(cls.tstart, cls.tstop, cls.nphot)

        time, cls.lc1 = \
            mp.lcurve.lcurve(events,
                             cls.bintime,
                             start_time=cls.tstart,
                             stop_time=cls.tstop)

        events = ra.uniform(cls.tstart, cls.tstop, cls.nphot)

        time, cls.lc2 = \
            mp.lcurve.lcurve(events,
                             cls.bintime,
                             start_time=cls.tstart,
                             stop_time=cls.tstop)
        cls.time = time

        cls.freq1, cls.pds1, cls.pdse1, dum = \
            mp.fspec.welch_pds(cls.time, cls.lc1, cls.bintime, 1024)

        cls.freq2, cls.pds2, cls.pdse2, dum = \
            mp.fspec.welch_pds(cls.time, cls.lc2, cls.bintime, 1024)

        dum, cls.cpds, cls.ec, dum = \
            mp.fspec.welch_cpds(cls.time, cls.lc1, cls.lc2,
                                cls.bintime, 1024)

        # Calculate the variance discarding the freq=0 Hz element
        cls.varp1 = np.var(cls.pds1[1:])
        cls.varp2 = np.var(cls.pds2[1:])
        cls.varcr = np.var(cls.cpds.real[1:])

    def test_pdsstat1(self):
        """Test that the Leahy PDS goes to 2."""
        from scipy.optimize import curve_fit

        def baseline_fun(x, a):
            return a

        freq, pds, epds = \
            mp.rebin.const_rebin(self.freq1[1:], self.pds1[1:], 16,
                                 self.pdse1[1:])

        p, pcov = curve_fit(baseline_fun, freq, pds,
                            p0=[2], sigma=epds, absolute_sigma=True)

        perr = np.sqrt(np.diag(pcov))

        assert np.abs(p - 2) < perr * 3, \
            ('PDS white level did not converge to 2')

    def test_pdsstat2(self):
        """Test the statistical properties of the PDS."""
        r = _ratio(self.varp1, np.mean(self.pdse1[1:] ** 2))
        assert r < 0.1, \
            "{0} {1} {2}".format(self.varp1, np.mean(self.pdse1[1:] ** 2), r)

    def test_pdsstat3(self):
        """Test the statistical properties of the PDS."""
        r = _ratio(self.varp2, np.mean(self.pdse2[1:] ** 2))
        assert r < 0.1, \
            "{0} {1} {2}".format(self.varp2, np.mean(self.pdse2[1:] ** 2), r)

    def test_pdsstat4(self):
        """Test the statistical properties of the cospectrum."""
        r = _ratio(self.varcr, np.mean(self.ec[1:] ** 2))
        assert r < 0.1, \
            "{0} {1} {2}".format(self.varcr, np.mean(self.ec[1:] ** 2), r)

    def test_pdsstat5(self):
        """Test the statistical properties of the cospectrum.

        In particular ,the standard deviation of the cospectrum is a factor
        ~sqrt(2) smaller than the standard deviation of the PDS.
        """
        geom_mean = np.sqrt(self.varp1 * self.varp2)
        r = _ratio(2 * self.varcr, geom_mean)
        assert r < 0.1, \
            "{0} {1} {2}".format(2 * self.varcr, geom_mean, r)


class TestAll(unittest.TestCase):

    """Real unit tests."""

    def test_crossgti1(self):
        """Test the basic working of the intersection of GTIs."""
        gti1 = np.array([[1, 4]])
        gti2 = np.array([[2, 5]])
        newgti = mp.base.cross_gtis([gti1, gti2])

        assert np.all(newgti == [[2, 4]]), 'GTIs do not coincide!'

    def test_crossgti2(self):
        """A more complicated example of intersection of GTIs."""
        gti1 = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        gti2 = np.array([[2, 5], [6, 9], [11.4, 14]])
        newgti = mp.base.cross_gtis([gti1, gti2])

        assert np.all(newgti == [[4.0, 5.0], [7.0, 9.0], [12.2, 13.2]]), \
            'GTIs do not coincide!'

    def test_bti(self):
        """Test the inversion of GTIs."""
        gti = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        bti = mp.base.get_btis(gti)

        assert np.all(bti == [[2, 4], [5, 7], [10, 11], [11.2, 12.2]]), \
            'BTI is wrong!, %s' % repr(bti)

    def test_common_name(self):
        """Test the common_name function."""
        a = 'A_3-50_A.nc'
        b = 'B_3-50_B.nc'
        assert mp.base.common_name(a, b) == '3-50'

    def test_geom_bin(self):
        """Test if geom_bin fails under some conditions."""
        freq = np.arange(0, 100, 0.1)
        pds = np.random.normal(2, 0.1, len(freq))
        _ = mp.rebin.geom_bin(freq, pds, 1.3, pds_err=pds)
        _ = mp.rebin.geom_bin(freq, pds, 1.3)

    def test_exposure_calculation1(self):
        """Test if the exposure calculator works correctly."""
        times = np.array([1., 2., 3.])
        events = np.array([2.])
        priors = np.array([2.])
        dt = np.array([1., 1., 1.])
        expo = mp.exposure.get_livetime_per_bin(times, events, priors, dt=dt,
                                                gti=None)
        np.testing.assert_almost_equal(expo, np.array([1, 0.5, 0.]))

    def test_exposure_calculation2(self):
        """Test if the exposure calculator works correctly."""
        times = np.array([1., 2.])
        events = np.array([2.1])
        priors = np.array([0.3])
        dt = np.array([1., 1.])
        expo = mp.exposure.get_livetime_per_bin(times, events, priors, dt=dt,
                                                gti=None)
        np.testing.assert_almost_equal(expo, np.array([0, 0.3]))

    def test_exposure_calculation3(self):
        """Test if the exposure calculator works correctly."""
        times = np.array([1., 2., 3.])
        events = np.array([2.1])
        priors = np.array([0.7])
        dt = np.array([1., 1., 1.])
        expo = mp.exposure.get_livetime_per_bin(times, events, priors, dt=dt,
                                                gti=None)
        np.testing.assert_almost_equal(expo, np.array([0.1, 0.6, 0.]))

if __name__ == '__main__':
    unittest.main(verbosity=2)
