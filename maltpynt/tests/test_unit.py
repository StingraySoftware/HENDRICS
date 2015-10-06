# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""First set of tests."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import maltpynt as mp
import numpy as np
import logging
import os
import unittest

MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION

logging.basicConfig(filename='MP.log', level=logging.DEBUG, filemode='w')
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


def _ratio(a, b):
    return np.abs(a - b) / np.abs(a + b)


class TestPDS(unittest.TestCase):

    """Test PDS statistics."""

    @classmethod
    def setUpClass(cls):
        """Produce common products for all subsequent tests."""
        print("Setting up.")
        import numpy.random as ra
        cls.length = 512000
        cls.tstart = 0
        cls.tstop = cls.tstart + cls.length
        cls.ctrate = 100
        cls.bintime = 1

        ra.seed(seed=1234)
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

        data = mp.fspec.welch_pds(cls.time, cls.lc1, cls.bintime, 1024)
        cls.freq1, cls.pds1, cls.pdse1 = data.f, data.pds, data.epds

        data = mp.fspec.welch_pds(cls.time, cls.lc2, cls.bintime, 1024)
        cls.freq2, cls.pds2, cls.pdse2 = data.f, data.pds, data.epds

        data = mp.fspec.welch_cpds(cls.time, cls.lc1, cls.lc2,
                                   cls.bintime, 1024)
        cls.cpds, cls.ec = data.cpds, data.ecpds

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
                            p0=[2], sigma=1 / epds**2)

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
        del _

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

    def test_exposure_calculation4(self):
        """Test if the exposure calculator works correctly."""
        times = np.array([1., 1.5, 2., 2.5, 3.])
        events = np.array([2.6])
        priors = np.array([1.5])
        dt = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        expected_expo = np.array([0.15, 0.5, 0.5, 0.35, 0])
        expo = mp.exposure.get_livetime_per_bin(times, events, priors, dt=dt,
                                                gti=None)
        np.testing.assert_almost_equal(expo, expected_expo)

    def test_high_precision_keyword(self):
        """Test high precision FITS keyword read."""
        from maltpynt.io import high_precision_keyword_read
        hdr = {"MJDTESTI": 100, "MJDTESTF": np.longdouble(0.5),
               "CIAO": np.longdouble(0.)}
        assert \
            high_precision_keyword_read(hdr,
                                        "MJDTEST") == np.longdouble(100.5), \
            "Keyword MJDTEST read incorrectly"
        assert \
            high_precision_keyword_read(hdr, "CIAO") == np.longdouble(0.), \
            "Keyword CIAO read incorrectly"

    def test_decide_spectrum_intervals(self):
        """Test the division of start and end times to calculate spectra."""
        start_times = \
            mp.fspec.decide_spectrum_intervals([[0, 400], [1022, 1200]], 128)
        assert np.all(start_times == np.array([0, 128, 256, 1022]))

    def test_filter_for_deadtime_nonpar(self):
        """Test dead time filter, non-paralyzable case."""
        events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
        filt_events = mp.fake.filter_for_deadtime(events, 0.11)
        expected = np.array([1, 2, 2.2, 3, 3.2])
        assert np.all(filt_events == expected), \
            "Wrong: {} vs {}".format(filt_events, expected)

    def test_filter_for_deadtime_nonpar_bkg(self):
        """Test dead time filter, non-paralyzable case, with background."""
        events = np.array([1.1, 2, 2.2, 3, 3.2])
        bkg_events = np.array([1, 3.1])
        filt_events, info = \
            mp.fake.filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events,
                                        return_all=True)
        expected_ev = np.array([2, 2.2, 3, 3.2])
        expected_bk = np.array([1])
        assert np.all(filt_events == expected_ev), \
            "Wrong: {} vs {}".format(filt_events, expected_ev)
        assert np.all(info.bkg == expected_bk), \
            "Wrong: {} vs {}".format(info.bkg, expected_bk)

    def test_filter_for_deadtime_par(self):
        """Test dead time filter, paralyzable case."""
        events = np.array([1, 1.1, 2, 2.2, 3, 3.1, 3.2])
        assert np.all(mp.fake.filter_for_deadtime(
            events, 0.11, paralyzable=True) == np.array([1, 2, 2.2, 3]))

    def test_filter_for_deadtime_par_bkg(self):
        """Test dead time filter, paralyzable case, with background."""
        events = np.array([1.1, 2, 2.2, 3, 3.2])
        bkg_events = np.array([1, 3.1])
        filt_events, info = \
            mp.fake.filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events,
                                        paralyzable=True, return_all=True)
        expected_ev = np.array([2, 2.2, 3])
        expected_bk = np.array([1])
        assert np.all(filt_events == expected_ev), \
            "Wrong: {} vs {}".format(filt_events, expected_ev)
        assert np.all(info.bkg == expected_bk), \
            "Wrong: {} vs {}".format(info.bkg, expected_bk)

    def test_event_simulation(self):
        """Test simulation of events."""
        times = np.array([0.5, 1.5])
        lc = np.array([1000, 2000])
        events = mp.fake.fake_events_from_lc(times, lc)
        newtime, newlc = mp.lcurve.lcurve(events, 1., start_time=0,
                                          stop_time=2)
        assert np.all(np.abs(newlc - lc) < 3 * np.sqrt(lc))
        np.testing.assert_almost_equal(newtime, times)

    def test_deadtime_mask_par(self):
        """Test dead time filter, paralyzable case, with background."""
        events = np.array([1.1, 2, 2.2, 3, 3.2])
        bkg_events = np.array([1, 3.1])
        filt_events, info = \
            mp.fake.filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events,
                                        paralyzable=True, return_all=True)

        assert np.all(filt_events == events[info.mask])

if __name__ == '__main__':
    unittest.main(verbosity=2)
