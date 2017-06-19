# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""First set of tests."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import maltpynt as mp
import numpy as np
import logging
import os
import unittest
import pytest

MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION

logging.basicConfig(filename='MP.log', level=logging.DEBUG, filemode='w')
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


def _ratio(a, b):
    return np.abs(a - b) / np.abs(a + b)


class TestAll(unittest.TestCase):

    """Real unit tests."""

    def test_common_name(self):
        """Test the common_name function."""
        a = 'A_3-50_A.nc'
        b = 'B_3-50_B.nc'
        assert mp.base.common_name(a, b) == '3-50'

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

    def test_exposure_calculation5(self):
        """Test if the exposure calculator works correctly."""
        times = np.array([1., 2., 3.])
        events = np.array([1.1, 1.2, 1.4, 1.5, 1.8, 4])
        # dead time = 0.05
        priors = np.array([0.55, 0.05, 0.15, 0.05, 0.25, 2.15])
        dt = np.array([1, 1, 1])
        expected_expo = np.array([0.8, 0.9, 1])
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

    def test_deadtime_mask_par(self):
        """Test dead time filter, paralyzable case, with background."""
        events = np.array([1.1, 2, 2.2, 3, 3.2])
        bkg_events = np.array([1, 3.1])
        filt_events, info = \
            mp.fake.filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events,
                                        paralyzable=True, return_all=True)

        assert np.all(filt_events == events[info.mask])

    def test_deadtime_conversion(self):
        """Test the functions for count rate conversion."""
        original_rate = np.arange(1, 1000, 10)
        deadtime = 2.5e-3
        rdet = mp.base.r_det(deadtime, original_rate)
        rin = mp.base.r_in(deadtime, rdet)
        np.testing.assert_almost_equal(rin, original_rate)

    def test_high_precision_split1(self):
        C_I, C_F, C_l, k = \
            mp.io._split_high_precision_number("C", np.double(0.01), 8)
        assert C_I == 1
        np.testing.assert_almost_equal(C_F, 0, 6)
        assert C_l == -2
        assert k == "double"

    def test_high_precision_split2(self):
        C_I, C_F, C_l, k = \
            mp.io._split_high_precision_number("C", np.double(1.01), 8)
        assert C_I == 1
        np.testing.assert_almost_equal(C_F, np.double(0.01), 6)
        assert C_l == 0
        assert k == "double"

if __name__ == '__main__':
    unittest.main(verbosity=2)
