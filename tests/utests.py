from __future__ import print_function, unicode_literals
import maltpynt as mp
import numpy as np
MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION
import logging
import os
import sys

PY2 = sys.version_info[0] == 2
PYX6 = sys.version_info[1] == 6

if PY2 and PYX6:
    import unittest2 as unittest
else:
    import unittest

logging.basicConfig(filename='MP.log', level=logging.DEBUG, filemode='w')
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestFullRun(unittest.TestCase):
    '''Monolithic test case. Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/
    python-unittest-testcase-execution-order'''

    def step01_load_events(self):
        '''Test event file reading'''
        try:
            mp.read_events.mp_treat_event_file(os.path.join(datadir, 'A.evt'))
            mp.read_events.mp_treat_event_file(os.path.join(datadir, 'B.evt'))
        except:
            raise(Exception('Loading event file failed'))

    def step02_calibrate(self):
        '''Test event file calibration'''
        try:
            mp.calibrate.mp_calibrate(os.path.join(datadir, 'A_ev' +
                                      MP_FILE_EXTENSION),
                                      os.path.join(datadir, 'A_ev_calib') +
                                      MP_FILE_EXTENSION)
            mp.calibrate.mp_calibrate(os.path.join(datadir, 'B_ev') +
                                      MP_FILE_EXTENSION,
                                      os.path.join(datadir, 'B_ev_calib') +
                                      MP_FILE_EXTENSION)
        except:
            raise(Exception('Calibrating event file failed'))

    def step03_lcurve(self):
        '''Test light curve production'''
        try:
            mp.lcurve.mp_lcurve_from_events(
                os.path.join(datadir, 'A_ev_calib') + MP_FILE_EXTENSION,
                e_interval=[3, 50],
                safe_interval=[100, 300])
            mp.lcurve.mp_lcurve_from_events(
                os.path.join(datadir, 'B_ev_calib') + MP_FILE_EXTENSION,
                e_interval=[3, 50],
                safe_interval=[100, 300])
        except:
            raise(Exception('Production of light curve failed'))

    def step04_pds(self):
        '''Test PDS production'''
        try:
            mp.fspec.mp_calc_pds(os.path.join(datadir, 'A_E3-50_lc') +
                                 MP_FILE_EXTENSION,
                                 128)
            mp.fspec.mp_calc_pds(os.path.join(datadir, 'B_E3-50_lc') +
                                 MP_FILE_EXTENSION,
                                 128)
        except:
            raise(Exception('Production of PDSs failed'))

    def step05_cpds(self):
        '''Test CPDS production'''
        try:
            mp.fspec.mp_calc_cpds(os.path.join(datadir, 'A_E3-50_lc') +
                                  MP_FILE_EXTENSION,
                                  os.path.join(datadir, 'B_E3-50_lc') +
                                  MP_FILE_EXTENSION,
                                  128,
                                  outname=os.path.join(datadir, 'E3-50_cpds') +
                                  MP_FILE_EXTENSION)
        except:
            raise(Exception('Production of CPDS failed'))

    def step06_rebinlc(self):
        '''Test LC rebinning'''
        try:
            mp.rebin.mp_rebin_file(os.path.join(datadir, 'A_E3-50_lc') +
                                   MP_FILE_EXTENSION,
                                   2)
        except Exception as e:
            self.fail("{} failed ({}: {})".format('LC rebin', type(e), e))

    def step07_rebinpds1(self):
        '''Test PDS rebinning 1'''
        try:
            mp.rebin.mp_rebin_file(os.path.join(datadir, 'A_E3-50_pds') +
                                   MP_FILE_EXTENSION,
                                   2)
        except Exception as e:
            self.fail("{} failed ({}: {})".format('PDS rebin Test 1', type(e),
                                                  e))

    def step08_rebinpds2(self):
        '''Test PDS rebinning 2'''
        try:
            mp.rebin.mp_rebin_file(os.path.join(datadir, 'A_E3-50_pds') +
                                   MP_FILE_EXTENSION, 1.03)
        except Exception as e:
            self.fail("{} failed ({}: {})".format('PDS rebin Test 2', type(e),
                                                  e))

    def step09_savexspec1(self):
        '''Test save as Xspec 1'''
        try:
            mp.save_as_xspec.mp_save_as_xspec(
                os.path.join(datadir, 'A_E3-50_pds_rebin2')
                + MP_FILE_EXTENSION)
        except Exception as e:
            self.fail("{} failed ({}: {})".format('MP2Xspec Test 1', type(e),
                                                  e))

    def step10_savexspec2(self):
        '''Test save as Xspec 2'''
        try:
            mp.save_as_xspec.mp_save_as_xspec(
                os.path.join(datadir, 'A_E3-50_pds_rebin1.03')
                + MP_FILE_EXTENSION)
        except Exception as e:
            self.fail("{} failed ({}: {})".format('MP2Xspec Test 2', type(e),
                                                  e))

    def steps(self):
        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        '''Test a full run of the codes on two event lists'''
        print('')
        for name, step in self.steps():
            try:
                print('- ', step.__doc__, '...', end=' ')
                step()
                print('OK')
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
                print('Failed')


class TestPDS(unittest.TestCase):
    # First define a class variable that determines
    # if setUp was ever run
    ClassIsSetup = False

    @classmethod
    def setUpClass(cls):

        print("Setting up")
        import numpy.random as ra
        cls.length = 512000
        cls.tstart = 0
        cls.tstop = cls.tstart + cls.length
        cls.ctrate = 100
        cls.bintime = 1
        cls.nphot = ra.poisson(cls.length * cls.ctrate)

        events = ra.uniform(cls.tstart, cls.tstop, cls.nphot)

        time, cls.lc1 = \
            mp.lcurve.mp_lcurve(events,
                                cls.bintime,
                                start_time=cls.tstart,
                                stop_time=cls.tstop)

        events = ra.uniform(cls.tstart, cls.tstop, cls.nphot)

        time, cls.lc2 = \
            mp.lcurve.mp_lcurve(events,
                                cls.bintime,
                                start_time=cls.tstart,
                                stop_time=cls.tstop)
        cls.time = time

        cls.freq1, cls.pds1, cls.pdse1, dum = \
            mp.fspec.mp_welch_pds(cls.time, cls.lc1, cls.bintime, 512)

        cls.freq2, cls.pds2, cls.pdse2, dum = \
            mp.fspec.mp_welch_pds(cls.time, cls.lc2, cls.bintime, 512)

        dum, cls.cpds, cls.ec, dum = \
            mp.fspec.mp_welch_cpds(cls.time, cls.lc1, cls.lc2,
                                   cls.bintime, 128)

        # Calculate the variance discarding the freq=0 Hz element
        cls.varp1 = np.var(cls.pds1[1:])
        cls.varp2 = np.var(cls.pds2[1:])
        cls.varcr = np.var(cls.cpds.real[1:])

    def test_pdsstat1(self):
        '''Test that the Leahy PDS goes to 2'''
        from scipy.optimize import curve_fit

        baseline_fun = lambda x, a: a
        p, pcov = curve_fit(baseline_fun, self.freq1[1:], self.pds1,
                            p0=[2], sigma=self.pdse1, absolute_sigma=True)

        perr = np.sqrt(np.diag(pcov))

        assert np.abs(p - 2) < perr * 3, \
            ('PDS white level did not converge to 2')

    def test_pdsstat2(self):
        '''Test the statistical properties of the PDS.'''
        assert np.abs(self.varp1 / np.mean(self.pdse1[1:] ** 2)) - 1 < 0.3

    def test_pdsstat3(self):
        '''Test the statistical properties of the PDS.'''
        assert np.abs(self.varp2 / np.mean(self.pdse2[1:] ** 2)) - 1 < 0.3

    def test_pdsstat4(self):
        '''Test the statistical properties of the cospectrum.'''
        assert np.abs(self.varcr / np.mean(self.ec[1:] ** 2)) - 1 < 0.3

    def test_pdsstat5(self):
        '''Test the statistical properties of the cospectrum.

        In particular ,the standard deviation of the cospectrum is a factor
        ~sqrt(2) smaller than the standard deviation of the PDS'''
        geom_mean = np.sqrt(self.varp1 * self.varp2)
        assert np.abs(2 * self.varcr / geom_mean) - 1 < 0.3


class TestAll(unittest.TestCase):

    def test_crossgti1(self):
        '''Test the basic working of the intersection of GTIs'''
        gti1 = np.array([[1, 4]])
        gti2 = np.array([[2, 5]])
        newgti = mp.base.mp_cross_gtis([gti1, gti2])

        assert np.all(newgti == [[2, 4]]), 'GTIs do not coincide!'

    def test_crossgti2(self):
        '''A more complicated example of intersection of GTIs'''
        gti1 = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        gti2 = np.array([[2, 5], [6, 9], [11.4, 14]])
        newgti = mp.base.mp_cross_gtis([gti1, gti2])

        assert np.all(newgti == [[4.0, 5.0], [7.0, 9.0], [12.2, 13.2]]), \
            'GTIs do not coincide!'

    def test_bti(self):
        '''Test the inversion of GTIs'''
        gti = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        bti = mp.base.get_btis(gti)

        assert np.all(bti == [[2, 4], [5, 7], [10, 11], [11.2, 12.2]]), \
            'BTI is wrong!, %s' % repr(bti)

    def test_common_name(self):
        '''Test the common_name function'''
        a = 'A_3-50_A.nc'
        b = 'B_3-50_B.nc'
        assert mp.base.common_name(a, b) == '3-50'

if __name__ == '__main__':
    unittest.main(verbosity=2)
