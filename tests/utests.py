from __future__ import print_function, unicode_literals
import unittest
import maltpynt as mp
import numpy as np
MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION
import logging
import os

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


class TestAll(unittest.TestCase):

    def test_pdsstat1(self):
        '''Test that the Leahy PDS goes to 2'''
        import numpy.random as ra
        from scipy.optimize import curve_fit
        length = 512
        tstart = 0
        tstop = tstart + length
        ctrate = 100
        bintime = 1

        tot_pds = 0

        iter_ns = np.arange(10000)
        for iter_n in iter_ns + 1:
            nphot = ra.poisson(ctrate * length)
            event_list = ra.uniform(tstart, tstop, nphot)

            time, lc = mp.lcurve.mp_lcurve(event_list,
                                           bintime,
                                           start_time=tstart,
                                           stop_time=tstop)

            freq, pds = mp.fspec.mp_leahy_pds(lc, bintime)

            tot_pds += pds[1:]

            tot_pdse = tot_pds / iter_n / np.sqrt(iter_n)

            baseline_fun = lambda x, a: a
            p, pcov = curve_fit(baseline_fun, freq[1:], tot_pds / iter_n,
                                p0=[2], sigma=tot_pdse, absolute_sigma=True)

            perr = np.sqrt(np.diag(pcov))

            if iter_n > 100 and np.abs(p - 2) < perr:
                return
        raise Exception('PDS white level did not converge to 2 after '
                        '10000 iterations')

    def test_pdsstat2(self):
        '''Test the statistical properties of the cospectrum.

        In particular ,the standard deviation of the cospectrum is a factor
        ~sqrt(2) smaller than the standard deviation of the PDS'''
        tMin = 0
        tMax = 512
        ctrate = 10000
        bintime = 0.01

        events = np.random.uniform(tMin, tMax, ctrate * (tMax - tMin))
        time1, lc1 = mp.lcurve.mp_lcurve(events, bintime, start_time=tMin,
                                         stop_time=tMax)
        events = np.random.uniform(tMin, tMax, ctrate * (tMax - tMin))
        time2, lc2 = mp.lcurve.mp_lcurve(events, bintime, start_time=tMin,
                                         stop_time=tMax)
        del time1
        time = time2
        dum, pds1, dum, dum = \
            mp.fspec.mp_welch_pds(time, lc1, bintime, 128)
        dum, pds2, dum, dum = \
            mp.fspec.mp_welch_pds(time, lc2, bintime, 128)

        dum, cpds, dum, dum = \
            mp.fspec.mp_welch_cpds(time, lc1, lc2, bintime, 128)

        # Calculate the variance discarding the freq=0 Hz element
        varp1 = np.var(pds1[1:])
        varp2 = np.var(pds2[1:])
        varcr = np.var(cpds.real[1:])

        assert np.rint(np.sqrt(varp1 * varp2) / varcr) == 2

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
    try:
        unittest.main(verbosity=2)
    except:
        # Python 2.6 does not accept the verbosity keyword. Let's try if this
        # is the case
        unittest.main()
