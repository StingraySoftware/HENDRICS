from __future__ import print_function, unicode_literals
import unittest
import maltpynt as mp
import numpy as np
MP_FILE_EXTENSION = mp.io.MP_FILE_EXTENSION


class TestFullRun(unittest.TestCase):
    '''Monolithic test case. Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/
    python-unittest-testcase-execution-order'''

    def step1_load_events(self):
        print ('--------------------------------')
        print ('Testing event file reading')
        print ('--------------------------------')
        try:
            mp.read_events.mp_treat_event_file('../data/A.evt')
            mp.read_events.mp_treat_event_file('../data/B.evt')
        except:
            raise(Exception('Loading event file failed'))
        print ('--------------------------------')

    def step2_calibrate(self):
        print ('--------------------------------')
        print ('Testing event file calibration')
        print ('--------------------------------')
        try:
            mp.calibrate.mp_calibrate('../data/A_ev' + MP_FILE_EXTENSION,
                                      '../data/A_ev_calib' +
                                      MP_FILE_EXTENSION)
            mp.calibrate.mp_calibrate('../data/B_ev' + MP_FILE_EXTENSION,
                                      '../data/B_ev_calib' +
                                      MP_FILE_EXTENSION)
        except:
            raise(Exception('Calibrating event file failed'))
        print ('--------------------------------')

    def step3_lcurve(self):
        print ('--------------------------------')
        print ('Testing light curve production')
        print ('--------------------------------')
        try:
            mp.lcurve.mp_lcurve_from_events('../data/A_ev_calib' +
                                            MP_FILE_EXTENSION,
                                            e_interval=[3, 50],
                                            safe_interval=[100, 300])
            mp.lcurve.mp_lcurve_from_events('../data/B_ev_calib' +
                                            MP_FILE_EXTENSION,
                                            e_interval=[3, 50],
                                            safe_interval=[100, 300])
        except:
            raise(Exception('Production of light curve failed'))
        print ('--------------------------------')

    def step4_pds(self):
        print ('--------------------------------')
        print ('Testing PDS production')
        print ('--------------------------------')
        try:
            mp.fspec.mp_calc_pds('../data/A_E3-50_lc' + MP_FILE_EXTENSION,
                                 128)
            mp.fspec.mp_calc_pds('../data/B_E3-50_lc' + MP_FILE_EXTENSION,
                                 128)
        except:
            raise(Exception('Production of PDSs failed'))
        print ('--------------------------------')

    def step5_cpds(self):
        print ('--------------------------------')
        print ('Testing CPDS production')
        print ('--------------------------------')
        try:
            mp.fspec.mp_calc_cpds('../data/A_E3-50_lc' + MP_FILE_EXTENSION,
                                  '../data/B_E3-50_lc' + MP_FILE_EXTENSION,
                                  128)
        except:
            raise(Exception('Production of CPDS failed'))
        print ('--------------------------------')

    def steps(self):
        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self.steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))


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

if __name__ == '__main__':
    unittest.main(verbosity=2)
