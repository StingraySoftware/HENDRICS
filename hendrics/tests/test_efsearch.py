from stingray.lightcurve import Lightcurve
from stingray.events import EventList
import numpy as np
from hendrics.io import save_events, HEN_FILE_EXTENSION, load_folding, \
    load_events
from hendrics.efsearch import main_efsearch, main_zsearch
from hendrics.efsearch import decide_binary_parameters, folding_orbital_search
from hendrics.fold import main_fold
from hendrics.plot import plot_folding
import os
import pytest
from astropy.tests.helper import remote_data
try:
    import pandas as pd
    HAS_PD = True
except:
    HAS_PD = False

from hendrics.fold import HAS_PINT


def _dummy_par(par):
    with open(par, 'a') as fobj:
        print("BINARY BT", file=fobj)
        print("PB  1e20", file=fobj)
        print("A1  0", file=fobj)
        print("T0  56000", file=fobj)
        print("EPHEM  DE200", file=fobj)
        print("RAJ  00:55:01", file=fobj)
        print("DECJ 12:00:40.2", file=fobj)


class TestEFsearch():
    def setup_class(cls):
        cls.pulse_frequency = 1/0.101
        cls.tstart = 0
        cls.tend = 25.25
        cls.tseg = cls.tend - cls.tstart
        cls.dt = 0.00606
        cls.times = np.arange(cls.tstart, cls.tend, cls.dt) + cls.dt / 2
        cls.counts = \
            100 + 20 * np.cos(2 * np.pi * cls.times * cls.pulse_frequency)
        lc = Lightcurve(cls.times, cls.counts, gti=[[cls.tstart, cls.tend]])
        events = EventList()
        events.mjdref = 56000
        events.simulate_times(lc)
        cls.event_times = events.time
        cls.dum_noe = 'events_noe' + HEN_FILE_EXTENSION
        save_events(events, cls.dum_noe)
        events.pi = np.random.uniform(0, 1000, len(events.time))
        cls.dum_pi = 'events_pi' + HEN_FILE_EXTENSION
        save_events(events, cls.dum_pi)
        events.energy = np.random.uniform(3, 79, len(events.time))
        cls.dum = 'events' + HEN_FILE_EXTENSION
        save_events(events, cls.dum)

    def test_fold(self):
        evfile = self.dum
        evfile_noe = self.dum_noe
        evfile_pi = self.dum_pi

        main_fold([evfile, '-f', str(self.pulse_frequency), '-n', '64',
                   '--test', '--norm', 'ratios'])
        outfile = 'Energyprofile_ratios.png'
        assert os.path.exists(outfile)
        os.unlink(outfile)
        main_fold([evfile, '-f', str(self.pulse_frequency), '-n', '64',
                   '--test', '--norm', 'to1'])
        outfile = 'Energyprofile_to1.png'
        assert os.path.exists(outfile)
        os.unlink(outfile)
        main_fold([evfile, '-f', str(self.pulse_frequency), '-n', '64',
                   '--test', '--norm', 'blablabla'])
        outfile = 'Energyprofile_to1.png'
        assert os.path.exists(outfile)
        os.unlink(outfile)
        main_fold([evfile_pi, '-f', str(self.pulse_frequency), '-n', '64',
                   '--test', '--norm', 'blablabla'])
        outfile = 'Energyprofile_to1.png'
        assert os.path.exists(outfile)
        os.unlink(outfile)
        main_fold([evfile_noe, '-f', str(self.pulse_frequency), '-n', '64',
                   '--test', '--norm', 'blablabla'])
        outfile = 'Energyprofile.png'
        assert os.path.exists(outfile)
        os.unlink(outfile)

    def test_efsearch(self):
        evfile = self.dum
        main_efsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                       '--emin', '3', '--emax', '79',
                       '--fit-candidates'])
        outfile = 'events_EF_3-79keV' + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency,
                          atol=1/25.25)
        os.unlink(outfile)

    def test_zsearch(self):
        evfile = self.dum
        main_zsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                       '--emin', '3', '--emax', '79',
                       '--fit-candidates', '--fit-frequency',
                       str(self.pulse_frequency),
                       '--dynstep', '5'])
        outfile = 'events_Z2n_3-79keV' + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency,
                          atol=1/25.25)
        # Defaults to 2 harmonics
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_fdots(self):
        evfile = self.dum
        main_zsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                      '--fdotmin', ' -0.1', '--fdotmax', '0.1',
                      '--fit-candidates', '--fit-frequency',
                      str(self.pulse_frequency)])
        outfile = 'events_Z2n' + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True, output_data_file='bla.qdp')
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency,
                          atol=1/25.25)
        # Defaults to 2 harmonics
        assert len(efperiod.fdots) > 1
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_zsearch_fdots_fast(self):
        evfile = self.dum
        main_zsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                      '--fast'])
        outfile = 'events_Z2n' + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True, output_data_file='bla.qdp')
        efperiod = load_folding(outfile)

        assert len(efperiod.fdots) > 1
        assert efperiod.N == 2
        os.unlink(outfile)

    def test_fold_fast_fails(self):
        evfile = self.dum

        with pytest.raises(ValueError) as excinfo:
            main_efsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                           '--fast'])
        assert 'The fast option is only available for z ' in str(excinfo.value)

    @pytest.mark.skipif('not HAS_PD')
    def test_orbital(self):
        import pandas as pd
        events = load_events(self.dum)
        csv_file = decide_binary_parameters(137430, [0.03, 0.035],
                                            [2. * 86400, 2.5 * 86400],
                                            [0., 1.],
                                            fdot_range=[0, 5e-10], reset=False,
                                            NMAX=10)
        table = pd.read_csv(csv_file)
        assert len(table) == 10
        folding_orbital_search(events, csv_file, chunksize=10,
                               outfile='out.csv')
        table = pd.read_csv('out.csv')
        assert len(table) == 10
        assert np.all(table['done'])
        os.unlink(csv_file)
        os.unlink('out.csv')

    @remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_efsearch_deorbit(self):
        evfile = self.dum
        par = 'bububububu.par'

        _dummy_par(par)

        ip = main_zsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                           '--deorbit-par', par])

        outfile = 'events_Z2n' + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        os.unlink(par)

    def test_efsearch_deorbit_invalid(self):
        evfile = self.dum
        with pytest.raises(FileNotFoundError) as excinfo:
            ip = main_efsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                               '--deorbit-par', "nonexistent.par"])
        assert "Parameter file" in str(excinfo.value)

    @classmethod
    def teardown_class(cls):
        os.unlink(cls.dum)
