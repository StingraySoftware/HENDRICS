from stingray.lightcurve import Lightcurve
from stingray.events import EventList
import numpy as np
from hendrics.io import save_events, HEN_FILE_EXTENSION, load_folding
from hendrics.efsearch import main_efsearch, main_zsearch
from hendrics.fold import main_fold
from hendrics.phaseogram import main_phaseogram, run_interactive_phaseogram
from hendrics.plot import plot_folding
import os
import pytest


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
                       '--fit-candidates'])
        outfile = 'events_EF' + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency,
                          atol=1/25.25)
        os.unlink(outfile)

    def test_zsearch(self):
        evfile = self.dum
        main_zsearch([evfile, '-f', '9.85', '-F', '9.95', '-n', '64',
                      '--fit-candidates', '--fit-frequency',
                      str(self.pulse_frequency),
                      '--dynstep', '5'])
        outfile = 'events_Z2n' + HEN_FILE_EXTENSION
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
        # os.unlink(outfile)

    @classmethod
    def teardown_class(cls):
        os.unlink(cls.dum)
