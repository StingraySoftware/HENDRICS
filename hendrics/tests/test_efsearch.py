from stingray.lightcurve import Lightcurve
from stingray.events import EventList
import numpy as np
from hendrics.io import save_events, HEN_FILE_EXTENSION, load_folding
from hendrics.efsearch import main_efsearch, main_zsearch, main_phaseogram
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
        cls.dum = 'events' + HEN_FILE_EXTENSION
        save_events(events, cls.dum)

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
                      str(self.pulse_frequency)])
        outfile = 'events_Z2n' + HEN_FILE_EXTENSION
        assert os.path.exists(outfile)
        plot_folding([outfile], ylog=True)
        efperiod = load_folding(outfile)
        assert np.isclose(efperiod.peaks[0], self.pulse_frequency,
                          atol=1/25.25)
        # Defaults to 2 harmonics
        assert efperiod.N == 2
        # os.unlink(outfile)

    def test_phaseogram_input_periodogram(self):
        evfile = self.dum
        main_phaseogram([evfile, '--periodogram',
                         'events_Z2n' + HEN_FILE_EXTENSION, '--test'])

    def test_phaseogram_input_f(self):
        evfile = self.dum
        main_phaseogram([evfile, '-f', '9.9', '--test'])

    def test_phaseogram_raises(self):
        evfile = self.dum
        with pytest.raises(ValueError):
            main_phaseogram([evfile, '--test'])

    @classmethod
    def teardown_class(cls):
        os.unlink(cls.dum)

