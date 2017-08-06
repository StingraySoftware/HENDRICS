# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function
from stingray.events import EventList
import numpy as np
import os
from maltpynt.read_events import treat_event_file
from maltpynt.io import MP_FILE_EXTENSION
from maltpynt.lcurve import lcurve_from_events
import maltpynt as mp
import glob


class TestLcurve():
    """Real unit tests."""
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, 'data')
        cls.fits_fileA = os.path.join(cls.datadir, 'monol_testA.evt')
        cls.new_filename = \
            os.path.join(cls.datadir,
                         'monol_testA_nustar_fpma_ev' + MP_FILE_EXTENSION)
        cls.calib_filename = \
            os.path.join(cls.datadir,
                         'monol_testA_nustar_fpma_ev_calib' + MP_FILE_EXTENSION)


    def test_treat_event_file_nustar(self):
        treat_event_file(self.fits_fileA)
        lcurve_from_events(self.new_filename)
        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testA_nustar_fpma_lc' +
                                           MP_FILE_EXTENSION))

    def test_treat_event_file_nustar_energy(self):
        command = '{0} -r {1} --nproc 2'.format(
            self.new_filename,
            os.path.join(self.datadir, 'test.rmf'))
        mp.calibrate.main(command.split())
        lcurve_from_events(self.calib_filename, e_interval=[3, 50])

        assert os.path.exists(os.path.join(self.datadir,
                                           'monol_testA_nustar_fpma_E3-50_lc' +
                                           MP_FILE_EXTENSION))
