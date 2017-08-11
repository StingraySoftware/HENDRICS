# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function
from stingray.events import EventList
import numpy as np
import os
from maltpynt.read_events import treat_event_file
from maltpynt.io import MP_FILE_EXTENSION, load_data
import hendrics as mp

class TestReadEvents():
    """Real unit tests."""
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, 'data')
        cls.fits_fileA = os.path.join(cls.datadir, 'monol_testA.evt')

    def test_treat_event_file_nustar(self):
        treat_event_file(self.fits_fileA)
        new_filename = 'monol_testA_nustar_fpma_ev' + MP_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir,
                                           new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert 'instr' in data

    def test_treat_event_file_xmm(self):
        fits_file = os.path.join(self.datadir, 'monol_test_fake.evt')
        mp.fake.main(['--deadtime', '1e-4', '-m', 'XMM', '-i', 'epn',
                      '--ctrate', '2000',
                      '-o', fits_file])

        treat_event_file(fits_file)
        new_filename = 'monol_test_fake_xmm_epn_det01_ev' + MP_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir,
                                           new_filename))
        treat_event_file(fits_file, gti_split=True)
        new_filename = \
            'monol_test_fake_xmm_epn_det01_gti0_ev' + MP_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir,
                                           new_filename))
