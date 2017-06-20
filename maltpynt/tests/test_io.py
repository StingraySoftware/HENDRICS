from stingray.events import EventList
import numpy as np
import os
from maltpynt.io import load_events, save_events
from maltpynt.io import save_data, load_data
from maltpynt.io import MP_FILE_EXTENSION, _split_high_precision_number

class TestIO():
    """Real unit tests."""
    @classmethod
    def setup_class(cls):
        cls.dum = 'bubu' + MP_FILE_EXTENSION

    def test_save_data(self):
        struct = {'a': 0.1, 'b': np.longdouble('123.4567890123456789'),
                  'c': np.longdouble([[-0.5, 3.5]]),
                  'd': 1}
        save_data(struct, self.dum)
        struct2 = load_data(self.dum)
        assert np.allclose(struct['a'], struct2['a'])
        assert np.allclose(struct['b'], struct2['b'])
        assert np.allclose(struct['c'], struct2['c'])
        assert np.allclose(struct['d'], struct2['d'])

    def test_load_events(self):
        events = EventList([0, 2, 3.], pi=[1, 2, 3], mjdref=54385.3254923845,
                           gti = np.longdouble([[-0.5, 3.5]]))
        save_events(events, self.dum)
        events2 = load_events(self.dum)
        assert np.allclose(events.time, events2.time)
        assert np.allclose(events.pi, events2.pi)
        assert np.allclose(events.mjdref, events2.mjdref)
        assert np.allclose(events.gti, events2.gti)

    def test_high_precision_split1(self):
        C_I, C_F, C_l, k = \
            _split_high_precision_number("C", np.double(0.01), 8)
        assert C_I == 1
        np.testing.assert_almost_equal(C_F, 0, 6)
        assert C_l == -2
        assert k == "double"

    def test_high_precision_split2(self):
        C_I, C_F, C_l, k = \
            _split_high_precision_number("C", np.double(1.01), 8)
        assert C_I == 1
        np.testing.assert_almost_equal(C_F, np.double(0.01), 6)
        assert C_l == 0
        assert k == "double"

    @classmethod
    def teardown_class(cls):
        os.unlink(cls.dum)