import os
import pytest
from astropy.tests.helper import remote_data
import numpy as np
from hendrics.base import deorbit_events
from stingray.events import EventList
from hendrics.tests import _dummy_par
from hendrics.fold import HAS_PINT


def test_deorbit_badpar():
    ev = np.asarray(1)
    with pytest.warns(UserWarning) as record:
        ev_deor = deorbit_events(ev, None)
    assert np.any(
        ["No parameter file specified" in r.message.args[0] for r in record]
    )
    assert ev_deor == ev


def test_deorbit_non_existing_par():
    ev = np.asarray(1)
    with pytest.raises(FileNotFoundError) as excinfo:
        ev_deor = deorbit_events(ev, "warjladsfjqpeifjsdk.par")
    assert "Parameter file warjladsfjqpeifjsdk.par does not exist" in str(
        excinfo.value
    )


@remote_data
@pytest.mark.skipif("not HAS_PINT")
def test_deorbit_bad_mjdref():
    from hendrics.base import deorbit_events

    ev = EventList(np.arange(100), gti=np.asarray([[0, 2]]))
    ev.mjdref = 2
    par = _dummy_par("bububu.par")
    with pytest.raises(ValueError) as excinfo:
        _ = deorbit_events(ev, par)
    assert "MJDREF is very low (<01-01-1950), " in str(excinfo.value)
    os.remove("bububu.par")


@remote_data
@pytest.mark.skipif("not HAS_PINT")
def test_deorbit_inverse():
    from hendrics.base import deorbit_events

    ev = EventList(
        np.sort(np.random.uniform(0, 1000, 10)),
        gti=np.asarray([[0, 1000]]),
        mjdref=55000,
    )
    par = _dummy_par("bububu.par", pb=1.0, a1=30)
    ev2 = deorbit_events(ev, par)
    ev3 = deorbit_events(ev, par, invert=True)
    assert np.allclose(ev.time - ev2.time, -(ev.time - ev3.time), atol=1e-6)
    os.remove("bububu.par")


@remote_data
@pytest.mark.skipif("not HAS_PINT")
def test_deorbit_run():
    from hendrics.base import deorbit_events

    ev = EventList(np.arange(0, 210000, 1000), gti=np.asarray([[0.0, 210000]]))

    ev.mjdref = 56000.0
    ev.ephem = "de200"
    par = _dummy_par("bububu.par")
    with pytest.warns(UserWarning) as record:
        _ = deorbit_events(ev, par)
    assert np.any(
        ["The observation is very long." in r.message.args[0] for r in record]
    )
    os.remove("bububu.par")
