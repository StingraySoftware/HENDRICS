import numpy as np
from hendrics import exposure


def test_exposure_calculation1():
    """Test if the exposure calculator works correctly."""
    times = np.array([1.0, 2.0, 3.0])
    events = np.array([2.0])
    priors = np.array([2.0])
    dt = np.array([1.0, 1.0, 1.0])
    expo = exposure.get_livetime_per_bin(times, events, priors, dt=dt, gti=None)
    np.testing.assert_almost_equal(expo, np.array([1, 0.5, 0.0]))


def test_exposure_calculation2():
    """Test if the exposure calculator works correctly."""
    times = np.array([1.0, 2.0])
    events = np.array([2.1])
    priors = np.array([0.3])
    dt = np.array([1.0, 1.0])
    expo = exposure.get_livetime_per_bin(times, events, priors, dt=dt, gti=None)
    np.testing.assert_almost_equal(expo, np.array([0, 0.3]))


def test_exposure_calculation3():
    """Test if the exposure calculator works correctly."""
    times = np.array([1.0, 2.0, 3.0])
    events = np.array([2.1])
    priors = np.array([0.7])
    dt = np.array([1.0, 1.0, 1.0])
    expo = exposure.get_livetime_per_bin(times, events, priors, dt=dt, gti=None)
    np.testing.assert_almost_equal(expo, np.array([0.1, 0.6, 0.0]))


def test_exposure_calculation4():
    """Test if the exposure calculator works correctly."""
    times = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    events = np.array([2.6])
    priors = np.array([1.5])
    dt = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    expected_expo = np.array([0.15, 0.5, 0.5, 0.35, 0])
    expo = exposure.get_livetime_per_bin(times, events, priors, dt=dt, gti=None)
    np.testing.assert_almost_equal(expo, expected_expo)


def test_exposure_calculation5():
    """Test if the exposure calculator works correctly."""
    times = np.array([1.0, 2.0, 3.0])
    events = np.array([1.1, 1.2, 1.4, 1.5, 1.8, 4])
    # dead time = 0.05
    priors = np.array([0.55, 0.05, 0.15, 0.05, 0.25, 2.15])
    dt = np.array([1, 1, 1])
    expected_expo = np.array([0.8, 0.9, 1])
    expo = exposure.get_livetime_per_bin(times, events, priors, dt=dt, gti=None)
    np.testing.assert_almost_equal(expo, expected_expo)
