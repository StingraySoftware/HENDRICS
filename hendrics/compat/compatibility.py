import copy
from functools import wraps
import numpy as np
import stingray.utils
from stingray.events import EventList

try:
    from numba import jit, njit, prange, vectorize
    from numba import float32, float64, int32, int64
    from numba import types
    from numba.extending import overload_method

    @overload_method(types.Array, "take")  # pragma: no cover
    def array_take(arr, indices):
        """Adapt np.take to arrays"""
        if isinstance(indices, types.Array):

            def take_impl(arr, indices):
                n = indices.shape[0]
                res = np.empty(n, arr.dtype)
                for i in range(n):
                    res[i] = arr[indices[i]]
                return res

            return take_impl

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(**kwargs):
        """Dummy decorator in case jit cannot be imported."""

        def true_decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                r = func(*args, **kwargs)
                return r

            return wrapped

        return true_decorator

    jit = njit

    def prange(*args):
        """Dummy decorator in case jit cannot be imported."""
        return range(*args)

    class vectorize(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            wrapped_f = np.vectorize(func)

            return wrapped_f

    float32 = float64 = int32 = int64 = lambda x, y: None

    def array_take(arr, indices):
        """Adapt np.take to arrays"""
        return np.take(arr, indices)

    # Patch stingray.utils


def _root_squared_mean(array):
    array = np.asarray(array)
    return np.sqrt(np.sum(array ** 2)) / len(array)


stingray.utils._root_squared_mean = _root_squared_mean


def _paralyzable_dead_time(event_list, dead_time):
    """Apply paralyzable dead time to an event list.

    Parameters
    ----------
    event_list : array of floats
        Event times of arrival
    dead_time: float
        Dead time (single value)

    Returns
    -------
    output_event_list : array of floats
        Filtered event times
    mask : array of bools
        Final mask, showing all good events in the original event list.
    """
    mask = np.ones(len(event_list), dtype=bool)
    dead_time_end = event_list + dead_time
    bad = dead_time_end[:-1] > event_list[1:]
    # Easy: paralyzable case. Here, events coming during dead time produce
    # more dead time. So...
    mask[1:][bad] = False

    return event_list[mask], mask


@njit()
def _nonpar_core(event_list, dead_time_end, mask):
    """Numba-compiled core of the non-paralyzable dead time calculation.

    Parameters
    ----------
    event_list : array of floats
        Event times of arrival
    dead_time_end : array of floats
        End of the dead time of each event
    mask : array of bools
        Final mask of good events. Initially, all entries must be ``True``

    Return
    ------
    mask : array of bools
        Final mask of good events
    """
    for i in range(1, len(event_list)):
        if event_list[i] < dead_time_end[i - 1]:
            dead_time_end[i] = dead_time_end[i - 1]
            mask[i] = False
    return mask


def _non_paralyzable_dead_time(event_list, dead_time):
    """Apply non-paralyzable dead time to an event list.

    Parameters
    ----------
    event_list : array of floats
        Event times of arrival
    dead_time: float
        Dead time (single value)

    Returns
    -------
    output_event_list : array of floats
        Filtered event times
    mask : array of bools
        Final mask, showing all good events in the original event list.
    """
    event_list_dbl = (event_list - event_list[0]).astype(np.double)
    dead_time_end = event_list_dbl + np.double(dead_time)
    mask = np.ones(event_list_dbl.size, dtype=bool)
    mask = _nonpar_core(event_list_dbl, dead_time_end, mask)
    return event_list[mask], mask


class _DeadtimeFilterOutput(object):
    uf_events = None
    is_event = None
    deadtime = None
    mask = None
    bkg = None


def get_deadtime_mask(
    ev_list,
    deadtime,
    bkg_ev_list=None,
    dt_sigma=None,
    paralyzable=False,
    return_all=False,
    verbose=False,
):
    """Filter an event list for a given dead time."""
    additional_output = _DeadtimeFilterOutput()

    # Create the total lightcurve, and a "kind" array that keeps track
    # of the events classified as "signal" (True) and "background" (False)
    if bkg_ev_list is not None:
        tot_ev_list = np.append(ev_list, bkg_ev_list)
        ev_kind = np.append(
            np.ones(len(ev_list), dtype=bool),
            np.zeros(len(bkg_ev_list), dtype=bool),
        )
        order = np.argsort(tot_ev_list)
        tot_ev_list = tot_ev_list[order]
        ev_kind = ev_kind[order]
        del order
    else:
        tot_ev_list = ev_list
        ev_kind = np.ones(len(ev_list), dtype=bool)

    additional_output.uf_events = tot_ev_list
    additional_output.is_event = ev_kind
    additional_output.deadtime = deadtime
    additional_output.uf_mask = np.ones(tot_ev_list.size, dtype=bool)
    additional_output.bkg = tot_ev_list[np.logical_not(ev_kind)]

    if deadtime <= 0.0:
        if deadtime < 0:
            raise ValueError("Dead time is less than 0. Please check.")
        retval = [np.ones(ev_list.size, dtype=bool), additional_output]
        return retval

    nevents = len(tot_ev_list)
    all_ev_kind = ev_kind.copy()

    if dt_sigma is not None:
        deadtime_values = ra.normal(deadtime, dt_sigma, nevents)
        deadtime_values[deadtime_values < 0] = 0.0
    else:
        deadtime_values = np.zeros(nevents) + deadtime

    initial_len = len(tot_ev_list)

    # Note: saved_mask gives the mask that produces tot_ev_list_filt from
    # tot_ev_list. The same mask can be used to also filter all other arrays.
    if paralyzable:
        tot_ev_list_filt, saved_mask = _paralyzable_dead_time(
            tot_ev_list, deadtime_values
        )

    else:
        tot_ev_list_filt, saved_mask = _non_paralyzable_dead_time(
            tot_ev_list, deadtime_values
        )
    del tot_ev_list

    ev_kind = ev_kind[saved_mask]
    deadtime_values = deadtime_values[saved_mask]
    final_len = tot_ev_list_filt.size
    if verbose:
        log.info(
            "filter_for_deadtime: "
            "{0}/{1} events rejected".format(
                initial_len - final_len, initial_len
            )
        )

    retval = saved_mask[all_ev_kind]

    if return_all:
        # uf_events: source and background events together
        # ev_kind : kind of each event in uf_events.
        # bkg : Background events
        additional_output.uf_events = tot_ev_list_filt
        additional_output.is_event = ev_kind
        additional_output.deadtime = deadtime_values
        additional_output.bkg = tot_ev_list_filt[np.logical_not(ev_kind)]
        retval = [retval, additional_output]

    return retval


def filter_for_deadtime(event_list, deadtime, **kwargs):
    """Filter an event list for a given dead time.

    This function accepts either a list of times or a
    `stingray.events.EventList` object.

    For the complete optional parameter list, see `get_deadtime_mask`

    Parameters
    ----------
    ev_list : array-like or class:`stingray.events.EventList`
        The event list
    deadtime: float
        The (mean, if not constant) value of deadtime

    Returns
    -------
    new_ev_list : class:`stingray.events.EventList` object
        The filtered event list
    additional_output : dict
        See `get_deadtime_mask`

    """
    # Need to import here to avoid circular imports in the top module.
    from stingray.events import EventList

    local_retall = kwargs.pop("return_all", False)

    if isinstance(event_list, EventList):
        retval = event_list.apply_deadtime(
            deadtime, return_all=local_retall, **kwargs
        )
    else:
        mask, retall = get_deadtime_mask(
            event_list, deadtime, return_all=True, **kwargs
        )
        retval = event_list[mask]
        if local_retall:
            retval = [retval, retall]

    return retval


class _MonkeyPatchedEventList(EventList):
    def __init__(self, *args, **kwargs):
        EventList.__init__(self, *args, **kwargs)
        if not hasattr(self, "mission"):
            self.mission = None
        if not hasattr(self, "instr"):
            self.instr = None
        if not hasattr(self, "timesys"):
            self.timesys = None
        if not hasattr(self, "timeref"):
            self.timeref = None
        if not hasattr(self, "ephem"):
            self.ephem = None

    def apply_mask(self, mask, inplace=False):  # pragma: no cover
        """For compatibility with old stingray version.

        Examples
        --------
        >>> evt = _MonkeyPatchedEventList(time=[0, 1, 2])
        >>> newev0 = evt.apply_mask([True, True, False], inplace=False);
        >>> newev1 = evt.apply_mask([True, True, False], inplace=True);
        >>> np.allclose(newev0.time, [0, 1])
        True
        >>> np.allclose(newev1.time, [0, 1])
        True
        >>> evt is newev1
        True
        """
        if inplace:
            new_ev = self
        else:
            new_ev = copy.deepcopy(self)
        for attr in "time", "energy", "pi", "cal_pi":
            if hasattr(new_ev, attr) and getattr(new_ev, attr) is not None:
                setattr(new_ev, attr, getattr(new_ev, attr)[mask])
        return new_ev

    def apply_deadtime(self, deadtime, inplace=False, **kwargs):
        """Apply deadtime filter to this event list.

        Additional arguments in ``kwargs`` are passed to `get_deadtime_mask`

        Parameters
        ----------
        deadtime : float
            Value of dead time to apply to data
        inplace : bool, default False
            If True, apply the deadtime to the current event list. Otherwise,
            return a new event list.

        Returns
        -------
        new_event_list : `EventList` object
            Filtered event list. if `inplace` is True, this is the input object
            filtered for deadtime, otherwise this is a new object.
        additional_output : object
            Only returned if `return_all` is True. See `get_deadtime_mask` for
            more details.

        Examples
        --------
        >>> events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
        >>> events = EventList(events)
        >>> events.pi=np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        >>> events.energy=np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        >>> events.mjdref = 10
        >>> filt_events, retval = events.apply_deadtime(0.11, inplace=False,
        ...                                             verbose=False,
        ...                                             return_all=True)
        >>> filt_events is events
        False
        >>> expected = np.array([1, 2, 2.2, 3, 3.2])
        >>> np.allclose(filt_events.time, expected)
        True
        >>> np.allclose(filt_events.pi, 1)
        True
        >>> np.allclose(filt_events.energy, 1)
        True
        >>> np.allclose(events.pi, 1)
        False
        >>> filt_events = events.apply_deadtime(0.11, inplace=True,
        ...                                     verbose=False)
        >>> filt_events is events
        True
        """
        local_retall = kwargs.pop("return_all", False)

        mask, retall = get_deadtime_mask(
            self.time, deadtime, return_all=True, **kwargs
        )

        new_ev = self.apply_mask(mask, inplace=inplace)

        if local_retall:
            new_ev = [new_ev, retall]

        return new_ev
