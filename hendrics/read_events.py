# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Read and save event lists from FITS files."""

import warnings
import os
import numpy as np
from astropy import log
from stingray.utils import assign_value_if_none
from stingray.events import EventList
from stingray.gti import cross_two_gtis
from .io import load_events
from .base import common_name
from .base import hen_root
from .io import save_events, load_events_and_gtis
from .io import HEN_FILE_EXTENSION


def treat_event_file(
    filename,
    noclobber=False,
    gti_split=False,
    min_length=4,
    gtistring=None,
    length_split=None,
    randomize_by=None,
):
    """Read data from an event file, with no external GTI information.

    Parameters
    ----------
    filename : str

    Other Parameters
    ----------------
    noclobber: bool
        if a file is present, do not overwrite it
    gtistring: str
        comma-separated set of GTI strings to consider
    gti_split: bool
        split the file in multiple chunks, containing one GTI each
    length_split: float, default None
        split the file in multiple chunks, with approximately this length
    min_length: float
        minimum length of GTIs accepted (only if gti_split is True or
        length_split is not None)
    """
    gtistring = assign_value_if_none(gtistring, "GTI,GTI0,STDGTI")
    log.info("Opening %s" % filename)

    data = load_events_and_gtis(filename, gtistring=gtistring)

    events = data.ev_list
    mission = events.mission
    instr = events.instr.lower()
    gtis = events.gti
    detector_id = data.detector_id

    if randomize_by is not None:
        events.time += np.random.uniform(
            -randomize_by / 2, randomize_by / 2, events.time.size
        )

    if detector_id is not None:
        detectors = np.array(list(set(detector_id)))
    else:
        detectors = [None]
    outfile_root = (
        hen_root(filename) + "_" + mission.lower() + "_" + instr.lower()
    )

    if randomize_by is not None:
        outfile_root += f"_randomize_by_{randomize_by:g}s"

    output_files = []
    for d in detectors:
        if d is not None:
            good_det = d == data.detector_id
            outroot_local = "{0}_det{1:02d}".format(outfile_root, d)

        else:
            good_det = np.ones_like(events.time, dtype=bool)
            outroot_local = outfile_root

        outfile = outroot_local + "_ev" + HEN_FILE_EXTENSION
        if (
            noclobber
            and os.path.exists(outfile)
            and (not (gti_split or length_split))
        ):
            warnings.warn(
                "{0} exists and using noclobber. Skipping".format(outfile)
            )
            return

        if gti_split or (length_split is not None):
            lengths = np.array([g1 - g0 for (g0, g1) in gtis])
            gtis = gtis[lengths >= min_length]

            if length_split:
                gti0 = np.arange(gtis[0, 0], gtis[-1, 1], length_split)
                gti1 = gti0 + length_split
                gti_chunks = np.array(
                    [[g0, g1] for (g0, g1) in zip(gti0, gti1)]
                )
                label = "chunk"
            else:
                gti_chunks = gtis
                label = "gti"

            for ig, g in enumerate(gti_chunks):
                outfile_local = (
                    "{0}_{1}{2:03d}_ev".format(outroot_local, label, ig)
                    + HEN_FILE_EXTENSION
                )

                good_gtis = cross_two_gtis([g], gtis)
                if noclobber and os.path.exists(outfile_local):
                    warnings.warn(
                        "{0} exists, ".format(outfile_local)
                        + "and noclobber option used. Skipping"
                    )
                    return
                good = np.logical_and(events.time >= g[0], events.time < g[1])
                all_good = good_det & good
                if len(events.time[all_good]) < 1:
                    continue
                events_filt = EventList(
                    events.time[all_good],
                    pi=events.pi[all_good],
                    gti=good_gtis,
                    mjdref=events.mjdref,
                )
                events_filt.instr = events.instr.lower()
                events_filt.header = events.header
                events_filt.mission = events.mission
                if hasattr(events, "cal_pi") and events.cal_pi is not None:
                    events_filt.cal_pi = events.cal_pi[all_good]
                save_events(events_filt, outfile_local)
                output_files.append(outfile_local)
        else:
            events_filt = EventList(
                events.time[good_det],
                pi=events.pi[good_det],
                gti=events.gti,
                mjdref=events.mjdref,
            )
            events_filt.instr = events.instr.lower()
            events_filt.header = events.header
            events_filt.mission = events.mission
            if hasattr(events, "cal_pi") and events.cal_pi is not None:
                events_filt.cal_pi = events.cal_pi[good_det]
            save_events(events_filt, outfile)
            output_files.append(outfile)
    return output_files


def _wrap_fun(arglist):
    f, kwargs = arglist
    try:
        return treat_event_file(f, **kwargs)
    except IndexError:
        log.error(f"Empty or corrupt event file: {f}")
    except Exception as e:
        log.error(f"Unknown error: {f}")
        log.error(f"{str(e)}")


def multiple_event_concatenate(event_lists):
    """
    Join multiple :class:`EventList` objects into one.

    If both are empty, an empty :class:`EventList` is returned.

    GTIs are crossed if the event lists are over a common time interval,
    and appended otherwise.

    ``pi`` and ``pha`` remain ``None`` if they are ``None`` in both.
    Otherwise, 0 is used as a default value for the :class:`EventList` where
    they were None.

    Parameters
    ----------
    event_lists : list of :class:`EventList` object
        :class:`EventList` objects that we are joining

    Returns
    -------
    `ev_new` : :class:`EventList` object
        The resulting :class:`EventList` object.
    """

    ev_new = EventList()

    gtis = np.concatenate([ev.gti for ev in event_lists])
    order = np.argsort(gtis[:, 0])
    gtis = gtis[order]

    ev_new.time = np.concatenate([ev.time for ev in event_lists])
    order = np.argsort(ev_new.time)
    ev_new.time = ev_new.time[order]

    if hasattr(event_lists[0], "pi") and event_lists[0].pi is not None:
        ev_new.pi = np.concatenate([ev.pi for ev in event_lists])[order]
    if hasattr(event_lists[0], "energy") and event_lists[0].energy is not None:
        ev_new.energy = np.concatenate([ev.energy for ev in event_lists])[
            order
        ]

    ev_new.mjdref = event_lists[0].mjdref
    ev_new.gti = gtis

    return ev_new


def join_eventlists(event_file1, event_file2, new_event_file=None):
    """Join two event files.

    Parameters
    ----------
    event_file1 : str
        First event file
    event_file2 : str
        Second event file

    Other parameters
    ----------------
    new_event_file : str, default None
        Output event file. If not specified uses `hendrics.utils.common_name`
        to figure out a good name to use mixing up the two input names.

    Returns
    -------
    new_event_file : str
        Output event file
    """
    if new_event_file is None:
        new_event_file = (
            common_name(event_file1, event_file2) + "_ev" + HEN_FILE_EXTENSION
        )

    events1 = load_events(event_file1)
    events2 = load_events(event_file2)
    if events2.time.size == 0 or events2.gti.size == 0:
        warnings.warn(f"{event_file2} has no good events")
        return None

    if events2.mjdref != events1.mjdref:
        warnings.warn("Different missions detected; changing MJDREF")
        time_diff = (events1.mjdref - events2.mjdref) * 86400
        events2.time -= time_diff
        events2.mjdref = events1.mjdref
        events2.gti -= time_diff

    events = events1.join(events2)
    if hasattr(events2, "header"):
        events.header = events1.header
    if events1.instr.lower() != events2.instr.lower():
        events.instr = ",".join([e.instr.lower() for e in [events1, events2]])

    save_events(events, new_event_file)

    return new_event_file


def join_many_eventlists(eventfiles, new_event_file=None):
    """Join two event files.

    Parameters
    ----------
    event_files : list of str
        List of event files

    Other parameters
    ----------------
    new_event_file : str, default None
        Output event file. If not specified ``joint_ev`` + HEN_FILE_EXTENSION

    Returns
    -------
    new_event_file : str
        Output event file
    """
    if new_event_file is None:
        new_event_file = "joint_ev" + HEN_FILE_EXTENSION

    N = len(eventfiles)
    first_events = load_events(eventfiles[0])
    all_events = [first_events]
    for i, event_file in enumerate(eventfiles[1:]):
        log.info(f"Reading {event_file} ({i + 1}/{N})")
        events = load_events(event_file)
        if not np.isclose(events.mjdref, first_events.mjdref):
            warnings.warn(f"{event_file} has a different MJDREF")
            continue
        if hasattr(events, "instr") and not events.instr == first_events.instr:
            warnings.warn(f"{event_file} is from a different instrument")
            continue
        if (
            events.time.size == 0
            or events.gti.size == 0
            or not np.all(
                [
                    events.time[0] < events.gti.max(),
                    events.time[-1] > events.gti.min(),
                ]
            )
        ):
            warnings.warn(f"{event_file} has no good events")
            continue

        all_events.append(events)

    events = multiple_event_concatenate(all_events)
    save_events(events, new_event_file)
    return new_event_file


def split_eventlist(fname, max_length, overlap=None):
    root = hen_root(fname)
    ev = load_events(fname)

    if overlap is None:
        overlap = 0
    if overlap >= 1:
        raise ValueError("Overlap cannot be >=1. Exiting.")

    event_times = ev.time
    GTI = ev.gti
    t0 = GTI[0, 0]
    count = 0
    from .base import nchars_in_int_value

    nchars = nchars_in_int_value((GTI.max() - t0) / max_length)

    all_files = []
    while t0 < GTI.max():
        t1 = min(t0 + max_length, GTI.max())
        if t1 - t0 < max_length / 2:
            break
        idx_start = np.searchsorted(event_times, t0)
        idx_stop = np.searchsorted(event_times, t1)
        gti_local = cross_two_gtis(GTI, [[t0, t1]])

        local_times = event_times[idx_start:idx_stop]
        new_ev = EventList(time=local_times, gti=gti_local)
        for attr in ["pi", "energy"]:
            if hasattr(ev, attr) and getattr(ev, attr) is not None:
                setattr(new_ev, attr, getattr(ev, attr)[idx_start:idx_stop])
        for attr in ["mission", "instr", "mjdref"]:
            if hasattr(ev, attr) and getattr(ev, attr) is not None:
                setattr(new_ev, attr, getattr(ev, attr))
        newfname = root + f"_{count:0{nchars}d}" + HEN_FILE_EXTENSION

        save_events(new_ev, newfname)
        all_files.append(newfname)
        t0 = t0 + max_length * (1.0 - overlap)
        count += 1
    return all_files


def main_join(args=None):
    """Main function called by the `HENjoinevents` command line script."""
    import argparse

    description = (
        "Read a cleaned event files and saves the relevant "
        "information in a standard format"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="Files to join", type=str, nargs="+")
    parser.add_argument(
        "-o", "--output", type=str, help="Name of output file", default=None
    )
    args = parser.parse_args(args)

    if len(args.files) == 2:
        return join_eventlists(
            args.files[0], args.files[1], new_event_file=args.output
        )
    else:
        return join_many_eventlists(args.files, new_event_file=args.output)


def main_splitevents(args=None):
    """Main function called by the `HENsplitevents` command line script."""
    import argparse

    description = (
        "Reads a cleaned event files and splits the file into "
        "overlapping multiple chunks of fixed length"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("fname", help="File 1", type=str)
    parser.add_argument(
        "-l",
        "--length-split",
        help="Split event list by GTI",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--overlap",
        type=float,
        help="Overlap factor. 0 for no overlap, 0.5 for "
        "half-interval overlap, and so on.",
        default=None,
    )

    args = parser.parse_args(args)

    return split_eventlist(
        args.fname, max_length=args.length_split, overlap=args.overlap
    )


def main(args=None):
    """Main function called by the `HENreadevents` command line script."""
    import argparse
    from multiprocessing import Pool
    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Read a cleaned event files and saves the relevant "
        "information in a standard format"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "--noclobber",
        help=("Do not overwrite existing event files"),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--gti-split",
        help="Split event list by GTI",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--length-split",
        help="Split event list by GTI",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--min-length",
        type=int,
        help="Minimum length of GTIs to consider",
        default=0,
    )
    parser.add_argument(
        "--gti-string", type=str, help="GTI string", default=None
    )
    parser.add_argument(
        "--randomize-by",
        type=float,
        help="Randomize event arrival times by this amount "
        "(e.g. it might be the 0.073-s frame time in "
        "XMM)",
        default=None,
    )

    _add_default_args(parser, ["output", "loglevel", "debug", "nproc"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    with log.log_to_file("HENreadevents.log"):
        argdict = {
            "noclobber": args.noclobber,
            "gti_split": args.gti_split,
            "min_length": args.min_length,
            "gtistring": args.gti_string,
            "length_split": args.length_split,
            "randomize_by": args.randomize_by,
        }

        arglist = [[f, argdict] for f in files]

        if os.name == "nt" or args.nproc == 1:
            [_wrap_fun(a) for a in arglist]
        else:
            pool = Pool(processes=args.nproc)
            for i in pool.imap_unordered(_wrap_fun, arglist):
                pass
            pool.close()
