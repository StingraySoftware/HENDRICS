import os
import subprocess as sp
import time
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from stingray import AveragedPowerspectrum, EventList
from stingray.fourier import positive_fft_bins
from stingray.gti import time_intervals_from_gtis
from stingray.io import FITSTimeseriesReader
from stingray.loggingconfig import logger
from stingray.utils import histogram


def get_data_intervals(interval_idxs, info=None, fname=None, sample_time=None):
    """
    Generate light curves for specified time intervals from event data.
    Parameters
    ----------
    interval_idxs : array-like
        Indices specifying which intervals to extract from `info["interval_times"]`.
    info : dict, optional
        Dictionary containing metadata, must include "interval_times" key with time intervals.
    fname : str, optional
        Path to the FITS file containing event data.
    sample_time : float, optional
        The time resolution (bin width) for the light curve histogram.
    Yields
    ------
    lc : array-like
        Histogrammed light curve for each specified interval.
    Notes
    -----
    - Uses FITSTimeseriesReader to read event data from the FITS file.
    - Each yielded light curve corresponds to one interval in `interval_idxs`.
    - The number of bins is determined by the interval duration and `sample_time`.
    """

    tsreader = FITSTimeseriesReader(fname, output_class=EventList)
    time_intervals = info["interval_times"][interval_idxs]
    if np.shape(time_intervals) == (2,):
        time_intervals = [time_intervals]

    # This creates a generator of event lists
    event_lists = tsreader.filter_at_time_intervals(time_intervals)

    for ev, t_int in zip(event_lists, time_intervals):
        nbin = int(np.rint((t_int[1] - t_int[0]) / sample_time))
        lc = histogram(ev.time, bins=nbin, range=(t_int[0], t_int[1]))
        yield lc


def single_rank_intervals(this_ranks_intervals, sample_time=None, info=None, fname=None):
    """
    Generate an averaged powerspectrum from light curve intervals for a single rank.
    Parameters
    ----------
    this_ranks_intervals : list or array-like
        List of intervals (e.g., time ranges) assigned to the current rank.
    sample_time : float, optional
        The time resolution (bin size) for the light curve data.
    info : dict, optional
        Dictionary containing metadata about the intervals, including "interval_times".
    fname : str or None, optional
        Filename or path to the data source, if required by `get_data_intervals`.
    Returns
    -------
    pds : AveragedPowerspectrum
        The averaged powerspectrum computed from the light curve intervals.
    nbin : int
        Number of bins in each interval, calculated from the interval duration and sample time.
    Notes
    -----
    This function extracts light curve data for the specified intervals, computes the number of bins,
    and returns the averaged powerspectrum using the "leahy" normalization.
    """

    # print(kwargs)
    t_int = info["interval_times"][0]
    nbin = int(np.rint((t_int[1] - t_int[0]) / sample_time))
    lc_iterable = get_data_intervals(
        this_ranks_intervals, info=info, fname=fname, sample_time=sample_time
    )
    intv = info["interval_times"][0]
    segment_size = intv[1] - intv[0]
    pds = AveragedPowerspectrum.from_lc_iterable(
        lc_iterable,
        segment_size=segment_size,
        dt=sample_time,
        norm="leahy",
        silent=True,
        use_common_mean=False,
    )
    return pds, nbin


def main_none(fname, sample_time, segment_size):
    """
    Process a FITS timeseries file and compute an averaged powerspectrum.
    This function reads event data from a FITS file, processes it using the Stingray
    library, and computes an averaged powerspectrum with Leahy normalization.

    Parameters
    ----------
    fname : str
        Path to the FITS file containing the timeseries data.
    sample_time : float
        The time resolution (bin size) to use when processing the data.
    segment_size : float
        The size of each segment (in seconds) for averaging the powerspectrum.
    Returns
    -------
    freq : numpy.ndarray
        Array of frequency values for the computed powerspectrum.
    power : numpy.ndarray
        Array of power values corresponding to each frequency.

    Notes
    -----
    This function uses the standard Stingray processing pipeline and does not
    parallelize the computation.
    """

    logger.info("Using standard Stingray processing")
    tsreader = FITSTimeseriesReader(fname, output_class=EventList)

    data = tsreader[:]
    pds = AveragedPowerspectrum.from_events(
        data,
        dt=sample_time,
        segment_size=segment_size,
        norm="leahy",
        use_common_mean=False,
    )

    return pds


def main_mpi(fname, sample_time, segment_size):
    """
    Perform parallel processing of time series data using MPI.
    This function distributes the processing of time series intervals across multiple MPI ranks.
    Each rank processes a subset of intervals, computes partial results, and then combines them
    using a binary tree reduction algorithm to obtain the final result.
    Algorithm:
        1. The root rank (rank 0) loads time series data and computes interval boundaries.
        2. The interval information is broadcasted to all ranks.
        3. Each rank determines its assigned intervals and processes them using `single_rank_intervals`.
        4. All ranks synchronize using MPI barriers.
        5. Partial results are combined using a binary tree reduction, where ranks pair up and
           send/receive data until only one rank holds the final result.
        6. The final result is normalized and frequency bins are computed.
    Parameters
    ----------
    fname : str
        Path to the FITS file containing the time series data.
    sample_time : float
        The sampling time for the time series analysis.
    segment_size : float
        The size of each segment (interval) to be processed.
    Returns
    -------
    output : AveragedPowerspectrum
        The averaged power spectrum computed across all intervals and processes.

    Notes
    -----
    - This function requires an MPI environment and assumes that the necessary MPI communicator
      and dependencies are available.
    - Only the rank responsible for the final reduction returns the results; other ranks return None.
    """

    tsreader = FITSTimeseriesReader(fname, output_class=EventList)

    def data_lookup():
        # This will also contain the boundaries of the data
        # to be loaded
        start, stop = time_intervals_from_gtis(tsreader.gti, segment_size)
        interval_times = np.array(list(zip(start, stop)))
        return {
            "gtis": tsreader.gti,
            "interval_times": interval_times,
            "n_intervals": len(interval_times),
        }

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    info = None
    if my_rank == 0:
        logger.debug(f"{my_rank}: Loading data")
        info = data_lookup()

    info = world_comm.bcast(info, root=0)

    if my_rank == 0:
        logger.debug(f"{my_rank}: Info:", info)

    total_n_intervals = info["n_intervals"]

    intervals_per_rank = total_n_intervals / world_size
    all_intervals = np.arange(total_n_intervals, dtype=int)

    this_ranks_intervals = all_intervals[
        (all_intervals >= my_rank * intervals_per_rank)
        & (all_intervals < (my_rank + 1) * intervals_per_rank)
    ]
    logger.debug(
        f"{my_rank}: Intervals {this_ranks_intervals[0] + 1} " f"to {this_ranks_intervals[-1] + 1}"
    )

    # data = get_data_intervals(this_ranks_intervals)
    result, data_size = single_rank_intervals(
        this_ranks_intervals, info=info, fname=fname, sample_time=sample_time
    )

    world_comm.Barrier()

    # NOW, the binary tree reduction
    # Perform binary tree reduction
    totals = result.power * result.m
    previous_processors = np.arange(world_size, dtype=int)

    while 1:
        new_processors = previous_processors[::2]
        parner_processors = previous_processors[1::2]

        if my_rank == 0:
            logger.debug(f"{my_rank}: New processors: {new_processors}")
        if my_rank == 0:
            logger.debug(f"{my_rank}: Partners: {parner_processors}")
        if len(parner_processors) == 0:
            if my_rank == 0:
                logger.debug(f"{my_rank}: Done.")
            break
        world_comm.Barrier()

        for i, (sender, receiver) in enumerate(zip(parner_processors, new_processors)):
            if my_rank == 0:
                logger.debug(f"Loop {i + 1}: {sender}, {receiver}")
            tag = 10000 + 100 * sender + receiver
            if my_rank == receiver:
                data_from_partners = np.zeros(totals.size)
                logger.debug(f"{my_rank}: Receiving from {sender} with tag {tag}")
                # Might be good to use a non-blocking receive here
                world_comm.Recv(
                    data_from_partners,
                    source=sender,
                    tag=tag,
                )
                totals += data_from_partners
                logger.debug(f"{my_rank}: New data are now {totals}")
            elif my_rank == sender:
                # Only one partner for now. Might be tweaked differently. The rest should work with
                # any number of partners for a given processing rank

                logger.debug(f"{my_rank}: Sending to {receiver} with tag {tag}")
                world_comm.Send(totals, dest=receiver, tag=tag)
            else:
                logger.debug(f"{my_rank}: Doing nothing")

            world_comm.Barrier()
            previous_processors = new_processors

    world_comm.Barrier()

    assert len(new_processors) == 1
    if my_rank == new_processors[0]:
        logger.debug("Results")
        totals /= total_n_intervals

        freq = np.fft.fftfreq(data_size, d=sample_time)[positive_fft_bins(data_size)]
        output = AveragedPowerspectrum()
        output.freq = freq
        output.power = totals
        output.m = total_n_intervals
        output.gti = info["gtis"]
        output.dt = sample_time

        return output

    return None


def main_multiprocessing(fname, sample_time, segment_size, world_size=8):
    """Run parallel analysis on time series data using Python's multiprocessing.

    This function divides the input time series data into segments and distributes the analysis
    across multiple processes. Each process computes results for a subset of intervals, and the
    results are aggregated to produce the final output.
    Algorithm:
        1. Load time series data and determine Good Time Intervals (GTIs).
        2. Split the data into segments of specified size.
        3. Assign segments to worker processes based on the number of available processes (`world_size`).
        4. Each process analyzes its assigned intervals using `single_rank_intervals`.
        5. Aggregate results from all processes and compute the average power spectrum.
        6. Return the frequency array and the averaged power spectrum.

    Parameters
    ----------
    fname : str
        Path to the FITS file containing the time series data.
    sample_time : float
        The time resolution of the data samples.
    segment_size : float
        The size (in seconds) of each segment to analyze.
    world_size : int, optional
        Number of parallel worker processes to use (default is 8).

    Returns
    -------
    output : AveragedPowerspectrum
        The averaged power spectrum computed across all intervals and processes.

    """

    def data_lookup():
        # This will also contain the boundaries of the data
        # to be loaded
        tsreader = FITSTimeseriesReader(fname, output_class=EventList)

        start, stop = time_intervals_from_gtis(tsreader.gti, segment_size)
        interval_times = np.array(list(zip(start, stop)))
        return {
            "gtis": tsreader.gti,
            "interval_times": interval_times,
            "n_intervals": len(interval_times),
        }

    info = data_lookup()
    data_size = np.rint(segment_size / sample_time).astype(int)

    logger.debug("Info:", info)

    total_n_intervals = info["n_intervals"]

    # intervals_per_rank = total_n_intervals / world_size
    all_intervals = np.arange(total_n_intervals, dtype=int)

    intervals_per_rank = total_n_intervals / world_size
    this_ranks_intervals = []
    for my_rank in range(world_size):
        this_ranks_intervals.append(
            all_intervals[
                (all_intervals >= my_rank * intervals_per_rank)
                & (all_intervals < (my_rank + 1) * intervals_per_rank)
            ]
        )

    p = Pool(world_size)

    totals = 0
    for results, data_size in p.imap_unordered(
        partial(
            single_rank_intervals,
            info=info,
            fname=fname,
            sample_time=sample_time,
        ),
        this_ranks_intervals,
    ):
        totals += results.power * results.m
    logger.debug("Results")
    totals /= total_n_intervals

    freq = np.fft.fftfreq(data_size, d=sample_time)[positive_fft_bins(data_size)]

    output = AveragedPowerspectrum()
    output.freq = freq
    output.power = totals
    output.m = total_n_intervals
    output.gti = info["gtis"]
    output.dt = sample_time

    return output


def main(args=None):
    import argparse

    from hendrics.base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Compute the Leahy-normalized power spectrum of an event list in parallel.\n"
        "The MPI version needs to be run with mpiexec, as follows:\n"
        "mpiexec -n 10 python HENparfspec filename.fits --method mpi\n"
        "To run the algorithm in parallel using multiprocessing, use:\n"
        "python HENparfspec filename.fits --method multiprocessing --nproc 10\n"
        "To run the algorithm sequentially, for testing purposes, just execute"
        "HENparfspec filename.fits\n"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("fname", help="Input FITS file name")
    parser.add_argument("-o", "--outfname", help="Output FITS file name", default="out_pds.fits")
    parser.add_argument(
        "-b",
        "--sample_time",
        type=float,
        default=1 / 8129 / 2,
        help="Light curve bin time; if negative, interpreted"
        + " as negative power of 2."
        + " Default: 2^-13, or keep input lc bin time"
        + " (whatever is larger)",
    )

    parser.add_argument(
        "-f",
        "--segment_size",
        type=float,
        default=128,
        help="Length of FFTs. Default: 16 s",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="leahy",
        help="Normalization to use" + " (Accepted: leahy and rms;" + ' Default: "leahy")',
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mpi", "multiprocessing", "none"],
        default="none",
        help="Computation distribution method",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="Number of processors to use",
    )
    _add_default_args(parser, ["loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    logger.setLevel(args.loglevel)

    fname = args.fname

    sample_time = args.sample_time
    segment_size = args.segment_size

    if args.method == "mpi":
        # This method needs to be run with mpiexec -n <nproc> python script.py
        pds = main_mpi(fname, sample_time, segment_size)
    elif args.method == "multiprocessing":
        pds = main_multiprocessing(fname, sample_time, segment_size, world_size=args.nproc)
    else:
        pds = main_none(fname, sample_time, segment_size)
    if pds is None:
        return

    pds.write(args.outfname)
