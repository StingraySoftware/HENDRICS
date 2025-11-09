import numpy as np
from stingray.utils import _allocate_array_or_memmap

from numba import config, cuda
import cupy as cp

config.CUDA_ENABLE_PYNVJITLINK =1

@cuda.jit()
def _hist1d_numba_gpu(H, tracks, bins, range_min, range_max):
    '''
    CUDA kernel for computing a 1D histogram on the GPU.

    Parameters
    ----------
    H : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Device array representing the histogram bins to be filled
    tracks : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Device array of input values for which the histogram is computed
    bins : int
        Total number of histogram bins
    range_min : float
        Lower bound of the histogram range
    range_max : float
        Upper bound of the histogram range
   '''
        
        delta = bins/(range_max-range_min)

        tid = cuda.grid(1)

        if tid <= tracks.size:
            i = (tracks[tid]-range_min) * delta
            if 0 <= i < bins:
                cuda.atomic.add(H, int(i), 1)


def hist1d_numba_gpu(a, bins, ranges, use_memmap=True, tmp=None):
    '''
    Compute a 1D histogram of input data using a custom CUDA kernel on the GPU

    Parameters
    ----------
    a : numpy.ndarray
        Input array of values to histogram
    bins : int
        Number of bins
    ranges : tuple of float
        The (min, max) range of values for the histogram
    use_memmap : bool, optional
        If True (default), allocate the histogram array as a memory-mapped 
        file using `_allocate_array_or_memmap`; If False, allocate a regular 
        NumPy array
    tmp : str or None, optional
        Temporary directory to use for memory-mapped file creation, if 
        `use_memmap=True`

    Returns
    -------
    H : numpy.ndarray
        The computed histogram as a 1D NumPy array of length `bins`
    '''
    
        hist_arr = _allocate_array_or_memmap((bins), a.dtype, use_memmap=use_memmap, tmp=tmp)

        d_tracks = cuda.to_device(a)
        d_hist_arr = cuda.to_device(hist_arr)


        threads_per_block = 256
        blocks_per_grid = (a.size+threads_per_block-1) // threads_per_block

        _hist1d_numba_gpu[blocks_per_grid, threads_per_block](d_hist_arr, d_tracks, bins,  ranges[0], ranges[1])
        H = d_hist_arr.copy_to_host()
        return H


def hist1d_numba_gpu_fft(a, bins, ranges, use_memmap=True, tmp=None):
    '''
    Compute a 1D histogram on the GPU using a CUDA kernel and return its FFT
    
    Parameters
    ----------
    a : numpy.ndarray
        Input array of values to histogram
    bins : int
        Number of bins
    ranges : tuple of float
        The (min, max) range of values for the histogram
    use_memmap : bool, optional
        If True (default), allocate the histogram array as a memory-mapped 
        file using `_allocate_array_or_memmap`; If False, allocate a regular 
        NumPy array
    tmp : str or None, optional
        Temporary directory to use for memory-mapped file creation, if 
        `use_memmap=True`
    
    Returns
    -------
    
    H_to_host : numpy.ndarray
    FFT of the computed histogram as a 1D NumPy array of complex numbers
    
    '''
    
        hist_arr = _allocate_array_or_memmap((bins), a.dtype, use_memmap=use_memmap, tmp=tmp)

        d_tracks = cp.asarray(a)
        d_hist_arr = cp.asarray(hist_arr)

        threads_per_block = 256
        blocks_per_grid = (a.size+threads_per_block-1) // threads_per_block

        _hist1d_numba_gpu[blocks_per_grid, threads_per_block](d_hist_arr, d_tracks, bins,  ranges[0], ranges[1])

        fft_H = cp.fft.fft(d_hist_arr)
        H_to_host = cp.asnumpy(fft_H)
        return H_to_host

 
