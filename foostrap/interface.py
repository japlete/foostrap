import numpy as np
from numba import njit, get_num_threads, prange
from dataclasses import dataclass
from .preproc import preproc_args

@dataclass
class BootRes:
    # Bootstrap results. Confidence interval tuple and array of samples
    ci: tuple
    boot_samples: np.array

@njit(parallel= True, cache=True)
def core_parallelizer(core_func, x1, x2, gens, n1, n2, n_boot):
    # Use numba to distribute bootstrap sampling iterations equally among threads. Each thread has its own generator
    nt = len(gens)
    thread_samps = np.array([n_boot // nt] * nt) + np.array([1] * (n_boot % nt) + [0] * (nt - (n_boot % nt)))
    idxs = [0] + list(np.cumsum(thread_samps))
    boot_stat = np.empty(n_boot)
    for i in prange(nt):
        boot_stat[idxs[i] : idxs[i+1]] = core_func(x1, x2, thread_samps[i], gens[i], n1, n2)
    return boot_stat

def foostrap(x1, x2= None, boot_samples= 10000, conf_lvl= 0.95, alternative= 'two-sided', ci_method= 'BCa', random_state= None, parallel= True, ignore_sparse_below= 0.1):
    '''
    Perform parallel bootstrap sampling and confidence interval estimation for one or two data samples.
    The available statistics are mean and ratio of sums, which get automatically recognized based on the number of dimensions of the input.
    If the data is sparse and using the mean statistic, the number of zeros in each sample is generated from a Binomial distribution, instead of resampling the zeros directly.

    Parameters:
    - x1 (numpy.ndarray): Primary sample array. If 1-D, the mean statistic is used. If 2-D, the ratio of sums is used, as x1[:,0].sum() / x1[:,1].sum()
    - x2 (numpy.ndarray, optional): Second sample to compare against x1, as statistic(x1) - statistic(x2). Default is None.
    - boot_samples (int): Number of bootstrap samples to generate. Must be greater than zero. Default is 10000.
    - conf_lvl (float): Confidence level for the confidence interval, between 0 and 1. Default is 0.95.
    - alternative (str): Type of confidence interval. 'two-sided': with upper and lower bound, 'less': only upper bound, 'greater': only lower bound. Default 'two-sided'.
    - ci_method (str): Method for confidence interval estimation. Options are 'BCa' for Bias-Corrected and Accelerated or 'percentile'. Default is 'BCa'.
    - random_state (None, int, numpy.random.Generator, or numpy.random.SeedSequence): Seed or numpy random generator for reproducibility. Default is None.
    - parallel (bool): Whether to use parallel processing for bootstrap sampling. Default is True.
    - ignore_sparse_below (float): Threshold under which sparse data is treated as dense, to avoid the overhead of a separate sampling. Between 0 and 1. Default is 0.1.
    
    Returns:
    BootRes: A dataclass containing the confidence interval (ci) as a tuple and the bootstrap samples (boot_samples) as a numpy array.
    
    Example:
    >>> x1 = np.random.normal(0, 1, size=100)
    >>> result = foostrap(x1)
    >>> print(result.ci)
    (-0.199, 0.204)
    '''
    # Validate and preprocess arguments
    x1t, x2t, rng, parallel, n1, n2, ci_alphas, core_func, jack_func, ci_func = preproc_args(x1, x2, boot_samples, conf_lvl, alternative, ci_method, random_state, parallel, ignore_sparse_below)

    # If parallel, split generators and call the numba multicore wrapper
    if parallel:
        nt = get_num_threads()
        gens = tuple(rng.spawn(nt))
        boot_stat = core_parallelizer(core_func, x1t, x2t, gens, n1, n2, boot_samples)
    else:
        # Single core directly calls core_func
        boot_stat = core_func(x1t, x2t, boot_samples, rng, n1, n2)

    # Get confidence intervals for the bootstrap samples
    quants = ci_func(boot_stat, ci_alphas, jack_func, x1t, x2t, n1, n2)
    return BootRes(ci= tuple(quants), boot_samples= boot_stat)
