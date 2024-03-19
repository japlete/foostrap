# -*- coding: utf-8 -*-
import numpy as np
from numba import get_num_threads
from dataclasses import dataclass
from .preproc import preproc_args

@dataclass
class BootRes:
    # Bootstrap results. Confidence interval tuple and array of samples
    ci: tuple
    boot_samples: np.array

def foostrap(x1, x2= None, statistic= 'auto', q= 0.5, boot_samples= 10000, conf_lvl= 0.95, alternative= 'two-sided', ci_method= 'BCa', random_state= None, parallel= True, ignore_sparse_below= 0.1):
    '''
    Perform parallel bootstrap sampling and confidence interval estimation.
    The available statistics are:
    - For 1-dimensional data: mean, standard deviation, quantile q
    - For 2-dimensional paired data: ratio of sums, weighted mean and pearson correlation
    To compare two independent samples (e.g. mean difference), use the argument x2 for the second sample.

    Parameters:
    - x1 (numpy.ndarray): Primary sample array. If observations are paired, the shape must be a 2-column array.
    - x2 (numpy.ndarray, optional): Second sample to compare against x1, as statistic(x1) - statistic(x2). Default is None.
    - statistic (one of 'mean','std','quantile','ratio','wmean','pearson','auto'): the statistic to compute over each sample. Default 'auto' ('mean' for 1D sample, 'ratio' for 2D sample).
    - q (float): probability for the 'quantile' statistic. Ignored otherwise. Default 0.5 (the median)
    - boot_samples (int): Number of bootstrap samples to generate. Must be greater than zero. Default is 10000.
    - conf_lvl (float): Confidence level for the confidence interval, between 0 and 1. Default is 0.95.
    - alternative (str): Type of confidence interval. 'two-sided': with upper and lower bound, 'less': only upper bound, 'greater': only lower bound. Default 'two-sided'.
    - ci_method (str): Method for confidence interval estimation. Options are 'BCa' for Bias-Corrected and Accelerated or 'percentile'. Default is 'BCa'.
    - random_state (None, int, numpy.random.Generator, or numpy.random.SeedSequence): Seed or numpy random generator for reproducibility. Default is None.
    - parallel (bool): Whether to use parallel processing for bootstrap sampling. Default is True.
    - ignore_sparse_below (float): Threshold under which sparse data is treated as dense, to avoid the overhead of a separate sampling. Between 0 and 1. Default is 0.1.
    
    Returns:
    BootRes: A dataclass containing the confidence interval (ci) as a tuple and the bootstrap samples (boot_samples) as a numpy array.
    
    Example 1: mean statistic, 1-sample
    >>> x1 = np.random.normal(0, 1, size=100)
    >>> result = foostrap(x1)
    >>> print(result.ci)
    (-0.199, 0.204)

    Example 2: weighted mean difference statistic, 2-sample
    >>> x1 = np.random.lognormal(0, 1, size= (100, 2))
    >>> x2 = np.random.lognormal(0, 1, size= (100, 2))
    >>> result = foostrap(x1, x2, statistic= 'wmean')
    >>> print(result.ci)
    (-1.04, 0.194)
    '''
    # Validate and preprocess arguments
    x1t, x2t, rng, parallel, n1, n2, ci_alphas, sampler_func1, sampler_func2, stat_name, jack_func, ci_func = \
        preproc_args(x1, x2, statistic, q, boot_samples, conf_lvl, alternative, ci_method, random_state, parallel, ignore_sparse_below)

    # If parallel, split generators and call the numba multicore wrapper
    if parallel:
        gens = tuple(rng.spawn(get_num_threads()))
    else:
        gens = (rng,)
    boot_stat = sampler_func1(stat_name, x1t, gens, n1, boot_samples, q)
    if n2 > 0:
        boot_stat -= sampler_func2(stat_name, x2t, gens, n2, boot_samples, q)

    # Get confidence intervals for the bootstrap samples
    quants = ci_func(boot_stat, ci_alphas, jack_func, x1t, x2t, n1, n2)
    return BootRes(ci= tuple(quants), boot_samples= boot_stat)
