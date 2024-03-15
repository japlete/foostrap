# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, objmode, prange

@njit(parallel= True, cache=True)
def core_parallelizer(sampler_func1, sampler_func2, stat_func1, stat_func2, x1, x2, gens, n1, n2, n_boot, arg):
    # Use numba prange to distribute bootstrap sampling iterations equally among threads. Each thread has its own generator
    nt = len(gens)
    thread_samps = np.array([n_boot // nt] * nt) + np.array([1] * (n_boot % nt) + [0] * (nt - (n_boot % nt)))
    idxs = [0] + list(np.cumsum(thread_samps))
    boot_stat = np.empty(n_boot)
    for i in prange(nt):
        local_res = sampler_func1(x1, thread_samps[i], gens[i], n1, stat_func1, arg)
        if n2 > 0:
            local_res -= sampler_func2(x2, thread_samps[i], gens[i], n2, stat_func2, arg)
        boot_stat[idxs[i] : idxs[i+1]] = local_res
    return boot_stat

@njit(fastmath=True, cache=True)
def bstrap_sampler_1d(x, n_boot, gen, n, stat_func, arg):
    # Generate n_boot bootstrap samples for 1D x using generator gen and compute the statistic stat_func
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        boot_stat[i] = stat_func(x[gen.integers(0, n, size= n, dtype= np.uint64)])
    return boot_stat

@njit(fastmath=True, cache=True)
def bstrap_sampler_1d_sparse(x, n_boot, gen, n, stat_func, arg):
    # Generate n_boot bootstrap samples for 1D sparse x using generator gen and compute the statistic stat_func
    nnz = len(x)
    p = nnz / n
    with objmode(nzboot='int64[:]'):
        nzboot = gen.binomial(n, p, n_boot)
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        boot_stat[i] = stat_func(x[gen.integers(0, nnz, size= nzboot[i], dtype= np.uint64)], n)
    return boot_stat

@njit(fastmath=True, cache=True)
def bstrap_sampler_1d_bin(x, n_boot, gen, n, stat_func, arg):
    # Generate n_boot bootstrap samples for 1D binary x using generator gen and compute the statistic stat_func
    p = x.sum() / n
    with objmode(n_ones='int64[:]'):
        n_ones = gen.binomial(n, p, n_boot)
    return stat_func(n_ones, n)

@njit(fastmath=True, cache=True)
def bstrap_sampler_1d_arg(x, n_boot, gen, n, stat_func, arg):
    # Generate n_boot bootstrap samples for 1D x using generator gen and compute the statistic stat_func, accepting an argument arg
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        boot_stat[i] = stat_func(x[gen.integers(0, n, size= n, dtype= np.uint64)], arg)
    return boot_stat

@njit(fastmath=True, cache=True)
def bstrap_sampler_1d_sparse_arg(x, n_boot, gen, n, stat_func, arg):
    # Generate n_boot bootstrap samples for 1D sparse x using generator gen and compute the statistic stat_func
    nnz = len(x)
    p = nnz / n
    with objmode(nzboot='int64[:]'):
        nzboot = gen.binomial(n, p, n_boot)
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        boot_stat[i] = stat_func(x[gen.integers(0, nnz, size= nzboot[i], dtype= np.uint64)], n, arg)
    return boot_stat

@njit(fastmath=True, cache=True)
def bstrap_sampler_1d_bin_arg(x, n_boot, gen, n, stat_func, arg):
    # Generate n_boot bootstrap samples for 1D binary x using generator gen and compute the statistic stat_func
    p = x.sum() / n
    with objmode(n_ones='int64[:]'):
        n_ones = gen.binomial(n, p, n_boot)
    return stat_func(n_ones, n, arg)

@njit(fastmath=True, cache=True)
def bstrap_sampler_2d(x, n_boot, gen, n, stat_func, arg):
    # Generate n_boot bootstrap samples for 2D x using generator gen and compute the statistic stat_func
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        boot_stat[i] = stat_func(x[:,gen.integers(0, n, size= n, dtype= np.uint64)])
    return boot_stat

sampler_map = {(False,False,False) : bstrap_sampler_1d,
               (False,True,False) : bstrap_sampler_1d_sparse,
               (True,False,False) : bstrap_sampler_1d_bin,
               (False,False,True) : bstrap_sampler_1d_arg,
               (False,True,True) : bstrap_sampler_1d_sparse_arg,
               (True,False,True) : bstrap_sampler_1d_bin_arg,
               '2d' : bstrap_sampler_2d,
              }