# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange

### STATISTICS SECTION ###

@njit(fastmath=True, cache=True)
def numba_mean(a):
    # Compute the mean of a
    n = len(a)
    s = a.dtype.type(0)
    for i in range(n):
        s += a[i]
    return s / n

@njit(fastmath=True, cache=True)
def numba_mean_sparse(a, n):
    # Compute the mean of sparse a
    nnz = len(a)
    s = a.dtype.type(0)
    for i in range(nnz):
        s += a[i]
    return s / n

@njit(fastmath=True, cache=True)
def numba_mean_bin(a, n):
    # Compute the mean of binary a
    return a / n

@njit(fastmath=True, cache=True)
def numba_std(a):
    # Compute the standard deviation of a
    n = len(a)
    sx = a.dtype.type(0)
    sx2 = sx
    for i in range(n):
        sx += a[i]
        sx2 += a[i] ** 2
    return np.sqrt(sx2 / n - (sx / n) ** 2)

@njit(fastmath=True, cache=True)
def numba_std_sparse(a, n):
    # Compute the standard deviation of sparse a
    nnz = len(a)
    sx = a.dtype.type(0)
    sx2 = sx
    for i in range(nnz):
        sx += a[i]
        sx2 += a[i] ** 2
    return np.sqrt(sx2 / n - (sx / n) ** 2)

@njit(fastmath=True, cache=True)
def numba_std_bin(a, n):
    # Compute the standard deviation of binary a
    p = a / n
    return np.sqrt(p * (1.0 - p))

@njit(fastmath=True, cache=True)
def quantile_dense(a, q):
    # Compute the q-quantile of dense a
    return np.quantile(a, q)

@njit(fastmath=True, cache=True)
def quantile_sparse(a, n, q):
    # Compute the q-quantile of sparse a
    return np.quantile(np.concatenate((np.zeros(n-len(a), dtype= a.dtype), a)), q)

@njit(fastmath=True, cache=True)
def quantile_bin(a, n, q):
    # Compute the q-quantile of binary a
    nz = n - a
    idx = q * (n-1) + 1.0
    idx_lo = int(np.floor(idx))
    return 1.0 * (nz < idx_lo) + (idx - idx_lo) * (nz == idx_lo)

@njit(fastmath= True, cache=True)
def ratio_rows(a):
    # Compute the ratio of sums of consecutive rows
    n = a.shape[1]
    sn = a.dtype.type(0)
    sd = sn
    for i in range(n):
        sn += a[0,i]
        sd += a[1,i]
    if sd != 0:
        return sn / sd
    else:
        return np.nan

@njit(fastmath= True, cache=True)
def wmean_rows(a):
    # Compute the weighted mean of consecutive rows
    n = a.shape[1]
    sxw = a.dtype.type(0)
    sw = sxw
    for i in range(n):
        sxw += a[0,i] * a[1,i]
        sw += a[1,i]
    if sw != 0:
        return sxw / sw
    else:
        return np.nan

@njit(fastmath= True, cache=True)
def pearson_rows(a):
    # Compute the pearson correlation of consecutive rows
    n = a.shape[1]
    sx = a.dtype.type(0)
    sy, sxy, sx2, sy2 = sx, sx, sx, sx
    for i in range(n):
        sx += a[0,i]
        sx2 += a[0,i] ** 2
        sy += a[1,i]
        sy2 += a[1,i] ** 2
        sxy += a[0,i] * a[1,i]
    denom = np.sqrt((n * sx2 - sx ** 2)*(n * sy2 - sy ** 2))
    if denom != 0.0:
        return (n * sxy - sx * sy) / denom
    else:
        return np.nan

### SAMPLERS SECTION ###

@njit(parallel=True, fastmath=True, cache=True)
def bstrap_sampler_1d(stat_name, x, gens, n, n_boot, arg):
    # Generate n_boot bootstrap samples for 1D array x using generators gens and compute the statistic
    nt = len(gens)
    thread_samps = np.array([n_boot // nt] * nt) + np.array([1] * (n_boot % nt) + [0] * (nt - (n_boot % nt)))
    idxs = [0] + list(np.cumsum(thread_samps))
    boot_stat = np.empty(n_boot)
    for i in prange(nt):
        local_n = thread_samps[i]
        local_gen = gens[i]
        local_res = np.empty(local_n)
        if stat_name == 'mean':
            for k in range(local_n):
                local_res[k] = numba_mean(x[local_gen.integers(0, n, size= n, dtype= np.uint64)])
        elif stat_name == 'std':
            for k in range(local_n):
                local_res[k] = numba_std(x[local_gen.integers(0, n, size= n, dtype= np.uint64)])
        else: # quantile
            for k in range(local_n):
                local_res[k] = quantile_dense(x[local_gen.integers(0, n, size= n, dtype= np.uint64)], arg)
        boot_stat[idxs[i] : idxs[i+1]] = local_res
    return boot_stat
    
@njit(parallel=True, fastmath=True, cache=True)
def bstrap_sampler_2d(stat_name, x, gens, n, n_boot, arg):
    # Generate n_boot bootstrap samples for 2D array x using generators gens and compute the statistic
    nt = len(gens)
    thread_samps = np.array([n_boot // nt] * nt) + np.array([1] * (n_boot % nt) + [0] * (nt - (n_boot % nt)))
    idxs = [0] + list(np.cumsum(thread_samps))
    boot_stat = np.empty(n_boot)
    for i in prange(nt):
        local_n = thread_samps[i]
        local_gen = gens[i]
        local_res = np.empty(local_n)
        if stat_name == 'ratio':
            for k in range(local_n):
                local_res[k] = ratio_rows(x[:,local_gen.integers(0, n, size= n, dtype= np.uint64)])
        elif stat_name == 'wmean':
            for k in range(local_n):
                local_res[k] = wmean_rows(x[:,local_gen.integers(0, n, size= n, dtype= np.uint64)])
        else: # pearson
            for k in range(local_n):
                local_res[k] = pearson_rows(x[:,local_gen.integers(0, n, size= n, dtype= np.uint64)])
        boot_stat[idxs[i] : idxs[i+1]] = local_res
    return boot_stat

@njit(parallel=True, fastmath=True, cache=True)
def bstrap_sampler_1d_sparse(stat_name, x, gens, n, n_boot, arg):
    # Generate n_boot bootstrap samples for 1D sparse array x using generators gens and compute the statistic
    nt = len(gens)
    nnz = len(x)
    p = nnz / n
    thread_samps = np.array([n_boot // nt] * nt) + np.array([1] * (n_boot % nt) + [0] * (nt - (n_boot % nt)))
    idxs = [0] + list(np.cumsum(thread_samps))
    boot_stat = np.empty(n_boot)
    nzboot = gens[0].binomial(n, p, n_boot)
    for i in prange(nt):
        local_n = thread_samps[i]
        local_gen = gens[i]
        local_res = np.empty(local_n)
        local_nzboot = nzboot[idxs[i] : idxs[i+1]]
        if stat_name == 'mean':
            for k in range(local_n):
                local_res[k] = numba_mean_sparse(x[local_gen.integers(0, nnz, size= local_nzboot[k], dtype= np.uint64)], n)
        elif stat_name == 'std':
            for k in range(local_n):
                local_res[k] = numba_std_sparse(x[local_gen.integers(0, nnz, size= local_nzboot[k], dtype= np.uint64)], n)
        else: # quantile
            for k in range(local_n):
                local_res[k] = quantile_sparse(x[local_gen.integers(0, nnz, size= local_nzboot[k], dtype= np.uint64)], n, arg)
        boot_stat[idxs[i] : idxs[i+1]] = local_res
    return boot_stat

@njit(fastmath=True, cache=True)
def bstrap_sampler_1d_bin(stat_name, x, gens, n, n_boot, arg):
    # Generate n_boot bootstrap samples for 1D binary array x using generators gens and compute the statistic
    p = x.sum() / n
    gen = gens[0]
    n_ones = gen.binomial(n, p, n_boot)
    if stat_name == 'mean':
        boot_stat = numba_mean_bin(n_ones, n)
    elif stat_name == 'std':
        boot_stat = numba_std_bin(n_ones, n)
    else: # quantile
        boot_stat = quantile_bin(n_ones, n, arg)
    return boot_stat

# (Dimensions, Binary, Sparse) -> sampler function
sampler_map = {(1, False, False) : bstrap_sampler_1d,
               (1, False, True) : bstrap_sampler_1d_sparse,
               (1, True, False) : bstrap_sampler_1d_bin,
               (2, False, False) : bstrap_sampler_2d,
              }