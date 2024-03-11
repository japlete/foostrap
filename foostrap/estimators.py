import numpy as np
from numba import njit, objmode
from scipy.stats import norm

@njit(fastmath=True, cache=True)
def bstrap_core_mean(x1, x2, n_boot, gen, n1, n2):
    # Generate bootstrap samples for the mean of x1 (or difference of means if x2 provided)
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        samp1 = x1[gen.integers(0, n1, size= n1, dtype= np.uint64)].mean()
        boot_stat[i] = samp1
        if n2 > 0:
            samp2 = x2[gen.integers(0, n2, size= n2, dtype= np.uint64)].mean()
            boot_stat[i] -= samp2
    return boot_stat

@njit(fastmath=True, cache=True)
def bstrap_core_mean_sparse(x1, x2, n_boot, gen, n1, n2):
    # Generate bootstrap samples for the mean of sparse sample x1 (or difference of means if x2 provided)
    nnz1 = len(x1)
    nnz2 = len(x2)
    p1 = nnz1 / n1
    with objmode(nzboot1='int64[:]',nzboot2='int64[:]'):
        nzboot1 = gen.binomial(n1, p1, n_boot)
        if n2 > 0:
            nzboot2 = gen.binomial(n2,  nnz2 / n2, n_boot)
        else:
            nzboot2 = np.array([], dtype= np.int64)
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        samp1 = x1[gen.integers(0, nnz1, size= nzboot1[i], dtype= np.uint64)].sum() / n1
        boot_stat[i] = samp1
        if n2 > 0:
            samp2 = x2[gen.integers(0, nnz2, size= nzboot2[i], dtype= np.uint64)].sum() / n2
            boot_stat[i] -= samp2
    return boot_stat

def bstrap_core_mean_bin(x1, x2, n_boot, gen, n1, n2):
    # Generate bootstrap samples for the mean of binary sample x1 (or difference of means if x2 provided)
    p1 = x1.sum() / n1
    nnz2 = x2.sum()
    boot_stat = gen.binomial(n1, p1, n_boot) / n1
    if n2 > 0:
        boot_stat -= gen.binomial(n2,  nnz2 / n2, n_boot) / n2
    return boot_stat
    
@njit(fastmath=True, cache=True)
def bstrap_core_ratio(x1, x2, n_boot, gen, n1, n2):
    # Generate bootstrap samples for the ratio of sums of x1 (or difference of ratios if x2 provided)
    boot_stat = np.empty(n_boot)
    for i in range(n_boot):
        samp1 = x1[:,gen.integers(0, n1, size= n1, dtype= np.uint64)].sum(axis= 1)
        boot_stat[i] = samp1[0]/samp1[1]
        if n2 > 0:
            samp2 = x2[:,gen.integers(0, n2, size= n2, dtype= np.uint64)].sum(axis= 1)
            boot_stat[i] -= samp2[0]/samp2[1]        
    return boot_stat

def jacknife_mean(x1, x2, n1, n2):
    # Generate jacknife estimates for the mean (or mean difference if x2 provided)
    s1 = x1.sum()
    samp_stat = s1 / n1
    jacks1 = np.full(n1+n2, s1, dtype=np.float64)
    nnz1 = len(x1)
    jacks1[:nnz1] = (jacks1[:nnz1] - x1) / (n1-1)
    jacks1[nnz1:n1] /= (n1-1)
    jacks1[n1:] /= n1
    if n2 > 0:
        s2 = x2.sum()
        jacks2 = np.full_like(jacks1, s2)
        jacks2[:n1] /= n2
        nnz2 = len(x2)
        jacks2[n1:n1+nnz2] = (jacks2[n1:n1+nnz2] - x2) / (n2-1)
        jacks2[n1+nnz2:] /= (n2-1)
        jacks1 -= jacks2
        samp_stat -= s2 / n2
    return samp_stat, jacks1

def jacknife_ratio(x1, x2, n1, n2):
    # Generate jacknife estimates for the ratio of sums (or difference of ratios if x2 provided)
    s1 = x1.sum(axis= 1)
    samp_stat = s1[0] / s1[1]
    jacks1 = np.full((2,n1+n2), s1.reshape((2,-1)), dtype=np.float64)
    jacks1[:,:n1] -= x1
    jacks = jacks1[0,:] / jacks1[1,:]
    if n2 > 0:
        s2 = x2.sum(axis= 1)
        jacks2 = np.full_like(jacks1, s2.reshape((2,1)))
        jacks2[:,n1:] -= x2
        jacks -= jacks2[0,:] / jacks2[1,:]
        samp_stat -= s2[0] / s2[1]
    return samp_stat, jacks

def ci_est_percent(samps, ci_alphas, jack_func, x1, x2, n1, n2):
    # Compute percentiles directly from bootstrap samples
    if ci_alphas[0] == -np.inf:
        return (-np.inf, np.quantile(samps, ci_alphas[1]))
    elif ci_alphas[1] == np.inf:
        return (np.quantile(samps, ci_alphas[0]), np.inf)
    else:
        return np.quantile(samps, ci_alphas)

def ci_est_bca(samps, ci_alphas, jack_func, x1, x2, n1, n2):
    # Get jacknife estimates and sample statistic
    samp_stat, jacks = jack_func(x1, x2, n1, n2)
    # Compute the bias term
    b = norm.ppf((1.0*(samps < samp_stat) + 0.5*(samps == samp_stat)).mean())
    # Compute the acceleration term
    jack_mean = jacks.mean()
    acc = ((jack_mean - jacks)**3).sum() / 6 / (((jack_mean - jacks)**2).sum())**1.5
    # Final formula for corrected quantiles
    ci_lo, ci_hi = -np.inf, np.inf
    if ci_alphas[0] != -np.inf:
        per_norm_lo = b + norm.ppf(ci_alphas[0])
        ci_lo = max(0.0, norm.cdf(b + per_norm_lo/(1.0 - acc*per_norm_lo)))
    if ci_alphas[1] != np.inf:
        per_norm_hi = b + norm.ppf(ci_alphas[1])
        ci_hi = min(1.0, norm.cdf(b + per_norm_hi/(1.0 - acc*per_norm_hi)))
    return ci_est_percent(samps, (ci_lo, ci_hi), jack_func, x1, x2, n1, n2)
