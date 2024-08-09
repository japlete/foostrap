# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm

### BCa AND QUANTILES METHODS ###

def jacknife_mean(x, n1, n2):
    # Generate jacknife estimates for the mean of x of length n1, leaving n2 samples unaltered for x2
    s = x.sum()
    return s / n1, \
        (s - np.pad(x, (0, n1 - len(x) + n2))) / \
        np.concatenate((np.full(n1,n1-1,dtype=np.float64), \
                        np.full(n2,n1,dtype=np.float64)))

def jacknife_std(x, n1, n2):
    # Generate jacknife estimates for the standard deviation of x of length n1, leaving n2 samples unaltered for x2
    s = x.sum()
    s2 = (x**2).sum()
    denom = np.concatenate((np.full(n1,n1-1,dtype=np.float64), np.full(n2,n1,dtype=np.float64)))
    x_ext = np.pad(x, (0, n1 - len(x) + n2))
    return np.sqrt(s2 / n1 - (s / n1)**2), \
        np.sqrt((s2 - x_ext**2)/denom - ((s - x_ext)/denom)**2)

def jacknife_quantile(x, n1, n2, q):
    # Generate jacknife estimates for the q-quantile of x of length n1, leaving n2 samples unaltered for x2
    xs = np.sort(np.pad(x, (n1 - len(x), 0)))
    stat = np.quantile(xs, q)
    idx = q * (n1-2)
    idx_lo, idx_hi = np.full(n1,int(np.floor(idx))), np.full(n1,int(np.ceil(idx)))
    idx_lo[:idx_lo[0]+1] += 1
    idx_hi[:idx_hi[0]+1] += 1
    xlo, xhi = xs[idx_lo], xs[idx_hi]
    return stat, np.pad((xhi - xlo) * (idx - idx_lo), (0, n2), constant_values= stat)

def jacknife_ratio(x, n1, n2):
    # Generate jacknife estimates for the ratio of sums of x of length n1, leaving n2 samples unaltered for x2
    s = x.sum(axis= 1)
    jacks = s.reshape(2,1) - np.pad(x, ((0,0),(0,n2)))
    return s[0] / s[1], jacks[0,:] / jacks[1,:]

def jacknife_wmean(x, n1, n2):
    # Generate jacknife estimates for the weighted mean of x of length n1, leaving n2 samples unaltered for x2
    xw = np.dot(x[0,:], x[1,:])
    w = x[1,:].sum()
    xw_ext = np.pad(x, ((0,0), (0,n2)))
    return xw / w, (xw - xw_ext[0,:] * xw_ext[1,:]) / (w - xw_ext[1:])

def jacknife_pearson(x, n1, n2):
    # Generate jacknife estimates for the pearson correlation of x of length n1, leaving n2 samples unaltered for x2
    sx, sy = x.sum(axis= 1)
    sx2, sy2 = (x ** 2).sum(axis= 1)
    sxy = np.dot(x[0,:], x[1,:])
    stat = (n1 * sxy - sx * sy) / np.sqrt((n1 * sx2 - sx ** 2)*(n1 * sy2 - sy ** 2))
    xy_ext = np.pad(x, ((0,0), (0,n2)))
    n_ext = np.concatenate((np.full(n1,n1-1,dtype=np.float64), np.full(n2,n1,dtype=np.float64)))
    jacks = (n_ext * (sxy - xy_ext[0,:] * xy_ext[1,:]) - (sx - xy_ext[0,:]) * (sy - xy_ext[1,:])) / \
        np.sqrt((n_ext * (sx2 - xy_ext[0,:] ** 2) - (sx - xy_ext[0,:])**2) * (n_ext * (sy2 - xy_ext[1,:] ** 2) - (sy - xy_ext[1,:])**2))
    return stat, jacks

# Statistic -> jackknife estimator
jack_func_map = {'mean' : jacknife_mean,
                 'std' : jacknife_std,
                 'quantile' : jacknife_quantile,
                 'ratio' : jacknife_ratio,
                 'wmean' : jacknife_wmean,
                 'pearson' : jacknife_pearson,
                }

def jacknife_wrapper(x1, x2, n1, n2, jack_func):
    # Call the jack_func statistic over 1 or 2 samples and compute the difference
    samp_stat, jacks = jack_func(x1, n1, n2)
    if n2 > 0:
        samp_stat2, jacks2 = jack_func(x2, n2, n1)
        samp_stat -= samp_stat2
        jacks -= np.flip(jacks2)
    return samp_stat, jacks

def ci_est_percent(samps, ci_alphas, jack_func, x1, x2, n1, n2):
    # Compute percentiles directly from bootstrap samples
    if ci_alphas[0] == -np.inf:
        return (-np.inf, np.nanquantile(samps, ci_alphas[1]))
    elif ci_alphas[1] == np.inf:
        return (np.nanquantile(samps, ci_alphas[0]), np.inf)
    else:
        return np.nanquantile(samps, ci_alphas)

def ci_est_bca(samps, ci_alphas, jack_func, x1, x2, n1, n2):
    # Get jacknife estimates and sample statistic
    samp_stat, jacks = jacknife_wrapper(x1, x2, n1, n2, jack_func)
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
