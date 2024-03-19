# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import mode
from .stats_numba import sampler_map
from .stats_py import jack_func_map, ci_est_percent, ci_est_bca
import warnings

def check_sparsity(x, n, min_zeros= 0.1):
    # Flag data if binary. If not binary, but sparse, remove the zeros from the array
    xz = x == 0
    is_bin = np.all(xz | (x == 1))
    if xz.sum() / n > min_zeros and n > 100:
        ret = x[~xz].copy()
    else:
        ret = x
    return is_bin, ret

def warn_pearson(x, n, nboot, threshold):
    # Warn user if probability of a sample of equal values exceeds threshold
    m,k = mode(x)
    prob = 1.0 - (1.0-(k/n)**n)**nboot
    if prob > threshold:
        warnings.warn(f'The value {m} is repeated {k} times out of {n} observations. There is a {np.round(prob, int(np.ceil(-np.log10(threshold))))*100} % probability of getting at least 1 nan sample out of {nboot} due to zero denominator.', stacklevel=2)

def preproc_args(x1, x2, statistic, q, boot_samples, conf_lvl, alternative, ci_method, random_state, parallel, ignore_sparse_below):
    # Check validity of each argument, and preprocess them
    
    # x1
    x1t = np.asarray(x1).squeeze()
    if x1t.ndim > 2:
        raise ValueError("x1 must be at most 2-dimensional after squeezing.")
    elif x1t.ndim == 2 and x1t.shape[1] != 2:
        raise ValueError("If x1 is 2-dimensional, the second dimension must be of length 2.")
    elif x1t.ndim == 0:
        x1t = x1t.reshape(-1)
    is_1d = x1t.ndim == 1
    x1_valid = np.isfinite(x1t) if is_1d else np.isfinite(x1t).all(axis=1)
    n1 = x1_valid.sum()
    if n1 == 0:
        if len(x1t) == 0:
            raise ValueError("x1 cannot be empty")
        else:
            raise ValueError("x1 must contain at least 1 valid number")
    x1t = x1t[x1_valid] if is_1d else x1t[x1_valid,:].T.copy()
    if x1t.dtype == bool:
        x1t = x1t.astype(np.uint8)
    
    # x2
    if x2 is not None:
        x2t = np.asarray(x2).squeeze()
        if x2t.ndim == 0:
            x2t = x2t.reshape(-1)
        elif x2t.ndim != x1t.ndim:
            raise ValueError("x2 must have the same number of dimensions as x1.")
        elif x2t.ndim == 2 and x2t.shape[1] != 2:
            raise ValueError("If x2 is 2-dimensional, the second dimension must be of length 2.")
        x2_valid = np.isfinite(x2t) if is_1d else np.isfinite(x2t).all(axis=1)
        n2 = x2_valid.sum()
        if n2 == 0:
            raise ValueError("x2 must have at least 1 valid number, or be omitted.")
        else:
            x2t = x2t[x2_valid] if is_1d else x2t[x2_valid,:].T.copy()
    else:
        n2 = 0
        x2t = np.array([], dtype= x1t.dtype) if is_1d else np.array([], dtype= x1t.dtype).reshape((2,-1))
    if x2t.dtype == bool:
        x2t = x2t.astype(np.uint8)
        
    # statistic
    if statistic not in ('auto','mean','std','quantile','ratio','wmean','pearson'):
        raise ValueError("statistic must be one of ('auto','mean','std','quantile','ratio','wmean','pearson')")
    if statistic in ('mean','std','quantile') and not is_1d:
        raise ValueError(f"statistic {statistic} is only available for 1-dimensional arrays")
    if statistic in ('ratio','wmean','pearson') and is_1d:
        raise ValueError(f"statistic {statistic} is only available for 2-dimensional arrays")
    if statistic == 'auto':
        statistic = 'mean' if is_1d else 'ratio'

    # q
    if statistic == 'quantile' and not (isinstance(q, (float,int)) and 0 <= q <= 1):
        raise ValueError("q must be a single number between 0 and 1")
    
    # boot_samples
    if not isinstance(boot_samples, int) or boot_samples <= 0:
        raise ValueError("boot_samples must be an integer greater than 0.")
    
    # conf_lvl
    if not (isinstance(conf_lvl, float) and 0 < conf_lvl < 1):
        raise ValueError("conf_lvl must be a float greater than 0 and less than 1.")

    # alternative
    if alternative not in ('two-sided','less','greater'):
        raise ValueError("alternative must be either 'two-sided', 'less' or 'greater'")
    if alternative == 'two-sided':
        ci_alphas = (0.5 - conf_lvl/2.0, 0.5 + conf_lvl/2.0)
    elif alternative == 'less':
        ci_alphas = (-np.inf, conf_lvl)
    else:
        ci_alphas = (1.0 - conf_lvl, np.inf)
    
    # ci_method
    if ci_method not in ['BCa', 'percentile']:
        raise ValueError("ci_method must be either 'BCa' or 'percentile'.")
    
    # random_state
    if not (random_state is None or isinstance(random_state, (int, np.random.Generator, np.random.SeedSequence))):
        raise ValueError("random_state must be None, an integer, numpy.random.Generator, or numpy.random.SeedSequence.")
    if random_state is None:
        rng = np.random.Generator(np.random.SFC64())
    elif isinstance(random_state, (int, np.random.SeedSequence)):
        rng = np.random.Generator(np.random.SFC64(random_state))
    else:
        rng = random_state
        
    # parallel
    if not isinstance(parallel, bool):
        raise ValueError("parallel must be a boolean.")
    
    # ignore_sparse_below
    if not (isinstance(ignore_sparse_below, (float, int)) and 0 <= ignore_sparse_below <= 1):
        raise ValueError("ignore_sparse_below must be a float greater or equal to 0 and less or equal to 1.")

    # If arrays are 1D, check sparsity of samples
    if is_1d:
        is_bin1, x1t = check_sparsity(x1t, n1, min_zeros= ignore_sparse_below)
        if n2 > 0:
            is_bin2, x2t = check_sparsity(x2t, n2, min_zeros= ignore_sparse_below)
        else:
            is_bin2 = True
        # Pick corresponding sampler and statistic function
        is_sparse1 = (not is_bin1) and (len(x1t) < n1)
        is_sparse2 = (not is_bin2) and (len(x2t) < n2)
        sampler_func1 = sampler_map[(1, is_bin1, is_sparse1)]
        sampler_func2 = sampler_map[(1, is_bin2, is_sparse2)]
    else:
        # If using paired statistics, sparsity doesn't apply
        sampler_func1, sampler_func2 = sampler_map[(2, False, False)], sampler_map[(2, False, False)]
    
    # Get jacknife estimator
    jack_func = jack_func_map[statistic] if statistic != 'quantile' else lambda x, n1, n2 : jack_func_map['quantile'](x, n1, n2, q)
    
    # Set function for confidence interval estimation (percentile or BCa)
    if n1 == 1 or n2 == 1:
        ci_method = 'percentile'
    ci_func = ci_est_percent if ci_method == 'percentile' else ci_est_bca

    # Warn user if doing dividing by sums that contain zeros
    if statistic in ('ratio','wmean') and (np.any(x1t[1,:] == 0) or np.any(x2t[1,:] == 0)):
        warnings.warn('Zeros detected in the denominator values. This could cause a division by zero over a random resampling. These samples will be recorded as nan', stacklevel=2)

    # Warn user if many obsevations for the pearson correlation are equal (risk of zero variance in denominator)
    if statistic in ('pearson',):
        warn_pearson(x1t[0,:], n1, boot_samples, 0.001)
        warn_pearson(x1t[1,:], n1, boot_samples, 0.001)
        if n2 > 0:
            warn_pearson(x2t[0,:], n2, boot_samples, 0.001)
            warn_pearson(x2t[1,:], n2, boot_samples, 0.001)
    
    return x1t, x2t, rng, parallel, n1, n2, ci_alphas, sampler_func1, sampler_func2, statistic, jack_func, ci_func
