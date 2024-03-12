import numpy as np
from .estimators import bstrap_core_mean_bin, bstrap_core_mean_sparse, bstrap_core_mean, jacknife_mean, bstrap_core_ratio, jacknife_ratio, ci_est_percent, ci_est_bca
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

def preproc_args(x1, x2, boot_samples, conf_lvl, alternative, ci_method, random_state, parallel, ignore_sparse_below):
    # Check validity of each argument, and preprocess them
    
    # x1
    x1t = np.asarray(x1).squeeze()
    if x1t.ndim > 2:
        raise ValueError("x1 must be at most 2-dimensional after squeezing.")
    elif x1t.ndim == 2 and x1t.shape[1] != 2:
        raise ValueError("If x1 is 2-dimensional, the second dimension must be of length 2.")
    elif x1t.ndim == 0:
        x1t = x1t.reshape(-1)
    stat_is_mean = x1t.ndim == 1
    x1_valid = np.isfinite(x1t) if stat_is_mean else np.isfinite(x1t).all(axis=1)
    n1 = x1_valid.sum()
    if n1 == 0:
        if len(x1t) == 0:
            raise ValueError("x1 cannot be empty")
        else:
            raise ValueError("x1 must contain at least 1 valid number")
    x1t = x1t[x1_valid] if stat_is_mean else x1t[x1_valid,:].T.copy()
    
    # x2
    if x2 is not None:
        x2t = np.asarray(x2).squeeze()
        if x2t.ndim == 0:
            x2t = x2t.reshape(-1)
        elif x2t.ndim != x1t.ndim:
            raise ValueError("x2 must have the same number of dimensions as x1.")
        elif x2t.ndim == 2 and x2t.shape[1] != 2:
            raise ValueError("If x2 is 2-dimensional, the second dimension must be of length 2.")
        x2_valid = np.isfinite(x2t) if stat_is_mean else np.isfinite(x2t).all(axis=1)
        n2 = x2_valid.sum()
        if n2 == 0:
            raise ValueError("x2 must have at least 1 valid number, or be omitted.")
        else:
            x2t = x2t[x2_valid] if stat_is_mean else x2t[x2_valid,:].T.copy()
    else:
        n2 = 0
        x2t = np.array([], dtype= x1t.dtype) if stat_is_mean else np.array([], dtype= x1t.dtype).reshape((2,-1))
        
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

    # If using mean statistic, check sparsity of samples and pick corresponding resampler (core_func) and jacknife estimator. 
    # If using ratio statistic, sparsity doesn't apply
    if stat_is_mean:
        is_bin1, x1t = check_sparsity(x1t, n1, min_zeros= ignore_sparse_below)
        if n2 > 0:
            is_bin2, x2t = check_sparsity(x2t, n2, min_zeros= ignore_sparse_below)
        else:
            is_bin2 = True
        if is_bin1 and is_bin2:
            core_func = bstrap_core_mean_bin
            parallel= False
        elif len(x2t) < n2 or len(x1t) < n1:
            core_func = bstrap_core_mean_sparse
        else:
            core_func = bstrap_core_mean
        jack_func = jacknife_mean
    else:
        core_func = bstrap_core_ratio
        jack_func = jacknife_ratio
    
    # Set function for confidence interval estimation (percentile or BCa)
    if n1 == 1 or n2 == 1:
        ci_method = 'percentile'
    ci_func = ci_est_percent if ci_method == 'percentile' else ci_est_bca

    # Warn user if doing ratio of sums with zeros in the denominator
    if not stat_is_mean and (np.any(x1t[1,:] == 0) or np.any(x2t[1,:] == 0)):
        warnings.warn('Zeros detected in the denominator values. This could cause a division by zero over a random resampling')
    
    return x1t, x2t, rng, parallel, n1, n2, ci_alphas, core_func, jack_func, ci_func
