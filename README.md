# Foostrap: Fast Bootstrap Resampling

[![PyPI version](https://badge.fury.io/py/foostrap.svg)](https://badge.fury.io/py/foostrap)
![Python Version](https://img.shields.io/badge/python-3.10-orange.svg)
![Numpy Version](https://img.shields.io/badge/numpy-1.25-blue.svg)
![Numba Version](https://img.shields.io/badge/numba-0.61-blue.svg)
![Scipy Version](https://img.shields.io/badge/scipy-1.11-blue.svg)
[![License](https://img.shields.io/pypi/l/foostrap.svg)](https://github.com/japlete/foostrap/blob/main/LICENSE)

## Overview

Foostrap is a simple Python library for efficient bootstrap resampling and confidence interval estimation.

## Features

- Parallel by default using Numba. Typically at least 4x faster than the current Scipy bootstrap. See benchmark notebook [here](https://github.com/japlete/foostrap/blob/main/benchmark.ipynb).
- Implements the Bias-Corrected and Accelerated (BCa) method for CI estimation. Can also use percentiles.
- Optimized for sparse and binary data. The number of zeros is drawn from a Binomial distribution, instead of resampling them individually.
- Supported statistics:
  - For 1-dimensional data: mean, standard deviation, quantile q
  - For 2-dimensional paired data: ratio of sums, weighted mean and pearson correlation
- Robust: unit tests validate edge cases and results within 2 decimal places from Scipy bootstrap.

## Installation

```bash
pip install foostrap
```

or optionally, if you also want icc-rt as recommended by Numba:

```bash
pip install foostrap[iccrt]
```

## Usage

The `foostrap` function can take 1 sample or 2 independent samples for comparison. The comparison is always the difference `statistic(sample 1) - statistic(sample 2)`.

If no statistic is specified, by default the mean is used for 1-D samples, and ratio of sums for 2-D samples.

### Example

```python
import numpy as np
from foostrap import foostrap

# Generate some data
x1 = np.random.normal(size=100)

# Performing bootstrap resampling (1-sample mean)
result = foostrap(x1)

# Displaying the confidence interval tuple
print(result.ci)
```

### Parameters

- `x1` (numpy.ndarray): Primary sample array. If observations are paired, the shape must be a 2-column array.
- `x2` (numpy.ndarray, optional): Second sample to compare against x1. Default is None.
- `statistic` (one of `'mean','std','quantile','ratio','wmean','pearson','auto'`): the statistic to compute over each sample. Default `'auto'` (see above).
- `q` (float): probability for the `'quantile'` statistic. Ignored otherwise. Default 0.5 (the median)
- `boot_samples` (int): Number of bootstrap samples to generate. Default 10 000
- `conf_lvl` (float): Confidence level for the interval estimation. Default is 0.95.
- `alternative` (str): Type of confidence interval. `'two-sided'`: with upper and lower bound, `'less'`: only upper bound, `'greater'`: only lower bound. Default `'two-sided'`.
- `ci_method` (str): Method for CI estimation, `'BCa'` (default) or `'percentile'`.
- `random_state`: (int, numpy Generator or SeedSequence): For reproducibility.
- `parallel` (bool): Whether to use parallel processing. Default True
- `ignore_sparse_below` (float): Threshold under which sparse data is treated as dense, to avoid the overhead of a separate sampling. Default 0.1

### Returns

A data class containing the confidence interval (`ci`) as a tuple and the bootstrap samples (`boot_samples`) as a numpy array.

### Notes

1. The first execution will take a few seconds longer since Numba takes time to compile the functions for the first time. The compiled functions are cached in the `__pycache__` in the library installation directory. You can save the cached functions and reuse them in another machine, as long as it has the same package versions and CPU.
2. Each thread gets a separate random generator, spawned from the user supplied or the default. This means that for the results to be reproducible, the number of CPU cores must remain constant.
3. Only the 1-D statistics have the sparse and binary data optimization, since paired data typically doesn't have zeros in both values of an observation.

### More examples

```python
# Generate some data
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)

# Bootstrap median(x1) - median(x2)
result = foostrap(x1, x2, statistic= 'quantile', q= 0.5)

# Displaying the confidence interval tuple
print(result.ci)

# Generate 2-column correlated data
x1 = np.random.normal(size=(100,2))
x1[:,1] += x1[:,0]

# Bootstrap pearson correlation coefficient
result = foostrap(x1, statistic= 'pearson')

# Displaying the confidence interval tuple
print(result.ci)
```

## Contributing

If you need other statistics to be supported, note that the current statistics have optimized functions for the sampling and jackknife method. So any new statistic will also need specialized functions.

## License

Foostrap is released under the MIT License. See the LICENSE file for more details.
