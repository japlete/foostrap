# Foostrap: Fast Bootstrap Resampling

## Overview
Foostrap is a simple Python library for efficient bootstrap resampling and confidence interval estimation.

## Features

- Parallel by default using Numba. Typically at least 4x faster than the current Scipy bootstrap. See benchmark notebook [here](https://github.com/japlete/foostrap/blob/main/benchmark.ipynb).
- Implements the Bias-Corrected and Accelerated (BCa) method for CI estimation. Can also use percentiles.
- Optimized for sparse and binary data. The number of zeros is drawn from a Binomial distribution, instead of resampling them individually.
- Supported statistics:
    - 1-sample mean
    - 2-sample mean difference
    - 1-sample ratio of sums
    - 2-sample ratio of sums difference
- Robust: unit tests validate edge cases and results within 2 decimal places from Scipy bootstrap.

## Installation

```
pip install foostrap
```

or optionally, if you also want icc-rt as recommended by Numba:

```
pip install foostrap[iccrt]
```

## Usage

The `foostrap` function recognizes the statistic to be computed (mean or ratio of sums) based on the dimensions of the input data.

1-D arrays use the mean statistic.

2-D arrays use the ratio of the columns, as `x[:,0].sum() / x[:,1].sum()`.

To provide a second sample for comparison, use the `x2` optional argument.

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

- `x1` (numpy.ndarray): Primary sample array.
- `x2` (numpy.ndarray, optional): Second sample array for comparison.
- `boot_samples` (int): Number of bootstrap samples to generate. Default 10 000
- `conf_lvl` (float): Confidence level for the interval estimation.
- `alternative` (str): Type of confidence interval (with only lower bound, only upper bound, or both). Default 'two-sided'
- `ci_method` (str): Method for CI estimation, `'BCa'` (default) or `'percentile'`.
- `random_state`: (int, Generator or SeedSequence): For reproducibility.
- `parallel` (bool): Whether to use parallel processing. Default True
- `ignore_sparse_below` (float): Threshold under which sparse data is treated as dense, to avoid the overhead of a separate sampling. Default 0.1

### Returns

A data class containing the confidence interval (`ci`) as a tuple and the bootstrap samples (`boot_samples`) as a numpy array.

### Notes
1. The first execution will take a few seconds longer since Numba takes time to compile the function for the first time. For benchmarking, use subsequent runs.
2. Each thread gets a separate random generator, spawned from the user supplied or the default. This means that for the results to be reproducible, the number of CPU cores must remain constant.
3. Only the mean and mean difference statistics have the sparse and binary data optimization, since it would be rare for a ratio statistic to have zeros in both values of an observation.

## Contributing

If you need other statistics to be supported, note that the current statistics have optimized functions for the sampling and jackknife method. So any new statistic will also need specialized functions.

## License

Foostrap is released under the MIT License. See the LICENSE file for more details.
