import unittest
import numpy as np
from scipy.stats import bootstrap as sboot
from foostrap import foostrap

drng = np.random.default_rng
rng = drng(0)
rand_arr1d = np.concatenate([np.zeros(500), rng.normal(size= 500)])
rand_arr2d = np.concatenate([rand_arr1d.reshape(-1,1), rng.lognormal(size= 1000).reshape(-1,1)], axis= 1)
print('Test array shapes', rand_arr1d.shape, rand_arr2d.shape)

class TestNoInput(unittest.TestCase):
    def test_empty1(self):
        with self.assertRaises(ValueError):
            foostrap(np.array([]))

    def test_empty2(self):
        with self.assertRaises(ValueError):
            foostrap(rand_arr1d, x2= np.array([]))

    def test_all_invalid1(self):
        with self.assertRaises(ValueError):
            foostrap(np.array([np.nan, np.inf]))

    def test_all_invalid2(self):
        with self.assertRaises(ValueError):
            foostrap(rand_arr2d, x2= np.array([[np.nan, 1], [1, np.inf]]))

class TestDimensions(unittest.TestCase):
    def test_mismatch(self):
        with self.assertRaises(ValueError):
            foostrap(rand_arr1d, rand_arr2d)

    def test_3d(self):
        with self.assertRaises(ValueError):
            foostrap(np.ones((3, 3, 3)))

class TestLowVariaion(unittest.TestCase):
    def test_all_equal1(self):
        foostrap(np.full(3, 0.5))

    def test_all_equal2(self):
        foostrap(rand_arr1d, np.full(3, 0.5))

    def test_single_value1(self):
        foostrap(np.array([1]))

    def test_single_value2(self):
        foostrap(np.array([1]), np.array([1]))

    def test_single_value_ratio1(self):
        foostrap(np.array([1,1]).reshape((1,2)))

    def test_single_value_ratio2(self):
        foostrap(np.array([1,1]).reshape((1,2)), np.array([1,1]).reshape((1,2)))

    def test_low_nboot(self):
        foostrap(rand_arr1d, rand_arr1d, boot_samples= 1)

class TestDtypes(unittest.TestCase):
    def test_floats(self):
        foostrap(rand_arr1d.astype(np.float64), rand_arr1d.astype(np.float32))

    def test_ints(self):
        foostrap(np.array([1,2,3]).astype(np.int64), np.array([1,2,3]).astype(np.int32))

class TestIllegalMath(unittest.TestCase):
    def test_zeros_denom(self):
        with self.assertWarns(UserWarning):
            foostrap(np.concatenate((np.ones((100,2)), np.array([1,0]).reshape(-1,2))), random_state=0, parallel=False)

class TestMixedSparsity(unittest.TestCase):
    def test_dense_sparse(self):
        res = foostrap(np.full(3, 2), rand_arr1d)
        self.assertFalse(np.isnan(res.ci).any() or np.isnan(res.boot_samples).any(), 'There are nan values in output')

    def test_bin_sparse(self):
        res = foostrap(rng.binomial(1,0.5,200), rand_arr1d)
        self.assertFalse(np.isnan(res.ci).any() or np.isnan(res.boot_samples).any(), 'There are nan values in output')

    def test_bin_dense(self):
        res = foostrap(rng.binomial(1,0.5,200), np.full(3, 2))
        self.assertFalse(np.isnan(res.ci).any() or np.isnan(res.boot_samples).any(), 'There are nan values in output')

#Statistic functions for the Scipy execution
def mean_dif(x, y, axis= -1):
    return np.mean(x, axis= axis) - np.mean(y, axis= axis)
def ratio_dif(x, y, axis= -1):
    return x[0].sum(axis= axis) / x[1].sum(axis= axis) - y[0].sum(axis= axis) / y[1].sum(axis= axis)

#Comparison of CIs against Scipy
class TestScipyClose(unittest.TestCase):
    def test_2sample_dense_2d(self):
        rfoo = foostrap(rand_arr2d, rand_arr2d+0.1, random_state=drng(1), parallel=False, ignore_sparse_below=1.0, boot_samples= 30000)
        rsci = sboot((rand_arr2d, rand_arr2d+0.1), ratio_dif, vectorized= True, random_state= drng(1), batch= 1000, n_resamples= 30000)
        self.assertAlmostEqual(rfoo.ci[0], rsci.confidence_interval.low, places=2)
        self.assertAlmostEqual(rfoo.ci[1], rsci.confidence_interval.high, places=2)

    def test_2sample_sparse_less_conflvl(self):
        rfoo = foostrap(rand_arr1d, rand_arr1d*1.1, random_state=drng(1), parallel=False, conf_lvl= 0.9, alternative= 'less')
        rsci = sboot((rand_arr1d, rand_arr1d*1.1), mean_dif, vectorized= True, random_state= drng(1), batch= 1000, n_resamples= 10000, alternative= 'less', confidence_level= 0.9)
        self.assertAlmostEqual(rfoo.ci[0], rsci.confidence_interval.low, places=2)
        self.assertAlmostEqual(rfoo.ci[1], rsci.confidence_interval.high, places=2)

    def test_2sample_binary_greater_conflvl_cimethod(self):
        rfoo = foostrap(rand_arr1d!=0, rand_arr1d!=0, random_state=drng(1), parallel=False, conf_lvl= 0.9, alternative= 'greater', ci_method='percentile')
        rsci = sboot((rand_arr1d!=0, rand_arr1d!=0), mean_dif, vectorized= True, random_state= drng(1), batch= 1000, n_resamples= 10000, alternative= 'greater', confidence_level= 0.9, method= 'percentile')
        self.assertAlmostEqual(rfoo.ci[0], rsci.confidence_interval.low, places=2)
        self.assertAlmostEqual(rfoo.ci[1], rsci.confidence_interval.high, places=2)

    def test_1sample_dense_less_conflvl(self):
        rfoo = foostrap(rand_arr1d, random_state=drng(1), parallel=False, conf_lvl= 0.85, alternative= 'less', ignore_sparse_below=1.0)
        rsci = sboot((rand_arr1d,), np.mean, vectorized= True, random_state= drng(1), batch= 1000, n_resamples= 10000, alternative= 'less', confidence_level= 0.85)
        self.assertAlmostEqual(rfoo.ci[0], rsci.confidence_interval.low, places=2)
        self.assertAlmostEqual(rfoo.ci[1], rsci.confidence_interval.high, places=2)

    def test_1sample_sparse_greater_cimethod(self):
        rfoo = foostrap(rand_arr1d, random_state=drng(1), parallel=False, alternative= 'greater', ci_method='percentile')
        rsci = sboot((rand_arr1d,), np.mean, vectorized= True, random_state= drng(1), batch= 1000, n_resamples= 10000, alternative= 'greater', method= 'percentile')
        self.assertAlmostEqual(rfoo.ci[0], rsci.confidence_interval.low, places=2)
        self.assertAlmostEqual(rfoo.ci[1], rsci.confidence_interval.high, places=2)

    def test_1sample_binary_conflvl(self):
        rfoo = foostrap(rand_arr1d!=0, random_state=drng(1), parallel=False, conf_lvl= 0.98)
        rsci = sboot((rand_arr1d!=0,), np.mean, vectorized= True, random_state= drng(1), batch= 1000, n_resamples= 10000, confidence_level= 0.98)
        self.assertAlmostEqual(rfoo.ci[0], rsci.confidence_interval.low, places=2)
        self.assertAlmostEqual(rfoo.ci[1], rsci.confidence_interval.high, places=2)

if __name__ == '__main__':
    unittest.main()