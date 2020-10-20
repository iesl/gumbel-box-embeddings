import pytest
from .metrics import *
import numpy as np
from scipy import stats
from hypothesis import given, example, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


same_len_arrays = st.integers(min_value=1, max_value=100).flatmap(
    lambda n: st.lists(arrays(np.float, shape=n, elements=st.floats(1e-1,1)), min_size=2, max_size=2)
    )


@given(same_len_arrays)
def test_pearson_r_same_as_scipy(arrays):
    p, q = arrays
    # Due to floating point errors, if the array is filled with a single value torch will sometimes return a value while
    # SciPy will not.
    # This happens with, for example, p = q = np.ones(8)*0.1, where torch.mean(p) != 0.1 while np.mean(p) == 0.1
    # To avoid these cases we make the following assumption:
    assume((p != p[0]).any() and (q != q[0]).any())
    torch_version = pearson_r(torch.from_numpy(p), torch.from_numpy(q))
    scipy_version, _ = stats.pearsonr(p,q)
    assert np.isclose(torch_version, scipy_version) or (torch.isnan(torch_version) and np.isnan(scipy_version))
