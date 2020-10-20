import pytest
from hypothesis import given, example
from hypothesis import strategies as st
from boxes.utils import *


# Actually a bit surprising it passes, given that the naive implementation suffers from stability issues
@given(st.floats(0,1,allow_nan=False, allow_infinity=False, exclude_min = True))
def test_log1mexp(x):
    x = torch.tensor(x)
    assert (torch.isclose(log1mexp(x), torch.log(1-torch.exp(-x)))).all()




