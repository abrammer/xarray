import xarray as xr
import numpy as np
import pytest

from . import raises_regex


def test_allclose_regression():
    x = xr.DataArray(1.01)
    y = xr.DataArray(1.02)
    xr.testing.assert_allclose(x, y, atol=0.01)


def test_allclose_string_data():
    x = xr.DataArray(np.array('0', 'S'))
    y = xr.DataArray(np.array('0'))
    xr.testing.assert_allclose(x, y)


@pytest.mark.parametrize(
    'func', [xr.testing.assert_allclose, xr.testing.assert_identical])
def test_typeerror_raises(func):
    x = 1.01
    y = 1.01
    with raises_regex(TypeError, "not supported by assertion comparison"):
        func(x, y)
