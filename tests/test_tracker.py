from ftle import tracker
import xarray as xr
import numpy as np


class Test_advection:
    def test_correct_when_homogeneous_field_and_euler(self):
        x = xr.Variable('pid', [1., 2., 3.])
        y = xr.Variable('pid', [1., 2., 3.])
        u = xr.DataArray(dims=('y', 'x'), data=2*np.ones((5, 6)))
        v = xr.DataArray(dims=('y', 'x'), data=np.ones((5, 6)))
        dt = 1
        x2, y2 = tracker.advection(x, y, u, v, dt)
        assert x2.values.tolist() == [3, 4, 5]
        assert y2.values.tolist() == [2, 3, 4]
