import pytest
from ftle import fields
import xarray as xr
import numpy as np


class Test_Fields_from_dict:
    def test_is_collection_of_functions(self):
        input_dict = dict(myfield=lambda x, y, z, t: x + 2*y + 3*z + 4*t)
        f = fields.Fields.from_dict(input_dict)
        assert f['myfield'](1, 2, 3, 4) == 1 + 4 + 9 + 16

    def test_can_be_converted_to_dict(self):
        input_dict = dict(myfield=lambda x, y, z, t: x + 2*y + 3*z + 4*t)
        f = fields.Fields.from_dict(input_dict)
        f_dict = dict(f)
        assert list(f_dict.keys()) == ['myfield']


class Test_get_interp_func_from_xr_data_array:
    @pytest.fixture(scope='class')
    def coords(self):
        x = np.array([.1, .2, .3])
        y = np.array([.4, .5, .6])
        z = np.array([.7, .8, .9])
        t = np.array([1.0] * len(x))
        return x, y, z, t

    @pytest.fixture(scope='class')
    def darr(self):
        data = np.arange(2*3*4*5, dtype='f4').reshape((2, 3, 4, 5))
        return xr.DataArray(data, dims=('t', 'z', 'y', 'x'))

    def test_returns_callable(self, coords, darr):
        fn = fields.get_interp_func_from_xr_data_array(darr)
        assert callable(fn)

    def test_return_value_is_correct_size(self, coords, darr):
        fn = fields.get_interp_func_from_xr_data_array(darr)
        assert fn(*coords).shape == coords[0].shape

    def test_return_value_is_dtype_float64(self, coords, darr):
        fn = fields.get_interp_func_from_xr_data_array(darr)
        assert fn(*coords).dtype == fn.dtype
        assert fn.dtype == np.dtype('f8')

    def test_interpolates_in_all_directions(self, darr):
        fn = fields.get_interp_func_from_xr_data_array(darr)
        zero = np.zeros(3)
        ramp = np.array([0, 0.5, 1])

        # Start at coordinate zero and make linear increase in all directions
        results = [
            fn(ramp, zero, zero, zero),
            fn(zero, ramp, zero, zero),
            fn(zero, zero, ramp, zero),
            fn(zero, zero, zero, ramp),
        ]

        # Check the result
        for i, result in enumerate(results):
            assert result[0] != result[1], f"Axis {i}: Constant output despite variable input"
            assert result.mean() == result[1], f"Axis {i}: Nonlinear change in output"

    def test_interpolates_lower_dimensional_arrays(self, darr):
        coords = [np.array([0, 0.5, 1])] * 4
        for i in range(4):
            lowdim_array = darr[(0, ) * i]
            fn = fields.get_interp_func_from_xr_data_array(lowdim_array)
            result = fn(*coords)

            assert len(result) == len(coords[0]), f"Dim {i}: Wrong output dimension"
            assert result[0] != result[1], f"Dim {i}: Constant output, variable input"
            assert result.mean() == result[1], f"Dim {i}: Nonlinear change in output"
            assert result.dtype == fn.dtype, f"Dim {i}: Wrong output data type"

    def test_accepts_nondimensional_array(self):
        coords = [np.array([0, 0.5, 1])] * 4
        darr = xr.DataArray(data=42, dims=())
        fn = fields.get_interp_func_from_xr_data_array(darr)
        result = fn(*coords)
        assert result == 42
        assert result.dtype == fn.dtype
