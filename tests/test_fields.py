import pytest
from ftle import fields
import xarray as xr
import numpy as np
from pathlib import Path


FORCING_1 = Path(__file__).parent / 'forcing_1.nc'
FORCING_2 = Path(__file__).parent / 'forcing_2.nc'


@pytest.fixture(scope='module')
def forcing_1():
    with xr.open_dataset(FORCING_1) as dset:
        yield dset


@pytest.fixture(scope='module')
def forcing_2():
    with xr.open_dataset(FORCING_2) as dset:
        yield dset


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


class Test_Fields_from_dataset:
    def test_creates_function_for_every_data_variable(self):
        dset = xr.Dataset(
            data_vars=dict(
                hc=xr.Variable((), 5),
                temp=xr.Variable(('t', 'z', 'y', 'x'), np.zeros((2, 3, 4, 5))),
                zeta=xr.Variable(('t', 'y', 'x'), np.zeros((2, 4, 5))),
                h=xr.Variable(('y', 'x'), np.zeros((4, 5))),
            ),
        )
        f = fields.Fields.from_dataset(dset)
        assert set(f.keys()) == {'hc', 'temp', 'zeta', 'h'}
        assert all(callable(fn) for fn in f.values())


class Test_Fields_from_roms_dataset:
    @pytest.fixture(scope='class')
    def coords(self):
        x = np.array([1, 1.5, 2])
        y = np.array([3, 3.5, 4])
        z = np.array([-.75, -.5, -.25])
        t = np.array([0, 0.5, 1])
        return t, z, y, x

    def test_variables_with_no_dims(self, forcing_1, coords):
        f = fields.Fields.from_roms_dataset(forcing_1)
        func = f['hc']
        result = func(*coords)
        assert result.mean() == result[1]

    def test_variables_with_dims_zrho(self, forcing_1, coords):
        f = fields.Fields.from_roms_dataset(forcing_1)
        func = f['Cs_r']
        result = func(*coords)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype


class Test_get_interp_func_from_xr_data_array:
    @pytest.fixture(scope='class')
    def coords(self):
        x = np.array([.1, .2, .3])
        y = np.array([.4, .5, .6])
        z = np.array([.7, .8, .9])
        t = np.array([1.0] * len(x))
        return t, z, y, x

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
        for i in range(5):
            lowdim_array = darr[(0, ) * i]
            fn = fields.get_interp_func_from_xr_data_array(lowdim_array)
            result = fn(*coords)

            assert len(result) == len(coords[0]), f"Dim {i}: Wrong output dimension"
            assert result.mean() == result[1], f"Dim {i}: Nonlinear change in output"
            assert result.dtype == fn.dtype, f"Dim {i}: Wrong output data type"
            if i < len(coords):
                assert result[0] != result[1], f"Dim {i}: Constant output, variable input"

    def test_accepts_nondimensional_array(self):
        coords = [np.array([0, 0.5, 1])] * 4
        darr = xr.DataArray(data=42, dims=())
        fn = fields.get_interp_func_from_xr_data_array(darr)
        result = fn(*coords)
        assert result == 42
        assert result.dtype == fn.dtype

    def test_accepts_mapping(self, coords, darr):
        mapping = dict(t='ocean_time', z='s_rho', y='eta_rho', x='xi_rho')
        inv_map = dict(ocean_time='t', s_rho='z', eta_rho='y', xi_rho='x')
        roms_darr = darr.rename(mapping)
        fn_roms = fields.get_interp_func_from_xr_data_array(roms_darr, mapping=inv_map)
        fn_regular = fields.get_interp_func_from_xr_data_array(darr)
        assert list(fn_roms(*coords)) == list(fn_regular(*coords))

    def test_accepts_offset(self, coords, darr):
        offset = dict(t=-100)
        coords2 = (coords[0] + 100, ) + coords[1:]
        fn_offset = fields.get_interp_func_from_xr_data_array(darr, offset=offset)
        fn_regular = fields.get_interp_func_from_xr_data_array(darr)
        assert list(fn_regular(*coords)) == list(fn_offset(*coords2))
