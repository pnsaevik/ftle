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
        z = np.array([5, 5.5, 6])
        t = np.array([0, 0.5, 1])
        return x, y, z, t

    @pytest.fixture(scope='class')
    def fields_1(self, forcing_1):
        return fields.Fields.from_roms_dataset(forcing_1)

    def test_correct_interpolation_when_variables_with_no_dims(self, fields_1, coords):
        func = fields_1['hc']
        result = func(*coords)
        assert result.mean() == result[1]
        assert not np.isnan(result.mean())

    def test_correct_interpolation_when_variables_with_dims_zrho(self, fields_1, coords):
        x, y, z, t = coords
        z = np.array([-1.0, -.5, 0])

        func = fields_1['Cs_r']
        result = func(x, y, z, t)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype
        assert np.array(np.isnan(result)).tolist() == [True, True, False]

    def test_correct_interpolation_when_variables_with_dims_zw(self, fields_1, coords):
        x, y, z, t = coords
        z = np.array([-1.0, -.5, 0])

        func = fields_1['Cs_w']
        result = func(x, y, z, t)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype
        assert np.array(np.isnan(result)).tolist() == [True, False, False]

    def test_correct_interpolation_when_variables_with_dims_time(self, fields_1, coords):
        x, y, z, t = coords
        t = np.array([-.5, 0, 0.5])

        func = fields_1['ocean_time']
        result = func(x, y, z, t)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype
        assert np.array(np.isnan(result)).tolist() == [True, False, False]

    def test_correct_interpolation_when_variables_with_dims_etarho_xirho(self, fields_1, coords):
        x, y, z, t = coords
        y = np.array([-.5, 0, 0])
        x = np.array([0, -.5, 0])

        func = fields_1['h']
        result = func(x, y, z, t)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype
        assert np.array(np.isnan(result)).tolist() == [True, True, False]

    def test_correct_interpolation_when_variables_with_dims_time_etarho_xirho(self, fields_1, coords):
        func = fields_1['zeta']
        result = func(*coords)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype

    def test_correct_interpolation_when_variables_with_fourdims_rho(self, fields_1, coords):
        func = fields_1['temp']
        result = func(*coords)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype

    def test_correct_boundaries_of_variable_u(self, fields_1, coords):
        x, y, z, t = coords
        y = np.array([-1, -.5, -.5])
        x = np.array([0.5, 0, 0.5])

        func = fields_1['u']
        result = func(x, y, z, t)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype
        assert np.array(np.isnan(result)).tolist() == [True, True, False]

    def test_correct_boundaries_of_variable_v(self, fields_1, coords):
        x, y, z, t = coords
        y = np.array([0.5, 0, 0.5])
        x = np.array([-1, -.5, -.5])

        func = fields_1['v']
        result = func(x, y, z, t)
        assert result.shape == (len(coords[0]), )
        assert result.dtype == func.dtype
        assert np.array(np.isnan(result)).tolist() == [True, True, False]

    def test_u_interpolates_by_nearest_algorithm_in_y_direction(self, fields_1):
        y = np.array([0, .4, .6])
        x = np.array([.5, .5, .5])
        z = np.ones_like(x)
        t = np.zeros_like(x)

        func = fields_1['u']
        result = func(x, y, z, t)
        assert result[0] == result[1]
        assert result[0] != result[2]

    def test_can_interpolate_using_depth(self, forcing_1):
        f_zdepth = fields.Fields.from_roms_dataset(forcing_1, z_coords='depth')
        f_regular = fields.Fields.from_roms_dataset(forcing_1)
        t = np.array([0, 0, 0, 0])
        z = np.array([-.5, 0, .5, forcing_1.dims['s_rho'] - .5])
        y = np.array([5, 5, 5, 5])
        x = np.array([5, 5, 5, 5])

        fn_zdepth = f_zdepth['Cs_w']
        fn_regular = f_regular['Cs_w']

        result_zdepth = fn_zdepth(x, y, z, t)
        result_regular = fn_regular(x, y, z, t)

        # Regular[-.5]: Ocean floor
        assert result_regular[0] == -1

        # Regular[0]: Middle of first layer
        assert -1 < result_regular[1]

        # Regular[.5]: First layer boundary
        assert result_regular[2] == forcing_1['Cs_w'][1]

        # Regular[35.5]: Surface
        assert result_regular[3] == 0

        # Depth[-.5]: Slightly above surface (but clipped to 0)
        assert result_zdepth[0] == 0

        # Depth[0]: Surface
        assert result_zdepth[1] == 0

        # Depth[.5]: Slightly below surface
        assert result_zdepth[2] < 0

        # Depth[35.5]: Below ocean floor (but clipped to ocean floor)
        assert result_zdepth[3] == -1


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

    def test_returns_callable(self, darr):
        fn = fields.get_interp_func_from_xr_data_array(darr)
        assert callable(fn)

    def test_return_value_is_correct_size(self, coords, darr):
        fn = fields.get_interp_func_from_xr_data_array(darr)
        assert fn(*coords).shape == coords[0].shape

    def test_return_value_is_dtype_float64(self, coords, darr):
        fn = fields.get_interp_func_from_xr_data_array(darr)
        assert fn(*coords).dtype == fn.dtype
        assert fn.dtype == np.dtype('f8')

    def test_interpolates_datetime64_array(self):
        y, z, t = np.zeros((3, 3))
        x = np.array([0, 0.5, .75])
        data = np.arange(2*3*4*5, dtype='f4').reshape((2, 3, 4, 5))
        data = np.datetime64('1970-01-01') + data.astype('timedelta64[D]')
        darr = xr.DataArray(data, dims=('t', 'z', 'y', 'x'))
        fn = fields.get_interp_func_from_xr_data_array(darr)
        assert fn(x, y, z, t).values.astype('datetime64[h]').astype(str).tolist() == [
            '1970-01-01T00', '1970-01-01T12', '1970-01-01T18',
        ]

    def test_return_value_is_datetime64ns_if_input_is_datetime64(self, coords, darr):
        epoch = np.datetime64('1970-01-01')
        one_sec = np.timedelta64(1, 's')
        fn = fields.get_interp_func_from_xr_data_array(epoch + one_sec * darr)
        assert fn(*coords).dtype == fn.dtype
        assert fn.dtype == np.dtype('datetime64[ns]')

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

    def test_accepts_mapping(self, coords, darr):
        mapping = dict(t='ocean_time', z='s_rho', y='eta_rho', x='xi_rho')
        inv_map = dict(ocean_time='t', s_rho='z', eta_rho='y', xi_rho='x')
        roms_darr = darr.rename(mapping)
        fn_roms = fields.get_interp_func_from_xr_data_array(roms_darr, mapping=inv_map)
        fn_regular = fields.get_interp_func_from_xr_data_array(darr)
        assert list(fn_roms(*coords)) == list(fn_regular(*coords))

    def test_accepts_offset(self, coords, darr):
        x, y, z, t = coords
        offset = dict(t=-100)
        coords2 = (x, y, z, t + 100)
        fn_offset = fields.get_interp_func_from_xr_data_array(darr, offset=offset)
        fn_regular = fields.get_interp_func_from_xr_data_array(darr)
        assert list(fn_regular(*coords)) == list(fn_offset(*coords2))

    def test_accepts_nearest(self, darr):
        nearest = ['x']

        darr = darr.isel(t=0, z=0)
        x = np.array([0, .4, .6, 0])
        y = np.array([0, 0, 0, .1])
        z = np.zeros_like(x)
        t = np.zeros_like(x)

        fn = fields.get_interp_func_from_xr_data_array(darr, nearest=nearest)
        result = fn(x, y, z, t)

        assert result[0] == result[1]  # Constant when small x change
        assert result[0] != result[2]  # Nonconstant when big x change
        assert result[0] != result[3]  # Nonconstant when small y change

    def test_accepts_transform(self, darr, coords):
        x, y, z, t = coords

        def negate(xx, yy, zz, tt):
            return -xx, -yy, -zz, -tt

        fn_1 = fields.get_interp_func_from_xr_data_array(darr)
        fn_2 = fields.get_interp_func_from_xr_data_array(darr, transform=negate)

        assert fn_1(x, y, z, t).values.tolist() == fn_2(-x, -y, -z, -t).values.tolist()

    def test_returns_transformed_coordinates(self, darr, coords):
        # The input coordinates are negative, but the output coordinates are positive

        x, y, z, t = coords

        def negate(xx, yy, zz, tt):
            return -xx, -yy, -zz, -tt

        fn_2 = fields.get_interp_func_from_xr_data_array(darr, transform=negate)

        coords_2 = fn_2(-x, -y, -z, -t).coords

        assert coords_2['x'].values.tolist() == x.tolist()
        assert coords_2['y'].values.tolist() == y.tolist()
        assert coords_2['z'].values.tolist() == z.tolist()
        assert coords_2['t'].values.tolist() == t.tolist()


class Test_data_vars_without_coords:
    def test_removes_coords_when_only_coords_dataset(self):
        dset = xr.Dataset(coords=dict(x=[1, 2, 3]))
        data_vars = fields.data_vars_without_coords(dset)
        assert data_vars['x'].values.tolist() == [1, 2, 3]
        assert data_vars['x'].coords == {}

    def test_removes_coords_when_datavars_coords_mixture_dataset(self):
        dset = xr.Dataset(
            data_vars=dict(data=xr.Variable(dims='x', data=[10, 20, 30])),
            coords=dict(x=[1, 2, 3])
        )
        data_vars = fields.data_vars_without_coords(dset)
        assert data_vars['x'].values.tolist() == [1, 2, 3]
        assert data_vars['x'].coords == {}
        assert data_vars['data'].values.tolist() == [10, 20, 30]
        assert data_vars['data'].coords == {}
