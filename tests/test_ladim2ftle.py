import pytest
import xarray as xr
from ftle import ladim2ftle
import numpy as np


@pytest.fixture(scope='module')
def dset():
    return xr.Dataset(
        coords=dict(
            time=xr.Variable(
                dims='time',
                data=np.array(['2016-01-01', '2016-01-02']).astype('datetime64[ns]'),
            ),
        ),
        data_vars=dict(
            particle_count=xr.Variable(
                dims='time',
                data=[5, 4],
            ),
            pid=xr.Variable(
                dims='particle_instance',
                data=[0, 1, 2, 3, 4, 0, 1, 2, 3],
            ),
            release_time=xr.Variable(
                dims='particle',
                data=[1000, 2000, 3000, 4000, 5000],
            ),
            X=xr.Variable(
                dims='particle_instance',
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            Y=xr.Variable(
                dims='particle_instance',
                data=[20, 20, 30, 40, 50, 60, 70, 80, 90],
            ),
        ),
    )


class Test_stack_by_time:
    def test_returns_reorganized_dataset(self, dset):
        out = ladim2ftle.stack_by_time(dset)
        assert out.dims == {'pid': 4, 'time': 2}
        assert out.time.dims == ('time', )
        assert out.pid.dims == ('pid',)
        assert out.X.dims == ('time', 'pid')
        assert out.Y.dims == ('time', 'pid')
        assert out.particle_count.dims == ('time', )


class Test_reshape_by_coords:
    @pytest.fixture(scope='class')
    def stack_dset(self, dset):
        return ladim2ftle.stack_by_time(dset)

    def test_has_correct_dimensions(self, stack_dset):
        out = ladim2ftle.reshape_by_coords(stack_dset)
        assert out.dims == {'Y0': 3, 'X0': 4, 'time': 2}

    def test_has_organized_endpos_in_a_grid(self, stack_dset):
        out = ladim2ftle.reshape_by_coords(stack_dset)
        assert out.X.dims == ('Y0', 'X0')
        assert out.Y.dims == ('Y0', 'X0')

    def test_missing_grid_particles_are_negative(self, stack_dset):
        out = ladim2ftle.reshape_by_coords(stack_dset)
        assert (out.X.values < 0).astype(int).tolist() == [
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]

    def test_has_organized_pid_in_a_grid(self, stack_dset):
        out = ladim2ftle.reshape_by_coords(stack_dset)
        assert out.pid.dims == ('Y0', 'X0')
