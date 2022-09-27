import pytest
import xarray as xr
from ftle import ladim2ftle
import numpy as np


@pytest.fixture(scope='module')
def ladim_dset():
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
                data=[1000, 1000, 1000, 1000, 1000],
            ),
            X=xr.Variable(
                dims='particle_instance',
                data=[1., 2., 1., 3., 4., 1.1, 2.2, 1.5, 3.1],
            ),
            Y=xr.Variable(
                dims='particle_instance',
                data=[20., 20., 30., 30., 30., 21., 25., 31., 35.],
            ),
        ),
    )


@pytest.fixture(scope='module')
def stack_dset(ladim_dset):
    return ladim2ftle.stack_by_time(ladim_dset)


@pytest.fixture(scope='module')
def grid_dset(stack_dset):
    return ladim2ftle.reshape_by_coords(stack_dset)


class Test_stack_by_time:
    def test_returns_reorganized_dataset(self, stack_dset):
        assert stack_dset.dims == {'pid': 4, 'time': 2}
        assert stack_dset.time.dims == ('time', )
        assert stack_dset.pid.dims == ('pid',)
        assert stack_dset.X.dims == ('time', 'pid')
        assert stack_dset.Y.dims == ('time', 'pid')
        assert stack_dset.particle_count.dims == ('time', )


class Test_reshape_by_coords:
    def test_has_correct_dimensions(self, grid_dset):
        assert grid_dset.dims == {'Y0': 2, 'X0': 3, 'time': 2}

    def test_has_organized_endpos_in_a_grid(self, grid_dset):
        assert grid_dset.X.dims == ('Y0', 'X0')
        assert grid_dset.Y.dims == ('Y0', 'X0')

    def test_missing_grid_particles_have_negative_pid(self, grid_dset):
        assert (grid_dset.pid.values < 0).astype(int).tolist() == [
            [0, 0, 1],
            [0, 1, 0],
        ]

    def test_missing_grid_particles_have_nan_coords(self, grid_dset):
        assert np.isnan(grid_dset.X.values).tolist() == np.isnan(grid_dset.Y.values).tolist()
        assert np.isnan(grid_dset.X.values).astype(int).tolist() == [
            [0, 0, 1],
            [0, 1, 0],
        ]

    def test_has_organized_pid_in_a_grid(self, grid_dset):
        assert grid_dset.pid.dims == ('Y0', 'X0')


@pytest.fixture(scope='module')
def dist_dset(grid_dset):
    return ladim2ftle.compute_neighbour_maxdist(grid_dset)


class Test_compute_neighbour_distances:
    def test_computes_correct_maximal_distance(self, dist_dset):
        assert dist_dset.maxdist.values.astype(int).tolist() == [
            [10, 4, 0], [10, 0, 0],
        ]
