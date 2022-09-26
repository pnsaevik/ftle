import pytest
import xarray as xr
from ftle import ladim2ftle
import numpy as np


@pytest.fixture(scope='module')
def dset():
    return xr.Dataset(
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


class Test_reorganize:
    def test_has_correct_dimensions(self, dset):
        out = ladim2ftle.reorganize(dset)
        assert 'time' in set(out.dims)
        assert 'X0' in set(out.dims)
        assert 'Y0' in set(out.dims)

    def test_has_organized_endpos_in_a_grid(self, dset):
        out = ladim2ftle.reorganize(dset)
        assert out.X.dims == ('Y0', 'X0')
        assert out.Y.dims == ('Y0', 'X0')

    def test_missing_grid_particles_are_negative(self, dset):
        out = ladim2ftle.reorganize(dset)
        assert (out.X.values < 0).astype(int).tolist() == [
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]

    def test_has_organized_pid_in_a_grid(self, dset):
        out = ladim2ftle.reorganize(dset)
        assert out.pid.dims == ('Y0', 'X0')
