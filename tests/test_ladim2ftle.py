import pytest
import xarray as xr
from ftle import ladim2ftle


@pytest.fixture(scope='module')
def dset():
    return xr.Dataset(
        data_vars=dict(
            particle_count=xr.Variable(
                dims='time',
                data=[4, 5],
            ),
            pid=xr.Variable(
                dims='particle_instance',
                data=[0, 1, 2, 3, 0, 1, 2, 3, 4],
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
                data=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            ),
        ),
    )


class Test_get_original_position:
    def test_returns_correct_xy_values(self, dset):
        out = ladim2ftle.get_original_position(dset)
        assert out.X.values.tolist() == [1, 2, 3, 4, 9]
        assert out.Y.values.tolist() == [10, 20, 30, 40, 90]
