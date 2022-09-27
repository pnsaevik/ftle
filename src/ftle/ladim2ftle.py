import xarray as xr
import numpy as np


def append(dset, outfile):
    pass


def run(ladim_dset):
    return xr.Dataset()


def stack_by_time(ladim_dset: xr.Dataset):
    """
    Reorganize the dataset by start/stop position
    :param ladim_dset: A ladim dataset
    :return: A dataset stacked by time
    """

    # Simplify input dataset
    dset = ladim_dset.drop_dims("particle").swap_dims(particle_instance='pid')

    # Extract start and stop positions
    count = dset.particle_count.values
    dset0 = dset.isel(pid=slice(None, count[0]), time=[0])
    dset1 = dset.isel(pid=slice(-count[-1], None), time=[-1])

    # Combine start and stop positions into a multidimensional array
    return xr.concat(objs=[dset0, dset1], dim='time', join='inner')


def reshape_by_coords(stack_dset: xr.Dataset):
    """
    Reshape the dataset by the gridded coordinates of the original positions
    :param stack_dset: A dataset stacked by time
    :return: A gridded dataset
    """

    dset0 = stack_dset.isel(time=0)
    dset1 = stack_dset.isel(time=-1)

    # Reindex dset1 by the position of origin (X0, Y0)
    dset_unstack = dset1.assign(
        X0=dset0.X,
        Y0=dset0.Y,
        particle=xr.Variable(dims='pid', data=dset1.pid.values),
    ).set_index(
        indexes={'pid': ('Y0', 'X0')},
    ).unstack(
        'pid', fill_value=np.nan,
    ).assign_coords(
        time=stack_dset.time.isel(time=[0, -1])
    ).rename_vars(particle='pid')

    return dset_unstack


def compute_neighbour_maxdist(grid_dset):
    """
    Compute max distance to neighbouring points
    :param grid_dset: A gridded dataset
    :return: The input dataset, but with appended variable `maxdist`
    """
    assert grid_dset.X.dims == ('Y0', 'X0')
    assert grid_dset.Y.dims == ('Y0', 'X0')

    # Extract coordinate values
    xy = np.stack([grid_dset.X.values, grid_dset.Y.values])

    # Compute squared distance between points
    dist2_lft_rgt = np.sum((xy[:, :, 1:] - xy[:, :, :-1])**2, axis=0)
    dist2_top_bot = np.sum((xy[:, 1:, :] - xy[:, :-1, :])**2, axis=0)

    # Compute squared distance to neighbour points
    dist2 = np.empty((4,) + grid_dset.X.shape, dtype=xy.dtype)
    dist2[:, :, [0, -1]] = np.nan
    dist2[:, [0, -1], :] = np.nan
    dist2[0, :, :-1] = dist2_lft_rgt  # right
    dist2[1, :, 1:] = dist2_lft_rgt   # left
    dist2[2, :-1, :] = dist2_top_bot  # bottom
    dist2[3, 1:, :] = dist2_top_bot   # top

    # Find the maximal neighbour distance
    dist_max = np.sqrt(np.nanmax(dist2, axis=0, initial=0))

    attrs = dict(long_name='maximal distance to neighbouring points')
    if 'units' in grid_dset.X.attrs:
        attrs['units'] = grid_dset.X.attrs['units']

    return grid_dset.assign(
        maxdist=xr.Variable(
            dims=grid_dset.X.dims,
            data=dist_max,
            attrs=attrs,
        ),
    )


def main(infile, outfile):
    dset_in = xr.load_dataset(infile)
    dset_out = run(dset_in)
    append(dset_out, outfile)


if __name__ == '__main__':
    main(
        infile=r"C:\Users\a5606\Downloads\lyapunov\NK800_20161221_005m.nc",
        outfile=r"C:\Users\a5606\Downloads\lyapunov\output.nc",
    )
