import xarray as xr
import netCDF4 as nc
import numpy as np
import logging
import glob


logger = logging.getLogger(__name__)


def append(dist_dset, outfile):
    logger.info(f'Append data to outfile "{outfile}"')
    with nc.Dataset(outfile, 'a') as out:
        num_times = len(out.variables['time'])
        idx_depth = np.flatnonzero(out.variables['depth'][:] == dist_dset.meta_depth)[0]
        idx_x_min = dist_dset.X0.min().values.astype(int).item()
        idx_y_min = dist_dset.Y0.min().values.astype(int).item()
        idx_x_max = idx_x_min + len(dist_dset.X0)
        idx_y_max = idx_y_min + len(dist_dset.Y0)
        idx_x = slice(idx_x_min, idx_x_max)
        idx_y = slice(idx_y_min, idx_y_max)

        datenum = nc.date2num(
            dates=np.datetime64(dist_dset.meta_date).astype('datetime64[h]').astype(object),
            units=out.variables['time'].units,
            calendar=getattr(out.variables['time'], 'calendar', 'standard'),
        )
        out.variables['time'][num_times] = datenum
        out.variables['ALCS'][num_times, idx_depth, idx_y, idx_x] = dist_dset.maxdist.values


def create_output_dataset(nk800_fname, outfile_fname):
    logger.info(f'Load grid file "{nk800_fname}"')
    nk800_dset = xr.load_dataset(nk800_fname)

    out = xr.Dataset(
        data_vars=dict(
            depth=xr.Variable(
                dims='z',
                data=[5, 100, 300],
                attrs=dict(long_name="Vertical z-level"),
            ),
            lat_rho=nk800_dset.lat_rho,
            lon_rho=nk800_dset.lon_rho,
            mask_rho=nk800_dset.mask_rho,
            time=xr.Variable(
                dims='time',
                data=[],
                attrs=dict(
                    long_name="time since initialization",
                    units="days since 1948-01-01 00:00:00",
                )
            ),
            ALCS=xr.Variable(
                dims=('time', 'z', 'y', 'x'),
                data=np.zeros((0, 3) + nk800_dset.lat_rho.shape, dtype='f8'),
                attrs=dict(
                    long_name="Attracting Lagrangian Coherent Structure",
                    units="meter",
                    coordinates='lon_rho lat_rho',
                ),
            ),
        ),
    )
    grid_mapping_name = nk800_dset.mask_rho.grid_mapping
    out[grid_mapping_name] = nk800_dset[grid_mapping_name]

    logger.info(f'Create output file "{outfile_fname}"')
    out.to_netcdf(
        outfile_fname,
        unlimited_dims='time',
        encoding={'ALCS': {'_FillValue': 0}}
    )


def load_ladim_dset(ladim_fname):
    import re
    m = re.match(r'.*?_([0-9]{8})_([0-9]{3})m.nc', ladim_fname)
    d = m.group(1)
    datestring = f'{d[0:4]}-{d[4:6]}-{d[6:8]}'
    z = int(m.group(2))

    out = xr.load_dataset(ladim_fname)
    out.attrs["meta_date"] = datestring
    out.attrs["meta_depth"] = z

    return out


def run(ladim_fnames, nk800_fname, outfile_name):
    create_output_dataset(nk800_fname, outfile_name)
    for ladim_fname in ladim_fnames:
        logger.info(f'Load ladim file "{ladim_fname}"')
        ladim_dset = load_ladim_dset(ladim_fname)
        run_single(ladim_dset, outfile_name)


def run_single(ladim_dset, outfile_name):
    logger.info(f'Compute ALCS')
    stack_dset = stack_by_time(ladim_dset)
    grid_dset = reshape_by_coords(stack_dset)
    dist_dset = compute_neighbour_maxdist(grid_dset)
    append(dist_dset, outfile_name)


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
    dist2[:, :, [0, -1]] = 0
    dist2[:, [0, -1], :] = 0
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


def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        prog='ladim2alcs',
        description="LADIM2ALCS  Create Attracting Lagrangian Coherent Structure file"
    )
    parser.add_argument(
        '--nk800',
        metavar="NK800.nc",
        default=None,
        help="NorKyst800 grid file (required if ALCS.nc is not present)",
    )
    parser.add_argument('fname_out', help="ALCS output file", metavar='out.nc')
    parser.add_argument('fnames_in', nargs='+', help="Ladim output files", metavar='ladim.nc')
    args = parser.parse_args(args)

    fnames_in = []
    for pattern in args.fnames_in:
        fnames_in += sorted(glob.glob(pattern))

    init_logger()

    run(ladim_fnames=fnames_in, nk800_fname=args.nk800, outfile_name=args.fname_out)


def init_logger():
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    main(*argv)
