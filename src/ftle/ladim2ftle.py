import xarray as xr
import numpy as np


def append(dset, outfile):
    pass


def run(ladim_dset):
    return xr.Dataset()


def reorganize(ladim_dset: xr.Dataset):
    """
    Reorganize the dataset by start/stop position
    :param ladim_dset:
    :return:
    """

    # Simplify input dataset
    dset = ladim_dset.drop_dims("particle").swap_dims(particle_instance='pid')

    # Extract start and stop positions
    count = dset.particle_count.values
    dset0 = dset.isel(pid=slice(None, count[0]), time=[0])
    dset1 = dset.isel(pid=slice(-count[-1], None), time=[-1])

    # Combine start and stop positions into a multidimensional array
    dset_con = xr.concat(objs=[dset0, dset1], dim='time', join='inner')

    # Reindex dset1 by the position of origin (X0, Y0)
    dset_unstack = dset1.assign(
        X0=dset_con.X.isel(time=0).astype('i4'),
        Y0=dset_con.Y.isel(time=0).astype('i4'),
        particle=xr.Variable(dims='pid', data=dset1.pid.values),
    ).set_index(
        indexes={'pid': ('Y0', 'X0')},
    ).unstack(
        'pid', fill_value=-1,
    ).drop_dims('time').assign_coords(
        time=dset_con.time.isel(time=[0, -1])
    ).rename_vars(particle='pid')

    return dset_unstack


def main(infile, outfile):
    dset_in = xr.load_dataset(infile)
    dset_out = run(dset_in)
    append(dset_out, outfile)


if __name__ == '__main__':
    main(
        infile=r"C:\Users\a5606\Downloads\lyapunov\NK800_20161221_005m.nc",
        outfile=r"C:\Users\a5606\Downloads\lyapunov\output.nc",
    )
