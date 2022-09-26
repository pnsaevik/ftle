import xarray as xr
import numpy as np


def append(dset, outfile):
    pass


def run(ladim_dset):
    return xr.Dataset()


def get_original_position(ladim_dset):
    """
    For each particle in ladim_dset, find the initial X, Y coordinate position.
    :param ladim_dset:
    :return:
    """
    particle, first_occurence = np.unique(ladim_dset.pid.values, return_index=True)
    x = ladim_dset.X.values[first_occurence]
    y = ladim_dset.Y.values[first_occurence]
    return xr.Dataset(
        coords=dict(
            particle=xr.Variable(dims='particle', data=particle),
        ),
        data_vars=dict(
            X=xr.Variable(dims='particle', data=x, attrs=dict(long_name='initial x position')),
            Y=xr.Variable(dims='particle', data=y, attrs=dict(long_name='initial y position')),
            particle_instance=xr.Variable(dims='particle', data=first_occurence),
            release_time=ladim_dset.release_time.isel(particle=particle),
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
