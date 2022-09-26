import xarray as xr


def append(dset, outfile):
    pass


def run(ladim_dset):
    return xr.Dataset()


def main(infile, outfile):
    dset_in = xr.load_dataset(infile)
    dset_out = run(dset_in)
    append(dset_out, outfile)


if __name__ == '__main__':
    main(
        infile=r"C:\Users\a5606\Downloads\lyapunov\NK800_20161221_005m.nc",
        outfile=r"C:\Users\a5606\Downloads\lyapunov\output.nc",
    )
