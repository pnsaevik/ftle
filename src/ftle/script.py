def ladim2alcs():
    import argparse
    import glob
    import sys

    args = sys.argv[1:]

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

    from .ladim2ftle import run
    run(ladim_fnames=fnames_in, nk800_fname=args.nk800, outfile_name=args.fname_out)


def init_logger():
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )


def main():
    import argparse
    import glob
    import sys

    args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='ftle',
        description="FTLE  Create Finite-time Lyapunov Exponents from grid files"
    )
    parser.add_argument('outfile', help="FTLE output file", metavar='out.nc')
    parser.add_argument('infiles', nargs='+', help="ROMS datasets")
    args = parser.parse_args(args)

    fnames_in = []
    for pattern in args.infiles:
        fnames_in += sorted(glob.glob(pattern))

    init_logger()

    run(infiles=fnames_in, outfile=args.outfile)


def run(infiles, outfile):
    # TODO: This is only a stub

    # Open input files
    import xarray as xr
    mfdset = xr.Dataset(infiles)

    # Read options file
    start_times = mfdset.ocean_time[0:3].values
    stop_times = mfdset.ocean_time[1:4].values

    # Create output dataset
    import netCDF4 as nc
    with nc.Dataset(outfile, 'w') as out_dset:

        # Run simulations
        for start_time, stop_time in zip(start_times, stop_times):
            result = run_single(mfdset, start_time, stop_time)
            # append_result_to_output_dataset(result, out_dset)


def run_single(grid, start_time, stop_time):
    # TODO: This is only a stub
    import xarray as xr

    # Run particle simulation
    particles = xr.Dataset()

    # Convert particle simulation to Lyapunov exponents
    lyapunov = xr.Dataset()

    return lyapunov
