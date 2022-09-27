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
    pass
