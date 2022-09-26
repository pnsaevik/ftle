def ladim2ftle():
    import sys
    args = sys.argv[1:]
    from . import ladim2ftle
    ladim2ftle.main(infile=args[0], outfile=args[1])


def main():
    pass
