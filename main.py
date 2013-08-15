import os
import sys
import argparse

import core.gmd as gmd

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source file path")
    args = parser.parse_args(argv)

    # Instantiate MMD object
    model = gmd.ModelDriver.from_input_file(args.source)
    model.setup()
    model.run()
    model.finish()


if __name__ == "__main__":
    main()
