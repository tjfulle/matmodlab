import sys
import os
import numpy as np

from .interface import read_file, write_file, transform


def test():
    test_head = ["TIME", "INTEGER", "FLOAT"]
    test_data = np.array([[_ / 10.0, _, _ / 9.0] for _ in range(0, 10)])

    base_f = "new_io_test."
    base_d = os.path.dirname(os.path.realpath(__file__))

    write_cols = ["INTEGER", 2, "TIME"]
    read_cols = [2, "INTEGER", "FLOAT"]

    print("\n{0:=^80s}\n".format(" FILEIO DIAGNOSTIC "))

    type_coverage = ["txt", "txt.gz", "pkl", "xls", "xlsx", "json"]

    # write file in format A, read file in format A, compare.
    for ext in type_coverage:
        filename = base_f + ext
        print("\n===== Writing {0}".format(filename))

        write_file(filename, test_head, test_data, columns=write_cols)
        head, data = read_file(filename, columns=read_cols)

        print("Headers: " + repr(head))
        print("Data\n" + repr(data))
        print("Same headers? " + repr(test_head == head))
        print("Data diff\n" + repr(data - test_data))

        os.remove(filename)
        print("Removed {0}".format(filename))

args = sys.argv[1:]
if "-h" in args or "--help" in args:
    print("Use '--test' to perform tests.")
    print("Can be used to convert between file types:")
    print("  $ python -m tabfileio input.xlsx output.txt")

elif "--test" in args:
    test()

elif len(args) == 2:
    file1 = os.path.realpath(args[0])
    file2 = os.path.realpath(args[1])
    if not os.path.isfile(file1):
        sys.exit("Input file {0} does not exist".format(file1))
    if os.path.isfile(file2):
        sys.exit("Output file {0} exists".format(file2))

    print("Converting {0} to {1}".format(file1, file2))
    transform(file1, file2)
    print("Successfully converted {0} to {1}".format(file1, file2))
