import os
import numpy as np

from .excelio import read_excel, write_excel
from .textio import read_text, write_text
from .pickleio import read_pickle, write_pickle
from .jsonio import read_json, write_json

test_head = ["TIME", "INTEGER", "FLOAT"]
test_data = np.array([[_ / 10.0, _, _ / 9.0] for _ in range(0, 10)])

base_f = "new_io_test."
base_d = os.path.dirname(os.path.realpath(__file__))

write_cols = ["INTEGER", 2, "TIME"]
read_cols = [2, "INTEGER", "FLOAT"]

print("\n{0:=^80s}\n".format(" FILEIO DIAGNOSTIC "))

type_coverage = [
                 ["txt", write_text, read_text],
                 ["txt.gz", write_text, read_text],
                 ["pkl", write_pickle, read_pickle],
                 ["xls", write_excel, read_excel],
                 ["xlsx", write_excel, read_excel],
                 ["json", write_json, read_json],
                ]

# write file in format A, read file in format A, compare.
for ext, fw, fr in type_coverage:
    filename = base_f + ext
    print("\n===== Writing {0}".format(filename))

    fw(filename, test_head, test_data, columns=write_cols)
    head, data = fr(filename, columns=read_cols)

    print("Headers: " + repr(head))
    print("Data\n" + repr(data))
    print("Same headers? " + repr(test_head == head))
    print("Data diff\n" + repr(data - test_data))

    os.remove(filename)
    print("Removed {0}".format(filename))
