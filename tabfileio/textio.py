import re
import gzip
import numpy as np


RE = re.compile('[ \,]')


def _split(string, comments, i=0):
    return [x for x in RE.split(string.strip().split(comments, 1)[i])
                                                         if x.split()]


def read_text(filename, skiprows=0, comments='#', columns=None, disp=1):

    # Check to see if we are looking at a gzipped text file
    if filename.lower().endswith(".txt.gz"):
        opener = gzip.open
    else:
        opener = open

    # Open the file in byte-mode and decode
    with opener(filename, 'rb') as F:
        content = F.read().decode("utf-8")
        lines = content.split("\n")

    # set the index past the lines that we don't want
    line_idx = skiprows

    # Check for headers
    headline = lines[line_idx].strip()
    if headline.startswith(comments):
        probably_header = True
        headline = headline.split(comments, 1)[1]
    else:
        probably_header = False
        try:
            [float(x) for x in _split(headline, comments)]
        except ValueError:
            probably_header = True

    if probably_header:
        head = _split(headline, comments)
        line_idx = skiprows + 1
    else:
        # first line not a header, rewind
        head = None
        line_idx = skiprows

    data = []
    try:
        for i in range(line_idx, len(lines)):
            line = _split(lines[i], comments)

            if not line:
                continue

            try:
                line = [float(x) for x in line]
            except ValueError:
                raise Exception('expected floats in line {0} '
                                'got {1}'.format(i+1, line))
            data.append(line)
    except:
        pass

    data = np.array(data)

    # If specific columns are requested, filter the data
    if columns is not None:
        if any(isinstance(x, str) for x in columns):
            h = [s.lower() for s in head]
            for (i, item) in enumerate(columns):
                if isinstance(item, str):
                    columns[i] = h.index(item.lower())

        if head is not None:
            head = [head[i] for i in columns]
        data = data[:, columns]

    if not disp:
        return data
    return head, data


def write_text(filename, head, data, columns=None):

    # This formatting is chosen because it can exactly represent a
    # double precision float. The width of 26 is chosen so as to give
    # at least one space between columns even when -1.0e+100 is used.
    def fltfmt(x):
        return "{0:26.17e}".format(x)

    def strfmt(x):
        return "{0:>26s}".format(x)

    #
    # If specific columns are requested, filter the data
    if columns is not None:
        if any(isinstance(x, str) for x in columns):
            h = [s.lower() for s in head]
            for (i, item) in enumerate(columns):
                if isinstance(item, str):
                    columns[i] = h.index(item.lower())
    else:
        columns = list(range(0, len(head)))

    # Check to see if we are looking at a gzipped text file
    if filename.lower().endswith(".txt.gz"):
        opener = gzip.open
    else:
        opener = open

    # Open the file in byte-mode and encode each line before writing
    with opener(filename, 'wb') as F:
        text = "".join([strfmt(head[i]) for i in columns]) + "\n"
        F.write(text.encode("utf-8"))
        for row in data:
            text = "".join([fltfmt(row[i]) for i in columns]) + "\n"
            F.write(text.encode("utf-8"))


if __name__ == '__main__':
    head, data = read_text("io_test.txt", columns=["TIME", "STRESS", 1])
    print(head)
    print(data)
