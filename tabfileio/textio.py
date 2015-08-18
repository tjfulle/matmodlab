import re
import numpy as np


RE = re.compile('[ \,]')
def _split(string, comments, i=0):
    return [x for x in RE.split(string.strip().split(comments,1)[i]) if x.split()]


def read_text(filename, skiprows=0, comments='#', columns=None, disp=1):

    fown = False
    try:
        if isinstance(filename, basestring):
            fown = True
            fh = iter(open(filename))
        else:
            fh = iter(filename)
    except (TypeError):
        message = 'filename must be a string, file handle, or generator'
        raise ValueError(message)

    a = fh.tell()
    # Take care of those pesky unwanted rows
    for _ in range(skiprows):
        fh.readline()

    # Check for headers
    headline = fh.readline().strip()
    if headline.startswith(comments):
        probably_header = True
        headline = headline.split(comments, 1)[1]
        #headline = _split(headline, comments,1)
    else:
        probably_header = False
        try:
            [float(x) for x in _split(headline, comments)]
        except ValueError:
            probably_header = True

    if probably_header:
        head = _split(headline, comments)
    else:
        # first line not a header, rewind
        head = None
        fh.seek(a)

    data = []
    try:
        for (i, line) in enumerate(fh.readlines()):
            line = _split(line, comments)
            if not line:
                continue

            try:
                line = [float(x) for x in line]
            except ValueError:
                raise ValueError('expected floats in line {0} '
                                 'got {1}'.format(i+1, line))
            data.append(line)

    finally:
        if fown:
            fh.close()

    data = np.array(data)

    # If specific columns are requested, filter the data
    if columns is not None:
        if any(isinstance(x, basestring) for x in columns):
            h = [s.lower() for s in head]
            for (i, item) in enumerate(columns):
                if isinstance(item, basestring):
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
    fltfmt = lambda x: "{0:26.17e}".format(x)
    strfmt = lambda x: "{0:>26s}".format(x)

    #
    # If specific columns are requested, filter the data
    if columns is not None:
        if any(isinstance(x, basestring) for x in columns):
            h = [s.lower() for s in head]
            for (i, item) in enumerate(columns):
                if isinstance(item, basestring):
                    columns[i] = h.index(item.lower())
    else:
        columns = list(range(0, len(head)))

    with open(filename, 'w') as F:
        F.write("".join([strfmt(head[i]) for i in columns]) + "\n")
        for row in data:
            F.write("".join([fltfmt(row[i]) for i in columns]) + "\n")


if __name__ == '__main__':
    head, data = read_text("io_test.txt", columns=["TIME", "STRESS", 1])
    print(head)
    print(data)
