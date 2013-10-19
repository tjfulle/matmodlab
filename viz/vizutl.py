def readtabular(source):
    """Read in the mml-tabular.dat file

    """
    from utils.mmltab import read_mml_evaldb
    sources, paraminfo, _ = read_mml_evaldb(source)
    for (i, info) in enumerate(paraminfo):
        paraminfo[i] = ", ".join("{0}={1:.2g}".format(n, v) for (n, v) in info)
    return sources, paraminfo


def loadcontents(filepath):
    if filepath.endswith((".exo", ".e", ".base_exo")):
        exof = ExodusIIReader.new_from_exofile(filepath)
        glob_var_names = exof.glob_var_names()
        elem_var_names = exof.elem_var_names()
        data = [exof.get_all_times()]
        for glob_var_name in glob_var_names:
            data.append(exof.get_glob_var_time(glob_var_name))
        for elem_var_name in elem_var_names:
            data.append(exof.get_elem_var_time(elem_var_name, 0))
        data = np.transpose(np.array(data))
        head = ["TIME"] + glob_var_names + elem_var_names
        exof.close()
    else:
        head = loadhead(filepath)
        data = loadtxt(filepath, skiprows=1)
    return head, data


def loadhead(filepath, comments="#"):
    """Get the file header

    """
    line = " ".join(x.strip() for x in linecache.getline(filepath, 1).split())
    if line.startswith(comments):
        line = line[1:]
    return line.split()


def loadtxt(f, skiprows=0, comments="#"):
    """Load text from output files

    """
    lines = []
    for (iline, line) in enumerate(open(f, "r").readlines()[skiprows:]):
        try:
            line = [float(x) for x in line.split(comments, 1)[0].split()]
        except ValueError:
            break
        if not lines:
            ncols = len(line)
        if len(line) < ncols:
            break
        if len(line) > ncols:
            stop("*** {0}: error: {1}: inconsistent data in row {1}".format(
                EXE, os.path.basename(f), iline))
        lines.append(line)
    return np.array(lines)


def get_sorted_fileinfo(filepaths):
    """Sort the fileinfo based on length of header in each file in filepaths so
    that the file with the longest header is first

    """
    fileinfo = []
    for filepath in filepaths:
        fnam = os.path.basename(filepath)
        fhead, fdata = loadcontents(filepath)
        if not np.any(fdata):
            logerr("No data found in {0}".format(filepath))
            continue
        fileinfo.append((fnam, fhead, fdata))
        continue
    if logerr():
        stop("***error: stopping due to previous errors")
    return sorted(fileinfo, key=lambda x: len(x[1]), reverse=True)


def common_prefix(strings):
    """Find the longest string that is a prefix of all the strings.

    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix
