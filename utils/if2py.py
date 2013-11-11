import os
import sys
from numpy.f2py import main as f2py_main

def f2py(name, source_files, signature, fc, incd, disp=0):
    """Build material model with f2py

    """
    # f2py pulls its arguments from sys.argv
    # f2py returns none if successful, it is an exception if not
    # successful
    hold = [x for x in sys.argv]
    fflags = "-fPIC -shared"
    argv = ["f2py", "-c",
            "-I{0}".format(incd) if incd else None,
            "-m", name,
            "--f90flags={0}".format(fflags),
            "--f77flags={0}".format(fflags),
            "--f77exec={0}".format(fc),
            "--f90exec={0}".format(fc),
            "-DIMPLNONE", "-Dpycallback",
            "--link-lapack_opt",
            signature if signature else None,
            "--quiet"]
    argv.extend(source_files)
    sys.argv = [x for x in argv if x]

    try:
        os.remove(os.path.join(os.path.dirname(source_files[0]), name + ".so"))
    except OSError:
        pass

    if disp == 0:
        f = os.devnull
    else:
        f = "f2py.con"
    with open(f, "w") as sys.stdout:
        with open(f, "a") as sys.stderr:
            try:
                f2py_main()
                built = 0
            except BaseException as e:
                print re.sub(r"error: ", "", e.message)
                built = 1
            except:
                print "failed to build {0} with f2py".format(name)
                built = 1

    sys.argv = [x for x in hold]
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    sys.stdout.flush()
    sys.stderr.flush()

    return built

if __name__ == "__main__":
    name = sys.argv[1]
    built = f2py(name, [name + ".f90"], None, "gfortran", os.getcwd())
