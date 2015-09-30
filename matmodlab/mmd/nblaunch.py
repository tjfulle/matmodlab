import os
import sys
from subprocess import Popen
from argparse import ArgumentParser
from os.path import dirname, isdir, isfile, join
import signal
from subprocess import Popen, PIPE

from ..product import EXMPL_D, PYEXE, IPY_D, TUT_D

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    p = ArgumentParser()
    p.add_argument('-d', default=os.getcwd(),
        help='Directory to launch notebook server [default: %(default)s]')
    p.add_argument('--examples', action='store_true', default=False,
        help=('Launch notebook server in matmodlab/examples '
              'directory [default: %(default)s]'))
    p.add_argument('--tutorial', action='store_true', default=False,
        help=('Launch notebook server in matmodlab/tutorial '
              'directory [default: %(default)s]'))
    args, other = p.parse_known_args(argv)

    if args.examples:
        d = EXMPL_D
    else:
        d = args.d

    a = other
    env = dict(os.environ)
    env['JUPYTER_CONFIG_DIR'] = IPY_D
    env['IPYTHONDIR'] = IPY_D
    command = 'ipython notebook'
    if args.examples:
        command += ' --notebook-dir={0}'.format(EXMPL_D)
    elif args.tutorial:
        command += ' --notebook-dir={0}'.format(TUT_D)
    kwds = {'env': env}
    try:
        kwds['preexec_fn'] = os.setsid
    except AttributeError:
        pass
    try:
        proc = Popen(command.split(), **kwds)
        proc.wait()
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGTERM)
    return 0

if __name__ == '__main__':
    main()
