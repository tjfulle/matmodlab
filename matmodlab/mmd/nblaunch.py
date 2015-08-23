import os
import sys
from subprocess import Popen
from argparse import ArgumentParser
from os.path import dirname, isdir, isfile, join
import signal
from subprocess import Popen, PIPE
try:
    from IPython.html import notebookapp
    from IPython.html.utils import url_path_join
except ImportError:
    raise SystemExit('IPython.html not found')

from matmodlab.product import EXMPL_D, PYEXE, IPY_D

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    p = ArgumentParser()
    p.add_argument('-d', default=os.getcwd(),
        help='Directory to launch notebook server [default: %(default)s]')
    p.add_argument('--examples', action='store_true', default=False,
        help=('Launch notebook server in matmodlab/examples '
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
    try:
        proc = Popen(command, env=env, shell=True, preexec_fn=os.setsid)
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGTERM)
    return 0
    #notebookapp.launch_new_instance(notebook_dir=d, open_browser=True, argv=a)

if __name__ == '__main__':
    main()
