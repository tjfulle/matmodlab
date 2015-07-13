import os
import sys
from subprocess import Popen
from argparse import ArgumentParser
from os.path import dirname, isdir, isfile, join
from IPython.html import notebookapp
from IPython.html.utils import url_path_join

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

    a = ['--profile-dir={0}'.format(IPY_D)] + other
    notebookapp.launch_new_instance(notebook_dir=d, open_browser=True, argv=a)

if __name__ == '__main__':
    main()
