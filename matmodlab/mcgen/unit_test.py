import sys
import logging
from argparse import ArgumentParser

from mc import *

def test():
    """example test case """
    # Baseline solution
    c = np.array([3.292, 181.82])
    p = np.array([[.0001, 2489],
                  [.001, 1482],
                  [.01, 803],
                  [.1, 402],
                  [1, 207],
                  [10, 124],
                  [100, 101],
                  [0, 222]], dtype=np.float64)
    mc = read_csv('mcgen.test', ref_temp=75., apply_log=True,
                  fitter=PRONY, optimizer=FMIN, optwlf=False)
    mc.to_csv('mcgen.csv')
    errors = 0
    if not np.allclose(mc.wlf_opt, c, rtol=1.e-3, atol=1.e-3):
        logging.error('WLF coefficients not within tolerance')
        errors += 1
    if not np.allclose(mc.mc_fit[:, 1], p[:, 1], rtol=1.e-2, atol=1.e-2):
        logging.error('Prony series not within tolerance')
        errors += 1
    if errors:
        print 'Failed'
    else:
        print 'Success'
    return

if __name__ == '__main__':
    sys.exit(test())
