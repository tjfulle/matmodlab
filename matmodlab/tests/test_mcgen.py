import sys
from testconf import *
try:
    import pandas
    from matmodlab.fitting.mcgen import *
except ImportError:
    pandas = None

@pytest.mark.mcgen
@pytest.mark.skipif(pandas is None, reason='pandas not imported')
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
    f = os.path.join(this_directory, 'mcgen.csv')
    mc = MasterCurve.Import(f, ref_temp=75., apply_log=True,
                            fitter=PRONY, optimizer=FMIN, optwlf=False)
    s1 = 'WLF coefficients not within tolerance'
    s2 = 'Prony series not within tolerance'
    mc.fit()
    assert np.allclose(mc.wlf_opt, c, rtol=1.e-3, atol=1.e-3), s1
    assert np.allclose(mc.mc_fit[:, 1], p[:, 1], rtol=1.e-2, atol=1.e-2), s2
