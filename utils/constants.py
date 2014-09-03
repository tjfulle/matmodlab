import numpy as np
NSYMM = 6
NTENS = 9
DEFAULT_TEMP = 298.
I6 = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
I9 = np.eye(3).reshape(9,)

ROOT2 = np.sqrt(2.0)
ROOT3 = np.sqrt(3.0)
TOOR2 = 1.0 / ROOT2
TOOR3 = 1.0 / ROOT3
ROOT23 = ROOT2 / ROOT3
