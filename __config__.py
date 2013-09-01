import os
__version__ = (1, 0, 0)
R = os.path.dirname(os.path.realpath(__file__))
MTL_PARAM_DB_FILE = os.path.join(R, "materials/material_properties.db")
from utils.namespace import Namespace
cfg = Namespace()
cfg.debug = False
cfg.sqa = False
cfg.I = None
