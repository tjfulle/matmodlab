import os
__version__ = (1, 0, 0)
ROOT_D = os.path.dirname(os.path.realpath(__file__))
LIB_D = os.path.join(ROOT_D, "lib")
MTL_DB_D = os.path.join(ROOT_D, "materials/db")
F_MTL_PARAM_DB = os.path.join(MTL_DB_D, "material_properties.db")
F_MTL_MODEL_DB = os.path.join(ROOT_D, "lib/material_models.db")
F_EVALDB = "mml-evaldb.xml"
RESTART = -2
from utils.namespace import Namespace
cfg = Namespace()
cfg.debug = False
cfg.sqa = False
cfg.I = None

SPLASH = """\
                  M           M    M           M    L
                 M M       M M    M M       M M    L
                M   M   M   M    M   M   M   M    L
               M     M     M    M     M     M    L
              M           M    M           M    L
             M           M    M           M    L
            M           M    M           M    L
           M           M    M           M    LLLLLLLLL
                     Material Model Laboratory v {0}

""".format(".".join("{0}".format(i) for i in __version__))
