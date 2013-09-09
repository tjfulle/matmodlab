import os
__version__ = (1, 0, 0)
ROOT_D = os.path.dirname(os.path.realpath(__file__))
LIB_D = os.path.join(ROOT_D, "lib")
MTL_LIB_D = os.path.join(ROOT_D, "materials/library")
F_MTL_PARAM_DB = os.path.join(MTL_LIB_D, "material_properties.db")
F_MTL_MODEL_DB = os.path.join(ROOT_D, "lib/material_models.db")
F_EVALDB = "gmd-evaldb.xml"
from utils.namespace import Namespace
cfg = Namespace()
cfg.debug = False
cfg.sqa = False
cfg.I = None

SPLASH = """\
                    GGGGGGG      M           M    DDDDDDD
                  G             M M       M M    D      D
                G              M   M   M   M    D       D
               G   GGGGGGG    M     M     M    D       D
              G         G    M           M    D       D
             G         G    M           M    D       D
             G       G     M           M    D      D
              GGGGGG      M           M    DDDDDDD
                 Generalized Model Driver v {0}

""".format(".".join("{0}".format(i) for i in __version__))
