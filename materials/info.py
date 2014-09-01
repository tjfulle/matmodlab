import os
import sys
from project import UMATS
from materials.materialdb import MaterialDB
D = os.path.dirname(os.path.realpath(__file__))
MTL_DB_D = os.path.join(D, "db")
F_MTL_PARAM_DB = os.path.join(MTL_DB_D, "material_properties.db")
MATERIAL_DB = MaterialDB.gen_db(UMATS)
sys.path.insert(0, MATERIAL_DB.path)
