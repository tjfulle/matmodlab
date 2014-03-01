from viz.mdldat import (MMLModel, MMLModelParameter,
                        MMLMaterial, MMLMaterialParameter)
from __config__ import F_MTL_PARAM_DB
from utils.mtldb import read_all_material_params_from_db
from utils.impmod import load_file
from materials.material import read_mtldb


def load_material_params(name):
    materials = []
    for (material, params) in read_all_material_params_from_db(name).items():
        parameters = [MMLMaterialParameter(name=k, default=v)
                      for (k, v) in params.items()]
        materials.append(MMLMaterial(name=material, defaults=parameters))
    return materials


def load_models():
    models = []
    fmt = lambda s: s.strip().upper()
    for (name, (filepath, mclass)) in read_mtldb().items():
        # get the parameter names for the model
        module = load_file(filepath)
        parameters = getattr(module, mclass).param_names
        params = [MMLModelParameter(name=fmt(p), distribution="Specified")
                  for p in parameters]
        materials = load_material_params(name)
        model = MMLModel(name=name, parameters=params, materials=materials,
                         model_type=["any"])
        models.append(model)

    return models
