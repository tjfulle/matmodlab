import Payette_model_index as pmi
import Payette_xml_parser as px

from viz.mdldat import GMDModel, GMDModelParameter, GMDMaterial, GMDMaterialParameter


def load_material_params(model_index, model_name):
    if model_name not in model_index.constitutive_models():
        return []

    materials = []
    material_database = model_index.control_file(model_name)
    if material_database is not None:
        xml_obj = px.XMLParser(material_database)
        mats = xml_obj.get_parameterized_materials()
        for mat in mats:
            name, aliases = mat

            parameters = []
            for param in xml_obj.get_material_parameterization(mat[0]):
                key, default = param
                if key == "Units":
                    continue
                parameters.append(
                    GMDMaterialParameter(name=key, default=default))
            materials.append(GMDMaterial(
                name=name, aliases=aliases, defaults=parameters))

    return materials


def load_models(model_type='any'):
    models = []
    model_index = pmi.ModelIndex()

    for model_name in model_index.constitutive_models():
        control_file = model_index.control_file(model_name)
        cmod = model_index.constitutive_model_object(model_name)
        cmod_obj = cmod(control_file)
        if model_type != 'any' and model_type not in cmod_obj.material_type:
            continue

        params = []
        for param in cmod_obj.get_parameter_names_and_values():
            params.append(
                GMDModelParameter(
                    name=param[0], description=param[1],
                    distribution='Specified', specified=param[2])
            )

        model = GMDModel(
            model_name=model_name, parameters=params, model_type=[
                cmod_obj.material_type],
            materials=load_materials(model_index, model_name))
        models.append(model)

    return models
