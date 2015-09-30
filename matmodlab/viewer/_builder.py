import os
import re
import numpy as np
from traits.api import *
from traitsui.api import *
from traitsui.menu import Menu, Action, Separator
from pyface.api import FileDialog, OK as pyOK

from ..utils.misc import load_file
from ..mmd.loader import MaterialLoader #, SET_AT_RUNTIME
from ..constants import DEFAULT_TEMP, NUM_TENSOR_3D

defined_steps = ('StrainStep', 'StrainRateStep',
                 'StressStep', 'StressRateStep',
                 'DefGradStep', 'DisplacementStep',
                 'MixedStep', 'DataSteps')

class StepSelector(HasTraits):
    steps = List(Str)
    selected = Str('StrainStep')
    def __init__(self, selected=None):
        kwds = {}
        kwds['steps'] = [s for s in defined_steps]
        if selected is not None:
            kwds['selected'] = selected
        super(StepSelector, self).__init__(**kwds)
    traits_view = View(Item('selected', show_label = False,
                            editor = EnumEditor(name='steps')),
                       style='simple', resizable=True)

class StepDefn(HasTraits):
    xx, xy, xz = Float, Float, Float
    yx, yy, yz = Float, Float, Float
    zx, zy, zz = Float, Float, Float
    descriptors = Str
    kappa = Float(0.)
    temperature = Float(DEFAULT_TEMP)
    scale = Float(1.)
    increment = Float(1.)
    frames = Int(1)
    filename_button = Button
    filename = Str
    time_column = Int(0)
    columns = List([1,2,3,4,5,6])
    nc = Int
    kind = Str
    name = Str
    kwds = Dict
    def __init__(self, step_kind, keywords=None):
        kwds = {}
        nc = {'DefGradStep': 9, 'DisplacementStep': 3}.get(step_kind, 6)
        xx = xy = xz = 0.
        yx = yy = yz = 0.
        zx = zy = zz = 0.
        if nc == 9:
            xx = yy = zz = 1.
        kwds['xx'], kwds['xy'], kwds['xz'] = xx, xy, xz
        kwds['yx'], kwds['yy'], kwds['yz'] = yx, yy, yz
        kwds['zx'], kwds['zy'], kwds['zz'] = zx, zy, zz
        kwds['nc'] = nc
        kwds['kind'] = step_kind

        super(StepDefn, self).__init__(**kwds)

        if keywords is not None:

            components = keywords.get('components')
            if components is not None:
                if nc == 9:
                    (self.xx, self.xy, self.xz,
                     self.yx, self.yy, self.yz,
                     self.zx, self.zy, self.zz) = components

                elif nc == 3:
                    self.xx, self.yy, self.zz = components

                elif nc == 6:
                    N = NUM_TENSOR_3D - len(components)
                    components = np.append(components, [0.] * N)
                    (self.xx, self.yy, self.zz,
                     self.xy, self.yz, self.xz) = components

            descriptors = keywords.get('descriptors')
            if descriptors is not None:
                self.descriptors = ''.join(str(x) for x in descriptors)
            self.kappa = keywords.get('kappa', self.kappa)
            self.temperature = keywords.get('temperature', self.temperature)
            self.scale = keywords.get('scale', self.scale)
            self.increment = keywords.get('increment', self.increment)
            self.frames = keywords.get('frames', self.frames)
            self.filename = keywords.get('filename', self.filename)
            self.time_column = keywords.get('time_column', self.time_column)
            self.columns = keywords.get('columns', self.columns)

    sym_group = VGroup(
        HGroup(spring,
               Item('xx', show_label=False, width=-80),
               Item('xy', show_label=False, width=-80),
               Item('xz', show_label=False, width=-80)),
        HGroup(spring,
               Item('yy', show_label=False, width=-80),
               Item('yz', show_label=False, width=-80)),
        HGroup(spring,
               Item('zz', show_label=False, width=-80)),
        label='Components',
        show_border=True,
        visible_when='nc == 6 and kind != "DataSteps"')
    ten_group = VGroup(
        HGroup(Item('xx', show_label=False),
               Item('xy', show_label=False),
               Item('xz', show_label=False)),
        HGroup(Item('yx', show_label=False),
               Item('yy', show_label=False),
               Item('yz', show_label=False)),
        HGroup(Item('zx', show_label=False),
               Item('zy', show_label=False),
               Item('zz', show_label=False)),
        label='Components',
        show_border=True,
        visible_when='nc == 9 and kind != "DataSteps"')
    vec_group = VGroup(
        Item('xx', show_label=False, padding=15, width=-50),
        Item('yy', show_label=False, padding=15, width=-50),
        Item('zz', show_label=False, padding=15, width=-50),
        label='Deformation Components',
        show_border=True, visible_when='nc == 3 and kind != "DataSteps"')
    desc_group = Group('descriptors',
                       visible_when='kind in ("MixedStep", "DataSteps")')
    file_group = VGroup(
        HGroup(Item('filename', show_label=False, width=.9),
               Item('filename_button',
                    editor=ButtonEditor(label='Find'), show_label=False)),
        Item('time_column', label='Time Column'),
        Item('columns', editor=ListEditor(), label='Data Columns'),
        label='DataSteps file', show_border=True,
        visible_when='kind == "DataSteps"'),
    traits_view = View(
        Group(sym_group, ten_group, vec_group), file_group, desc_group, '10',
        Item('increment'),
        Item('frames'),
        Item('kappa', visible_when='"Stress" not in kind and "Mixed" not in kind'),
        Item('scale'),
        Item('temperature', visible_when='kind != "DataSteps"'))

    def _filename_button_fired(self):
        paths = open_file()
        if not paths:
            return
        self.filename = paths[0]

#-- StepEditor Class ---------------------------------------------------------
class StepEditorHandler(Handler):
    def closed(self, info, is_ok):
        if not is_ok:
            return
        obj = info.object
        obj.done = True
        obj.step_defn = {'name': obj.name,
                         'kwds': obj.create_step_kwds(),
                         'kind': obj.step.kind}

        if obj.edit_mode:
            obj.step_defn['update'] = obj.index

class StepEditor(HasTraits):
    name = Str
    step_selector = Instance(StepSelector)
    step = Instance(StepDefn)
    step_defn = Dict
    ikwds = Dict
    edit_mode = Int(0)
    index = Int
    done = Bool(False)
    editable = Bool(True)
    edit_view = View(
        Item('name', editor=TextEditor(multi_line=False)),
        '5', '_',
        VGroup(
            Group(Item('step_selector', show_label=False),
                  label='Step Kind', show_border=True, enabled_when='editable'),
            Group(Item('step', show_label=False),
                  label='Step Options', show_border=True, enabled_when='editable')),
            style='custom', resizable=True, buttons=['OK', 'Cancel'],
            scrollable=True, height=500, width=350, title='Step Editor',
            handler=StepEditorHandler())

    def create_step(self, name):
        self.name = name
        self.step_selector = StepSelector()
        self.edit_traits(view='edit_view')

    def edit_step(self, name, kind, kwds, index, editable):
        self.name = name
        self.edit_mode = 1
        self.ikwds = kwds
        self.index = index
        self.step_selector = StepSelector(selected=kind)
        self.editable = editable
        self.edit_traits(view='edit_view')

    @on_trait_change('step_selector.selected')
    def update_step_options(self, new):
        if new not in defined_steps:
            new = defined_steps[0]
        keywords = None if not self.edit_mode else dict(self.ikwds)
        self.step = StepDefn(new, keywords=keywords)

    def create_step_kwds(self):
        s = self.step
        kwds = {}
        kwds['frames'] = s.frames
        kwds['scale'] = s.scale
        if s.kind == 'DataSteps':
            kwds.update(self._data_steps_kwds())
        else:
            kwds.update(self._other_steps_kwds())

        return kwds

    def format_descriptors(self, desc):
        return ''.join([x for x in re.split(r' \,', desc) if x.split()])

    def _other_steps_kwds(self):
        s = self.step
        kwds = {}
        if s.nc == 3:
            components = [s.xx, s.yy, s.zz]
        elif s.nc == 9:
            components = [s.xx, s.xy, s.xz,
                          s.yx, s.yy, s.yz,
                          s.zx, s.zy, s.zz]
        else:
            components = [s.xx, s.yy, s.zz, s.xy, s.yz, s.xz]
        kwds['components'] = components

        if s.kind == 'MixedStep':
            kwds['descriptors'] = self.format_descriptors(s.descriptors)

        kwds['increment'] = s.increment
        kwds['temperature'] = s.temperature

        if 'Stress' not in s.kind and 'Mixed' not in s.kind:
            kwds['kappa'] = s.kappa

        return kwds

    def _data_steps_kwds(self):
        s = self.step
        kwds = {}
        kwds['descriptors'] = s.descriptors
        kwds['filename'] = s.filename
        kwds['columns'] = s.columns

        return kwds

class Parameters(HasTraits):
    pass

class ParameterEditor(HasTraits):
    """A simple instance editor for parameters"""
    parameters = Instance(Parameters)
    def __init__(self, param_names, ivals=None):
        dict_param_val = dict(zip(param_names, [0.] * len(param_names)))
        if ivals is not None:
            dict_param_val.update(ivals)
        kwds = {'parameters': Parameters(**dict_param_val)}
        super(ParameterEditor, self).__init__(**kwds)

    def update(self, dict):
        self.parameters = Parameters(**dict)

    traits_view = View(HGroup(Item('parameters',
                                   editor=InstanceEditor(),
                                   show_label=False),
                              spring),
                       style='custom', resizable=True)

class MaterialSelector(HasTraits):
    """The material model selector

    Parameters
    ----------
    materials : list(str)
        model names

    """
    materials = List(Str)
    selected = Str
    def __init__(self, materials, selected=None):
        selected = selected or materials[0]
        kwds = {'materials': materials,
                'selected': selected}
        super(MaterialSelector, self).__init__(**kwds)
    traits_view = View(Item('selected', show_label = False,
                            editor = EnumEditor(name='materials')),
                       style='simple', resizable=True)

#-- MaterialModelEditor Class -------------------------------------------------
def pretty(s):
    regex = re.compile('[_\- ]')
    return ' '.join(s.strip().title() for s in regex.split(s) if s.split())
class MaterialModelEditor(HasTraits):
    meta_info = Dict(Str, Dict)
    material_selector = Instance(MaterialSelector)
    parameter_editor = Instance(ParameterEditor)
    matinfo = Tuple(Str, Dict, Dict)
    rebuild = Bool(False)
    def __init__(self):
        kwds = {}

        # save all the information for editing later
        materials = load_matmodlab_materials()
        meta_info = {}
        for (name, params) in materials.items():
            meta_info[pretty(name)] = {'name': name, 'parameters': params}
        kwds['meta_info'] = meta_info

        material_names = sorted(meta_info.keys())
        ms = MaterialSelector(material_names)
        kwds['material_selector'] = ms

        parameter_names = meta_info[ms.selected]['parameters']
        pe = ParameterEditor(parameter_names)
        kwds['parameter_editor'] = pe

        super(MaterialModelEditor, self).__init__(**kwds)

    def edit(self, *update):
        if update:
            self.material_selector.selected = pretty(update[0])
            self.parameter_editor.update(update[1])
        self.edit_traits(view='edit_view')

    class MyHandler(Handler):
        def closed(self, info, is_ok):
            if not is_ok:
                return
            object = info.object
            name = object.material_selector.selected
            parameters = object.parameter_editor.parameters
            keys = object.meta_info[name]['parameters']
            dict_param_val = {}
            kwds = {'rebuild': object.rebuild}
            for key in keys:
                try:
                    dict_param_val[key] = float(getattr(parameters, key))
                except AttributeError:
                    continue
            object.matinfo = (object.meta_info[name]['name'], dict_param_val, kwds)

    edit_view = View(
        VGroup(
            Group(Item('material_selector', show_label=False),
                  label='Material Model', show_border=True),
            Group(Item('parameter_editor', show_label=False),
                  label='Parameters', show_border=True),
            Item('rebuild')),
            style='custom', resizable=True, buttons=['OK', 'Cancel'],
            height=.6, width=.2, scrollable=True,
            title='Material Model Selector',
        handler=MyHandler)

    @on_trait_change('material_selector.selected')
    def update_parameter_editor(self, new):
        try:
            prop_names = self.meta_info[new]['parameters']
            self.parameter_editor = ParameterEditor(prop_names)
        except KeyError:
            pass

def load_matmodlab_materials():
    all_mats = MaterialLoader.load_materials()
    models = {}
    for (matname, info) in all_mats.data.items():
        models[matname] = info.mat_class.param_names(79)
    return models

def open_file():
    """Open file"""
    dialog = FileDialog(action="open")
    if dialog.open() != pyOK:
        return []
    return dialog.paths
