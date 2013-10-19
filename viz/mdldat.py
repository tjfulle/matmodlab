from enthought.traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant, Enum, Event, Tuple, File, Dict, cached_property
from enthought.traits.ui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, TableEditor, InstanceEditor, ListEditor, Spring, ObjectColumn, EnumEditor, CheckListEditor
from enthought.traits.ui.tabular_adapter import TabularAdapter

import viz.vizutl as vu

import random


class TraitPositiveInteger(BaseInt):

    default_value = 1

    def validate(self, obj, name, value):
        value = super(TraitPositiveInteger, self).validate(object, name, value)
        if value > 0:
            return value
        self.error(obj, name, value)

    def info(self):
        return 'a positive integer'


class MMLModelParameter(HasStrictTraits):
    name = String
    description = String
    distribution = Enum('Specified', 'Percentage', 'Range', 'Uniform',
                        'Normal', 'AbsGaussian', 'Weibull')
    value = Property
    default = Property
    specified = String("0")
    percent = Float(10.0)
    minimum = Float(0.0)
    maximum = Float(1.0)
    mean = Property(Float)
    stdev = Float(1.0)
    scale = Float(1.0)
    shape = Float(1.0)
    samples = Int(10)

    def _get_value(self):
        ffmt = lambda r: "{0:12.6E}".format(r)
        default = ffmt(float(self.specified))
        if self.distribution == 'Specified':
            return default
        elif self.distribution == 'Percentage':
            return "{0} +/- {1}%".format(default, self.percent)
        elif self.distribution == 'Range':
            return "Range(start={0}, end={1}, N={2})".format(
                ffmt(self.minimum), ffmt(self.maximum), self.samples)
        elif self.distribution == 'Uniform':
            return "Uniform(min={0}, max={1}, N={2})".format(
                ffmt(self.minimum), ffmt(self.maximum), self.samples)
        elif self.distribution == 'Normal':
            return "Normal(mean={0}, std. dev={1}, N={2})".format(
                ffmt(self.mean), ffmt(self.stdev), self.samples)
        elif self.distribution == 'AbsGaussian':
            return "Abs. Gaussian(mean={0}, std. dev={1}, N={2})".format(
                ffmt(self.mean), ffmt(self.stdev), self.samples)
        elif self.distribution == 'Weibull':
            return "Weibull(scale={0}, shape={1}, N={2})".format(
                self.scale, self.shape, self.samples)
        return "#ERR"

    def _get_default(self):
        if self.distribution == 'Specified':
            return self.specified
        elif self.distribution == 'Percentage':
            return self.specified
        elif self.distribution == 'Range':
            return self.minimum
        elif self.distribution == 'Uniform':
            return self.minimum
        elif self.distribution == 'Normal':
            return self.mean
        elif self.distribution == 'AbsGaussian':
            return self.mean
        elif self.distribution == 'Weibull':
            return random.weibullvariate(self.scale, self.shape)
        return 0.0

    def _get_mean(self):
        return float(self.specified)

    def _set_mean(self, val):
        self.specified = str(val)

    edit_view = View(
        Item('name', label='Parameter', style='readonly'),
        Item('description', style='readonly'),
        Item('distribution'),
        Group(
            VGroup(
                Item('specified', label='Value'),
                visible_when="distribution == 'Specified'"
            ),
            VGroup(Item('specified', label='Value'),
                   Item('percent'), visible_when="distribution == 'percentage'"
            ),
            VGroup(Item('minimum', label='Start'), Item('maximum', label='End'),
                   Item('samples', label='Steps'),
                   visible_when="distribution == 'Range'"
            ),
            VGroup(Item('minimum', label='Min'), Item('maximum', label='Max'),
                   Item('samples'), visible_when="distribution == 'Uniform'"
            ),
            VGroup(Item('mean'), Item('stdev'), Item('samples'),
                   visible_when="distribution == 'Normal' or distribution == 'AbsGaussian'"
            ),
            VGroup(
                Item('scale'),
                Item('shape'),
                Item('samples'),
                visible_when="distribution == 'Weibull'"
            ),
            layout='tabbed'
        ),
        buttons=['OK', 'Cancel']
    )


class MMLMaterialParameter(HasStrictTraits):
    name = String
    default = String


class MMLMaterial(HasStrictTraits):
    name = String
    defaults = List(Instance(MMLMaterialParameter))


class MMLMaterialAdapter(TabularAdapter):
    columns = [('Materials Defined for Selected Model', 'name')]


class MMLEOSBoundary(HasStrictTraits):
    path_increments = TraitPositiveInteger(10000)
    surface_increments = TraitPositiveInteger(20)
    min_density = Float(8.0)
    max_density = Float(16.0)
    min_temperature = Float(200)
    max_temperature = Float(2000)
    isotherm = Bool(True)
    hugoniot = Bool(True)
    isentrope = Bool(False)
    auto_density = Bool(True)

    def _min_density_changed(self, info):
        self.auto_density = False

    def _max_density_changed(self, info):
        self.auto_density = False


class MMLLeg(HasStrictTraits):
    time = Float(0.0)
    nsteps = Int(0)
    types = String
    components = String

    view = View(
        HGroup(
            Item('time'),
            Item('nsteps'),
            Item('types'),
            Item('components')
        )
    )


class MMLModel(HasStrictTraits):
    name = String
    model_type = List(String)
    parameters = List(Instance(MMLModelParameter))
    selected_parameter = Instance(MMLModelParameter)
    materials = List(Instance(MMLMaterial))
    selected_material = Instance(MMLMaterial)
    eos_boundary = Instance(MMLEOSBoundary, MMLEOSBoundary())
    TSTAR = Float(1.0)
    FSTAR = Float(1.0)
    SSTAR = Float(1.0)
    DSTAR = Float(1.0)
    ESTAR = Float(1.0)
    EFSTAR = Float(1.0)
    AMPL = Float(1.0)
    RATFAC = Float(1.0)
    leg_defaults = String
    leg_default_types = List(String,
                             ['Custom', 'Uniaxial Strain', 'Biaxial Strain',
                              'Spherical Strain', 'Uniaxial Stress',
                              'Biaxial Stress', 'Spherical Stress'])
    legs = List(MMLLeg, [MMLLeg()])
    supplied_data = List(Tuple(String, File))
    supplied_data_names = Property(List(String), depends_on='supplied_data')
    leg_data_name = String
    leg_data_file = Property(File, depends_on='leg_data_name')
    leg_data_time = Dict(String, String,
                         {
                             'Time': 'time',
                             'Delta Time': 'dtime'
                         }
                         )
    leg_data_time_names = Property(String, depends_on='leg_data_time')
    selected_leg_data_time = String
    leg_data_type = Dict(String, Tuple(String, Int),
                         {'Strain Rate': ('strain rate', 6),
                          'Strain': ('strain', 6),
                          'Stress Rate': ('stress rate', 6),
                          'Stress': ('stress', 6),
                          'Deformation Gradient': ('deformation gradient', 9),
                          'Electric Field': ('electric field', 3),
                          'Displacement': ('displacement', 3),
                          'VStrain': ('vstrain', 1),
                          'Pressure': ('pressure', 1),})
    leg_data_type_names = Property(String, depends_on='leg_data_type')
    selected_leg_data_type = String
    leg_data_columns = List(String)
    selected_leg_data_columns = List(String)

    permutation_method = Enum('Zip', 'Combine')
    cell = Event

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)

        for param in self.parameters:
            if param.name == "R0" and param.value is not None:
                param.on_trait_change(self.update_density(param), 'value')

    def _supplied_data_changed(self, trait):
        if 'Data' not in self.leg_default_types:
            self.leg_default_types.append('Data')

    @cached_property
    def _get_supplied_data_names(self):
        return map(lambda x: x[0], self.supplied_data)

    @cached_property
    def _get_leg_data_file(self):
        name = self.leg_data_name
        for v in self.supplied_data:
            if v[0] == name:
                return v[1]
        return ''

    @cached_property
    def _get_leg_data_time_names(self):
        return self.leg_data_time.keys()

    @cached_property
    def _get_leg_data_type_names(self):
        return self.leg_data_type.keys()

    def _leg_data_name_changed(self, value):
        if len(self.leg_data_file) < 1:
            return ['']

        columns = vu.loadhead(self.leg_data_file)
        if columns is None:
            columns = ['']
        self.leg_data_columns = columns

    def _selected_material_changed(self, info):
        if info is None:
            return

        for default in info.defaults:
            for param in self.parameters:
                if param.name == default.name:
                    param.distribution = 'Specified'
                    default = float(default.default)
                    param.specified = "{0:12.6E}".format(default)
                    param.minimum = default - .1 * default
                    param.maximum = default + .1 * default
                    param.mean = default
                    break

    def update_density(self, param):
        def do_update():
            if self.eos_boundary.auto_density:
                try:
                    r0 = float(param.value)
                    self.eos_boundary.min_density = r0 * 0.9
                    self.eos_boundary.max_density = r0 * 1.1
                except:
                    pass
                self.eos_boundary.auto_density = True

        return do_update

    def _cell_fired(self, info):
        info[0].configure_traits()

    def _leg_defaults_changed(self, info):
        if info == 'Uniaxial Strain':
            self.legs = [
                MMLLeg(time=0, nsteps=0,
                       types='222222', components='0 0 0 0 0 0'),
                MMLLeg(time=1, nsteps=100,
                       types='222222', components='1 0 0 0 0 0'),
            ]
        elif info == 'Biaxial Strain':
            self.legs = [
                MMLLeg(time=0, nsteps=0,
                       types='222222', components='0 0 0 0 0 0'),
                MMLLeg(time=1, nsteps=100,
                       types='222222', components='1 1 0 0 0 0'),
            ]
        elif info == 'Spherical Strain':
            self.legs = [
                MMLLeg(time=0, nsteps=0,
                       types='222222', components='0 0 0 0 0 0'),
                MMLLeg(time=1, nsteps=100,
                       types='222222', components='1 1 1 0 0 0'),
            ]
        elif info == 'Uniaxial Stress':
            self.legs = [
                MMLLeg(time=0, nsteps=0,
                       types='444444', components='0 0 0 0 0 0'),
                MMLLeg(time=1, nsteps=100,
                       types='444444', components='1 0 0 0 0 0'),
            ]
        elif info == 'Biaxial Stress':
            self.legs = [
                MMLLeg(time=0, nsteps=0,
                           types='444444', components='0 0 0 0 0 0'),
                MMLLeg(time=1, nsteps=100,
                           types='444444', components='1 1 0 0 0 0'),
            ]
        elif info == 'Spherical Stress':
            self.legs = [
                MMLLeg(time=0, nsteps=0,
                           types='444444', components='0 0 0 0 0 0'),
                MMLLeg(time=1, nsteps=100,
                           types='444444', components='1 1 1 0 0 0'),
            ]

    def _legs_changed(self, info):
        for i in range(len(info)):
            if info[i] is None:
                info[i] = MMLLeg()

    param_view = View(
        UItem('parameters',
              editor=TableEditor(
              auto_size=False,
              reorderable=False,
              sortable=True,
              click='cell',
              columns=[
                    ObjectColumn(name='name', editable=False, width=0.3),
                    ObjectColumn(name='value', editable=False,
                                 width=0.7, horizontal_alignment='right')
              ],
              )
              ),
        Label('Permutation Method'),
        UItem('permutation_method', style='custom')
    )

    material_view = View(
        UItem('materials',
              editor=TabularEditor(
              show_titles=True,
              selected='selected_material',
              editable=False,
              adapter=MMLMaterialAdapter()),
              )
    )

    eos_boundary_view = View(
        UItem('eos_boundary'),
        style='custom',
    )

    solid_path_view = View(
        HGroup(
            Item('leg_defaults',
                 editor=EnumEditor(name='leg_default_types'),
                 style='simple'),
            VGroup(
                Item('TSTAR', label='TSTAR'),
                Item('FSTAR', label='FSTAR')
            ),
            VGroup(
                Item('SSTAR', label='SSTAR'),
                Item('DSTAR', label='DSTAR')
            ),
            VGroup(
                Item('ESTAR', label='ESTAR'),
                Item('EFSTAR', label='EFSTAR')
            ),
            VGroup(
                Item('AMPL', label='AMPL'),
                Item('RATFAC', label='RATFAC')
            ),
            style='simple'
        ),
        UItem('legs',
              editor=ListEditor(
              style='custom'
              ),
              visible_when='leg_defaults != "Data"',
              ),
        VGroup(
            Item('leg_data_name',
                 editor=EnumEditor(name='supplied_data_names'),
                 label='Data to use',
                 style='simple',
                 ),
            HGroup(
                Item('selected_leg_data_time',
                     editor=EnumEditor(name='leg_data_time_names'),
                     label='Time',
                     style='simple'
                     ),
                Item('selected_leg_data_type',
                     editor=EnumEditor(name='leg_data_type_names'),
                     label='Deformation Type',
                     style='simple'
                     ),
            ),
            Item('selected_leg_data_columns',
                 editor=CheckListEditor(name='leg_data_columns', cols=10),
                 label='Columns to use',
                 style='custom',
                 ),
            visible_when='leg_defaults == "Data"',
        ),
        style='custom',
    )
