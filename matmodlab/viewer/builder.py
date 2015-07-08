import os
import re
import sys
import datetime
import numpy as np
from os.path import basename, isfile, join, splitext, dirname, realpath
from traits.api import *
from traitsui.api import *
from traitsui.message import error, message
from traitsui.menu import Action
from pyface.api import FileDialog, OK as pyOK
from pyface.gui import GUI

from matmodlab.mml_siteenv import environ
from matmodlab.mmd.mdb import mdb, ModelCaptured as ModelCaptured

__all__ = ['Model', 'Step', 'Material', 'PermModel', 'OptModel', 'import_models']

from _builder import *

class Step(HasTraits):
    name = Str
    kind = Str
    keywords = Dict
    editable = Bool(True)
    def asstring(self):
        if not self.kind:
            return ''
        kwds = ', '.join('{0}={1}'.format(k, repr(v))
                         for (k, v) in self.keywords.items())
        return '{0}(name={1}, {2})'.format(self.kind, repr(self.name), kwds)

class Steps(HasTraits):
    steps = List(Step)
    nsteps = Int
    editor = Instance(StepEditor)
    locked = Bool(False)
    def __init__(self):
        kwds = {'editor': StepEditor()}
        super(Steps, self).__init__(**kwds)

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def create_step(self, name=None):
        if name is None:
            name = 'Step-{0}'.format(len(self.steps)+1)
        self.editor.create_step(name)

    def edit_step(self, name):
        for (i, step) in enumerate(self.steps):
            if step.name == name:
                break
        else:
            message('***warning: attempting to edit nonexistent step')
            return
        self.editor.edit_step(step.name, step.kind, step.keywords, i, step.editable)

    def append(self, step):
        self.steps.append(step)
        self.nsteps = len(self.steps)

    @on_trait_change('editor.step_defn')
    def update_on_close(self):
        if not self.editor.done:
            return
        if not self.editor.editable:
            self.editor.editable = True
            return
        name = self.editor.step_defn['name']
        kind = self.editor.step_defn['kind']
        keywords = self.editor.step_defn['kwds']
        step = Step(name=name, kind=kind, keywords=keywords)
        index = self.editor.step_defn.get('update')
        if index is not None:
            self.steps[index] = step
        else:
            self.append(step)

class Material(HasTraits):
    name = Str
    label = Str
    parameters = Dict(Str, Float)
    keywords = Dict
    editor = Instance(MaterialModelEditor)

    def __init__(self):
        kwds = {}
        kwds['name'] = 'empty'
        kwds['label'] = 'Material (empty)'
        kwds['parameters'] = {}
        kwds['editor'] = MaterialModelEditor()
        super(Material, self).__init__(**kwds)

    def asstring(self):
        if self.name == 'empty':
            return ''
        kwds = ', '.join('{0}={1}'.format(k, repr(v))
                         for (k, v) in self.keywords.items())
        return 'Material({0}, {1}, {2})'.format(
            repr(self.name), self.parameters, kwds)

    def _name_changed(self):
        self.label = 'Material ({0})'.format(self.name)

    def edit(self):
        args = []
        if self.name != 'empty':
            args.append(self.name)
            args.append(self.parameters)
        self.editor.edit(*args)

    @on_trait_change('editor.matinfo')
    def update_self(self):
        name, parameters, keywords = self.editor.matinfo
        self.update(name, parameters, keywords)

    def update(self, name, parameters, keywords):
        self.name = name
        self.parameters = parameters
        self.keywords = keywords
        self.label = 'Material ({0})'.format(name)

class Model(HasTraits):
    name = Str
    steps = Instance(Steps)
    material = Instance(Material)
    initial_temperature = Float(DEFAULT_TEMP)
    directory = Str(os.getcwd())
    matlabel = Str
    nsteps = Int
    def __init__(self, name):
        kwds = {'name': name, 'material': Material(), 'steps': Steps()}
        kwds['matlabel'] = kwds['material'].label
        super(Model, self).__init__(**kwds)

    edit_view = View(
        Item('name'), '5', '_', '5',
        VGroup(Item('directory',
                    label='Simulation Diretory',
                    visible_when='not len(model.steps)'),
               Item('initial_temperature',
                    label='Initial Temperature',
                    enabled_when='nsteps <= 1'),
               show_border=True, label='Model Options'), style='simple',
        title='New Model', buttons=['OK', 'Cancel'], resizable=True)

    def edit(self):
        self.configure_traits(view='edit_view')

    def add_step(self, name):
        self.steps.create_step(name)
        self.nsteps = len(self.steps)

    def edit_step(self, name):
        self.steps.edit_step(name)
        self.nsteps = len(self.steps)

    def asstring(self):
        today = datetime.date.today().strftime('%a %b %d %Y')
        a = ['# Model generated by the Material Model Laboratory on {4}',
             'from matmodlab import *',
             'model = MaterialPointSimulator({0}, ',
             '    d={1},',
             '    initial_temperature={2}']
        string = '\n'.join(a).format(repr(self.name), repr(self.directory),
                                     self.initial_temperature, today)

        # Add the material
        material = self.material.asstring()
        if material:
            material = 'model.{1}'.format(repr(self.name), material)
            string += '\n{0}'.format(material)

        # Add steps
        steps = []
        for step in self.steps:
            sstr = step.asstring()
            if not sstr:
                continue
            steps.append('model.{1}'.format(repr(self.name), sstr))
        if steps:
            string += '\n{0}'.format('\n'.join(steps))

        return string

    @on_trait_change('material.label')
    def update_matlabel(self):
        self.matlabel = self.material.label

    def display_input(self):
        viewer = ModelInputPreview(input_string=self.asstring())
        viewer.edit_traits(view='traits_view')

    def write_input(self):
        string = self.asstring()
        string += '\nmodel.run()'
        with open(self.name + '.mml', 'w') as fh:
            fh.write(string)

    def run(self):
        '''Create the model and execute it'''
        cwd = os.getcwd()
        os.chdir(self.directory)
        e = None
        try:
            GUI.set_busy(busy=True)
            code = compile(self.asstring(), '<string>', 'exec')
            exec code in globals()
            model.run()
        except BaseException as e:
            string = 'Failed to run simulation with the following error: {0}'
            string = string.format(' '.join('{0}'.format(x) for x in e.args))
            error(string)
        finally:
            os.chdir(cwd)
            GUI.set_busy(busy=False)

        if e is not None:
            return

        return model.exodus_file

class ModelInputPreview(HasStrictTraits):
    input_string = String
    traits_view = View(
        VGroup(Item('input_string', style='custom', show_label=False)),
        width=800,
        height=600,
        buttons=['OK'], resizable=True)

class PermModel(HasTraits):
    name = String
    filename = String
    contents = String
    view_view = View(
        VGroup(Item('contents', style='custom', show_label=False)),
        buttons=['OK'],
        width=800,
        height=600,
        resizable=True)
    def __init__(self, filename, job):
        kwds = {'filename': filename,
                'name': job,
                'contents': open(filename).read()}
        super(PermModel, self).__init__(**kwds)

    def display_input(self):
        self.edit_traits(view='view_view')

    def run(self):
        '''Create the model and execute it'''
        cwd = os.getcwd()
        os.chdir(self.directory)
        e = None
        try:
            GUI.set_busy(busy=True)
            gdict = globals()
            gdict['__name__'] = '__main__'
            code = compile(self.contents, '<string>', 'exec')
            exec(code) in gdict
        except BaseException as e:
            string = 'Failed to run simulation with the following error: {0}'
            string = string.format(' '.join('{0}'.format(x) for x in e.args))
            error(string)
        finally:
            GUI.set_busy(busy=False)
            os.chdir(cwd)

        if e is not None:
            return

        # determine the permutator object
        job = mdb.get_permutator()
        if job is None:
            return
        return job.output

class OptModel(HasTraits):
    name = String
    filename = String
    contents = String
    view_view = View(
        VGroup(Item('contents', style='custom', show_label=False)),
        buttons=['OK'],
        width=800,
        height=600,
        resizable=True)
    def __init__(self, filename, job):
        kwds = {'filename': filename,
                'name': job,
                'contents': open(filename).read()}
        super(OptModel, self).__init__(**kwds)

    def display_input(self):
        self.edit_traits(view='view_view')

    def run(self):
        '''Create the model and execute it'''
        cwd = os.getcwd()
        os.chdir(self.directory)
        e = None
        try:
            GUI.set_busy(busy=True)
            gdict = globals()
            gdict['__name__'] = '__main__'
            code = compile(self.contents, '<string>', 'exec')
            exec(code) in gdict
        except BaseException as e:
            string = 'Failed to run simulation with the following error: {0}'
            string = string.format(' '.join('{0}'.format(x) for x in e.args))
            error(string)
        finally:
            GUI.set_busy(busy=False)
            os.chdir(cwd)

        if e is not None:
            return

        # determine the optimzer object
        job = mdb.get_optimizer()
        if job is None:
            return
        self.output = job.output
        name = splitext(job.job)[0]
        self.baseline = None
        d = dirname(job.rootd)
        for ext in ('.base_exo', '.base_dat', '.xls', '.xlsx'):
            f = join(d, name + ext)
            if isfile(f):
                self.baseline = f
                break
        return self.output

def import_models(filename):

    # import the file
    mdb.stash()
    environ.capture_model = 1

    d = dirname(realpath(filename))
    if d not in sys.path:
        sys.path.insert(0, d)

    cwd = os.getcwd()
    os.chdir(d)
    try:
        gdict = globals()
        gdict['__name__'] = '__main__'
        code = compile(open(filename, 'r').read(), '<string>', 'exec')
        exec(code) in gdict
    except ModelCaptured:
        pass
    finally:
        os.chdir(cwd)

    models = []

    # lets get the models from mdb
    for item in mdb.optimizers:
        model = OptModel(filename, item.job)
        model.directory = item.directory
        model.output = item.output
        models.append(model)

    for item in mdb.permutators:
        model = PermModel(filename, item.job)
        model.directory = item.directory
        model.output = item.output
        models.append(model)

    for item in mdb.models:
        # The model
        model = Model(item.job)
        model.directory = item.directory
        model.initial_temperature = item.initial_temperature

        # Its material
        mat = item.current_mat
        parameters = dict(zip(mat.parameter_names, mat.initial_parameters))
        model.material.update(mat.name, parameters, {})

        # And steps
        for step in item.steps.values():
            if step.name == 'Initial':
                continue
            if step.kind in ('StressStep', 'StressRateStep', 'MixedStep'):
                step.keywords.pop('kappa', None)
            mystep = Step(name=step.name, kind=step.kind,
                          keywords=step.keywords, editable=False)
            model.steps.append(mystep)

        models.append(model)

    environ.capture_model = 0
    mdb.pop_stash()

    return models

if __name__ == '__main__':
    #demo = MaterialModelEditor()
    #demo.configure_traits()
    #the_model.configure_traits(view='edit_view')
    #demo = StepEditor('Step-1')
    #demo.configure_traits()

    the_model = Model(name='The Model')
    the_model.add_step('Step-1')
