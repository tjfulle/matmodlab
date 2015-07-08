import sys
import numpy as np
from os.path import basename, realpath, split, dirname, sep

from traits.api import *
from traitsui.api import *
from traitsui.message import error, message
from pyface.api import FileDialog, OK as pyOK
from traitsui.menu import Menu, Action, Separator

from builder import *
from matmodlab.utils.fileio import loadfile
from matmodlab.utils.mmltab import read_mml_evaldb, is_evaldb

class OutputDB(HasTraits):
    id = Int
    name  = Str
    filepath = Str
    names = List(Str)
    data = Array
    vmap = Dict(Str, Int)
    info = Str
    hidden = Bool
    def __init__(self, filename, info=None, name=None):
        if name is None:
            name = basename(filename)
        filepath = realpath(filename)
        names, data = loadfile(filepath)
        vmap = dict([(s.upper(), i) for (i, s) in enumerate(names)])
        kwds = {'name': name, "filepath": filepath,
                'info': info or '', 'names': names, 'data': data,
                'vmap': vmap, 'id': hashf(filename), 'hidden': False}

        super(OutputDB, self).__init__(**kwds)

    def get(self, name, time=None):
        j = self.vmap.get(name.upper())
        if j is None:
            return
        data = self.data[:, j]
        if time is None:
            return data
        i = np.argmin(np.abs(time - self.data[:, self.vmap["TIME"]]))
        return data[i]

    def legend(self, name):
        if name.upper() not in self.vmap:
            return
        legend = name.upper()
        if self.info:
            legend += " ({0})".format(self.info)
        return legend

    @property
    def choices(self):
        return sorted(self.vmap.keys(), key=lambda k: self.vmap[k])

    def reload_data(self):
        _, data = loadfile(self.filepath)
        self.data[:] = data
        return

def hashf(filename):
    return hash(realpath(filename))

def readtabular(source):
    """Read in the mml-tabular.dat file

    """
    sources, paraminfo, _ = read_mml_evaldb(source)
    for (key, info) in paraminfo.items():
        paraminfo[key] = ", ".join("{0}={1:.2g}".format(n, v) for (n, v) in info)
    return sources, paraminfo

class MyTreeNode(TreeNode):
    def get_icon(self, node, state):
        if node.hidden:
            return 'icon/hidden.png'
        return 'icon/file.png'

class DNDTreeNode(TreeNode):
    def on_drop(self, *args):
        raise SystemExit()
        if node.hidden:
            return 'icon/hidden.png'
        return 'icon/file.png'

# DATA CLASSES
class Root(HasTraits):
    name        = Str('Root')
    models      = List
    outputdbs   = List(OutputDB)
    steps       = List(Step)
    choices     = List(Str)
    def __init__(self, files=None, models=None, owner=None):

        kwds = {'owner': owner}

        if models is not None:
            kwds['models'] = models

        if files is not None:
            pks = []
            dbs = []
            for file in files:
                dbs.append(file)
                pks.extend([k for k in dbs[-1].choices if k not in pks])
            kwds['choices'] = pks
            kwds['outputdbs'] = dbs

        super(Root, self).__init__(**kwds)

    def open_outputdb(self):
        files = open_outputdbs()
        if not files:
            return
        for file in files:
            self.add_outputdb(file)
        self.refresh()

    def refresh(self):
        pass

    def add_outputdb(self, filename):
        if hashf(filename) in [f.id for f in self.outputdbs]:
            self.reload_outputdb(filename)
            return
        if is_evaldb(filename):
            filepaths, variables = readtabular(filename)
            d = dirname(filename) + sep
            names = dict([(f, f.replace(d, '')) for f in filepaths])
            files = [OutputDB(f, info=variables[f], name=names[f])
                     for f in filepaths]
            self.outputdbs.extend(files)
            f = files[0]
        else:
            f = OutputDB(filename)
            self.outputdbs.append(f)
        self.choices.extend([c for c in f.choices if c not in self.choices])
        if self.owner:
            self.owner.onadd()

    def remove_outputdb(self, filename):
        fid = hashf(filename)
        for (i, file) in enumerate(self.outputdbs):
            if fid == file.id:
                break
        else:
            return
        self.outputdbs.pop(i)
        if self.outputdbs:
            self.choices = [c for c in self.outputdbs[0].choices]
            for file in self.outputdbs[1:]:
                self.choices.extend([c for c in self.outputdbs[0].choices
                                     if c not in self.choices])
        if self.owner:
            self.owner.onremove()

    def _index(self, filename):
        fid = hashf(filename)
        for (i, file) in enumerate(self.outputdbs):
            if  fid == file.id:
                return i

    def reload_outputdb(self, filename):
        i = self._index(filename)
        if i is None:
            return
        file = self.outputdbs[i]
        file.reload_data()
        if self.owner:
            self.owner.onreload()
        return

    def unhide_outputdb(self, filename):
        i = self._index(filename)
        if i is None:
            return
        file = self.outputdbs[i]
        file.hidden = False
        if self.owner:
            self.owner.update()
        return

    def hide_outputdb(self, filename):
        i = self._index(filename)
        if i is not None:
            self.outputdbs[i].hidden = True
        if self.owner:
            self.owner.update()

    def show_outputdb(self, filename):
        i = self._index(filename)
        if i is not None:
            self.outputdbs[i].hidden = False
        if self.owner:
            self.owner.update()

    def hide_others(self, filename):
        j = self._index(filename)
        for (i, file) in enumerate(self.outputdbs):
            if j == i:
                self.outputdbs[i].hidden = False
            else:
                self.outputdbs[i].hidden = True
        if self.owner:
            self.owner.update()
        return

    def is_hidden(self, filename):
        i = self._index(filename)
        return self.outputdbs[i].hidden

# View for objects that aren't edited
no_view = View()

class TreeHandler(Handler):

    def refresh(self, editor, object):
        editor.update_editor()

    # Model actions
    def new_step(self, editor, object):
        names = [x.name for x in object.steps]
        j = len(names) or 1
        while 1:
            name = "Step-{0}".format(j)
            if name not in names:
                break
            j += 1
        object.add_step(name)
        editor.update_editor()

    def edit_step(self, editor, object):
        parent = editor.get_parent(object)
        parent.edit_step(object.name)

    def new_model(self, editor, object):
        names = [x.name for x in object.models]
        j = len(names)
        while 1:
            name = "Model-{0}".format(j+1)
            if name not in names:
                break
            j += 1
        model = Model(name=name)
        model.edit()
        object.models.append(model)

    def import_models(self, editor, object):
        wildcard = ('Python files (*.py)|*.py|'
                    'All files (*)|*')
        dialog = FileDialog(action="open files", wildcard=wildcard)
        if dialog.open() != pyOK:
            return
        for file in dialog.paths:
            try:
                models = import_models(file)
            except BaseException as e:
                string = 'Failed to import model with the following error: {0}'
                string = string.format(e.args[0])
                error(string)
                continue
            object.models.extend(models)

    def remove_model(self, editor, object):
        parent = editor.get_parent(object)
        for (i, model) in enumerate(parent.models):
            if model.name == object.name:
                break
        else:
            return
        parent.models.pop(i)

    def edit_model_attributes(self, editor, object):
        object.edit()

    def run_model(self, editor, object):
        output = object.run()
        parent = editor.get_parent(object)
        if output is None:
            return
        try:
            parent.add_outputdb(object.baseline)
        except AttributeError:
            pass
        parent.add_outputdb(output)
        editor.update_editor()

    def display_input(self, editor, object):
        object.display_input()

    def write_input(self, editor, object):
        object.write_input()

    def edit_material(self, editor, object):
        object.material.edit()

    # Output database actions
    def open_outputdb(self, editor, object):
        """Open file"""
        object.open_outputdb()
        editor.update_editor()

    def reload_all_outputdbs(self, editor, object):
        for file in object.outputdbs:
            object.reload_outputdb(file.filepath)

    def show_all_outputdbs(self, editor, object):
        for file in object.outputdbs:
            object.unhide_outputdb(file.filepath)
        editor.update_editor()

    def remove_outputdb(self, editor, object):
        """Close outputdbs"""
        # Remove the file from the list
        container = editor.get_parent(object)
        container.remove_outputdb(object.filepath)

    def reload_outputdb(self, editor, object):
        container = editor.get_parent(object)
        container.reload_outputdb(object.filepath)
        return

    def hide_outputdb(self, editor, object):
        container = editor.get_parent(object)
        container.hide_outputdb(object.filepath)
        editor.update_editor()
        return

    def show_outputdb(self, editor, object):
        container = editor.get_parent(object)
        container.show_outputdb(object.filepath)
        editor.update_editor()
        return

    def hide_others(self, editor, object):
        container = editor.get_parent(object)
        container.hide_others(object.filepath)
        editor.update_editor()
        return

    def is_hidden(self, editor, object):
        parent = editor.get_parent(object)
        return parent.is_hidden(object.filepath)

# Actions used by tree editor context menu
open_outputdb_action = Action(
    name='Open output database',
    action='handler.open_outputdb(editor,object)')
reload_all_outputdb_action = Action(
    name='Reload All',
    action='handler.reload_all_outputdbs(editor,object)',
    enabled_when='len(object.outputdbs)')
show_all_outputdb_action = Action(
    name='Show All',
    action='handler.show_all_outputdbs(editor,object)',
    enabled_when='len(object.outputdbs)')
reload_outputdb_action = Action(
    name='Reload',
    action='handler.reload_outputdb(editor,object)')
delete_outputdb_action = Action(
    name='Remove',
    action='handler.remove_outputdb(editor,object)')
hide_outputdb_action = Action(
    name='Hide',
    action='handler.hide_outputdb(editor,object)',
    enabled_when='not handler.is_hidden(editor,object)')
hide_others_action = Action(
    name='Hide Others',
    action='handler.hide_others(editor,object)',
    enabled_when='not handler.is_hidden(editor,object)')
show_outputdb_action = Action(
    name='Show',
    action='handler.show_outputdb(editor,object)',
    enabled_when='handler.is_hidden(editor,object)')
new_step_action = Action(
    name='New Step',
    enabled_when='not object.steps.locked',
    action='handler.new_step(editor,object)')
new_model_action = Action(
    name='New Model Wizard',
    action='handler.new_model(editor,object)')
import_models_action = Action(
    name='Import Models',
    action='handler.import_models(editor,object)')
remove_model_action = Action(
    name='Remove',
    action='handler.remove_model(editor,object)')
edit_model_action = Action(
    name='Edit Attributes',
    action='handler.edit_model_attributes(editor,object)')
run_model_action = Action(
    name='Run',
    action='handler.run_model(editor,object)')
display_input_action = Action(
    name='Display Input',
    action='handler.display_input(editor,object)')
write_input_action = Action(
    name='Write Input',
    action='handler.write_input(editor,object)')
edit_material_action = Action(
    name='Edit Properties',
    action='handler.edit_material(editor,object)')
edit_step_action = Action(
    name='Edit',
    action='handler.edit_step(editor,object)')
refresh_action = Action(
    name='Refresh Tree',
    action='handler.refresh(editor,object)')
rename_action = Action(name='Rename', action='editor._menu_rename_node',
                       enabled_when='editor._is_renameable(object)')
delete_action = Action(name = 'Delete', action='editor._menu_delete_node',
                       enabled_when='editor._is_deletable(object)')

# Tree editor
tree_editor = TreeEditor(
    nodes = [
        TreeNode(node_for  = [Root],
                  children  = '',
                  label     = 'name',
                  view      = View(Group('name',
                                   orientation='vertical',
                                   show_left=True))),
        TreeNode(node_for  = [Root],
                  children  = 'models',
                  auto_open = True,
                  label     = '=Models',
                  icon_group='icon/folder.png',
                  icon_open ='icon/folder.png',
                  view      = no_view,
                  menu      = Menu(
                                   refresh_action,
                                   Separator(),
                                   import_models_action,
                                   Separator(),
                                   new_model_action,
                                   )),
        TreeNode(node_for  = [Root],
                  children  = 'outputdbs',
                  label     = '=Output Databases',
                  view      = no_view,
                  icon_group='icon/graph.png',
                  icon_open ='icon/graph.png',
                  menu      = Menu(open_outputdb_action,
                                   reload_all_outputdb_action,
                                   show_all_outputdb_action)),
        TreeNode(node_for   = [Model],
                  children  = '',
                  label     = 'name',
                  icon_item ='icon/box.png',
                  menu      = Menu(run_model_action,
                                   remove_model_action,
                                   Separator(),
                                   edit_model_action,
                                   write_input_action,
                                   display_input_action)),
        TreeNode(node_for  = [Model],
                  auto_open = True,
                  children  = '',
                  menu      = Menu(edit_material_action),
                  label     = 'matlabel',
                  icon_item ='icon/atom.png'),
        TreeNode(node_for  = [Model],
                  auto_open = True,
                  children  = 'steps',
                  label     = '=Steps',
                  icon_group='icon/step.png',
                  icon_open ='icon/step.png',
                  menu      = Menu(new_step_action)),
        TreeNode(node_for   = [Step],
                 auto_open = True,
                 label     = 'name',
                 icon_item ='',
                 view      = no_view,
                 menu      = Menu(edit_step_action)),
        TreeNode(node_for  = [PermModel],
                 label     = 'name',
                 icon_item='icon/permutate.png',
                 view      = no_view,
                 menu      = Menu(run_model_action,
                                  display_input_action)),
        TreeNode(node_for  = [OptModel],
                 label     = 'name',
                 icon_item='icon/opt.png',
                 view      = no_view,
                 menu      = Menu(run_model_action,
                                  display_input_action)),
        MyTreeNode(node_for   = [OutputDB],
                  label     = 'name',
                  menu      = Menu(hide_outputdb_action,
                                   hide_others_action,
                                   show_outputdb_action,
                                   reload_outputdb_action,
                                   delete_outputdb_action),
                 view = no_view),
    ], editable=False, hide_root=True, auto_open=2
)

class InfoPane(HasTraits):
    name    = Str('Information')
    root = Instance(Root)
    editor = tree_editor

    def __init__(self, files=None, models=None, owner=None):
        kwds = {'root': Root(files=files, models=models, owner=owner)}
        super(InfoPane, self).__init__(**kwds)

    # The main view
    view = View(
           Group(
               Item(
                    name = 'root',
                    id = 'root',
                    editor = tree_editor,
                    resizable = True),
                orientation = 'vertical',
                show_labels = False,
                show_left = True,),
            dock = 'horizontal',
            drop_class = HasTraits,
            handler = TreeHandler(),
            buttons = ['Undo', 'OK', 'Cancel'],
            resizable = True)

    def __getattr__(self, attr):
        # return this objects attribute first. if it doesn't exist, return the
        # root items
        try:
            HasTraits.__getattr__(self, attr)
        except AttributeError:
            return getattr(self.root, attr)

def open_outputdbs():
    """Open file"""
    wildcard = ('DBX files (*.dbx)|*.dbx|'
                'Exodus files (*.exo *.e *.base_exo *.exo.*)|'
                              '*.exo *.e *.base_exo *.exo.*|'
                'Data files (*.out *.dat *.csv *.base_out *.base_dat)|'
                            '*.out *.dat *.csv *.base_out *.base_dat|'
                'All files (*)|*')
    dialog = FileDialog(action="open files", wildcard=wildcard)
    if dialog.open() != pyOK:
        return []
    return dialog.paths

if __name__ == '__main__':

    import glob
    from os.path import join, realpath, dirname
    d = join(dirname(realpath(__file__)), '../../inputs')
    files = glob.glob(join(d, "*.dbx"))

    # INSTANCES
    step_1 = Step(name='Step-1')
    step_2 = Step(name='Step-2')
    step_3 = Step(name='Step-3')
    step_4 = Step(name='Step-4')

    model_1 = Model('Model-1', steps=[step_1, step_2])
    model_2 = Model('Model-2', steps=[step_1, step_2, step_3])
    ipane = InfoPane(models=[model_1, model_2], files=files)
    ipane = InfoPane() #files=files)
    ipane.configure_traits()
