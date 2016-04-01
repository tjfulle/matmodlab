import sys
import numpy as np
from os.path import basename, realpath, split, dirname, sep

from traits.api import *
from traitsui.api import *
from traitsui.message import error, message
from pyface.api import FileDialog, OK as pyOK
from traitsui.menu import Menu, Action, Separator

from tabfileio import read_file

def loadfile(filename):
    """Load the data file"""
    if filename.endswith(('.csv', '.rpk', '.out')):
        # Matmodlab files
        try:
            from matmodlab.utils.fileio import loadfile as lf
        except ImportError:
            raise ValueError('file type requires matmodlab package')
        return lf(filename, disp=1)
    else:
        return read_file(filename, disp=1)


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
        if name not in self.vmap:
            return
        legend = name
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
    try:
        from matmodlab.utils.mmltab import read_mml_evaldb
    except ImportError:
        raise ValueError('file type requires matmodlab package')
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
    outputdbs   = List(OutputDB)
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
        if filename.endswith('.edb'):
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
                  children  = 'outputdbs',
                  label     = '=Output Databases',
                  view      = no_view,
                  icon_group='icon/graph.png',
                  icon_open ='icon/graph.png',
                  menu      = Menu(open_outputdb_action,
                                   reload_all_outputdb_action,
                                   show_all_outputdb_action)),
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
