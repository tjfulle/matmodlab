# tree_editor.py -- Example of a tree editor
from traits.api \
    import HasTraits, Str, Regex, List, Instance
from traitsui.api \
    import TreeEditor, TreeNode, View, Item, VSplit, \
           HGroup, Handler, Group
from traitsui.menu \
    import Menu, Action, Separator

# DATA CLASSES

class Scalar(HasTraits):
    key  = Str
    name  = Str

class Component(HasTraits):
    label = Str

class Tensor(HasTraits):
    key  = Str
    name  = Str
    components = List(Component)
    def __init__(self, name, key=None):
        key = key or name
        labels = ('XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ')
        components = [Component(label='{0}.{1}'.format(key, l)) for l in labels]
        super(Tensor, self).__init__(name=name, key=key, components=components)

class FieldOutputs(HasTraits):
    name = 'Field Outputs'
    fields = List

class Container(HasTraits):
    name = 'Data Container'
    fields = Instance(FieldOutputs)

# View for objects that aren't edited
no_view = View()

class TreeHandler(Handler):
    pass

# Tree editor
tree_editor = TreeEditor(
    nodes = [
        TreeNode(node_for=[FieldOutputs],
                 label='=Data Container',
                 auto_open=False,
                 children='',
                 view=no_view),
        TreeNode(node_for=[FieldOutputs],
                 label='=Field Outputs',
                 auto_open=True,
                 icon_group='icon/field.png',
                 icon_open='icon/field.png',
                 children='fields',
                 view=no_view),
        TreeNode(node_for=[Scalar],
                  children  = '',
                  icon_item = '',
                  auto_open = False,
                  label     = 'key'),
        TreeNode(node_for=[Tensor],
                 icon_group = '',
                 icon_open ='',
                 children  = 'components',
                 auto_open = False,
                 label     = 'key'),
        TreeNode(node_for=[Component],
                  icon_item = '',
                  children  = '',
                  auto_open = False,
                  label     = 'label'),
    ], editable=False, hide_root=True
)

# The main view
view = View(
           Group(
               Item(
                    name = 'fields',
                    id = 'fields',
                    editor = tree_editor,
                    resizable = True ),
                orientation = 'vertical',
                show_labels = False,
                show_left = True, ),
            title = 'Field Outputs',
            dock = 'horizontal',
            drop_class = HasTraits,
            handler = TreeHandler(),
            buttons=['Undo', 'OK', 'Cancel'],
            resizable=True,
            width = .3,
            height = .7 )

if __name__ == '__main__':

    # INSTANCES
    t = Scalar(key='t', name='Time')
    dt = Scalar(key='dt', name='dTime')
    T = Scalar(key='T', name='Temperature')
    S = Tensor(key='S', name='Stress')
    E = Tensor(key='E', name='Strain')

    container = Container(
        fields = FieldOutputs(
            fields=[t, dt, T, S, E]))

    container.configure_traits( view = view )
