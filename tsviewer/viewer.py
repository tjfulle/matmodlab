import os
import math
import random
import numpy as np
from os.path import basename

# chaco and traits imports are in the enthought directory in EPD/Canopy
try: from enthought import chaco, traits
except ImportError: pass

from chaco.api import *
from traits.api import *
from traitsui.api import *
from traitsui.message import error
from chaco.tools.api import *
from enable.api import ComponentEditor
from pyface.api import FileDialog, OK as pyOK
from chaco.example_support import COLOR_PALETTE
from traitsui.tabular_adapter import TabularAdapter

from .infopane import InfoPane
icns = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'icon')

def Ones(n):
    return [1. for _ in range(n)]

__all__ = ["MMLPostViewer", "SingleSelect", 'MultiSelect']

LS = ['solid', 'dot dash', 'dash', 'long dash'] # , 'dot']
LDICT = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
         "asin": math.asin, "acos": math.acos,
         "atan": math.atan, "atan2": math.atan2, "pi": math.pi,
         "log": math.log, "exp": math.exp, "floor": math.floor,
         "ceil": math.ceil, "abs": math.fabs, "random": random.random, }
GDICT = {"__builtins__": None}
EPSILON = np.finfo(np.float).eps

def get_color(i=0, rand=False, _i=[0], reset=False):
    c = ["Blue", "Red", "Purple", "Green", "Orange", "HotPink", "Cyan",
         "Magenta", "Chocolate", "Yellow", "Black", "DodgerBlue", "DarkRed",
         "DarkViolet", "DarkGreen", "OrangeRed", "Teal", "DarkSlateGray",
         "RoyalBlue", "Crimson", "SeaGreen", "Plum", "DarkGoldenRod",
         "MidnightBlue", "DarkOliveGreen", "DarkMagenta", "DarkOrchid",
         "DarkTurquoise", "Lime", "Turquoise", "DarkCyan", "Maroon"]
    if reset:
        _i[0] = 0
        return
    if rand:
        color = c[random.randint(0, len(c)-1)]
    else:
        color = c[_i[0] % (len(c) - 1)]
        _i[0] += 1
    return color.lower()

class XYScales(HasTraits):
    xscale = String("1.0")
    yscale = String("1.0")
    _apply = Action(name='Apply', action='_apply')

    def __init__(self, **kwds):
        super(XYScales, self).__init__(**kwds)

    class MyHandler(Handler):

        def _apply(self, info):
            xscale = info.object.xscale.strip()
            yscale = info.object.yscale.strip()
            info.object.set_scales(xscale, yscale)

        def closed(self, info, is_ok):
            if not is_ok:
                return
            xscale = info.object.xscale.strip()
            yscale = info.object.yscale.strip()
            info.object.set_scales(xscale, yscale)

    edit_view = View(
            HGroup(VGroup(
            Item("xscale", label="X Scale", editor=TextEditor(multi_line=False)),
            Item("yscale", label="Y Scale", editor=TextEditor(multi_line=False))),
            spring),
            title='XY Scales',
            buttons=['OK', 'Cancel', _apply],
            handler=MyHandler())

    def set_scales(self, xscale, yscale):
        """Detect if the x-axis scale was changed and let the plotter know

        """
        if not self.plot:
            return
        try:
            xs = self.plot.format_scale('x', xscale)
            self.xscale = xscale
            self.plot.set_xscale(xs)
        except:
            self.xscale = '1.0'
            error('Invalid X scale request')
        try:
            ys = self.plot.format_scale('y', yscale)
            self.yscale = yscale
            self.plot.set_yscale(ys)
        except:
            self.yscale = '1.0'
            error('Invalid Y scale request')

        self.plot.change_plot()
        return

class MMLPostViewer(HasTraits):
    container = Instance(Plot)
    ipane = Instance(InfoPane)
    choices = List(Str)
    indices = List(Int)
    x_idx = Int
    y_idx = Int
    Time = Float
    high_time = Float
    low_time = Float
    times = List(Float)
    frame = Int(0)
    frames = List(Int)
    time_data_labels = Dict(Tuple, List)
    xscale = List(Float)
    yscale = List(Float)
    _line_style_index = Int
    rand = Bool
    legend_visible = Bool
    xyscales = Instance(XYScales)

    traits_view = View(
        Item('container', editor=ComponentEditor(), show_label=False),
        resizable=True)

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.time_data_labels = {}
        self.x_idx = 0
        self.y_idx = 0
        self._refresh = 1
        self.choices = self.ipane.choices
        self.xyscales = XYScales(plot=self)
        self.xscale = Ones(len(self.choices))
        self.yscale = Ones(len(self.choices))
        self.rand = True
        self.legend_visible = True
        pass

    def _choices_changed(self):
        if self.choices:
             return
        self.time_data_labels = {}
        self.x_idx = 0
        self.y_idx = 0
        self._refresh = 1
        self.xscale = Ones(len(self.choices))
        self.yscale = Ones(len(self.choices))
        self.rand = True
        self.legend_visible = True
        try:
             self.container.invalidate_and_redraw()
        except AttributeError:
             pass

    def format_scale(self, x_or_y, item):
        scale = Ones(len(self.choices))
        for (i, v) in enumerate(self.choices):
            data = self.ipane.outputdbs[0].get(v)
            if item[:4] == "norm":
                _max = np.amax(np.abs(data))
                _max = 1. if _max < EPSILON else _max
                a = 1. / _max
            elif item == "max":
                a = np.amax(data)
            elif item == "min":
                a = np.amin(data)
            else:
                a = float(eval(item, GDICT, LDICT))

            scale[i] = a
        return scale

    def set_xscale(self, item):
        assert len(item) == len(self.choices)
        self.xscale = item

    def set_yscale(self, item):
        assert len(item) == len(self.choices)
        self.yscale = item

    @on_trait_change('Time')
    def change_data_markers(self):
        xname = self.choices[self.x_idx]
        for (d, pd) in enumerate(self.ipane.outputdbs):
            if (pd.name, d) not in self.time_data_labels:
                continue

            for (i, y_idx) in enumerate(self.indices):
                yname = self.choices[y_idx]
                self.time_data_labels[(pd.name, d)][i].data_point = (
                    pd.get(xname, self.Time) * self.xscale[self.x_idx],
                    pd.get(yname, self.Time) * self.yscale[y_idx])

        self.container.invalidate_and_redraw()
        self.frame = f_index(self.times, self.Time)
        return

    def create_container(self):
        container = Plot(padding=(50,5,5,35), fill_padding=True,
                         bgcolor="white", use_backbuffer=True,
                         border_visible=True)
        return container

    def change_axis(self, index):
        """Change the x-axis of the current plot

        Parameters
        ----------
        index : int
            The column containing the new x-axis data

        Returns
        -------
        None

        """
        self.x_idx = index
        self.change_plot(self.indices)
        return

    def create_data_label(self, xp, yp, d, di, name):
        nform = "[%(x).5g, %(y).5g]"
        if self.nfiles - 1:
            lform = "{1} ({0})".format(name, nform)
        else:
            lform = nform
        label = DataLabel(component=self.container, data_point=(xp, yp),
                          label_position="bottom right",
                          border_visible=False,
                          bgcolor="transparent",
                          label_format=lform,
                          marker_color=tuple(COLOR_PALETTE[(d + di) % 10]),
                          marker_line_color="transparent",
                          marker="diamond", arrow_visible=False)
        self.time_data_labels[(name, d)].append(label)
        self.container.overlays.append(label)
        return

    def create_plot(self, x, y, c, ls, yvar_name, lw=2.5):
        self.container.data.set_data("x " + yvar_name, x)
        self.container.data.set_data("y " + yvar_name, y)
        self.container.plot(
            ("x " + yvar_name, "y " + yvar_name),
            line_width=lw, name=yvar_name,
            color=c, bgcolor="white", border_visible=True, line_style=ls)
        self._refresh = 0
        return

    def change_plot(self, indices=None):
        if indices is None:
            indices = self.indices
        self.indices = indices
        self.container = self.create_container()
        if self.ipane.outputdbs:
             self.times = [float(x) for x in self.ipane.outputdbs[0].get("TIME")]
             self.low_time = min(self.times)
             self.high_time = max(self.times)
             self.frames = range(len(self.times))
        else:
             self.times = []
             self.high_time = self.low_time = 0.
             self.frames = range(len(self.times))
        self.container.data = ArrayPlotData()
        self.time_data_labels = {}
        if len(indices) == 0:
            return
        self._refresh = 1

        xname = self.choices[self.x_idx]

        # loop through plot data and plot it
        fnams = []
        for (d, pd) in enumerate(self.ipane.outputdbs):

            if pd.hidden:
                continue

            self.y_idx = indices[0]
            self.time_data_labels[(pd.name, d)] = []

            # indices is an integer list containing the columns of the data to
            # be plotted. The indices are wrt to the FIRST file in parsed, not
            # necessarily the same for every file.
            fnam = pd.name
            if fnam in fnams:
                fnam += "-{0}".format(len(fnams))
            if fnam not in fnams:
                fnams.append(fnam)
            get_color(reset=1)

            for i, idx in enumerate(indices):

                yname = self.choices[idx]

                # get the data
                if pd.get(xname) is None or pd.get(yname) is None:
                    continue
                x = pd.get(xname) * self.xscale[self.x_idx]
                y = pd.get(yname) * self.yscale[idx]

                # legend entry
                legend = pd.legend(yname)
                if self.nfiles - 1:
                    entry = "{1} ({0})".format(fnam, legend)
                else:
                    entry = legend
                color = get_color(rand=self.rand)
                ls = LS[(d + i) % len(LS)]
                self.create_plot(x, y, color, ls, entry)

                # create point marker
                xp = pd.get(xname, self.Time) * self.xscale[self.x_idx]
                yp = pd.get(yname, self.Time) * self.yscale[idx]
                yp_idx = pd.choices.index(yname)
                self.create_data_label(xp, yp, d, yp_idx, pd.name)

                continue

        add_default_grids(self.container)

        self.container.index_range.tight_bounds = True
        self.container.index_range.refresh()
        self.container.value_range.tight_bounds = True
        self.container.value_range.refresh()

        self.container.tools.append(PanTool(self.container))

        zoom = ZoomTool(self.container, tool_mode="box", always_on=False)
        self.container.overlays.append(zoom)

        dragzoom = DragZoom(self.container, drag_button="right")
        self.container.tools.append(dragzoom)

        self.container.legend.visible = self.legend_visible

        def tickfmt(val, label):
            return '{0:g}'.format(val)

        self.container.x_axis.tick_label_formatter = lambda x, y="x": tickfmt(x, y)
        self.container.y_axis.tick_label_formatter = lambda x, y="y": tickfmt(x, y)
        self.container.x_axis.title = self.choices[self.x_idx]

        self.container.invalidate_and_redraw()
        return

    def update(self, choices=None, dframe=None, frame=None, time=None):
        if choices is not None:
            self.choices = [x for x in choices]
            self.xscale = Ones(len(choices))
            self.yscale = Ones(len(choices))

        elif dframe is not None:
            frame = self.frame + dframe
            if frame < 0:
                frame = len(self.frames) - 1
            elif frame > len(self.frames) - 1:
                frame = 0
            self.frame = frame
            self.Time = self.times[self.frame]

        elif frame is not None:
            if frame < 0:
                frame = len(self.frames) - 1
            elif frame > len(self.frames) - 1:
                frame = 0
            self.frame = frame
            self.Time = self.times[self.frame]

        elif time is not None:
            self.frame = f_index(self.times, time)
            self.Time = self.times[self.frame]

    @property
    def min_x(self):
        name = self.choices[self.x_idx]
        return np.amin(self.ipane.outputdbs[0].get(name))

    @property
    def max_x(self):
        name = self.choices[self.x_idx]
        return np.amax(self.ipane.outputdbs[0].get(name))

    @property
    def abs_max_x(self):
        name = self.choices[self.x_idx]
        return np.amax(np.abs(self.ipane.outputdbs[0].get(name)))

    @property
    def min_y(self):
        name = self.choices[self.y_idx]
        return np.amin(self.ipane.outputdbs[0].get(name))

    @property
    def max_y(self):
        name = self.choices[self.y_idx]
        return np.amax(self.ipane.outputdbs[0].get(name))

    @property
    def abs_max_y(self):
        name = self.choices[self.y_idx]
        return np.amax(np.abs(self.ipane.outputdbs[0].get(name)))

    @property
    def nfiles(self):
        return len(self.ipane.outputdbs)

class SingleSelectHandler(Handler):
    def closed(self, info, is_ok):
        if hasattr(info.object, 'caller'):
            info.object.caller.can_fire = True
        else:
            info.object.can_fire = True
    def object_selected_changed(self, info):
        if not info.object.choices:
            return
        selected = info.object.selected
        is_string = 1
        i = info.object.choices.index(selected)
        info.object.plot.change_axis(i)

class SingleSelect(HasTraits):
    choices = List(Str)
    selected = Str
    plot = Instance(MMLPostViewer)
    can_fire = Bool(True)
    traits_view = View(
        Item(name='selected', editor=EnumEditor(name='choices'), show_label=False),
        buttons   = ['OK'],
        style     = 'simple',
        resizable = True,
        handler   = SingleSelectHandler())

    @on_trait_change('plot.choices')
    def _on_plot_choices_change(self):
        '''Change the x and y variables'''
        self.choices = self.plot.choices
        if not self.choices:
            return
        single = None
        good = lambda s: 'time' not in s and 'step' not in s
        for (i, label) in enumerate(self.choices):
            l = label.lower()
            if single is None and l == 'time':
                single = label
        self.selected = single or self.choices[0]

        # update the plot
        self.plot.change_axis(self.choices.index(self.selected))

class MultiSelectAdapter(TabularAdapter):
    columns = [('Output Fields', 'myvalue')]
    myvalue_text = Property
    def _get_myvalue_text(self):
        return self.item

class MultiSelect(HasPrivateTraits):
    choices = List(Str)
    selected = List(Str)
    plot = Instance(MMLPostViewer)
    view = View(HGroup(
            UItem('choices',
                         editor=TabularEditor(
                             show_titles=True,
                             selected='selected',
                             editable=False,
                             multi_select=True,
                             horizontal_lines=True,
                             adapter=MultiSelectAdapter()))),
        width=224, height=568, resizable=True)

    @on_trait_change('selected')
    def _selected_modified(self, object, name, new):
        ind = []
        for i in object.selected:
            ind.append(object.choices.index(i))
        self.plot.change_plot(ind)

    @on_trait_change('plot.choices')
    def _choices_modified(self, object, name, new):
        self.choices = [x for x in self.plot.choices]


def f_index(a, b):
    a = np.asarray(a)
    if b >= np.amax(a):
        return a.shape[0] - 1
    elif b <= np.amin(a):
        return 0
    try:
        return np.where(np.abs(a - b) <= 1.e-12)[0][0]
    except IndexError:
        for (i, c) in enumerate(a):
            if c >= b: return i

# Run the demo (if invoked from the command line):
if __name__== '__main__':
    # Create the demo:
    demo = XYScales()
    demo.configure_traits(view='edit_view')
