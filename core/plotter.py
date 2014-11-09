import os
import sys
import math
import random
import argparse
import warnings
import linecache
import numpy as np

warnings.simplefilter("ignore")

# chaco and traits imports are in the enthought directory in EPD/Canopy
try:
    from enthought import chaco, traits
except ImportError:
    pass

from chaco.api import *
from traits.api import *
from traitsui.api import *
from chaco.tools.api import *
from enable.api import ComponentEditor
from pyface.api import FileDialog, OK as pyOK
from chaco.example_support import COLOR_PALETTE
from traitsui.tabular_adapter import TabularAdapter

from utils.exojac import ExodusIIFile

SIZE = (700, 600)
H1, W1 = 868., 1124.
r = W1 / H1
H = 700
W = int(r * H)

EXE = "plot2d"
Change_X_Axis_Enabled = True
LDICT = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
         "asin": math.asin, "acos": math.acos,
         "atan": math.atan, "atan2": math.atan2, "pi": math.pi,
         "log": math.log, "exp": math.exp, "floor": math.floor,
         "ceil": math.ceil, "abs": math.fabs, "random": random.random, }
GDICT = {"__builtins__": None}
EPSILON = np.finfo(np.float).eps

LS = ['solid', 'dot dash', 'dash', 'long dash'] # , 'dot']
F_EVALDB = "mml-evaldb.xml"
SCALE = True


class Namespace(object):
    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)
    def __repr__(self):
        string = ", ".join("{0}={1}".format(k, repr(v)) for (k, v) in
                           self.__dict__.items())
        return "Namespace({0})".format(string)
    def items(self):
        return self.__dict__.items()

class Logger:
    errors = 0
    def __init__(self):
        self.ch = sys.stdout

    def write(self, message):
        self.ch.write(message + "\n")

    def info(message):
        self.write("plot2d: {0}".format(message))

    def error(message, stop=0):
        self.write("*** plot2d: error: {0}".format(message))
        if stop:
            sys.exit(1)
        self.errors += 1
logger = Logger()

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


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args(argv)
    sources = []
    for source in args.sources:
        if source.rstrip(os.path.sep).endswith(".eval"):
            source = os.path.join(source, F_EVALDB)
        if not os.path.isfile(source):
            logger.error("{0}: no such file".format(source))
            continue
        sources.append(os.path.realpath(source))
    if logger.errors:
        logger.error("stopping due to previous errors", stop=1)

    create_model_plot(sources)

class _SingleFileData:
    def __init__(self, filepath, legend_info=None):
        self.name = os.path.splitext(os.path.basename(filepath))[0]
        self.source = filepath
        self.legend_info = legend_info
        names, self.data = loadcontents(filepath)

        # data contains columns of data
        self.pv = dict([(s.upper(), i) for (i, s) in enumerate(names)])

    def __call__(self, name, time=None):
        j = self.pv.get(name.upper())
        if j is None:
            return
        data = self.data[:, j]
        if time is None:
            return data
        i = np.argmin(np.abs(time - self.data[:, self.pv["TIME"]]))
        return data[i]

    def legend(self, name):
        if name.upper() not in self.pv:
            return
        legend = name.upper()
        if self.legend_info:
            legend += " ({0})".format(self.legend_info)
        return legend

    @property
    def plotable_vars(self):
        return sorted(self.pv.keys(), key=lambda k: self.pv[k])

class FileData:
    def __init__(self, files, legend_info=None):
        self.pv = []
        self.data = []
        self.files = []
        for f in files:
            li = None if legend_info is None else legend_info[f]
            fd = _SingleFileData(f, legend_info=li)
            self.data.append(fd)
            self.pv.extend([x for x in fd.plotable_vars if x not in self.pv])
            self.files.append(f)

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return self.data.__len__()

    def __nonzero__(self):
        return bool(self.data)

    def add(self, filename):
        self.data.append(_SingleFileData(filename))

    def remove(self, filename):
        for (i, d) in enumerate(self.data):
            if filename in (d.source, d.name):
                break
        else:
            i = None
        if i is not None:
            self.data.pop(i)

    @property
    def plotable_vars(self):
        return self.pv


class Plot2D(HasTraits):
    container = Instance(Plot)
    file_data = Instance(FileData)
    overlay_file_data = Instance(FileData)
    variables = List(Str)
    plot_indices = List(Int)
    x_idx = Int
    y_idx = Int
    Time = Float
    high_time = Float
    low_time = Float
    time_data_labels = Dict(Tuple, List)
    runs_shown = List(Bool)
    x_scale = Float
    y_scale = Float
    _line_style_index = Int
    rand = Bool
    show_legend = Bool

    traits_view = View(
        Group(Item('container', editor=ComponentEditor(size=SIZE),
                   show_label=False),
              VGroup(
                  Item('Time',
                       editor=RangeEditor(low_name='low_time',
                                          high_name='high_time',
                                          format='%.1f',
                                          #label_width=28,
                                          mode='auto'),
                       show_label=False),
                  label="Time", show_border=True),
            orientation="vertical"),
        resizable=True, width=SIZE[0], height=SIZE[1] + 100)

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.time_data_labels = {}
        self.x_idx = 0
        self.y_idx = 0
        self._refresh = 1
        self.plotable_vars = self.file_data.plotable_vars
        self.runs_shown = [True] * len(self.plotable_vars)
        self.x_scale, self.y_scale = 1., 1.
        self.rand = False
        self.show_legend = True
        pass

    @on_trait_change('Time')
    def change_data_markers(self):
        xname = self.plotable_vars[self.x_idx]
        for (d, pd) in enumerate(self.file_data):
            if (pd.name, d) not in self.time_data_labels:
                continue

            for (i, y_idx) in enumerate(self.plot_indices):
                yname = self.plotable_vars[y_idx]
                self.time_data_labels[(pd.name, d)][i].data_point = (
                    pd(xname, self.Time) * self.x_scale,
                    pd(yname, self.Time) * self.y_scale)

        if self.overlay_file_data:
            # plot the overlay data
            for (d, od) in enumerate(self.overlay_file_data):
                # get the x and y indeces corresponding to what is
                # being plotted
                if (od.name, d) not in self.time_data_labels:
                    continue

                for (i, y_idx) in enumerate(self.plot_indices):
                    yname = self.plotable_vars[y_idx]
                    self.time_data_labels[(od.name, d)][i].data_point = (
                        od(xname, self.Time) * self.x_scale,
                        od(yname, self.Time) * self.y_scale)

        self.container.invalidate_and_redraw()
        return

    def create_container(self):
        container = Plot(padding=75, fill_padding=True,
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
        self.change_plot(self.plot_indices)
        return

    def create_data_label(self, xp, yp, d, di, name):
        nform = "[%(x).5g, %(y).5g]"
        if self.nfiles - 1 or self.overlay_file_data:
            lform = "({0}) {1}".format(name, nform)
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

    def create_plot(self, x, y, c, ls, yvar_name, lw=2.0):
        self.container.data.set_data("x " + yvar_name, x)
        self.container.data.set_data("y " + yvar_name, y)
        self.container.plot(
            ("x " + yvar_name, "y " + yvar_name),
            line_width=lw, name=yvar_name,
            color=c, bgcolor="white", border_visible=True, line_style=ls)
        self._refresh = 0
        return

    def change_plot(self, indices, x_scale=None, y_scale=None):
        self.plot_indices = indices
        self.container = self.create_container()
        self.high_time = float(max(self.file_data[0]("TIME")))
        self.low_time = float(min(self.file_data[0]("TIME")))
        self.container.data = ArrayPlotData()
        self.time_data_labels = {}
        if len(indices) == 0:
            return
        self._refresh = 1
        x_scale, y_scale = self.get_axis_scales(x_scale, y_scale)

        # loop through plot data and plot it
        overlays_plotted = False
        fnams = []
        for (d, pd) in enumerate(self.file_data):

            xname = self.plotable_vars[self.x_idx]
            self.y_idx = indices[0]

            self.time_data_labels[(pd.name, d)] = []

            # indices is an integer list containing the columns of the data to
            # be plotted. The indices are wrt to the FIRST file in parsed, not
            # necessarily the same for every file. Here, we loop through the
            # indices, determine the name from the first file's header and
            # find the x and y index in the file of interest
            fnam = pd.name
            header = pd.plotable_vars
            if fnam in fnams:
                fnam += "-{0}".format(len(fnams))
            fnams.append(fnam)
            get_color(reset=1)
            for i, idx in enumerate(indices):

                yname = self.plotable_vars[idx]

                # get the data
                x = pd(xname)
                y = pd(yname)

                if x is None or y is None:
                    continue

                x *= x_scale
                y *= y_scale

                legend = pd.legend(yname)
                if self.nfiles - 1 or self.overlay_file_data:
                    entry = "({0}) {1}".format(fnam, legend)
                else:
                    entry = legend

                color = get_color(rand=self.rand)
                ls = LS[(d + i) % len(LS)]
                self.create_plot(x, y, color, ls, entry)

                # create point marker
                xp = pd(xname, self.Time) * x_scale
                yp = pd(yname, self.Time) * y_scale
                yp_idx = pd.plotable_vars.index(yname)
                self.create_data_label(xp, yp, d, yp_idx, pd.name)

                if not overlays_plotted and self.overlay_file_data:
                    # plot the overlay data
                    overlays_plotted = True
                    ii = i + 1
                    for (dd, od) in enumerate(self.overlay_file_data):
                        # get the x and y indeces corresponding to what is
                        # being plotted
                        xo = od(xname)
                        yo = od(yname)
                        if xo is None or yo is None:
                            continue

                        # legend entry
                        entry = "({0}) {1}".format(od.name, yname)
                        color = get_color(rand=self.rand)
                        ls = "dot" #LS[(d + ii) % len(LS)]
                        self.create_plot(xo, yo, color, ls, entry, lw=1.0)

                        # create point marker
                        self.time_data_labels[(od.name, dd)] = []
                        xp = od(xname, self.Time) * x_scale
                        yp = od(yname, self.Time) * y_scale
                        yp_idx = od.plotable_vars.index(yname)
                        self.create_data_label(xp, yp, dd, yp_idx, od.name)

                        ii += 1
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

        self.container.legend.visible = self.show_legend

        def tickfmt(val, label):
            m = self.max_y if label == "y" else self.max_x
            if m > 1000:
                s = "{0:.4e}".format(val)
                s = s.split("e")
                s[0] = s[0].rstrip("0").rstrip(".")
                return "e".join(s)
            return "{0:f}".format(val).rstrip("0").rstrip(".")

        self.container.x_axis.tick_label_formatter = lambda x, y="x": tickfmt(x, y)
        self.container.y_axis.tick_label_formatter = lambda x, y="y": tickfmt(x, y)
        self.container.x_axis.title = self.plotable_vars[self.x_idx]

        self.container.invalidate_and_redraw()
        return

    @property
    def min_x(self):
        return np.amin(self.file_data[0](self.plotable_vars[self.x_idx]))

    @property
    def max_x(self):
        return np.amax(self.file_data[0](self.plotable_vars[self.x_idx]))

    @property
    def abs_max_x(self):
        return np.amax(np.abs(self.file_data[0](self.plotable_vars[self.x_idx])))

    @property
    def min_y(self):
        return np.amin(self.file_data[0](self.plotable_vars[self.y_idx]))

    @property
    def max_y(self):
        return np.amax(self.file_data[0](self.plotable_vars[self.y_idx]))

    @property
    def abs_max_y(self):
        return np.amax(np.abs(self.file_data[0](self.plotable_vars[self.y_idx])))

    @property
    def nfiles(self):
        return len(self.file_data)

    def get_axis_scales(self, x_scale, y_scale):
        """Get/Set the scales for the x and y axis

        Parameters
        ----------
        x_scale : float, optional
        y_scale : float, optional

        Returns
        -------
        x_scale : float
        y_scale : float

        """
        # get/set x_scale
        if x_scale is None:
            x_scale = self.x_scale
        else:
            self.x_scale = x_scale

        # get/set y_scale
        if y_scale is None:
            y_scale = self.y_scale
        else:
            self.y_scale = y_scale

        return x_scale, y_scale


class ChangeAxisHandler(Handler):

    def closed(self, info, is_ok):
        global Change_X_Axis_Enabled
        Change_X_Axis_Enabled = True


class ChangeAxis(HasStrictTraits):

    Change_X_Axis = Button('Change X-axis')
    Plot_Data = Instance(Plot2D)
    headers = List(Str)

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)

    def _Change_X_Axis_fired(self):
        global Change_X_Axis_Enabled
        Change_X_Axis_Enabled = False
        ms = SingleSelect(choices=self.headers, plot=self.Plot_Data)
        ms.configure_traits(handler=ChangeAxisHandler())

    view = View(Item('Change_X_Axis', enabled_when='Change_X_Axis_Enabled==True',
                     show_label=False))


class SingleSelectAdapter(TabularAdapter):
    columns = [('Plotable Variables', 'myvalue')]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item


class SingleSelect(HasPrivateTraits):
    choices = List(Str)
    selected = Str
    plot = Instance(Plot2D)

    view = View(
        HGroup(
            UItem('choices',
                         editor=TabularEditor(
                             show_titles=True, selected='selected', editable=False,
                             multi_select=False, adapter=SingleSelectAdapter()))),
        width=224, height=H-200, resizable=True, title='Change X-axis')

    @on_trait_change('selected')
    def _selected_modified(self, object, name, new):
        self.plot.change_axis(object.choices.index(object.selected))


class SingleSelectOverlayFilesAdapter(TabularAdapter):
    columns = [('Overlay File Name', 'myvalue')]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item


class SingleSelectOverlayFiles(HasPrivateTraits):
    choices = List(Str)
    selected = Str

    view = View(
        HGroup(
            UItem('choices',
                         editor=TabularEditor(
                             show_titles=True, selected='selected',
                             editable=False, multi_select=False,
                             adapter=SingleSelectOverlayFilesAdapter()))),
        width=224, height=100)


class MultiSelectAdapter(TabularAdapter):
    columns = [('Plotable Variables', 'myvalue')]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item


class MultiSelect(HasPrivateTraits):
    choices = List(Str)
    selected = List(Str)
    plot = Instance(Plot2D)

    view = View(
        HGroup(
            UItem('choices',
                         editor=TabularEditor(
                             show_titles=True,
                             selected='selected',
                             editable=False,
                             multi_select=True,
                             adapter=MultiSelectAdapter()))),
        width=224, height=H-300, resizable=True)

    @on_trait_change('selected')
    def _selected_modified(self, object, name, new):
        ind = []
        for i in object.selected:
            ind.append(object.choices.index(i))
        self.plot.change_plot(ind)


class ModelPlot(HasStrictTraits):

    Plot_Data = Instance(Plot2D)
    Multi_Select = Instance(MultiSelect)
    Change_Axis = Instance(ChangeAxis)
    Reset_Zoom = Button('Reset Zoom')
    Reload_Data = Button('Reload Data')
    Print_to_PDF = Button('Print to PDF')
    Random_Colors = Bool
    Show_Legend = Bool
    Load_Overlay = Button('Open Overlay')
    Close_Overlay = Button('Close Overlay')
    X_Scale = String("1.0")
    Y_Scale = String("1.0")
    Single_Select_Overlay_Files = Instance(SingleSelectOverlayFiles)
    file_data = Instance(FileData)

    def __init__(self, **traits):
        """Put together information to be sent to Plot2D information
        needed:

        variables : list
           list of variables that changed from one simulation to another
        x_idx : int
           column containing x variable to be plotted

        """
        HasStrictTraits.__init__(self, **traits)
        self.Plot_Data = Plot2D(file_data=self.file_data, x_idx=0)
        self.Multi_Select = MultiSelect(choices=self.file_data.plotable_vars,
                                        plot=self.Plot_Data)
        self.Change_Axis = ChangeAxis(
            Plot_Data=self.Plot_Data, headers=self.file_data.plotable_vars)
        self.Single_Select_Overlay_Files = SingleSelectOverlayFiles(choices=[])
        self.Random_Colors = False
        self.Show_Legend = True
        pass

    def _Reset_Zoom_fired(self):
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _X_Scale_changed(self, scale):
        """Detect if the x-axis scale was changed and let the plotter know

        Parameters
        ----------
        scale : str
           The user entered scale

        Returns
        -------
        None

        Notes
        -----
        scale should be a float, one of the operations in LDICT, or one of the
        optional magic keywords: min, max, normalize. On entry, scale is
        stripped, and if an empty string is sent in, it is reset to 1.0. If
        the magic words min or max are specified, the scale is set to the min
        or max of the x-axis data for the FIRST set of data. If the magic
        keyword normalize is specified, scale is set to 1 / max.

        """
        scale = scale.strip()
        if not scale:
            scale = self.X_Scale = "1.0"
        if scale == "max":
            scale = str(self.Plot_Data.max_x)
        elif scale == "min":
            scale = str(self.Plot_Data.min_x)
        elif scale == "normalize":
            _max = self.Plot_Data.abs_max_x
            _max = 1. if _max < EPSILON else _max
            scale = str(1. / _max)
        try:
            scale = float(eval(scale, GDICT, LDICT))
        except:
            return
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices, x_scale=scale)
        return

    def _Y_Scale_changed(self, scale):
        """Detect if the y-axis scale was changed and let the plotter know

        Parameters
        ----------
        scale : str
           The user entered scale

        Returns
        -------
        None

        Notes
        -----

        scale should be a float, one of the operations in LDICT, or one of the
        optional magic keywords: min, max, normalize. On entry, scale is
        stripped, and if an empty string is sent in, it is reset to 1.0. If
        the magic words min or max are specified, the scale is set to the min
        or max of the y-axis data for the FIRST set of data. If the magic
        keyword normalize is specified, scale is set to 1 / max.

        """
        scale = scale.strip()
        if not scale:
            scale = self.Y_Scale = "1.0"
        if scale == "max":
            scale = str(self.Plot_Data.max_y)
        elif scale == "min":
            scale = str(self.Plot_Data.min_y)
        elif scale == "normalize":
            _max = self.Plot_Data.abs_max_y
            _max = 1. if _max < EPSILON else _max
            scale = str(1. / _max)
        try:
            scale = float(eval(scale, GDICT, LDICT))
        except:
            return
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices, y_scale=scale)
        return

    def _Reload_Data_fired(self):
        self.reload_data()

    def reload_data(self):
        filepaths = [d.source for d in self.Plot_Data.file_data]
        file_data = FileData(filepaths)
        self.Plot_Data.file_data = file_data
        self.Multi_Select.choices = file_data.plotable_vars
        self.Change_Axis.headers = file_data.plotable_vars
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _Random_Colors_fired(self):
        self.Plot_Data.rand = not self.Plot_Data.rand
        return
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _Show_Legend_fired(self, *args):
        if self.Plot_Data.container is None:
            return
        self.Plot_Data.show_legend = not self.Plot_Data.show_legend
        self.Plot_Data.container.legend.visible = self.Plot_Data.show_legend
        self.Plot_Data.container.invalidate_and_redraw()

    def _Print_to_PDF_fired(self):
        import matplotlib.pyplot as plt
        if not self.Plot_Data.plot_indices:
            return

        data = self.Plot_Data.file_data
        indices = self.Plot_Data.plot_indices

        xname = self.Plot_Data.plotable_vars[self.Plot_Data.x_idx]
        yname = self.Plot_Data.plotable_vars[self.Plot_Data.y_idx]

        # get the maximum of Y for normalization
        ymax = max(np.amax(np.abs(d(yname))) for d in data)

        # setup figure
        plt.figure(0)
        plt.cla()
        plt.clf()

        # plot y value for each plot on window
        if self.Plot_Data.overlay_file_data:
            # plot the overlay data
            for od in self.Plot_Data.overlay_file_data:
                # get the x and y indeces corresponding to what is
                # being plotted
                xo = od(xname)
                yo = od(yname)
                if xo is None or yo is None:
                    continue
                # legend entry
                label = od.name + ":" + yname if len(indices) > 1 else yname
                plt.plot(xo, yo, label=label, lw=3)

        ynames = []
        for d in data:
            label = d.name + ":" + yname if len(indices) > 1 else yname
            ynames.append(yname)
            if SCALE:
                plt.plot(d(xname), d(yname) / ymax, label=label, lw=1)
            else:
                plt.plot(d(xname), d(yname), label=label, lw=1)

        yname = common_prefix(ynames)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.legend(loc="best")
        plt.savefig("{0}-vs-{1}.pdf".format(yname, xname))

    def _Close_Overlay_fired(self):
        if self.Single_Select_Overlay_Files.selected:
            index = self.Single_Select_Overlay_Files.choices.index(
                self.Single_Select_Overlay_Files.selected)
            self.Single_Select_Overlay_Files.choices.remove(
                self.Single_Select_Overlay_Files.selected)
            self.Plot_Data.overlay_file_data.remove(
                self.Single_Select_Overlay_Files.selected)
            if not self.Single_Select_Overlay_Files.choices:
                self.Single_Select_Overlay_Files.selected = ""
            else:
                if index >= len(self.Single_Select_Overlay_Files.choices):
                    index = len(self.Single_Select_Overlay_Files.choices) - 1
                self.Single_Select_Overlay_Files.selected = self.Single_Select_Overlay_Files.choices[index]
            self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _Load_Overlay_fired(self):
        dialog = FileDialog(action="open")
        dialog.open()
        info = {}
        if dialog.return_code == pyOK:
            if not self.Plot_Data.overlay_file_data:
                self.Plot_Data.overlay_file_data = FileData(dialog.paths)
                for d in self.Plot_Data.overlay_file_data:
                    self.Single_Select_Overlay_Files.choices.append(d.name)
            else:
                for eachfile in dialog.paths:
                    self.Plot_Data.overlay_file_data.add(eachfile)
                    name = self.Plot_Data.overlay_file_data[-1].name
                    self.Single_Select_Overlay_Files.choices.append(name)
            self.Plot_Data.change_plot(self.Plot_Data.plot_indices)
        return

def create_model_plot(sources, handler=None):
    """Create the plot window

    Parameters
    ----------

    """
    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    def genrunid(path):
        return os.path.splitext(os.path.basename(path))[0]

    if [source for source in sources if F_EVALDB in os.path.basename(source)]:
        if len(sources) > 1:
            logger.error("only one source allowed with {0}".format(F_EVALDB),
                         stop=1)
        source = sources[0]
        if not os.path.isfile(source):
            logger.error("{0}: no such file".format(source), stop=1)
        filepaths, variables = readtabular(source)
        file_data = FileData(filepaths, legend_info=variables)
        runid = genrunid(filepaths[0])

    else:
        filepaths = []
        for source in sources:
            if not os.path.isfile(source):
                logger.error("{0}: {1}: no such file".format(iam, source))
                continue
            filepaths.append(source)
        if logger.errors:
            logger.error("stopping due to previous errors", stop=1)
        variables = [""] * len(filepaths)
        runid = ("Material Model Laboratory" if len(filepaths) > 1
                 else genrunid(filepaths[0]))
        file_data = FileData(filepaths)

    ww = 100
    hh = 50
    view = View(HSplit(
        VGroup(
            Item('Multi_Select', show_label=False),
            Item('_'),
            VGroup(HGroup(
                Item('Random_Colors', style="simple"),
                Item('_'),
                Item('Show_Legend', style="simple", label="Legend")),
                VGrid(
                    Item('Change_Axis', show_label=False),
                    Item('Reset_Zoom', style="simple", show_label=False),
                    Item('Reload_Data', style="simple", show_label=False),
                    Item('Print_to_PDF', style="simple", show_label=False)),
                label="Options", show_border=True),
            VGroup(
                HGroup(Item("X_Scale", label="X Scale",
                                          editor=TextEditor(
                                              multi_line=False)),
                              Item("Y_Scale", label="Y Scale",
                                          editor=TextEditor(
                                              multi_line=False))),
                show_border=True, label="Scaling"),
            VGroup(
                HGroup(
                    Item('Load_Overlay', style="simple",
                         show_label=False, springy=True),
                    Item(
                        'Close_Overlay', style="simple",
                        show_label=False, springy=True),),
                Item('Single_Select_Overlay_Files', show_label=False,
                            resizable=True), show_border=True,
                            label="Overlay Files")),
        Item('Plot_Data', show_label=False, width=W-ww, height=H-hh,
                    springy=True, resizable=True)),
        style='custom', width=W, height=H,
        resizable=True, title=runid)

    main_window = ModelPlot(file_data=file_data)
    main_window.configure_traits(view=view, handler=handler)
    return main_window

def readtabular(source):
    """Read in the mml-tabular.dat file

    """
    from utils.mmltab import read_mml_evaldb
    sources, paraminfo, _ = read_mml_evaldb(source)
    for (key, info) in paraminfo.items():
        paraminfo[key] = ", ".join("{0}={1:.2g}".format(n, v) for (n, v) in info)
    return sources, paraminfo


def loadcontents(filepath):
    if filepath.endswith((".exo", ".e", ".base_exo")):
        exof = ExodusIIFile(filepath, "r")
        glob_var_names = exof.glob_var_names
        elem_var_names = exof.elem_var_names
        data = [exof.get_all_times()]
        for glob_var_name in glob_var_names:
            data.append(exof.get_glob_var_time(glob_var_name))
        for elem_var_name in elem_var_names:
            data.append(exof.get_elem_var_time(elem_var_name, 0))
        data = np.transpose(np.array(data))
        head = ["TIME"] + glob_var_names + elem_var_names
        exof.close()
    else:
        # treat all other files as ascii text and cross fingers...
        head = loadhead(filepath)
        data = loadtxt(filepath, skiprows=1)
    return head, data


def loadhead(filepath, comments="#"):
    """Get the file header

    """
    line = " ".join(x.strip() for x in linecache.getline(filepath, 1).split())
    if line.startswith(comments):
        line = line[1:]
    return line.split()


def loadtxt(f, skiprows=0, comments="#"):
    """Load text from output files

    """
    lines = []
    for (iline, line) in enumerate(open(f, "r").readlines()[skiprows:]):
        try:
            line = [float(x) for x in line.split(comments, 1)[0].split()]
        except ValueError:
            break
        if not lines:
            ncols = len(line)
        if len(line) < ncols:
            break
        if len(line) > ncols:
            logger.error("{0}: inconsistent data in row {1}".format(
                os.path.basename(f), iline), stop=1)
        lines.append(line)
    return np.array(lines)



def common_prefix(strings):
    """Find the longest string that is a prefix of all the strings.

    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix


if __name__ == "__main__":
    main()
