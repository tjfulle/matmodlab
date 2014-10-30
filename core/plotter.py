import os
import sys
import math
import random
import argparse
import linecache
import numpy as np

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

LS = ['solid', 'dot dash', 'dash', 'dot', 'long dash']
F_EVALDB = "mml-evaldb.xml"
XY_DATA = None
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
            logerr("{0}: no such file".format(source))
            continue
        sources.append(os.path.realpath(source))
    if logerr():
        stop("*** error: stopping due to previous errors")

    create_model_plot(sources)


class Plot2D(HasTraits):
    container = Instance(Plot)
    plot_info = Dict(Int, Dict(Str, List(Str)))
    plot_data = List(Array)
    overlay_headers = Dict(Str, List(Str))
    overlay_plot_data = Dict(Str, Array)
    variables = List(Str)
    plot_indices = List(Int)
    x_idx = Int
    y_idx = Int
    Time = Float
    high_time = Float
    low_time = Float
    time_data_labels = Dict(Int, List)
    runs_shown = List(Bool)
    x_scale = Float
    y_scale = Float
    _line_style_index = Int

    traits_view = View(
        Group(
            Item('container', editor=ComponentEditor(size=SIZE),
                        show_label=False),
            Item('Time', editor=RangeEditor(low_name='low_time',
                                                          high_name='high_time',
                                                          format='%.1f',
                                                          label_width=28,
                                                          mode='auto')),
            orientation="vertical"),
        resizable=True,
        width=SIZE[0], height=SIZE[1] + 100)

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.time_data_labels = {}
        self.x_idx = 0
        self.y_idx = 0
        self._refresh = 1
        self.runs_shown = [True] * len(self.variables)
        self.x_scale, self.y_scale = 1., 1.
        pass

    @on_trait_change('Time')
    def change_data_markers(self):
        ti = self.find_time_index()
        for d in range(len(self.plot_data)):
            if d not in self.time_data_labels:
                continue

            for i, y_idx in enumerate(self.plot_indices):
                self.time_data_labels[d][i].data_point = (
                    self.plot_data[d][ti, self.x_idx] * self.x_scale,
                    self.plot_data[d][ti, y_idx] * self.y_scale)

        self.container.invalidate_and_redraw()
        return

    def create_container(self):
        container = Plot(padding=80, fill_padding=True,
                         bgcolor="white", use_backbuffer=True,
            border_visible=True)
        return container

    def find_time_index(self):
        list_of_diffs = [abs(x - self.Time) for x in self.plot_data[0][:, 0]]
        tdx = list_of_diffs.index(min(list_of_diffs))
        return tdx

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

    def create_data_label(self, xp, yp, d, di):
        nform = "[%(x).5g, %(y).5g]"
        if self.nfiles() - 1 or self.overlay_plot_data:
            lform = "({0}) {1}".format(self.get_file_name(d), nform)
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

        self.time_data_labels[d].append(label)
        self.container.overlays.append(label)
        return

    def create_plot(self, x, y, c, ls, plot_name):
        self.container.data.set_data("x " + plot_name, x)
        self.container.data.set_data("y " + plot_name, y)
        self.container.plot(
            ("x " + plot_name, "y " + plot_name),
            line_width=2.0, name=plot_name,
            color=c, bgcolor="white", border_visible=True, line_style=ls)
        self._refresh = 0
        return

    def change_plot(self, indices, x_scale=None, y_scale=None):
        global XY_DATA
        self.plot_indices = indices
        self.container = self.create_container()
        self.high_time = float(max(self.plot_data[0][:, 0]))
        self.low_time = float(min(self.plot_data[0][:, 0]))
        self.container.data = ArrayPlotData()
        self.time_data_labels = {}
        if len(indices) == 0:
            return
        self._refresh = 1
        XY_DATA = []
        x_scale, y_scale = self.get_axis_scales(x_scale, y_scale)

        # loop through plot data and plot it
        overlays_plotted = False
        fnams = []
        for d in range(len(self.plot_data)):

            if not self.runs_shown[d]:
                continue

            # The legend entry for each file is one of the following forms:
            #   1) [file basename] VAR
            #   2) [file basename:] VAR variables
            # depending on if variabes were perturbed for this run.
            variables = self.variables[d]
            if len(variables) > 30:
                variables = ", ".join(variables.split(",")[:-1])
            if variables:
                variables = ": {0}".format(variables)

            self.time_data_labels[d] = []
            ti = self.find_time_index()
            mheader = self._mheader()
            xname = mheader[self.x_idx]
            self.y_idx = getidx(mheader, mheader[indices[0]])

            # indices is an integer list containing the columns of the data to
            # be plotted. The indices are wrt to the FIRST file in parsed, not
            # necessarily the same for every file. Here, we loop through the
            # indices, determine the name from the first file's header and
            # find the x and y index in the file of interest
            fnam, header = self.get_info(d)
            if fnam in fnams:
                fnam += "-{0}".format(len(fnams))
            fnams.append(fnam)
            get_color(reset=1)
            for i, idx in enumerate(indices):
                yname = mheader[idx]

                # get the indices for this file
                xp_idx = getidx(header, xname)
                yp_idx = getidx(header, yname)
                if xp_idx is None or yp_idx is None:
                    continue

                x = self.plot_data[d][:, xp_idx] * x_scale
                y = self.plot_data[d][:, yp_idx] * y_scale
                if self.nfiles() - 1 or self.overlay_plot_data:
                    entry = "({0}) {1}{2}".format(fnam, yname, variables)
                else:
                    entry = "{0} {1}".format(yname, variables)
                color = get_color(rand=True)
                ls = LS[(d + i) % len(LS)]
                self.create_plot(x, y, color, ls, entry)
                XY_DATA.append(Namespace(key=fnam, xname=xname, x=x,
                                         yname=yname, y=y, lw=1))

                # create point marker
                xp = self.plot_data[d][ti, xp_idx] * x_scale
                yp = self.plot_data[d][ti, yp_idx] * y_scale
                self.create_data_label(xp, yp, d, yp_idx)

                if not overlays_plotted:
                    # plot the overlay data
                    overlays_plotted = True
                    ii = i + 1
                    for fnam, head in self.overlay_headers.items():
                        # get the x and y indeces corresponding to what is
                        # being plotted
                        xo_idx = getidx(head, xname)
                        yo_idx = getidx(head, yname)
                        if xo_idx is None or yo_idx is None:
                            continue
                        xo = self.overlay_plot_data[fnam][:, xo_idx] * x_scale
                        yo = self.overlay_plot_data[fnam][:, yo_idx] * y_scale
                        # legend entry
                        entry = "({0}) {1}".format(fnam, head[yo_idx])
                        _i = d + len(self.plot_data) + 3
                        color = get_color(rand=True)
                        ls = LS[(d + ii) % len(LS)]
                        self.create_plot(xo, yo, color, ls, entry)
                        XY_DATA.append(Namespace(key=fnam, xname=xname, x=xo,
                                                 yname=yname, y=yo, lw=3))
                        ii += 1
                        continue

        add_default_grids(self.container)
        add_default_axes(self.container, htitle=mheader[self.x_idx])

        self.container.index_range.tight_bounds = False
        self.container.index_range.refresh()
        self.container.value_range.tight_bounds = False
        self.container.value_range.refresh()

        self.container.tools.append(PanTool(self.container))

        zoom = ZoomTool(self.container, tool_mode="box", always_on=False)
        self.container.overlays.append(zoom)

        dragzoom = DragZoom(self.container, drag_button="right")
        self.container.tools.append(dragzoom)

        self.container.legend.visible = True

        self.container.invalidate_and_redraw()
        return

    def _mheader(self):
        """Returns the "master" header - the header of the first file

        Returns
        -------
        header : list
        """
        return self.get_info(0)[1]

    def min_x(self):
        return np.amin(self.plot_data[0][:, self.x_idx])

    def max_x(self):
        return np.amax(self.plot_data[0][:, self.x_idx])

    def abs_max_x(self):
        return np.amax(np.abs(self.plot_data[0][:, self.x_idx]))

    def min_y(self):
        return np.amin(self.plot_data[0][:, self.y_idx])

    def max_y(self):
        return np.amax(self.plot_data[0][:, self.y_idx])

    def abs_max_y(self):
        return np.amax(np.abs(self.plot_data[0][:, self.y_idx]))

    def get_info(self, i):
        """Return the info for index i

        Parameters
        ----------
        i : int
            The location in self.plot_info

        Returns
        -------
        fnam : str
            the file name
        header : list
            the file header

        """
        return self.plot_info[i].items()[0]

    def get_file_name(self, i):
        return self.get_info(i)[0]

    def nfiles(self):
        return len(self.plot_info)

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

    view = View(Item('Change_X_Axis',
                                   enabled_when='Change_X_Axis_Enabled==True',
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
    plot_info = Dict(Int, Dict(Str, List(Str)))
    Multi_Select = Instance(MultiSelect)
    Change_Axis = Instance(ChangeAxis)
    Reset_Zoom = Button('Reset Zoom')
    Reload_Data = Button('Reload Data')
    Print_to_PDF = Button('Print to PDF')
    Load_Overlay = Button('Open Overlay')
    Close_Overlay = Button('Close Overlay')
    X_Scale = String("1.0")
    Y_Scale = String("1.0")
    Single_Select_Overlay_Files = Instance(SingleSelectOverlayFiles)
    filepaths = List(String)
    file_variables = List(String)

    def __init__(self, **traits):
        """Put together information to be sent to Plot2D information
        needed:

        plot_info : dict
           {0: {file_0: header_0}}
           {1: {file_1: header_1}}
           ...
           {n: {file_n: header_n}}
        variables : list
           list of variables that changed from one simulation to another
        x_idx : int
           column containing x variable to be plotted

        """

        HasStrictTraits.__init__(self, **traits)
        fileinfo = get_sorted_fileinfo(self.filepaths)
        data = []
        for idx, (fnam, fhead, fdata) in enumerate(fileinfo):
            if idx == 0: mheader = fhead
            self.plot_info[idx] = {fnam: fhead}
            data.append(fdata)

        self.Plot_Data = Plot2D(
            plot_data=data, variables=self.file_variables,
            x_idx=0, plot_info=self.plot_info)
        self.Multi_Select = MultiSelect(choices=mheader, plot=self.Plot_Data)
        self.Change_Axis = ChangeAxis(
            Plot_Data=self.Plot_Data, headers=mheader)
        self.Single_Select_Overlay_Files = SingleSelectOverlayFiles(choices=[])
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
            scale = str(self.Plot_Data.max_x())
        elif scale == "min":
            scale = str(self.Plot_Data.min_x())
        elif scale == "normalize":
            _max = self.Plot_Data.abs_max_x()
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
            scale = str(self.Plot_Data.max_y())
        elif scale == "min":
            scale = str(self.Plot_Data.min_y())
        elif scale == "normalize":
            _max = self.Plot_Data.abs_max_y()
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
        fileinfo = get_sorted_fileinfo(self.filepaths)
        data = []
        for idx, (fnam, fhead, fdata) in enumerate(fileinfo):
            if idx == 0: mheader = fhead
            self.plot_info[idx] = {fnam: fhead}
            data.append(fdata)
        self.Plot_Data.plot_data = data
        self.Plot_Data.plot_info = self.plot_info
        self.Multi_Select.choices = mheader
        self.Change_Axis.headers = mheader
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _Print_to_PDF_fired(self):
        import matplotlib.pyplot as plt
        if not XY_DATA:
            return

        # get the maximum of Y for normalization
        ymax = max(np.amax(np.abs(xyd.y)) for xyd in XY_DATA)

        # setup figure
        plt.figure(0)
        plt.cla()
        plt.clf()

        # plot y value for each plot on window
        ynames = []
        for xyd in sorted(XY_DATA, key=lambda x: x.lw, reverse=True):
            label = xyd.key + ":" + xyd.yname if len(XY_DATA) > 1 else xyd.yname
            ynames.append(xyd.yname)
            if SCALE:
                plt.plot(xyd.x, xyd.y / ymax, label=label, lw=xyd.lw)
            else:
                plt.plot(xyd.x, xyd.y, label=label, lw=xyd.lw)
        yname = common_prefix(ynames)
        plt.xlabel(xyd.xname)
        plt.ylabel(yname)
        plt.legend(loc="best")
        plt.savefig("{0}-vs-{1}.pdf".format(yname, xyd.xname))

    def _Close_Overlay_fired(self):
        if self.Single_Select_Overlay_Files.selected:
            index = self.Single_Select_Overlay_Files.choices.index(
                self.Single_Select_Overlay_Files.selected)
            self.Single_Select_Overlay_Files.choices.remove(
                self.Single_Select_Overlay_Files.selected)
            del self.Plot_Data.overlay_plot_data[
                self.Single_Select_Overlay_Files.selected]
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
            for eachfile in dialog.paths:
                try:
                    fhead, fdata = loadcontents(eachfile)
                except:
                    logmes("{0}: Error reading overlay data".format(eachfile))
                    continue
                fnam = os.path.basename(eachfile)
                self.Plot_Data.overlay_plot_data[fnam] = fdata
                self.Plot_Data.overlay_headers[fnam] = fhead
                self.Single_Select_Overlay_Files.choices.append(fnam)
                continue
            self.Plot_Data.change_plot(self.Plot_Data.plot_indices)
        return


def create_view(window_name):
    view = View(HSplit(
        VGroup(
            Item('Multi_Select', show_label=False, width=224,
                        height=H-200, springy=True, resizable=True),
            Item('Change_Axis', show_label=False), ),
        Item('Plot_Data', show_label=False, width=W-300, height=H-100,
                    springy=True, resizable=True)),
                       style='custom', width=W, height=H,
                       resizable=True, title=window_name)
    return view


def create_model_plot(sources, handler=None, metadata=None):
    """Create the plot window

    Parameters
    ----------

    """
    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    def genrunid(path):
        return os.path.splitext(os.path.basename(path))[0]

    if metadata is not None:
        stop("*** error: call create_view directly")
        metadata.plot.configure_traits(view=view)
        return

    if [source for source in sources if F_EVALDB in os.path.basename(source)]:
        if len(sources) > 1:
            stop("*** error: only one source allowed with {0}".format(F_EVALDB))
        source = sources[0]
        if not os.path.isfile(source):
            stop("*** error: {0}: no such file".format(source))
        filepaths, variables = readtabular(source)
        runid = genrunid(filepaths[0])

    else:
        filepaths = []
        for source in sources:
            if not os.path.isfile(source):
                logerr("{0}: {1}: no such file".format(iam, source))
                continue
            filepaths.append(source)
        if logerr():
            stop("***error: stopping due to previous errors")
        variables = [""] * len(filepaths)
        runid = ("Material Model Laboratory" if len(filepaths) > 1
                 else genrunid(filepaths[0]))

    view = View(HSplit(
        VGroup(
            Item('Multi_Select', show_label=False),
            Item('Change_Axis', show_label=False),
            Item('Reset_Zoom', show_label=False),
            Item('Reload_Data', show_label=False),
            Item('Print_to_PDF', show_label=False),
            VGroup(
                HGroup(Item("X_Scale", label="X Scale",
                                          editor=TextEditor(
                                              multi_line=False)),
                              Item("Y_Scale", label="Y Scale",
                                          editor=TextEditor(
                                              multi_line=False))),
                show_border=True),
            VGroup(
                HGroup(
                    Item('Load_Overlay', show_label=False, springy=True),
                    Item(
                        'Close_Overlay', show_label=False, springy=True),),
                Item('Single_Select_Overlay_Files', show_label=False,
                            resizable=False), show_border=True)),
        Item('Plot_Data', show_label=False, width=W-300, height=H-100,
                    springy=True, resizable=True)),
        style='custom', width=W, height=H,
        resizable=True, title=runid)

    main_window = ModelPlot(filepaths=filepaths, file_variables=variables)
    main_window.configure_traits(view=view, handler=handler)
    return main_window


def getidx(a, name, comments="#"):
    """Return the index for name in a"""
    try:
        return [x.lower() for x in a if x != comments].index(name.lower())
    except ValueError:
        return None


def stop(message):
    raise SystemExit(message)


def logmes(message):
    sys.stdout.write("plot2d: {0}\n".format(message))


def logerr(message=None, errors=[0]):
    if message is None:
        return errors[0]
    sys.stderr.write("*** {0}: error: {1}\n".format(EXE, message))
    errors[0] += 1


def readtabular(source):
    """Read in the mml-tabular.dat file

    """
    from utils.mmltab import read_mml_evaldb
    sources, paraminfo, _ = read_mml_evaldb(source)
    for (i, info) in enumerate(paraminfo):
        paraminfo[i] = ", ".join("{0}={1:.2g}".format(n, v) for (n, v) in info)
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
            stop("*** {0}: error: {1}: inconsistent data in row {1}".format(
                EXE, os.path.basename(f), iline))
        lines.append(line)
    return np.array(lines)


def get_sorted_fileinfo(filepaths):
    """Sort the fileinfo based on length of header in each file in filepaths so
    that the file with the longest header is first

    """
    fileinfo = []
    for filepath in filepaths:
        fnam = os.path.basename(filepath)
        fhead, fdata = loadcontents(filepath)
        if not np.any(fdata):
            logerr("No data found in {0}".format(filepath))
            continue
        fileinfo.append((fnam, fhead, fdata))
        continue
    if logerr():
        stop("***error: stopping due to previous errors")
    return sorted(fileinfo, key=lambda x: len(x[1]), reverse=True)


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
