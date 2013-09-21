import os
import sys
import math
import random
import argparse
import linecache
import numpy as np
import matplotlib.pyplot as plt

from enthought.chaco.example_support import COLOR_PALETTE
from enthought.enable.example_support import DemoFrame, demo_main
import enthought.enable.api as eapi
import enthought.traits.api as tapi
import enthought.traits.ui.api as tuiapi
import enthought.traits.ui.tabular_adapter as tuit
import enthought.chaco.api as capi
import enthought.chaco.tools.api as ctapi
import enthought.pyface.api as papi

from exoreader import ExodusIIReader

EXE = "plot2d"
Change_X_Axis_Enabled = True
LDICT = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
         "asin": math.asin, "acos": math.acos,
         "atan": math.atan, "atan2": math.atan2, "pi": math.pi,
         "log": math.log, "exp": math.exp, "floor": math.floor,
         "ceil": math.ceil, "abs": math.fabs, "random": random.random, }
GDICT = {"__builtins__": None}
EPSILON = np.finfo(np.float).eps
SIZE = (700, 600)
LS = ['dot dash', 'dash', 'dot', 'long dash']
F_EVALDB = "gmd-evaldb.xml"
L_EXO_EXT = [".exo", ".base_exo", ".e"]
L_DAT_EXT = [".out",]
L_REC_EXT = L_EXO_EXT + L_DAT_EXT
XY_PAIRS = {}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()
    for source in args.sources:
        if not os.path.isfile(source):
            logerr("{0}: no such file".format(source))
    if logerr():
        stop("*** error: stopping due to previous errors")

    create_model_plot(args.sources)


class Plot2D(tapi.HasTraits):
    container = tapi.Instance(capi.Plot)
    plot_info = tapi.Dict(tapi.Int, tapi.Dict(tapi.Str, tapi.List(tapi.Str)))
    plot_data = tapi.List(tapi.Array)
    overlay_headers = tapi.Dict(tapi.Str, tapi.List(tapi.Str))
    overlay_plot_data = tapi.Dict(tapi.Str, tapi.Array)
    variables = tapi.List(tapi.Str)
    plot_indices = tapi.List(tapi.Int)
    x_idx = tapi.Int
    y_idx = tapi.Int
    Time = tapi.Float
    high_time = tapi.Float
    low_time = tapi.Float
    time_data_labels = tapi.Dict(tapi.Int, tapi.List)
    runs_shown = tapi.List(tapi.Bool)
    x_scale = tapi.Float
    y_scale = tapi.Float

    traits_view = tuiapi.View(
        tuiapi.Group(
            tuiapi.Item('container', editor=eapi.ComponentEditor(size=SIZE),
                        show_label=False),
            tuiapi.Item('Time', editor=tuiapi.RangeEditor(low_name='low_time',
                                                          high_name='high_time',
                                                          format='%.1f',
                                                          label_width=28,
                                                          mode='auto')),
            orientation="vertical"),
        resizable=True,
        width=SIZE[0], height=SIZE[1] + 100)

    def __init__(self, **traits):
        tapi.HasTraits.__init__(self, **traits)
        self.time_data_labels = {}
        self.x_idx = 0
        self.y_idx = 0
        self.runs_shown = [True] * len(self.variables)
        self.x_scale, self.y_scale = 1., 1.
        pass

    @tapi.on_trait_change('Time')
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
        container = capi.Plot(padding=80, fill_padding=True,
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
        nform = "[%(x).2g, %(y).2g]"
        if self.nfiles() - 1 or self.overlay_plot_data:
            lform = "({0}) {1}".format(self.get_file_name(d), nform)
        else:
            lform = nform
        label = capi.DataLabel(component=self.container, data_point=(xp, yp),
                               label_position="bottom right",
                               border_visible=False,
                               bgcolor="transparent",
                               label_format=lform,
                               marker_color=tuple(COLOR_PALETTE[(d + di) % 10]),
                               marker_line_color="transparent",
                               marker="diamond",
                               arrow_visible=False)

        self.time_data_labels[d].append(label)
        self.container.overlays.append(label)
        return

    def create_plot(self, x, y, di, d, plot_name, ls):
        self.container.data.set_data("x " + plot_name, x)
        self.container.data.set_data("y " + plot_name, y)
        self.container.plot(
            ("x " + plot_name, "y " + plot_name),
            line_width=2.0, name=plot_name,
            color=tuple(COLOR_PALETTE[(d + di) % 10]),
            bgcolor="white", border_visible=True, line_style=ls)
        return

    def change_plot(self, indices, x_scale=None, y_scale=None):
        global XY_PAIRS
        self.plot_indices = indices
        self.container = self.create_container()
        self.high_time = float(max(self.plot_data[0][:, 0]))
        self.low_time = float(min(self.plot_data[0][:, 0]))
        self.container.data = capi.ArrayPlotData()
        self.time_data_labels = {}
        if len(indices) == 0:
            return
        XY_PAIRS = {}
        x_scale, y_scale = self.get_axis_scales(x_scale, y_scale)

        # loop through plot data and plot it
        overlays_plotted = False
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
                self.create_plot(x, y, yp_idx, d, entry, "solid")
                XY_PAIRS.setdefault(fnam, []).append([xname, x, yname, y])

                # create point marker
                xp = self.plot_data[d][ti, xp_idx] * x_scale
                yp = self.plot_data[d][ti, yp_idx] * y_scale
                self.create_data_label(xp, yp, d, yp_idx)

                if not overlays_plotted:
                    # plot the overlay data
                    overlays_plotted = True
                    ls_ = 0
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
                        self.create_plot(xo, yo, yo_idx, d, entry, LS[ls_ % 4])
                        XY_PAIRS.setdefault(fnam, []).append([xname, x, yname, y])
                        ls_ += 1
                        continue

        capi.add_default_grids(self.container)
        capi.add_default_axes(self.container, htitle=mheader[self.x_idx])

        self.container.index_range.tight_bounds = False
        self.container.index_range.refresh()
        self.container.value_range.tight_bounds = False
        self.container.value_range.refresh()

        self.container.tools.append(ctapi.PanTool(self.container))

        zoom = ctapi.ZoomTool(self.container, tool_mode="box", always_on=False)
        self.container.overlays.append(zoom)

        dragzoom = ctapi.DragZoom(self.container, drag_button="right")
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




class ChangeAxisHandler(tuiapi.Handler):

    def closed(self, info, is_ok):
        global Change_X_Axis_Enabled
        Change_X_Axis_Enabled = True


class ChangeAxis(tapi.HasStrictTraits):

    Change_X_Axis = tapi.Button('Change X-axis')
    Plot_Data = tapi.Instance(Plot2D)
    headers = tapi.List(tapi.Str)

    def __init__(self, **traits):
        tapi.HasStrictTraits.__init__(self, **traits)

    def _Change_X_Axis_fired(self):
        global Change_X_Axis_Enabled
        Change_X_Axis_Enabled = False
        ms = SingleSelect(choices=self.headers, plot=self.Plot_Data)
        ms.configure_traits(handler=ChangeAxisHandler())

    view = tuiapi.View(tuiapi.Item('Change_X_Axis',
                                   enabled_when='Change_X_Axis_Enabled==True',
                                   show_label=False))


class SingleSelectAdapter(tuit.TabularAdapter):
    columns = [('Plotable Variables', 'myvalue')]

    myvalue_text = tapi.Property

    def _get_myvalue_text(self):
        return self.item


class SingleSelect(tapi.HasPrivateTraits):
    choices = tapi.List(tapi.Str)
    selected = tapi.Str
    plot = tapi.Instance(Plot2D)

    view = tuiapi.View(
        tuiapi.HGroup(
            tuiapi.UItem('choices',
                         editor=tuiapi.TabularEditor(
                             show_titles=True, selected='selected', editable=False,
                             multi_select=False, adapter=SingleSelectAdapter()))),
        width=224, height=668, resizable=True, title='Change X-axis')

    @tapi.on_trait_change('selected')
    def _selected_modified(self, object, name, new):
        self.plot.change_axis(object.choices.index(object.selected))


class SingleSelectOverlayFilesAdapter(tuit.TabularAdapter):
    columns = [('Overlay File Name', 'myvalue')]

    myvalue_text = tapi.Property

    def _get_myvalue_text(self):
        return self.item


class SingleSelectOverlayFiles(tapi.HasPrivateTraits):
    choices = tapi.List(tapi.Str)
    selected = tapi.Str

    view = tuiapi.View(
        tuiapi.HGroup(
            tuiapi.UItem('choices',
                         editor=tuiapi.TabularEditor(
                             show_titles=True, selected='selected',
                             editable=False, multi_select=False,
                             adapter=SingleSelectOverlayFilesAdapter()))),
        width=224, height=100)


class MultiSelectAdapter(tuit.TabularAdapter):
    columns = [('Plotable Variables', 'myvalue')]

    myvalue_text = tapi.Property

    def _get_myvalue_text(self):
        return self.item


class MultiSelect(tapi.HasPrivateTraits):
    choices = tapi.List(tapi.Str)
    selected = tapi.List(tapi.Str)
    plot = tapi.Instance(Plot2D)

    view = tuiapi.View(
        tuiapi.HGroup(
            tuiapi.UItem('choices',
                         editor=tuiapi.TabularEditor(
                             show_titles=True,
                             selected='selected',
                             editable=False,
                             multi_select=True,
                             adapter=MultiSelectAdapter()))),
        width=224, height=568, resizable=True)

    @tapi.on_trait_change('selected')
    def _selected_modified(self, object, name, new):
        ind = []
        for i in object.selected:
            ind.append(object.choices.index(i))
        self.plot.change_plot(ind)


class ModelPlot(tapi.HasStrictTraits):

    Plot_Data = tapi.Instance(Plot2D)
    plot_info = tapi.Dict(tapi.Int, tapi.Dict(tapi.Str, tapi.List(tapi.Str)))
    Multi_Select = tapi.Instance(MultiSelect)
    Change_Axis = tapi.Instance(ChangeAxis)
    Reset_Zoom = tapi.Button('Reset Zoom')
    Reload_Data = tapi.Button('Reload Data')
    Print_to_PDF = tapi.Button('Print to PDF')
    Load_Overlay = tapi.Button('Open Overlay')
    Close_Overlay = tapi.Button('Close Overlay')
    X_Scale = tapi.String("1.0")
    Y_Scale = tapi.String("1.0")
    Single_Select_Overlay_Files = tapi.Instance(SingleSelectOverlayFiles)
    filepaths = tapi.List(tapi.String)
    file_variables = tapi.List(tapi.String)

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

        tapi.HasStrictTraits.__init__(self, **traits)
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
        if not XY_PAIRS:
            return

        # get the maximum of Y for normalization
        ymax = max(np.amax(np.abs(xy_pair[-1]))
                   for (label, item) in XY_PAIRS.items() for xy_pair in item)

        # setup figure
        plt.figure(0)
        plt.cla()
        plt.clf()

        # plot y value for each plot on window
        ynames = []
        for (fnam, info) in XY_PAIRS.items():
            for (xname, x, yname, y) in info:
                label = fnam + ":" + yname if len(XY_PAIRS) > 1 else yname
                ynames.append(yname)
                plt.plot(x, y / ymax, label=label)
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
        dialog = papi.FileDialog(action="open")
        dialog.open()
        info = {}
        if dialog.return_code == papi.OK:
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
    view = tuiapi.View(tuiapi.HSplit(
        tuiapi.VGroup(
            tuiapi.Item('Multi_Select', show_label=False, width=224,
                        height=668, springy=True, resizable=True),
            tuiapi.Item('Change_Axis', show_label=False), ),
        tuiapi.Item('Plot_Data', show_label=False, width=800, height=768,
                    springy=True, resizable=True)),
                       style='custom', width=1124, height=868,
                       resizable=True, title=window_name)
    return view


def create_model_plot(sources, handler=None, metadata=None):
    """Create the plot window

    Parameters
    ----------

    """
    if not isinstance(sources, (list)):
        stop("*** error: sources must be list of files")

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
            fname, fext = os.path.splitext(source)
            if fext not in L_REC_EXT:
                logerr("{0}: unrecognized file extension".format(source))
                continue
            filepaths.append(source)
        if logerr():
            stop("***error: stopping due to previous errors")
        variables = [""] * len(filepaths)
        runid = "GMD VIZ" if len(filepaths) > 1 else genrunid(filepaths[0])

    view = tuiapi.View(tuiapi.HSplit(
        tuiapi.VGroup(
            tuiapi.Item('Multi_Select', show_label=False),
            tuiapi.Item('Change_Axis', show_label=False),
            tuiapi.Item('Reset_Zoom', show_label=False),
            tuiapi.Item('Reload_Data', show_label=False),
            tuiapi.Item('Print_to_PDF', show_label=False),
            tuiapi.VGroup(
                tuiapi.HGroup(tuiapi.Item("X_Scale", label="X Scale",
                                          editor=tuiapi.TextEditor(
                                              multi_line=False)),
                              tuiapi.Item("Y_Scale", label="Y Scale",
                                          editor=tuiapi.TextEditor(
                                              multi_line=False))),
                show_border=True),
            tuiapi.VGroup(
                tuiapi.HGroup(
                    tuiapi.Item('Load_Overlay', show_label=False, springy=True),
                    tuiapi.Item(
                        'Close_Overlay', show_label=False, springy=True),),
                tuiapi.Item('Single_Select_Overlay_Files', show_label=False,
                            resizable=False), show_border=True)),
        tuiapi.Item('Plot_Data', show_label=False, width=800, height=768,
                    springy=True, resizable=True)),
        style='custom', width=1124, height=868,
        resizable=True, title=runid)

    main_window = ModelPlot(filepaths=filepaths, file_variables=variables)
    main_window.configure_traits(view=view, handler=handler)


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
    """Read in the gmd-tabular.dat file

    """
    from utils.gmdtab import read_gmd_evaldb
    sources, paraminfo, _ = read_gmd_evaldb(source)
    for (i, info) in enumerate(paraminfo):
        paraminfo[i] = ", ".join("{0}={1:.2g}".format(n, v) for (n, v) in info)
    return sources, paraminfo


def loadcontents(filepath):
    if filepath.endswith((".exo", ".e", ".base_exo")):
        exof = ExodusIIReader.new_from_exofile(filepath)
        glob_var_names = exof.glob_var_names()
        elem_var_names = exof.elem_var_names()
        data = [exof.get_all_times()]
        for glob_var_name in glob_var_names:
            data.append(exof.get_glob_var_time(glob_var_name))
        for elem_var_name in elem_var_names:
            data.append(exof.get_elem_var_time(elem_var_name, 0))
        data = np.transpose(np.array(data))
        head = ["TIME"] + glob_var_names + elem_var_names
        exof.close()
    else:
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
