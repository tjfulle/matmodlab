import os
import sys
import glob
import warnings
import logging
import numpy as np
from argparse import ArgumentParser
from os.path import join, dirname, realpath, sep, isfile, basename, split, isdir
warnings.simplefilter("ignore")

try:
    from traits.etsconfig.api import ETSConfig
except ImportError:
    raise SystemExit('traitsui not found')
toolkit = os.getenv('ETS_TOOLKIT', 'qt4')
ETSConfig.toolkit = toolkit
os.environ['ETS_TOOLKIT'] = toolkit

from chaco.api import *
from traits.api import *
from traitsui.api import *
from chaco.tools.api import *
from enable.api import ComponentEditor
from pyface.api import FileDialog, OK as pyOK
from chaco.example_support import COLOR_PALETTE
from traitsui.tabular_adapter import TabularAdapter
from traitsui.menu import MenuBar, Menu, Action, NoButtons

from .viewer import *
from .infopane import *
from .log import winstream

icns = join(dirname(realpath(__file__)), 'icon')

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = ArgumentParser()
    parser.add_argument("sources", nargs="*")
    args = parser.parse_args(argv)
    sources = []
    errors = 0
    for source in args.sources:
        filename = realpath(source)
        if not isfile(filename):
            # check for known extensions
            filename = filename.rstrip(".")
            for ext in ('.dbx', '.exo', '.base_exo', '.base_dat', '.dat', '.out'):
                if isfile(filename + ext):
                    filename = filename + ext
                    break
            else:
                logging.error("{0}: no such file".format(source))
                errors += 1
                continue
        sources.append(filename)
    if errors:
        raise ValueError('stopping due to previous errors')

    launch(sources)

class Application(HasStrictTraits):
    plot = Instance(MMLPostViewer)
    xaxis = Instance(SingleSelect)
    yaxis = Instance(MultiSelect)
    xscale = String("1.0")
    yscale = String("1.0")
    ipane = Instance(InfoPane)
    logbutton = Button('Log')
    can_fire = Bool(True)
    plotscale = Button(image=join(icns,'plot-scale.png'), style='toolbar')
    time = Float(0.)
    low = Float(0.)
    high = Float(0.)
    frame_first = Button(image=join(icns,'frame-first.png'), style='toolbar')
    frame_prev = Button(image=join(icns,'frame-previous.png'), style='toolbar')
    frame_next = Button(image=join(icns,'frame-next.png'), style='toolbar')
    frame_last = Button(image=join(icns,'frame-last.png'), style='toolbar')
    plotscale = Button(image=join(icns,'plot-scale.png'), style='toolbar')
    plotp = Button(image=join(icns,'camera.png'), style='toolbar')
    plotl = Button(image=join(icns,'legend.png'), style='toolbar')
    plotr = Button(image=join(icns,'random.png'), style='toolbar')

    def __init__(self, sources):
        """Put together information to be sent to MMLPostViewer information
        needed:

        variables : list
           list of variables that changed from one simulation to another
        x_idx : int
           column containing x variable to be plotted

        """
        traits = {"ipane": self.init(sources)}
        HasStrictTraits.__init__(self, **traits)
        self.plot = MMLPostViewer(ipane=self.ipane)
        self.xaxis = SingleSelect(plot=self.plot)
        self.yaxis = MultiSelect(choices=self.plot.choices, plot=self.plot)

    def init(self, sources):

        if not sources:
            return InfoPane(owner=self)

        filepaths = []
        for source in sources:
            if not isfile(source):
                raise IOError("{0}: no such file".format(source))

            if source.endswith('.edb'):
                d = dirname(source)
                if len(sources) > 1:
                    raise ValueError('only one evaluation db allowed')
                filepaths, variables = readtabular(source)
                names = dict([(f, f.replace(d, '')) for f in filepaths])
                break
            else:
                filepaths.append(source)
                variables, names = {}, {}

        files = [OutputDB(f, info=variables.get(f,''), name=names.get(f))
                 for f in filepaths]
        ipane = InfoPane(files=files, owner=self)
        return ipane

    def display_log(self):
        winstream.popup()

    def _xscale_changed(self, scale):
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
            scale = self.xscale = "1.0"
        self.plot.set_xscale(scale)
        self.update()
        return

    def _yscale_changed(self, scale):
        """See _xscale_changed"""
        scale = scale.strip()
        if not scale:
            scale = self.yscale = "1.0"
        self.plot.set_yscale(scale)
        self.update()
        return

    def onadd(self):
        """File added, update the view"""
        self.plot.update(self.ipane.choices)
        self.update()
        return

    def update(self):
        """File added, update the view"""
        self.plot.change_plot()
        return

    def onreload(self):
        """File reloaded, update the view"""
        self.update()
        return

    def adjust_plot_scales(self):
        self.plot.xyscales.edit_traits(view='edit_view')

    def _plotscale_fired(self):
        self.adjust_plot_scales()

    def onremove(self):
        """File added, update the view"""
        if not self.ipane.outputdbs:
            self.plot.update([])
            self.plot.change_plot([])
        else:
            self.update()
        return

    def adjust_random_coloring(self, rand=None):
        if rand is None:
            rand = not self.plot.rand
        self.plot.rand = rand
        self.update()
        return

    def random_coloring(self):
        self.adjust_random_coloring(rand=True)

    def uniform_coloring(self):
        self.adjust_random_coloring(rand=False)

    def _plotr_fired(self):
        self.adjust_random_coloring()

    def _plotl_fired(self):
        self.adjust_legend_visibility()

    def hide_legend(self):
        self.adjust_legend_visibility(visible=False)

    def show_legend(self):
        self.adjust_legend_visibility(visible=True)

    def adjust_legend_visibility(self, visible=None):
        if self.plot.container is None:
            return
        if visible is None:
            visible = not self.plot.legend_visible
        self.plot.legend_visible = visible
        self.plot.container.legend.visible = visible
        self.plot.container.invalidate_and_redraw()

    def _plotp_fired(self):
        self.print_screen()

    def print_screen(self):
        """Open file"""
        untitled = 'Untitled'
        files = glob.glob('*.png')
        i = 1
        while 1:
            default = untitled + ' {0}.png'.format(i)
            if default not in files:
                break
            i += 1
        wildcard = ('Screen Capture (*.png)|*.png|')
        dialog = FileDialog(action="save as", wildcard=wildcard,
                            default_filename=default)
        if dialog.open() != pyOK:
            return
        if not dialog.filename:
            return
        filename = dialog.filename
        if not filename.endswith('.png'):
            filename += '.png'

        width, height = self.plot.container.outer_bounds
        self.plot.container.do_layout(force=True)
        gc = PlotGraphicsContext((width, height), dpi=72)
        gc.render_component(self.plot.container)
        gc.save(filename, file_format='PNG')
        return


    @on_trait_change('time')
    def _time_up(self):
        self.plot.Time = self.time

    @on_trait_change('plot.times')
    def _times_up(self):
        if self.plot.times:
            self.time = self.plot.times[0]
            self.low = min(self.plot.times)
            self.high = max(self.plot.times)
        else:
            self.time = self.low = self.high = 0.

    def _frame_first_fired(self):
        self.plot.update(frame=0)
        self.time = self.plot.Time

    def _frame_prev_fired(self):
        self.plot.update(dframe=-1)
        self.time = self.plot.Time

    def _frame_next_fired(self):
        self.plot.update(dframe=1)
        self.time = self.plot.Time

    def _frame_last_fired(self):
        self.plot.update(frame=-1)
        self.time = self.plot.Time

    def open_file(self):
        self.ipane.open_outputdb()

class ApplicationHandler(Handler):

    legend_visible = Bool(True)
    randomly_colored = Bool(True)

    def open_file(self, info):
        info.object.open_file()

    def random_coloring(self, info):
        info.object.random_coloring()
        self.randomly_colored = True

    def uniform_coloring(self, info):
        info.object.uniform_coloring()
        self.randomly_colored = False

    def hide_legend(self, info):
        info.object.hide_legend()
        self.legend_visible = False

    def show_legend(self, info):
        info.object.show_legend()
        self.legend_visible = True

    def quit(self, info):
        info.ui.dispose()
        raise SystemExit(0)

    def print_screen(self, info):
        info.object.print_screen()

    def closed(self, info, is_ok):
        raise SystemExit(0)

def launch(sources=None):
    """Create the plot window

    Parameters
    ----------

    """
    if sources is None:
        sources = []

    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    h = -20
    info_pane = Item('ipane', show_label=False, resizable=False)
    plot_window = VGroup(
        HGroup(
            Item("xaxis", show_label=False),
            Item('plotscale', show_label=False),
            Item('plotl', show_label=False),
            Item('plotr', show_label=False),
            Item('plotp', show_label=False),
            spring,
            Item('time', show_label=False,
                 editor=RangeEditor(low_name='low', high_name='high',
                                    format='%.2f', mode='slider')),
            Item('frame_first', show_label=False),
            Item('frame_prev', show_label=False),
            Item('frame_next', show_label=False),
            Item('frame_last', show_label=False),
            ),
            Item('plot', show_label=False, springy=True, resizable=True,
                 width=900, height=600))

    title = "Time Series Viewer"
    ms = Item('yaxis', show_label=False, height=.75)
    view = View(HSplit(VSplit(info_pane, ms), plot_window),
                style='custom', resizable=True, title=title)
    main_window = Application(sources=sources)
    main_window.configure_traits(view=view, handler=ApplicationHandler)
    return main_window

if __name__ == "__main__":
    main()
