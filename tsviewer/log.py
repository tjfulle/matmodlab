from traits.api import *
from traitsui.api import *

# --- A logging window
class LoggingWindow(HasStrictTraits):
    contents = String
    def __init__(self):
        kwds = {'contents': ''}
        super(LoggingWindow, self).__init__(**kwds)

    traits_view = View(
        Item('contents', style='custom', show_label=False),
        buttons=['OK'],
        title='Material Model Laboratory Log',
        width=800,
        height=600,
        resizable=True)

    def write(self, string):
        self.contents += string + '\n'

    def flush(self, *args):
        pass

    def close(self, *args):
        pass

    def popup(self):
        self.edit_traits()

winstream = LoggingWindow()
