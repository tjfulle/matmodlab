try: from enthought import *
except ImportError: pass
from traits.api import HasTraits, List, Button, String, Dict
from traits.ui.api import View, VGroup, HGroup, Label, UItem, Spring, Handler
from traits.ui.menu import NoButtons


class MessageBox(HasTraits):
    """
    Similar to the traits ui dialog box, this one has slightly better
    formatting, and returns which button was clicked.

    Please use the function message_box(...) below to display the window
    """
    message = String
    button_return_values = Dict(String, String)
    button_ = Button
    result = String


class MessageBoxHandler(Handler):
    def setattr(self, info, object, name, value):
        if 'button' in name:
            object.result = object.button_return_values[name]
            info.ui.dispose()

    def close(self, info, is_ok):
        return


def message_box(message, title='Message', buttons=['OK']):
    button_group = HGroup(padding=10, springy=True)
    view = View(
        VGroup(
            UItem('message', style='readonly'),
            button_group,
            padding=15,
        ),
        title=title,
        width=480,
        kind='livemodal',
        buttons=NoButtons,
        handler=MessageBoxHandler()
    )

    button_return_values = {}
    box = MessageBox(message=message)
    button_group.content.append(Spring())
    for i in range(len(buttons)):
        if buttons[i] is None:
            button_group.content.append(Spring())
        else:
            name = 'button_%d' % i
            box.add_trait(name, Button())
            button_group.content.append(UItem(name, label=buttons[i]))
            button_return_values[name] = buttons[i]
    box.button_return_values = button_return_values

    button_group.content.append(Spring())

    box.configure_traits(view=view)

    return box.result


if __name__ == '__main__':
    print message_box('Do you wish to save?', title='Save your work', buttons=['Cancel', None, 'Yes', 'No'])
