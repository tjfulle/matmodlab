import numpy as np
from collections import OrderedDict

from utils.errors import MatModLabError
from utils.constants import DEFAULT_TEMP, NSYMM

class SingleLeg(object):
    def __init__(self, start_time, time_step, nc, control, components,
                 num_steps, num_io_dumps, elec_field, temp, user_field):
        self.start_time = start_time
        self.dtime = time_step
        self.termination_time = self.start_time + self.dtime
        self.num_steps = num_steps or 1
        self.num_dumps = 1000000 if num_io_dumps == "all" else num_io_dumps

        n, N = len(control), len(components)
        if n != N or n != nc:
            raise MatModLabError("expected len(components)="
                                 "len(control)={0}".format(nc))
        self.components = np.array(components, dtype=np.float64)
        self.control = np.array(control, dtype=np.int)

        if elec_field is None or len(elec_field) == 0:
            elec_field = np.zeros(3)

        n = len(elec_field)
        if n != 3:
            raise MatModLabError("expected len(elec_field)==3, got {0}".format(n))
        self.elec_field = np.array(elec_field)

        if temp is None:
            temp = DEFAULT_TEMP
        self.temp = temp

        if user_field is None:
            user_field = []
        self.user_field = np.array(user_field)

    @classmethod
    def zero_leg(cls, nc=NSYMM):
        return cls(0, 0, nc, np.ones(nc)*2, np.zeros(nc), 1, 1, None, None, None)

class LegRepository(OrderedDict):

    def __init__(self, legs):
        super(LegRepository, self).__init__()
        j = 0
        for (i, leg) in enumerate(legs):
            if i == 0 and leg.termination_time > 0.:
                self[i] = SingleLeg.zero_leg()
                j = 1
            self[i+j] = leg
