from os import remove
from os.path import splitext, isfile

import numpy as np
from femlib.data import *
from femlib.mesh import Mesh
from femlib.constants import *

import tabfileio

__all__ = ['File']

__version__ = (0, 1, 0)

def File(filename, mode='r'):
    if mode not in 'wr':
        raise ValueError('unknown File mode {0}'.format(mode))
    if mode == 'r':
        return TabularFileReader(filename)
    return TabularFileWriter(filename)

class _TabularFile(object):
    mode = None
    def view(self):
        from matmodlab.viewer import launch_viewer
        launch_viewer([self.filename])

    def __del__(self):
        self.close()

    def close(self):
        tabfileio.write_file(self.filename, self.head, self.data)
        pass

class TabularFileWriter(_TabularFile):
    mode = 'w'
    def __init__(self, filename):
        self.filename = filename
        self.head = []
        self.data = []
        self.initialized = False

    def put_steps(self, steps):
        for step in steps:
            self.put_step(step)
        self.close()

    def initialize(self, dimension, num_node, nodes, vertices,
                   num_elem, elements, connect, element_blocks, fields=None):
        self.initialized = True

    def put_step(self, step):
        assert self.initialized

        if self.head == []:
            self.head = ["TIME", "INCREMENT"]
            for fo in step.frames[0].field_outputs.values():
                for label in fo.labels:
                    if label == 1:
                        lname = fo.name
                    else:
                        lname = "{0:s}_{1:d}".format(fo.name, label)

                    if fo.type != SCALAR:
                        for component_label in fo.component_labels:
                            self.head.append(lname + "_" + component_label)
                    else:
                        self.head.append(lname)


        for (i, frame) in enumerate(step.frames, start=1):
            tmpdata = [frame.time + frame.increment, frame.increment]
            for (j, fo) in enumerate(frame.field_outputs.values(), start=1):
                for val in fo.data:
                    if hasattr(val, '__iter__'):
                        tmpdata = tmpdata + list(val)
                    else:
                        tmpdata.append(val)
            self.data.append(tmpdata)


class TabularFileReader(_TabularFile):
    mode = 'r'
    def __init__(self, filename):
        raise NotImplementedError("TabularFileReader has not been implemented yet.")
