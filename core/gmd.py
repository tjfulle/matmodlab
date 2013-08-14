import os

from utils.errors import Error1
import parser.parser as parser


class ModelDriver(object):

    def __init__(self, solcntrl, mtlid, mtlprops):
        """Initialize the ModelDriver object

        Parameters
        ----------
        solcntrl : ndarray

        material : tuple
            material[0] -> material name
            material[1] -> dict of material properties

        """
        pass

    @classmethod
    def from_input_file(cls, filepath):
        try:
            lines = open(filepath, "r").read()
        except OSError:
            raise errors.Error1("{0}: no such file".format(filepath))
        mm_input = parser.parse_input(lines)

        print mm_input.legs
        print mm_input.mtlid
        print mm_input.mtlprops
