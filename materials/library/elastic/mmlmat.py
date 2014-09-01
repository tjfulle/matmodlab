import os
_D = os.path.dirname(os.path.realpath(__file__))
NAME = "elastic"
SOURCE_FILES = [os.path.join(_D, f) for f in
                ("elastic_interface.f90", "elastic.f90", "elastic.pyf")]
INTERFACE = os.path.join(_D, "elastic.py")
CLASS = "Elastic"
