""" Base Module Help

The best way to interface with this package is to use the read_file(),
write_file(), and transform() functions. Example usages are:

  >>> tabfileio.read_file("read_me.txt")

  >>> tabfileio.write_file("write_me.pkl", head, data)

  >>> tabfileio.transform("file_in.xls", "file_out.json")

For more information, see the functions in tabfileio.interface.
"""
from .interface import read_file, write_file, transform

__version__ = "0.1.3"
