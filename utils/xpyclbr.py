import re
import imp
import sys

"""Similar to pyclbr, but only returns mapping of class names and supers"""

regex = r"(?ms)\bclass\s+(\w+)\((.*?)\)"

class Class:
    '''Class to represent a Python class.'''
    def __init__(self, module, name, super, file, attrs):
        self.module = module
        self.class_name = name
        if super is None:
            super = []
        self.super = super
        self.methods = {}
        self.file = file
        for (key, val) in attrs.items():
            setattr(self, key, str(val))

def readmodule(module, path=None, ancestors=None, reqattrs=None):
    dict = {}
    path = path or []
    attrs = reqattrs or []
    ancestors = ancestors or []
    f, fname, (_s, _m, ty) = imp.find_module(module, path + sys.path)
    if ty != imp.PY_SOURCE:
        # not Python source, can't do anything with this module
        f.close()
        return dict
    exists = lambda x: bool(x.strip().split())
    content = f.read()
    for (class_name, inherit) in re.findall(regex, content):
        inherit = [x.strip() for x in inherit.split(",") if exists(x)]
        if ancestors and not any(x in ancestors for x in inherit):
            continue
        # look for specific attributes. ***NOTE*** this function is really
        # only intended for files containing a single class definition, the
        # below would fail for multiple classes in a file
        my_attrs = {}
        for attr in attrs:
            rx = r'self\.{0} = (?P<n>.*)'.format(attr)
            match = re.search(rx, content)
            if match:
                my_attrs[attr] = re.sub(r"[\"\']", "", match.group("n")).strip()
            else:
                raise ValueError("requested attribute {0} not found "
                                 "in {1}.{2}".format(attr, module, class_name))
        dict[class_name] = Class(module, class_name, inherit, fname, my_attrs)
    f.close()
    return dict
