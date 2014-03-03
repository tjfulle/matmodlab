import os
import sys
import xml.dom.minidom as xdom
from xml.parsers.expat import ExpatError

from __config__ import F_MTL_MODEL_DB, UMATS
from core.io import Error1
from utils.impmod import load_file

D = os.path.dirname(os.path.realpath(__file__))
def cout(string):
    sys.stdout.write(string + "\n")


class _Material:
    """Class to hold some material meta data"""
    def __init__(self, name, **kwargs):
        self.source_files = None
        self.requires_lapack = False
        self.include_dir = None
        self.name = name
        for (k, v) in kwargs.items():
            if k == "class": k = "class_name"
            setattr(self, k, v)
        self.python_model = not self.source_files

    def __getitem__(self, attr):
        return self.__dict__[attr]

    def items(self):
        return self.__dict__.items()

    @property
    def dirname(self):
        return os.path.dirname(self.interface_file)

    def instantiate_material(self, params, options):
        """Instantiate the material model"""

        mtlmod = load_file(self.interface_file)
        mclass = getattr(mtlmod, self.class_name)

        mat = mclass()
        mat.setup_new_material(params)
        mat.set_constant_jacobian()
        mat.set_options(**options)

        return mat

    def create_material(matname, matparams, matopts):
        """Create a material object from the material name

        """
        # Instantiate the material object
        material = self.get_material_from_db(matname)
        if material is None:
            return
        mtlmod = load_file(material.interface_file)
        mclass = getattr(mtlmod, material.class_name)



class MaterialDB(object):
    """Holder for material info"""
    _xmldb = None
    db_file = F_MTL_MODEL_DB
    def __init__(self, *materials):
        """

        Parameters
        ----------
        materials : list of _Material instances

        """
        self._materials = []
        for m in materials:
            if not isinstance(m, _Material):
                raise TypeError("Materials must be _Material type")
            if m in self._materials:
                raise AttributeError("{0}: duplicate material name".format(m.name))
            self._materials.append(m)

    def __repr__(self):
        return "MaterialDB({0})".format(", ".join(m.name for m in self._materials))

    def __getitem__(self, name):
        m = self.get(name)
        if m is None:
            raise ValueError("{0}: not in DB".format(name))
        return m

    def get(self, name):
        for material in self._materials:
            if name == material.name:
                return material

    def remove(self, material):
        try:
            self._materials.remove(material)
        except ValueError:
            raise ValueError("MaterialDB.remove(material): material not in DB")

    def __iter__(self):
        return iter(self._materials)

    def __getitem__(self, name):
        for m in self._materials:
            if name == m.name:
                return m

    def __len__(self):
        return len(self._materials)

    def put(self, material):
        for (i, m) in enumerate(self._materials):
            if m.name == material.name:
                self._materials[i] = material
                break
        else:
            self._materials.append(material)

    @property
    def interface_files(self):
        """Path to interface files"""
        return [m.interface_file for m in self._materials]

    @property
    def materials(self):
        """Path to interface files"""
        return [m.name for m in self._materials]

    @property
    def path(self):
        """Directory names of all materials, can be used to set sys.path"""
        return [m.dirname for m in self._materials]

    def get_material_from_db(cls, matname):
        matname = matname.lower()
        mtldb = cls.gen_from_xmldb()
        for m in mtldb.materials:
            if m.name == matname:
                return m

    def write_db(self, wipe=False):
        """Write the F_MTL_MODEL_DB database file

        """
        if wipe and os.path.isfile(F_MTL_MODEL_DB):
            os.remove(F_MTL_MODEL_DB)

        # read in previously created db since it may differ from the current db
        mats_from_db = _read_db()
        if not mats_from_db:
            mats_from_db = []

        for mat in mats_from_db:
            self.put(mat)

        doc = xdom.Document()
        root = doc.createElement("Materials")

        for material in self._materials:
            # create element
            child = doc.createElement("Material")
            for (key, val) in material.items():
                child.setAttribute(key, "{0}".format(val))
            root.appendChild(child)

        doc.appendChild(root)
        doc.writexml(open(F_MTL_MODEL_DB, "w"), addindent="  ", newl="\n")
        doc.unlink()
        return

    @classmethod
    def gen_from_xmldb(cls):
        materials = _read_db()
        return cls(*materials)

    @classmethod
    def gen_from_search(cls, mats_to_build="all"):
        """Gather all of the matmodlab materials

        Parameters
        ----------
        mats_to_build : list or str
          list of materials to build, or 'all' if all materials are to be built

        """
        build_all = mats_to_build == "all"

        materials = []

        # --- builtin materials are all described in the mmats file
        mmats = load_file(os.path.join(D, "library/mmats.py"))
        for name in mmats.NAMES:
            if not build_all and name not in mats_to_build:
                continue
            materials.append(mmats.conf(name))

        # --- user materials
        for dirname in UMATS:
            if not os.path.isdir(dirname):
                cout("  *** warning: {0}: no such directory".format(dirname))
                continue
            if "umat.py" not in os.listdir(dirname):
                cout("  *** warning: umat.py not found in {0}".format(dirname))
                continue
            filename = os.path.join(dirname, "umat.py")
            umat = load_file(filename)
            try:
                name = umat.NAME
            except AttributeError:
                cout("  ***error: {0}: NAME not defined".format(filename))
                continue

            if not build_all and name not in mats_to_build:
                continue

            try:
                conf = umat.conf()
            except ValueError:
                cout("  ***error: {0}: failed to gather "
                     "information".format(filename))
                continue
            except AttributeError:
                cout("  ***error: {0}: conf function not defined".format(filename))
                continue

            materials.append(_Material(name, **conf))

        return cls(*materials)

def _read_db():
    """Read the material xml db

    """
    materials = []
    try:
        doc = xdom.parse(F_MTL_MODEL_DB)
    except IOError:
        return
    mats = doc.getElementsByTagName("Materials")[0]
    for mtl in mats.getElementsByTagName("Material"):
        name = str(mtl.attributes.getNamedItem("name").value).lower()
        ad = {}
        for (key, val) in mtl.attributes.items():
            if key == "name": continue
            try:
                ad[key] = eval(val)
            except (NameError, SyntaxError):
                ad[key] = "{0}".format(val)

        # instantiate the material to get param names
        mtlmod = load_file(ad["interface_file"])
        mtlmdl = getattr(mtlmod, ad["class_name"])
        ad["parse_table"] = mtlmdl.param_parse_table()
        del mtlmdl
        materials.append(_Material(name, **ad))
    doc.unlink()
    return materials

