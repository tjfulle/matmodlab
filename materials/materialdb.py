import os
import sys
import xml.dom.minidom as xdom
from xml.parsers.expat import ExpatError

from utils.misc import load_file
import __config__ as cfg

D = os.path.dirname(os.path.realpath(__file__))
def cout(string):
    sys.stdout.write(string + "\n")


class _Material:
    """Class to hold some material meta data"""
    def __init__(self, name, **kwargs):
        self.source_files = None
        self.requires_lapack = False
        self.include_dir = None
        self.python_alternative = None
        self.type = "default"
        self.name = name
        for (k, v) in kwargs.items():
            if k == "class": k = "class_name"
            setattr(self, k, v)

        if kwargs.get("python_model", None) is None:
            self.python_model = not self.source_files
        if self.python_model:
            self.so_lib = None

        elif name != "umat":
            self.so_lib = os.path.join(cfg.PKG_D, self.name + cfg.SO_EXT)

            # assume fortran model if source files are given
            if "abaqus" in self.type:
                self.source_files.append(cfg.ABQIO)
                # determine if signature file is present
                for f in self.source_files:
                    if f.endswith(".pyf"): break
                else:
                    sigf = {"abaqus_umat": cfg.ABQUMAT,
                            "abaqus_uanioshyper_inv": cfg.ABQUAHI}[self.type]
                    self.source_files.append(sigf)
            else:
                self.source_files.append(cfg.FIO)
        pass

    def __getitem__(self, attr):
        return self.__dict__[attr]

    def items(self):
        return self.__dict__.items()

    @property
    def dirname(self):
        return os.path.dirname(self.interface_file)

    @property
    def so_exists(self):
        if self.python_model:
            return True
        return os.path.isfile(self.so_lib)

    @property
    def ext_module(self):
        if not self.so_lib: return
        return os.path.basename(self.so_lib)

    def instantiate_material(self, params, options):
        """Instantiate the material model"""

        mtlmod = load_file(self.interface_file)
        mclass = getattr(mtlmod, self.class_name)

        mat = mclass()
        mat.set_options(**options)
        if options.get("umat") is not None:
            statev = options["umat"]
            mat.setup_umat(params, statev)
            del options["umat"]
        mat.setup_new_material(params)
        mat.set_constant_jacobian()

        return mat


class MaterialDB(object):
    """Holder for material info"""
    _xmldb = None
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
        if name in self._materials:
            return material
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

    def __getitem__(self, mat):
        try: mat = mat.name
        except AttributeError: pass
        for m in self._materials:
            if mat == m.name:
                return m
        raise AttributeError("{0}: not in material database".format(mat))

    def __len__(self):
        return len(self._materials)

    def put(self, material):
        for (i, m) in enumerate(self._materials):
            if m.name == material.name:
                self._materials[i] = material
                break
        else:
            self._materials.append(material)

    def material_from_name(self, name):
        for m in self._materials:
            if m.name == name:
                return m

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

    @classmethod
    def gen_db(cls, search_dirs):
        db = cls.gen_from_search(search_dirs)
        for mat in db:
            # instantiate the material to get param names
            mtlmod = load_file(mat.interface_file)
            mtlmdl = getattr(mtlmod, mat.class_name)
            mat.parse_table = mtlmdl.param_parse_table()
            del mtlmdl
        return db

    @classmethod
    def gen_from_search(cls, search_dirs, mats_to_build="all"):
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
        hasfile = lambda d, f: f in os.listdir(d)
        for dirname in search_dirs:
            if not os.path.isdir(dirname):
                cout("  *** warning: {0}: no such directory".format(dirname))
                continue
            # prefer mmlmat.py, but look for umat.py
            if not any(hasfile(dirname, f) for f in ("umat.py", "mmlmat.py")):
                cout("  *** warning: mmlmat.py not found in {0}".format(dirname))
                continue
            f = "umat.py" if "umat.py" in os.listdir(dirname) else "mmlmat.py"
            filename = os.path.join(dirname, f)
            mmlmat = load_file(filename)
            try:
                name = mmlmat.NAME
            except AttributeError:
                cout("  ***error: {0}: NAME not defined".format(filename))
                continue

            if not build_all and name not in mats_to_build:
                continue

            try:
                conf = mmlmat.conf()
            except ValueError:
                cout("  ***error: {0}: failed to gather "
                     "information".format(filename))
                continue
            except AttributeError:
                cout("  ***error: {0}: conf function not defined".format(filename))
                continue

            materials.append(_Material(name, **conf))

        return cls(*materials)
