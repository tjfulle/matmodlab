import os
import datetime
import numpy as np
import scipy.io.netcdf as nc

#from __config__ import __version__
__version__ = (2, 0, 0)
from exoconst import *


GET = 1
PUT = 2


class BadActionError(Exception):
    def __init__(self, action):
        self.message = "{0}: invalid action. expected one of 1, 2".format(action)
        super(BadActionError, self).__init__(self.message)


class NotYetImplemented(Exception):
    def __init__(self, meth):
        self.message = "{0}: ExodusIIFile method not yet implemented".format(meth)
        super(BadActionError, self).__init__(self.message)

class ExodusIIFileError(Exception):
    pass


class ExodusIIFile(object):
    """Exodus output databse manager

    """
    def open_db(self, filepath, mode="r"):
        """Open the netcdf database file"""
        if mode not in "rw":
            raise ExodusIIFileError("{0}: bad read/write mode".format(mode))
        return nc.netcdf_file(filepath, mode)

    def ex_gp_init(self, action, *args):
        """Read/write initial data"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == PUT:
            (title, num_dim, num_nodes, num_elem, num_el_blk,
             num_node_sets, num_side_sets) = args

            self.db.title = title

            # Create required dimensions
            self.db.createDimension("num_dim", num_dim)
            self.db.createDimension("num_nodes", num_nodes)
            self.db.createDimension("num_elem", num_elem)

            num_el_blk = max(num_el_blk, 1)
            self.db.createDimension("num_el_blk", num_el_blk)

            if num_node_sets:
                self.db.createDimension("num_node_sets", num_node_sets)

            if num_side_sets:
                self.db.createDimension("num_side_sets", num_node_sets)

            # element block meta data
            self.db.createVariable("eb_status", "i", ("num_el_blk",))
            self.db.variables["eb_status"][:] = np.array([1], dtype=np.int32)
            self.db.createVariable("eb_prop1", "i", ("num_el_blk",))
            self.db.variables["eb_prop1"][:] = np.array([1], dtype=np.int32)
            self.db.createVariable("eb_names", "c", ("num_el_blk", "len_string"))
            self.db.variables["eb_names"][0][:] = "block 1"
        return

    def ex_gp_time(self, action, *args):
        """Read/write time at time step"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        name = "time_whole"
        time_step = args[0]
        if action == GET:
            if time_step is None:
                return self.db.variables[name].data[:]
            return self.db.variables[name].data[time_step]

        time_value = args[1]
        self.db.variables[name][time_step] = time_value
        return

    def ex_gp_coord_names(self, action, *args):
        """Put the coordinate names in the ExodusII database"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == PUT:
            coord_names = args[0]
            num_dim = self.db.dimensions.get("num_dim", -1)
            num_nodes = self.db.dimensions.get("num_nodes", -1)
            if len(coord_names) != num_dim:
                raise ExodusIIFileError("len(coord_names) != num_dim")

            self.db.createVariable("coor_names", "c", ("num_dim", "len_string"))
            for (i, coor_name) in enumerate(coord_names):
                self.db.variables["coor_names"][i][:] = coor_name
                self.db.createVariable(coor_name.lower(), "d", ("num_nodes",))
        return

    def ex_gp_coord(self, action, *args):
        """Put nodal coordinates in the ExodusII database"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        name = "coor_names"
        if action == PUT:
            coords = args
            num_dim = self.db.dimensions.get("num_dim", -1)
            if len(coords) != num_dim:
                raise ExodusIIFileError("len(coords) != num_dim")

            for (i, coord) in enumerate(coords):
                coord_name = self.chara_to_text(self.db.variables[name][i])
                self.db.variables[coord_name.lower()][:] = coords[i][:]

        return

    def ex_gp_elem_block(self, action, *args):
        """Put element block information in to the ExodusII database"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        namee = "num_el_in_blk"
        namen = "num_nod_per_el"
        if action == GET:
            # Array of the element blocks IDs. The order of the IDs in this
            # array reflects the sequence that the element blocks were
            # introduced into the file.
            elem_blk_params = {}
            i = 0
            for (dim, n) in self.db.dimensions.items():
                if namee not in dim:
                    continue
                elem_blk_id = int(dim.lstrip(namee))
                elem_blk_params[elem_blk_id] = {}
                elem_blk_params[elem_blk_id]["NEL"] = n
                nnod = self.db.dimensions[namen + str(elem_blk_id)]
                elem_blk_params[elem_blk_id]["NNOD"] = nnod
                elem_blk_params[elem_blk_id]["O"] = i
                i += 1
            return elem_blk_params

        (elem_blk_id, elem_type, num_elem_this_blk,
         num_nodes_per_elem, num_attr) = args
        if num_attr:
            raise ExodusIIFileError("attributes not yet supported")

        num_el_blk = self.db.dimensions.get("num_el_blk", -1)
        if elem_blk_id > num_el_blk:
            raise ExodusIIFileError("number element blocks exceeded")

        # dimensioning
        self.db.createDimension(namee + str(elem_blk_id), num_elem_this_blk)
        self.db.createDimension(namen + str(elem_blk_id), num_nodes_per_elem)
        return

    def ex_gp_elem_conn(self, action, *args):
        """Put this elment block's connectivity into the ExodusII database"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == PUT:
            elem_blk_id, blk_conn = args
            num_el_blk = self.db.dimensions.get("num_el_blk", -1)
            if elem_blk_id > num_el_blk:
                raise ExodusIIFileError("number element blocks exceeded")

            # dimensions
            num_el_in_blk_char = "num_el_in_blk{0}".format(elem_blk_id)
            num_el_in_blkI = self.db.dimensions.get(num_el_in_blk_char, -1)

            num_nod_per_el_char = "num_nod_per_el{0}".format(elem_blk_id)
            num_nod_per_elI = self.db.dimensions.get(num_nod_per_el_char, -1)

            ID = elem_blk_id - 1
            CONN = blk_conn[ID] + 1
            if len(CONN) != num_nod_per_elI:
                raise ExodusIIFileError(
                    "expected {0} nodes in element block {1}, "
                    "got {2}".format(num_nod_per_elI, elem_blk_id, len(blk_conn)))

            # connectivity
            var = "connect{0}".format(elem_blk_id)
            self.db.createVariable(var, "i",
                                   (num_el_in_blk_char, num_nod_per_el_char))
            self.db.variables[var][ID, :] = CONN

        return

    def ex_gp_qa(self, action, *args):
        """Reads/writes the QA records to the database."""
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == PUT:
            num_qa_records, qa_records = args
            self.db.createDimension("num_qa_rec", num_qa_records)
            self.db.createVariable("qa_records", "c",
                                   ("num_qa_rec", "four", "len_string"))
            for (i, qa_record) in enumerate(qa_records):
                self.db.variables["qa_records"][i, 0, :] = qa_record[0]
                self.db.variables["qa_records"][i, 1, :] = qa_record[1]
                self.db.variables["qa_records"][i, 2, :] = qa_record[2]
                self.db.variables["qa_records"][i, 3, :] = qa_record[3]
        return

    def ex_gp_info(self, action, *args):
        """Reads/writes information records to the database"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == PUT:
            info = args[0]
            for (i, info_record) in enumerate(info):
                var = "info_rec_{0}".format(i)
                setattr(self.db, var, info_record[:MAX_LINE_LENGTH])
        return

    def ex_gp_var_param(self, action, *args):
        """Reads/writes the number of global, nodal, nodeset, sideset, or element
        variables """
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == PUT:
            var_type, num_vars = args
            if var_type == "G":
                dim  = "num_glo_var"
            elif var_type == "E":
                dim = "num_elem_var"
            elif var_type == "N":
                dim = "num_nod_var"
            else:
                raise ExodusIIFileError("{0}: unrecognized variable "
                                        "type".format(var_type))
            self.db.createDimension(dim, num_vars)
        return

    def ex_gp_var_names(self, action, *args):
        """Read/write names of the results variables from/to the database"""
        if action not in (GET, PUT):
            raise BadActionError(action)

        var_type = args[0]
        # create variable to hold names
        if var_type == "G":
            dim = "num_glo_var"
            name = "name_glo_var"
            var = "vals_glo_var"
        elif var_type == "N":
            dim = "num_nod_var"
            name = "name_nod_var"
            var = "vals_nod_var{0}"
        elif var_type == "E":
            dim = "num_elem_var"
            name = "name_elem_var"
            var = "vals_elem_var{0}eb{1}"
        else:
            raise ExodusIIFileError("{0}: unrecognized variable "
                                    "type".format(var_type))
        if action == GET:
            if args[1:]:
                return self.db.dimensions[dim]
            return self.chara_to_text(self.db.variables[name].data, arr=1)

        num_vars, var_names = args[1:]
        self.db.createVariable(name, "c", (dim, "len_string"))
        # store the names
        for (i, var_name) in enumerate(var_names):
            self.db.variables[name][i, :] = var_name
            if var_type == "G":
                self.db.createVariable(var, "d", ("time_step", dim))
            elif var_type == "N":
                _var = var.format(i+1)
                self.db.createVariable(_var, "d", ("time_step", "num_nodes"))
            elif var_type == "E":
                for j in range(self.db.dimensions["num_el_blk"]):
                    _var = var.format(i+1, j+1)
                    dim = ("time_step", "num_el_in_blk{0}".format(j+1))
                    self.db.createVariable(_var, "d", dim)
        return

    def ex_gp_elem_var_tab(self, action, *args):
        """Reads/write element variable table"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == GET:
            raise NotYetImplemented("get_elem_var_tab")

        num_elem_blk, num_elem_var, elem_var_tab = args
        if num_elem_blk != self.db.dimensions.get("num_el_blk", -1):
            raise ExodusIIFileError("wrong num_elem_blk")
        if num_elem_var != self.db.dimensions.get("num_elem_var", -1):
            raise ExodusIIFileError("wrong num_elem_var")
        self.db.createVariable("elem_var_tab", "i",
                               ("num_el_blk", "num_elem_var"))
        for i in range(self.db.dimensions["num_el_blk"]):
            self.db.variables["elem_var_tab"][i] = elem_var_tab[i]
        return

    def ex_gp_glob_vars(self, action, *args):
        """Reads/writes values of global variables for a single time step."""
        if action not in (PUT, GET):
            raise BadActionError(action)

        name = "vals_glo_var"
        time_step = args[0]
        if action == GET:
            if time_step is None:
                glo_var_name = args[1]
                glo_var_idx = self.get_var_index("G", glo_var_name)
                return self.db.variables[name].data[:, glo_var_idx]
            disp = args[1]
            try:
                var_values = self.db.variables[name].data[time_step]
            except IndexError:
                raise ExodusIIFileError("Error getting global variables at "
                                        "step {0}".format(time_step))
            if not disp:
                return var_values
            return dict(zip(self.glob_var_names(), var_values))

        num_glo_var, val_glo_var = args[1:]
        self.db.variables[name][time_step, :num_glo_var] = val_glo_var
        return

    def ex_gp_nodal_var(self, action, *args):
        """Reads/writes nodal variable for a single time step"""
        if action not in (PUT, GET):
            raise BadActionError(action)

        if action == PUT:
            time_step, nodal_var_index, num_nodes, nodal_var_vals = args
            name = "vals_nod_var{0}".format(nodal_var_index+1)
            self.db.variables[name][time_step, :num_nodes] = nodal_var_vals
        return

    def ex_gp_elem_var(self, action, *args):
        """Reads/writes element variable for a single time step"""
        if action not in (PUT, GET):
            raise BadActionError(action)
        time_step, elem_var_index, elem_blk_id, num_elem_this_blk = args[:4]
        name = "vals_elem_var{0}eb{1}".format(elem_var_index+1, elem_blk_id)
        if action == GET:
            if time_step is None:
                # num_elem_this_blk -> element number
                return self.db.variables[name].data[:, num_elem_this_blk]
            return self.db.variables[name].data[time_step, :num_elem_this_blk]

        elem_var_vals = args[4]
        self.db.variables[name][time_step, :num_elem_this_blk] = elem_var_vals
        return

    def close(self):
        self.db.close()

    def update(self):
        pass

    def get_vara_text(self, var_name):
        vara_text = self.db.variables.get(var_name)
        if var_name is None:
            return
        return np.array(["".join(s for s in row if s.split())
                         for row in vara_text])

    @staticmethod
    def _vara_to_text(vara):
        return "".join(s for s in vara if s.split())

    @staticmethod
    def chara_to_text(chara, arr=0):
        _join = lambda a: "".join(s for s in a if s.split())
        if not arr:
            return _join(chara)
        return [_join(x) for x in chara]

    @classmethod
    def new_from_runid(cls, runid):
        exof = cls(runid)
        return exof


class ExodusIIWriter(ExodusIIFile):
    """Defines pass through methods to the parent class"""
    def __init__(self, runid, d=None):
        d = os.getcwd() if d is None else d
        filepath = os.path.join(d, runid + ".exo")
        self.db = self.open_db(filepath, mode="w")

        self.db = nc.netcdf_file(filepath, "w")
        self.runid = os.path.splitext(os.path.basename(filepath))[0]
        self.filename = os.path.basename(filepath)
        self.db.filename = self.filename
        self.db.runid = self.runid

        # standard ExodusII dimensioning
        self.db.createDimension("time_step", None)
        self.db.createDimension("len_string", MAX_STR_LENGTH+1)
        self.db.createDimension("len_line", MAX_LINE_LENGTH+1)
        self.db.createDimension("four", 4)
        self.db.version = 5.0300002

        # initialize internal variables
        # time
        self.db.createVariable("time_whole", "d", ("time_step",))
        pass

    def put_init(self, title, num_dim, num_nodes, num_elem,
                 num_el_blk, num_node_sets, num_side_sets):
        """Put initial data in to the ExodusII database

        """
        self.ex_gp_init(PUT, title, num_dim, num_nodes, num_elem,
                        num_el_blk, num_node_sets, num_side_sets)
        return

    def put_time(self, time_step, time_value):
        """Writes the time value for a specified time step.

        Parameters
        ----------
        time_step : int
            The time step number.
            This is essentially a counter that is incremented only when
            results variables are output to the data file. The first time step
            is 1.

        time_value : float
            The time at the specified time step.

        """
        self.ex_gp_time(PUT, time_step, time_value)

    def put_coord_names(self, coord_names):
        """Writes the names of the coordinate arrays to the database.

        Parameters
        ----------
        coord_names : array_like
            Array containing num_dim names (of length MAX_STR_LENGTH) of the
            nodal coordinate arrays.

        """
        self.ex_gp_coord_names(PUT, coord_names)
        return

    def put_coord(self, *coords):
        """Write the names of the coordinate arrays

        Parameters
        ----------
        x, y, z : array_like
            x, y, z coordinates

        """
        self.ex_gp_coord(PUT, *coords)
        return

    def put_elem_block(self, elem_blk_id, elem_type, num_elem_this_blk,
                       num_nodes_per_elem, num_attr):
        """Write parameters used to describe an element block

        Parameters
        ----------
        elem_blk_id : int
            The element block ID.

        elem_type : str
            The type of elements in the element block. The maximum length of
            this string is MAX_STR_LENGTH. For historical reasons, this
            string should be all upper case.

        num_elem_this_blk : int
            The number of elements in the element block.

        num_nodes_per_elem : int
            Number of nodes per element in the element block

        num_attr : int
            The number of attributes per element in the element block.

        """
        self.ex_gp_elem_block(PUT, elem_blk_id, elem_type, num_elem_this_blk,
                              num_nodes_per_elem, num_attr)
        return

    def put_elem_conn(self, elem_blk_id, blk_conn):
        """writes the connectivity array for an element block

        Parameters
        ----------
        elem_blk_id : int
            The element block ID

        connect : array_like
            Connectivity array, list of nodes that define each element in the
            block

        """
        self.ex_gp_elem_conn(PUT, elem_blk_id, blk_conn)
        return

    def put_qa(self, num_qa_records, qa_records):
        """Writes the QA records to the database.

        Parameters
        ----------
        num_qa_records : int
            Then number of QA records

        qa_record : array_like, (num_qa_records, 4)
            Array containing the QA records

        Notes
        -----
        Each QA record contains for MAX_STR_LENGTH-byte character strings. The
        character strings are

          1) the analysis code name
          2) the analysis code QA descriptor
          3) the analysis date
          4) the analysis time

        """
        self.ex_gp_qa(PUT, num_qa_records, qa_records)
        return

    def put_info(self, info):
        """Writes information records to the database. The records are
        MAX_LINE_LENGTH-character strings.

        Parameters
        ----------
        info : array_like, (num_info, )
            Array containing the information records

        """
        self.ex_gp_info(PUT, info)
        return

    def put_var_param(self, var_type, num_vars):
        """Writes the number of global, nodal, nodeset, sideset, or element
        variables that will be written to the database.

        Parameters
        ----------
        var_type : str
            Character indicating the type of variable which is described.
            Use one of the following options:
              "g" (or "G")
              "n" (or "N")
              "e" (or "E")
              "m" (or "M")
              "s" (or "S")
            For global, nodal, element, nodeset variables, and sideset
            variables, respectively.

        num_vars : int
            The number of var_type variables that will be written to the
            database.

        """
        self.ex_gp_var_param(PUT, var_type.upper(), num_vars)
        return

    def put_var_names(self, var_type, num_vars, var_names):
        """Writes the names of the results variables to the database. The
        names are MAX_STR_LENGTH-characters in length.

        Parameters
        ----------


        Notes
        -----
        The function put_var_param must be called before this function is
        invoked.

        """
        self.ex_gp_var_names(PUT, var_type.upper(), num_vars, var_names)

    def put_elem_var_tab(self, num_elem_blk, num_elem_var, elem_var_tab):
        """Writes the EXODUS II element variable truth table to the database.

        The element variable truth table indicates whether a particular
        element result is written for the elements in a particular element
        block. A 0 (zero) entry indicates that no results will be output for
        that element variable for that element block. A non-zero entry
        indicates that the appropriate results will be output.

        Parameters
        ----------
        num_elem_blk : int
            The number of element blocks.

        num_elem_var : int
            The number of element variables.

        elem_var_tab : array_like, (num_elem_blk, num_elem_var)
             A 2-dimensional array containing the element variable truth
             table.

        Notes
        -----
        Although writing the element variable truth table is optional, it is
        encouraged because it creates at one time all the necessary netCDF
        variables in which to hold the EXODUS element variable values. This
        results in significant time savings. See Appendix A for a discussion
        of efficiency issues. Calling the function put_var_tab with an
        object type of "E" results in the same behavior as calling this
        function.

        The function put_var_param (or EXPVP for Fortran) must be called
        before this routine in order to define the number of element
        variables.

        """
        self.ex_gp_elem_var_tab(PUT, num_elem_blk, num_elem_var, elem_var_tab)

    def put_glob_vars(self, time_step, num_glo_var, vals_glo_var):
        """Writes the values of all the global variables for a single time step.

        time_step : int
            The time step number, as described under put_time.
            This is essentially a counter that is incremented when results
            variables are output. The first time step is 1.

        num_glob_vars : int
            The number of global variables to be written to the database.

        glob_var_vals : array_like
            Array of num_glob_vars global variable values for the time_stepth
            time step.

        Notes
        -----
        The function put_var_param must be invoked before this call is made.

        """
        self.ex_gp_glob_vars(PUT, time_step, num_glo_var, vals_glo_var)
        return

    def put_nodal_var(self, time_step, nodal_var_index, num_nodes,
                      nodal_var_vals):
        """Writes the values of a single nodal variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.

        nodal_var_index : int
            The index of the nodal variable.
            The first variable has an index of 1.

        num_nodes : int
            The number of nodal points.

        nodal_var_vals : array_like
            Array of num_nodes values of the nodal_var_indexth nodal variable
            for the time_stepth time step.

        Notes
        -----
        The function put_var_param must be invoked before this call is made.

        """
        self.ex_gp_nodal_var(PUT, time_step, nodal_var_index, num_nodes,
                             nodal_var_vals)
        return

    def put_elem_var(self, time_step, elem_var_index, elem_blk_id,
                     num_elem_this_blk, elem_var_vals):
        """Writes the values of a single elemental variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.

        elem_var_index : int
            The index of the element variable.
            The first variable has an index of 1.

        elem_blk_id : int
            The element block ID

        num_elem_this_blk : int
            The number of elements in the given element block

        elem_var_vals : array_like
            Array of num_elem_this_blk values of the elem_var_indexth element
            variable for the element block with ID of elem_blk_id at the
            time_stepth time step

        Notes
        -----
        The function put_var_param must be invoked before this call is
        made.

        It is recommended, but not required, to write the element variable
        truth table before this function is invoked for better efficiency.

        """
        self.ex_gp_elem_var(PUT, time_step, elem_var_index, elem_blk_id,
                            num_elem_this_blk, elem_var_vals)


class ExodusIIReader(ExodusIIFile):
    def __init__(self, filepath):
        if not os.path.isfile(filepath):
            raise IOError("{0}: no such file".format(filepath))

        self.filepath = filepath
        self.db = self.open_db(filepath, mode="r")

        self.filename = getattr(self.db, "filename", os.path.basename(filepath))
        self.runid = getattr(self.db, "runid", os.path.splitext(self.filename)[0])

        self.all_times = self.get_all_times()
        self.num_time_steps = self.all_times.shape[0]

        self.num_glo_var = self.ex_gp_var_names(GET, "G", 1)
        self.num_nod_var = self.ex_gp_var_names(GET, "N", 1)
        self.num_elem_var = self.ex_gp_var_names(GET, "E", 1)

        # get the index for each variable
        self.var_index = {}
        self.var_names = {}
        for var_type in "GEN":
            self.var_index[var_type] = {}
            self.var_names[var_type] = []
            for (idx, name) in enumerate(self.ex_gp_var_names(GET, var_type)):
                self.var_index[var_type][name] = idx
                self.var_names[var_type].append(name)

        # element block IDs and info
        self.elem_blk_params = self.ex_gp_elem_block(GET)
        self.elem_blk_ids = sorted(self.elem_blk_params,
                                   key=lambda k: self.elem_blk_params[k]["O"])

    def glob_var_names(self):
        return self.var_names["G"]

    def elem_var_names(self):
        return self.var_names["E"]

    def nod_var_names(self):
        return self.var_names["N"]

    def get_time(self, time_step):
        """Reads the time value for a specified time step

        Parameters
        ----------
        time_step : int
            Time step, 0 based indexing

        Returns
        -------
        time : float
            Time at time step

        """
        return self.ex_gp_time(GET, time_step)

    def get_all_times(self):
        """Reads the time values for all time steps

        Returns
        -------
        times : ndarray
            Array of times for all time steps

        """
        return self.ex_gp_time(GET, None)

    def get_var_index(self, var_type, var_name):
        if var_type not in "GEN":
            raise ExodusIIFileError("{0}: unrecognized var_type".format(var_type))
        try:
            return self.var_index[var_type][var_name]
        except KeyError:
            raise ExodusIIFileError("{0}: unrecognized var_name".format(var_name))

    def get_elem_blk_ids(self):
        """Reads the IDs of all of the element blocks from the database"""
        return self.elem_blk_ids

    def get_glob_vars(self, time_step, disp=0):
        """Read all global variables at one time step

        Parameters
        ----------
        time_step : int
            Time step, 0 based indexing

        disp : int, optional
            If disp > 0, return dictionary of {glob_var: glob_var_val}

        Returns
        -------
        var_values : ndarray, (num_glob_vars,)
            Global variable values for the stepth time step

        """
        return self.ex_gp_glob_vars(GET, time_step, disp)

    def get_glob_var_time(self, glo_var_name):
        """Read the global variable glo_var_name through all time

        Returns
        -------
        var_values : ndarray, (time_step,)
            Global variable values for the stepth time step

        """
        return self.ex_gp_glob_vars(GET, None, glo_var_name)

    def get_elem_var_time(self, elem_var, elem_num):
        """Read the element variable elemo_var through all time

        Returns
        -------
        var_values : ndarray, (time_step,)
            Global variable values for the stepth time step

        """
        # @tjf: hardcoded for 1 element block -> need to find element block
        # associated with elem_num
        elem_blk_id = 1

        # Get the element block[s]
        var_index = self.get_var_index("E", elem_var)
        return self.ex_gp_elem_var(GET, None, var_index, elem_blk_id, elem_num)

    def get_elem_var(self, time_step, elem_var, elem_blk_ids=None):
        """Read element variable at one time step

        Parameters
        ----------
        time_step : int
            The time step, 0 indexing

        elem_var : str
            The element variable

        elem_blk_ids : list, optional
            If given, list of element block IDs

        Returns
        -------
        elem_var : list

        """
        # Get the element block[s]
        if elem_blk_ids is None:
            elem_blk_ids = self.elem_blk_ids
        if not isinstance(elem_blk_ids, (list, tuple)):
            elem_blk_ids = [elem_blk_ids]

        var_index = self.get_var_index("E", elem_var)
        elem_var = []
        for elem_blk_id in elem_blk_ids:
            num_elem_this_blk = self.elem_blk_params[elem_blk_id]["NEL"]
            elem_var.append(self.ex_gp_elem_var(GET, time_step, var_index,
                                                elem_blk_id, num_elem_this_blk))
        return elem_var

    @classmethod
    def new_from_exofile(cls, filepath):
        return cls(filepath)


def test():
    runid = "foo"
    title = "my title"
    glo_vars = ("TIME_STEP", "STEP_NUM", "LEG_NUM")
    elem_vars = ["STRESS", "STRAIN", "DEFGRAD"]
    f = ExodusIIWriter.new_from_runid(runid)

    # "mesh" information
    num_dim = 3
    num_nodes = 8
    num_elem = 1
    num_elem_blk = 1
    num_node_sets = 0
    num_side_sets = 0

    # initialize file with parameters
    f.put_init(title, num_dim, num_nodes, num_elem,
               num_elem_blk, num_node_sets, num_side_sets)

    # write nodal coordinates values and names to database
    coords = np.array([[-1, -1, -1], [ 1, -1, -1],
                       [ 1,  1, -1], [-1,  1, -1],
                       [-1, -1,  1], [ 1, -1,  1],
                       [ 1,  1,  1], [-1,  1,  1]],
                      dtype=np.float64) * .5
    coord_names = np.array(["COORDX", "COORDY", "COORDZ"])
    f.put_coord_names(coord_names)
    f.put_coord(coords[:, 0], coords[:, 1], coords[:, 2])

    # write element block parameters
    conn = np.array([range(8)], dtype=np.int)
    num_attr = 0 # for now, we do not use attributes
    elem_blk_id = 1
    elem_blk_els = [0]
    num_elem_this_blk = 1
    elem_type = "HEX"
    num_nodes_per_elem = num_nodes
    f.put_elem_block(elem_blk_id, elem_type, num_elem_this_blk,
                     num_nodes_per_elem, num_attr)
    # write element connectivity for each element block
    blk_conn = conn[elem_blk_els][:, :num_nodes_per_elem]
    f.put_elem_conn(elem_blk_id, blk_conn)

    # write QA records
    now = datetime.datetime.now()
    day = now.strftime("%m/%d/%y")
    hour = now.strftime("%H:%M:%S")
    num_qa_rec = 1
    vers = ".".join(str(x) for x in __version__)
    qa_title = "MML {0} simulation".format(vers)
    qa_record = np.array([[qa_title, runid, day, hour]])
    f.put_qa(num_qa_rec, qa_record)

    # write results variables parameters and names
    names_glo_var = ["TIME_STEP", "STEP_NUM", "LEG_NUM"]
    num_glo_var = len(names_glo_var)
    f.put_var_param("g", num_glo_var)
    f.put_var_names("g", num_glo_var, names_glo_var)

    nod_var_names = ["DISPLX", "DISPLY", "DISPLZ"]
    num_nod_var = len(nod_var_names)
    f.put_var_param("n", num_nod_var)
    f.put_var_names("n", num_nod_var, nod_var_names)

    elem_var_names = ("STRESS", "STRAIN", "DEFGRAD", "TMPR")
    num_elem_var = len(elem_var_names)
    f.put_var_param("e", num_elem_var)
    f.put_var_names("e", num_elem_var, elem_var_names)

    # write element variable truth table
    truth_tab = np.empty((num_elem_blk, num_elem_var), dtype=np.int)
    for i in range(num_elem_blk):
        for j in range(num_elem_var):
            truth_tab[i, j] = 1
    f.put_elem_var_tab(num_elem_blk, num_elem_var, truth_tab)


    # loop through time and update variables
    times = np.linspace(0, 100, 35)
    vals_glo_var = np.zeros(num_glo_var)

    u = np.zeros((num_nodes, num_nod_var))

    num_el_this_blk = 1
    vals_elem_var = np.zeros((num_elem_var, num_el_this_blk))

    for (time_step, time) in enumerate(times):
        f.put_time(time_step, time)

        # write global variables
        vals_glo_var += 2.
        f.put_glob_vars(time_step, len(vals_glo_var), vals_glo_var)

        # write nodal variables
        u += 3.
        for k in range(num_nod_var):
            f.put_nodal_var(time_step, k, num_nodes, u[:, k])

        # write element variables
        vals_elem_var += 1.
        elem_blk_id = 1
        for k in range(num_elem_var):
            f.put_elem_var(time_step, k, elem_blk_id,
                           num_el_this_blk, vals_elem_var[k])
            continue

    f.close()

    ex = ExodusIIReader.new_from_exofile(f.filename)
    print ex.filename
    print ex.db.variables.keys()
    print ex.db.dimensions

if __name__ == "__main__":
    test()


