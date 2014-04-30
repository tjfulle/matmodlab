import os
import numpy as np

import netcdf as nc
from exoinc import *


class NotYetImplemented(Exception):
    def __init__(self, meth):
        self.message = "{0}: ExodusIIFile method not yet implemented".format(meth)
        super(NotYetImplemented, self).__init__(self.message)


class ExodusIIReader(object):
    """Exodus output databse reader

    """
    def __init__(self, filepath):
        self.db = self.open_db(filepath)
        self.set_id_idx_map = self._get_set_ids()
        self.var_id_idx_map = self._get_var_ids()
        pass

    def __repr__(self):
        return self.summary()

    def open_db(self, filepath):
        """Open the netcdf database file"""
        return nc.netcdf_file(filepath, "r")

    def close(self):
        self.db.close()

    def update(self):
        pass

    @property
    def filename(self):
        return getattr(self.db, ATT_FILENAME)

    @property
    def title(self):
        return getattr(self.db, ATT_TITLE)

    @property
    def num_dim(self):
        return self.db.dimensions[DIM_NUM_DIM]

    @property
    def num_nodes(self):
        return self.db.dimensions[DIM_NUM_NODES]

    @property
    def num_elem(self):
        return self.db.dimensions[DIM_NUM_ELEM]

    @property
    def num_elem_blk(self):
        return self.db.dimensions[DIM_NUM_EL_BLK]

    @property
    def num_node_sets(self):
        return self.db.dimensions.get(DIM_NUM_NS, 0)

    @property
    def num_side_sets(self):
        return self.db.dimensions.get(DIM_NUM_SS, 0)

    @property
    def glob_var_names(self):
        return chara_to_text(self.db.variables[VAR_NAME_GLO_VAR].data, aslist=True)

    @property
    def elem_var_names(self):
        return chara_to_text(self.db.variables[VAR_NAME_ELE_VAR].data, aslist=True)

    @property
    def node_var_names(self):
        return chara_to_text(self.db.variables[VAR_NAME_NOD_VAR].data, aslist=True)

    @property
    def coord_names(self):
        return self.get_coord_names()

    @property
    def coords(self):
        return self.get_coord()

    @property
    def num_time_steps(self):
        return self.db.variables[VAR_WHOLE_TIME].shape[0]

    @property
    def elem_blk_ids(self):
        return self.db.variables[VAR_ID_EL_BLK].data

    @property
    def node_set_ids(self):
        try: return self.db.variables[VAR_NS_IDS].data
        except KeyError: return []

    @property
    def side_set_ids(self):
        try: return self.db.variables[VAR_SS_IDS].data
        except KeyError: return []

    @property
    def elem_num_map(self):
        return self.get_elem_num_map()

    @property
    def connect(self):
        return self.get_db_conn()

    def setidx(self, obj_type, obj_id):
        return self.set_id_idx_map[obj_type].get(obj_id)

    def varidx(self, var_typ, var_name):
        return self.var_id_idx_map[var_typ].get(var_name)

    def _get_set_ids(self):
        set_id_idx_map = {}
        set_id_idx_map[PX_ELEM_BLK] = dict((j, i)
                                   for (i, j) in enumerate(self.elem_blk_ids))
        set_id_idx_map[PX_NODE_SET] = dict((j, i)
                                   for (i, j) in enumerate(self.node_set_ids))
        set_id_idx_map[PX_SIDE_SET] = dict((j, i)
                                   for (i, j) in enumerate(self.side_set_ids))
        return set_id_idx_map

    def _get_var_ids(self):
        var_id_idx_map = {}
        var_id_idx_map[PX_VAR_ELE] = dict((j, i)
                                  for (i, j) in enumerate(self.elem_var_names))
        var_id_idx_map[PX_VAR_NOD] = dict((j, i)
                                  for (i, j) in enumerate(self.node_var_names))
        var_id_idx_map[PX_VAR_GLO] = dict((j, i)
                                  for (i, j) in enumerate(self.glob_var_names))
        return var_id_idx_map

    def get_init(self):
        """Gets the initialization parameters from the ExodusII file

        Parameters
        ----------
        exoid : int
            File ID as returned by one of the factory initialization methods

        Returns
        -------
        title : str
            Title

        num_dim : int
            Number of spatial dimensions [1, 2, 3]

        num_nodes : int
            Number of nodes

        num_elem : int
            Number of elements

        num_elem_blk : int
            Number of element blocks

        num_node_sets : int
            Number of node sets

        num_side_sets : int
            Number of side sets

        """
        return (self.title, self.num_dim, self.num_nodes, self.num_elem,
                self.num_elem_blk, self.num_node_sets, self.num_side_sets)

    def get_all_times(self):
        """reads the time value for all times

        Parameters
        ----------

        Returns
        -------
        times : ndarray of floats
            Times for all steps

        """
        return self.db.variables[VAR_WHOLE_TIME].data[:]

    def get_time(self, step):
        """reads the time value for a specified time step

        Parameters
        ----------
        step : int
            Time step, 0 based indexing

        Returns
        -------
        time : float
            Time at time step

        """
        return self.db.variables[VAR_WHOLE_TIME].data[step]

    def get_coord_names(self):
        """Reads the names of the coordinate arrays from the database.

        Returns
        -------
        coord_names : array_like
            Array containing num_dim names (of length MAX_STR_LENGTH) of the
            nodal coordinate arrays.

        """
        return chara_to_text(self.db.variables[VAR_NAME_COOR].data[:])

    def get_coord(self, idx=None):
        """Read the coordinates of the nodes

        Returns
        -------
        x, y, z : array_like
            x, y, z coordinates

        """
        if idx is not None:
            return self.db.variables[VAR_COORDS(idx)].data[:]

        coords = []
        for i in range(self.num_dim):
            coords.append(self.db.variables[VAR_COORDS(i)].data[:])
        return coords

    def get_elem_block(self, elem_blk_id):
        """Returns parameters used to describe an element block

        Parameters
        ----------
        elem_blk_id : int
            The element block ID.

        Returns
        -------
        elem_type : str
            The type of elements in the element block.

        num_elem_this_blk : int
            The number of elements in the element block.

        num_nodes_per_elem : int
            Number of nodes per element in the element block

        num_attr : int
            The number of attributes per element in the element block.

        """
        i = self.setidx(PX_ELEM_BLK, elem_blk_id)
        if i is None:
            raise ExodusIIFileError("{0}: not a valid element block "
                                    "ID".format(elem_blk_id))
        elem_type = self.db.variables[VAR_CONN(i+1)].elem_type
        num_elem_this_blk = self.db.dimensions[DIM_NUM_EL_IN_BLK(i+1)]
        num_nodes_per_elem = self.db.dimensions[DIM_NUM_NOD_PER_EL(i+1)]
        num_attr = self.db.dimensions[DIM_NUM_ATT_IN_BLK(i+1)]
        return elem_type, num_elem_this_blk, num_nodes_per_elem, num_attr

    def get_elem_conn(self, elem_blk_id, disp=0):
        """reads the connectivity array for an element block

        Parameters
        ----------
        elem_blk_id : int
            The element block ID

        Returns
        -------
        connect : ndarray, (num_elem_this_blk, num_nodes_per_elem)
            Connectivity array; a list of nodes (internal node IDs; see Node
            Number Map) that define each element. The element index cycles faster
            than the node index.

        """
        i = self.setidx(PX_ELEM_BLK, elem_blk_id)
        if i is None:
            raise ExodusIIFileError("{0}: not a valid element block "
                                    "ID".format(elem_blk_id))
        var = self.db.variables[VAR_CONN(i+1)]
        if not disp:
            return var.data
        return {"connect": var.data, "elem_type": var.elem_type}

    def get_db_conn(self):
        """Reads the connectivity array for all element blocks from the database

        Returns
        -------
        connect : ndarray, (num_elem_this_blk, num_nodes_per_elem)
            Connectivity array; a list of nodes (internal node IDs; see Node
            Number Map) that define each element. The element index cycles faster
            than the node index.

        """
        connect = []
        for (i, elem_blk_id) in enumerate(self.elem_blk_ids):
            conn = self.get_elem_conn(elem_blk_id)
            connect.append(conn)
        return connect

    def get_elem_num_map(self):
        """Returns the element map attribute

        Returns
        -------
        elem_num_map : array_like
            The element number map

        """
        try:
            return self.db.variables[VAR_ELEM_MAP(1)].data[:]
        except KeyError:
            return

    def get_map(self):
        """Returns the optimized element order map attribute

        Returns
        -------
        elem_map : array_like
            The element order map

        """
        try:
            return self.db.variables[PX_VAR_EL_MAP].data[:]
        except KeyError:
            return

    def get_info(self):
        """Reads information records from the database. The records are
        MAX_LINE_LENGTH-character strings. Memory must be allocated for the
        information records before this call is made. The number of records
        can be determined by invoking inquire

        Returns
        -------
        info : array of strings
            information stored to exodus database

        """
        try:
            info = self.db.variables[VAR_INFO].data[:]
        except KeyError:
            return
        return self.chara_to_text(info)

    def get_glob_vars(self, step, disp=0):
        """Read all global variables at one time step

        Parameters
        ----------
        step : int
            Time step, 1 based indexing, it is changed here to 0 based

        disp : int, optional
            If disp > 0, return dictionary of {glob_var: glob_var_val}

        Returns
        -------
        var_values : ndarray, (num_glob_vars,)
            Global variable values for the stepth time step

        """
        try:
            data = self.db.variables[VAR_GLO_VAR].data[step]
        except IndexError:
            raise ExodusIIFileError("Error getting global variables at "
                                    "step {0}".format(step))
        if not disp:
            return data

        return dict(zip(self.glob_var_names, data))

    def get_glob_var_time(self, glob_var):
        """Read global variable through time

        glob_var : str
            The desired global variable

        Returns
        -------
        var_values : ndarray
            Array of the global variable

        """
        i = self.varidx(PX_VAR_GLO, glob_var)
        if i is None:
            raise ExodusIIFileError("{0}: global var not found".format(glob_var))
        return self.db.variables[VAR_GLO_VAR].data[:, i]

    def get_nodal_var(self, step, nodal_var):
        """Read nodal variable at one time step

        Parameters
        ----------
        step : int
            The time step, 0 indexing

        nodal_var : str
            The nodal variable

        Returns
        -------
        var_values : ndarray
            num_nodes values of the nodal_var_indexth nodal variable for the
            stepth time step.

        """
        if step == 0: step = 1
        i = self.varidx(PX_VAR_NOD, nodal_var)
        if i is None:
            raise ExodusIIFileError("{0}: nodal var not found".format(nodal_var))
        return self.db.variables[VAR_NOD_VAR_NEW(i+1)].data[step]

    def get_nodal_var_time(self, nodal_var, node_num):
        """Reads the values of a nodal variable for a single node through a
        specified number of time steps

        Parameters
        ----------
        nodal_var : str
            The desired nodal variable

        node_num : int
            The internal ID (see Node Number Map) of the desired node

        Returns
        -------
        var_vals : ndarray
            Array of (end_time_step - beg_time_step + 1) values of the
            node_numberth node for the nodal_var_indexth nodal variable.

        """
        i = self.varidx(PX_VAR_NOD, nodal_var)
        if i is None:
            raise ExodusIIFileError("{0}: nodal var not found".format(nodal_var))
        return self.db.variables[VAR_NOD_VAR_NEW(i+1)].data[:, node_num-PX_OFFSET]

    def get_elem_var(self, step, elem_var):
        """Read element variable at one time step

        Parameters
        ----------
        step : int
            The time step, 0 indexing

        elem_var : str
            The element variable

        Returns
        -------
        elem_var : ndarray

        """
        i = self.varidx(PX_VAR_ELE, elem_var)
        if i is None:
            raise ExodusIIFileError("{0}: elem var not found".format(elem_var))
        elem_var = []
        for (elem_blk_id, j) in self.set_id_idx_map[PX_ELEM_BLK].items():
            n = self.db.dimensions[DIM_NUM_EL_IN_BLK(j+1)]
            name = VAR_ELEM_VAR(i+1, j+1)
            elem_var.append(self.db.variables[name][step, :n])
        return np.array(elem_var)

    def get_elem_var_time(self, elem_var, elem_num, elem_blk_id=None):
        """Read element variable through time

        Parameters
        ----------
        elem_var : str
            The desired element variable

        elem_num : int
            The internal ID (see Element Number Map) of the desired element

        Returns
        -------
        var_vals : ndarray
            Array of (end_time_step - beg_time_step + 1) values of the
            node_numberth node for the nodal_var_indexth nodal variable.

        """
        i = self.varidx(PX_VAR_ELE, elem_var)
        if i is None:
            raise ExodusIIFileError("{0}: elem var not found".format(elem_var))

        if elem_blk_id is None:
            # find element block number that has this element assume that
            # elements are numbered contiguously with element block
            num_els = [self.db.dimensions[DIM_NUM_EL_IN_BLK(j+1)]
                       for j in range(self.num_elem_blk)]
            n = 0
            for (j, num_el) in enumerate(num_els):
                n += num_el
                if elem_num < n + PX_OFFSET:
                    break
            # j is now the element block id, now find the element number
            # relative the element block
            e = elem_num - sum(num_els[:j])
        else:
            j = self.setidx(PX_ELEM_BLK, elem_blk_id)
            if j is None:
                raise ExodusIIFileError("{0}: invalid element "
                                        "block".format(elem_blk_id))
            e = elem_num

        name = VAR_ELEM_VAR(i+1, j+1)
        return self.db.variables[name].data[:, e-PX_OFFSET]

    def summary(self):
        """return a summary string

        """
        S = ["Summary", "=" * 80]
        S.append("Exodus file name: {0}".format(self.filename))
        S.append("Title: {0}".format(self.title))
        S.append("Number of dimensions: {0}".format(self.num_dim))
        S.append("Coordinate names: {0}".format(" ".join(self.coord_names)))

        S.append("Number of nodes: {0}".format(self.num_nodes))
        S.append("Number of node sets: {0}, Ids={1}".format(
                self.num_node_sets,
                " ".join("{0}".format(x) for x in self.node_set_ids)))

        S.append("Number of elements: {0}".format(self.num_elem))
        S.append("Number of element blocks: {0} Ids={1}".format(
                self.num_elem_blk,
                " ".join("{0}".format(x) for x in self.elem_blk_ids)))

        S.append("Number of side sets: {0}".format(self.num_side_sets))

        S.append("Global Variables: {0}".format(", ".join(self.glob_var_names)))
        S.append("Element Variables: {0}".format(", ".join(self.elem_var_names)))
        S.append("Nodal Variables: {0}".format(", ".join(self.node_var_names)))

        S.append("Number of time steps: {0}".format(self.num_time_steps))
        for i in range(self.num_time_steps):
            S.append("    {0} {1}".format(i, self.get_time(i)))

        return "\n".join(S)

    # -------------- TJF: below methods need updating to PYTHON API
    def nodes_in_node_set(self, node_set_id):
        """Return a list of nodes in the node set

        Parameters
        ----------
        node_set_id : int
            The node set ID

        Returns
        -------
        node_list : ndarray
            Array of node IDs

        """
        raise NotYetImplemented("nodes_in_node_set")
        # Get only those nodes in the requested IDs
        node_set_params = self.node_set_params.get(node_set_id)
        if node_set_params is None:
            valid = ", ".join(["{0}".format(x)
                               for x in self.node_set_params])
            raise ExodusReaderError("{0}: invalid node set ID.  Valid IDs "
                                    "are: {1}".format(node_set_id, valid))
        return np.array(node_set_params["NODE LIST"])

    def nodes_in_region(self, xlims=(-PX_HUGE, PX_HUGE), ylims=(-PX_HUGE, PX_HUGE),
                        zlims=(-PX_HUGE, PX_HUGE), node_set_ids=None):
        """Return a list of nodes in the region bounded by (xlims, ylims, zlims)

        Parameters
        ----------
        xlims, ylims, zlims : tuple of floats
            Floats defining ([xyz]min, [xyz]max)

        node_set_id : list of ints, optional
            Node set IDs

        Returns
        -------
        node_list : ndarray
            Array of node IDs

        """
        raise NotYetImplemented("nodes_in_region")
        xmin, xmax = xlims
        ymin, ymax = ylims
        zmin, zmax = zlims
        if node_set_ids is None:
            node_list = np.arange(self.num_nodes)
        else:
            # Get only those nodes in the requested IDs
            if not isinstance(node_set_ids, (list, tuple)):
                node_set_ids = [node_set_ids]
            node_lists = [self.nodes_in_node_set(x) for x in node_set_ids]
            node_list = np.array([node for node_list in node_lists
                                  for node in node_list], dtype=np.int)
        return node_list[(self.XYZ[node_list, PY_X_COMP] >= xmin) &
                         (self.XYZ[node_list, PY_X_COMP] <= xmax) &
                         (self.XYZ[node_list, PY_Y_COMP] >= ymin) &
                         (self.XYZ[node_list, PY_Y_COMP] <= ymax) &
                         (self.XYZ[node_list, PY_Z_COMP] >= zmin) &
                         (self.XYZ[node_list, PY_Z_COMP] <= zmax)]

    def elems_in_region(self, xlims=(-PX_HUGE, PX_HUGE), ylims=(-PX_HUGE, PX_HUGE),
                        zlims=(-PX_HUGE, PX_HUGE), node_set_ids=None):
        """Return a list of elements in the region bounded by
        (xlims, ylims, zlims)

        Parameters
        ----------
        xlims, ylims, zlims : tuple of floats
            Floats defining ([xyz]min, [xyz]max)

        node_set_id : list of ints, optional
            Node set IDs

        Returns
        -------
        elem_list : ndarray
            Array of element IDs

        """
        raise NotYetImplemented("elems_in_region")
        # get the nodes in the bounding box
        node_list = self.nodes_in_region(xlims=xlims, ylims=ylims, zlims=zlims,
                                         node_set_ids=node_set_ids)
        # get elements connected to nodes
        return self.elems_from_nodes(node_list, strict=False)

    def elems_from_nodes(self, node_list, strict=True):
        """Return a list of elements whose nodes are in node_list

        Parameters
        ----------
        node_list : ndarray of ints
            Array of nodes

        strict : bool, optional
            If False, return element if more than half its nodes are in the
            node_list

        Returns
        -------
        elem_list : ndarray
            Array of element IDs

        """
        raise NotYetImplemented("elems_from_nodes")
        # loop through each element block, finding elements whose nodes are
        # in node_list.  Convert to global element IDs
        def issubset(a, b):
            """Return True if the set a is a subset of the set b, else False"""
            return a <= b if strict else len(a & b) >= len(a) / 2.

        elem_list, num_elems_seen, node_set = [], 0, set(node_list)
        for i, blk_conn in enumerate(self.connect):
            elem_list.extend([num_elems_seen + i_elem
                              for (i_elem, elem_conn) in enumerate(blk_conn)
                              if issubset(set(elem_conn), node_set)])
            num_elems_seen += blk_conn.shape[0]
        return np.array(elem_list, dtype=np.int)

    def elems_in_blk(self, elem_blk_id):
        """Return a list of elements in block elem_blk_id

        Parameters
        ----------
        elem_blk_id : int
            Element block ID

        Returns
        -------
        elem_list : ndarray
            Array of element IDs in elem_blk_id

        """
        i = self.setidx(PX_ELEM_BLK, elem_blk_id)
        if i is None:
            raise ExodusIIFileError("{0}: not a valid element block "
                                    "ID".format(elem_blk_id))

        # number of elements in blocks preceding elem_blk_id
        n = sum(len(x) for x in self.connect[:i])

        # element IDs for elements in elem_blk_id
        elem_ids = np.array([n + i for i in range(len(self.connect[i]))])
        elem_ids = elem_ids
        return self.elem_num_map[elem_ids]

