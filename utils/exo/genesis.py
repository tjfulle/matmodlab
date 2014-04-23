import os
import sys
import imp
import numpy as np

import xnetcdf as nc
from exoinc import *


class Genesis(object):

    def __init__(self, runid, d=None):
        """Instantiate the Genesis object

        Parameters
        ----------
        runid : int
            Runid (usually file basename)

        Notes
        -----
        The Genesis class is an interface to the Exodus II api. Its methods
        are named after the analogous method from the Exodus II C bindings,
        minus the prefix 'ex_'.

        """
        d = os.getcwd() or d
        filepath = os.path.join(d, runid + ".exo")
        self.db = self.open_db(filepath, mode="w")

        version = 5.0300002
        setattr(self.db, ATT_API_VERSION, version)
        setattr(self.db, ATT_VERSION, version)
        setattr(self.db, ATT_FLT_WORDSIZE, 4)
        setattr(self.db, ATT_FILESIZE, 1)

        setattr(self.db, ATT_FILENAME, os.path.basename(filepath))
        setattr(self.db, ATT_RUNID, runid)

        # standard ExodusII dimensioning
        self.db.createDimension(DIM_STR, MAX_STR_LENGTH + 1)
        self.db.createDimension(DIM_LIN, MAX_LINE_LENGTH + 1)
        self.db.createDimension(DIM_N4, 4)

        # initialize internal variables
        # internal counters
        self.counter = {PX_ELEM_BLK: 0, PX_NODE_SET: 0, PX_SIDE_SET: 0}
        self.objids = {PX_ELEM_BLK: {}, PX_NODE_SET: {}, PX_SIDE_SET: {}}
        pass

    def open_db(self, filepath, mode="r"):
        """Open the netcdf database file"""
        if mode not in "rw":
            raise ExodusIIFileError("{0}: bad read/write mode".format(mode))
        return nc.netcdf_file(filepath, mode)

    def close(self):
        self.db.close()

    def update(self):
        pass

    def register_id(self, obj_type, obj_id, obj_idx):
        if obj_id in self.objids[obj_type]:
            raise ExodusIIFileError("{0}: duplicate {1} block  "
                                    "ID".format(elem_blk_id, obj_type))
        self.objids[obj_type][obj_id] = obj_idx

    def get_obj_idx(self, obj_type, obj_id):
        return self.objids[obj_type].get(obj_id)

    @property
    def filename(self):
        return self.db.filename

    # ---------------------------------------------------- GENESIS OUTPUT --- #
    def put_init(self, title, num_dim, num_nodes, num_elem, num_elem_blk,
                 num_node_sets, num_side_sets):
        """Writes the initialization parameters to the EXODUS II file

        Parameters
        ----------
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
        self.db.title = title
        setattr(self.db, ATT_TITLE, title)

        # Create required dimensions
        self.db.createDimension(DIM_NUM_DIM, num_dim)
        self.db.createDimension(DIM_NUM_NODES, num_nodes)
        self.db.createDimension(DIM_NUM_ELEM, num_elem)

        def create_and_alloc(num, dim, var_id, var_stat, var_name, nz=0):
            if not num:
                return

            # block/set meta data
            self.db.createDimension(dim, num)

            if nz: prop1 = np.arange(num, dtype=np.int32)
            else: prop1 = np.zeros(num, dtype=np.int32)
            self.db.createVariable(var_id, DTYPE_INT, (dim,))
            self.db.variables[var_id][:] = prop1
            setattr(self.db.variables[var_id], ATT_PROP_NAME, "ID")

            if nz: status = np.ones(num, dtype=np.int32)
            else: status = np.zeros(num, dtype=np.int32)
            self.db.createVariable(var_stat, DTYPE_INT, (dim,))
            self.db.variables[var_stat][:] = status

            names = np.array([" " * MAX_STR_LENGTH for _ in prop1])
            self.db.createVariable(var_name, DTYPE_TXT, (dim, DIM_STR))
            for (i, name) in enumerate(names):
                self.db.variables[var_name][i][:] = name

        # element block meta data
        num_elem_blk = max(num_elem_blk, 1)
        create_and_alloc(num_elem_blk, DIM_NUM_EL_BLK, VAR_ID_EL_BLK,
                         VAR_STAT_EL_BLK, VAR_NAME_EL_BLK, nz=1)

        # node set meta data
        create_and_alloc(num_node_sets, DIM_NUM_NS, VAR_NS_IDS,
                         VAR_NS_STAT, VAR_NAME_NS)

        # side set meta data
        create_and_alloc(num_side_sets, DIM_NUM_SS, VAR_SS_IDS,
                         VAR_SS_STAT, VAR_NAME_SS)

        # set defaults
        self.db.createVariable(VAR_NAME_COOR, DTYPE_TXT, (DIM_NUM_DIM, DIM_STR))
        for i in range(num_dim):
            self.db.createVariable(PX_VAR_COORDS(i), DTYPE_FLT, (DIM_NUM_NODES,))

        self.db.createDimension(DIM_NUM_EM, 1)
        self.db.createVariable(VAR_ELEM_MAP(1), DTYPE_INT, (DIM_NUM_ELEM,))
        elem_map = np.arange(num_elem) + 1
        self.put_elem_num_map(elem_map)

        self.db.createVariable(PX_VAR_EL_MAP, DTYPE_INT, (DIM_NUM_ELEM,))
        self.put_map(elem_map)

    def put_coord_names(self, coord_names):
        """Writes the names of the coordinate arrays to the database.

        Parameters
        ----------
        coord_names : array_like
            Array containing num_dim names (of length MAX_STR_LENGTH) of the
            nodal coordinate arrays.

        """
        num_dim = self.db.dimensions[DIM_NUM_DIM]
        for i in range(num_dim):
            self.db.variables[VAR_NAME_COOR][i][:] = coord_names[i]
        return

    def put_coord(self, *coords):
        """Write the names of the coordinate arrays

        Parameters
        ----------
        coords: x, y, z : each array_like
            x, y, z coordinates

        """
        num_dim = self.db.dimensions[DIM_NUM_DIM]
        for i in range(num_dim):
            self.db.variables[PX_VAR_COORDS(i)][:] = coords[i]

        return

    def put_map(self, elem_map):
        """Writes out the optional element order map to the database

        Parameters
        ----------
        elem_map : array_like
            The element map

        Notes
        -----
        The following code generates a default element order map and outputs
        it to an open EXODUS II file. This is a trivial case and included just
        for illustration. Since this map is optional, it should be written out
        only if it contains something other than the default map.

        elem_map = []
        for i in range(num_elem):
            elem_map.append(i)

        """
        num_elem = self.db.dimensions[DIM_NUM_ELEM]
        if len(elem_map) > num_elem:
            raise ExodusIIFileError("len(elem_map) > num_elem")
        self.db.variables[PX_VAR_EL_MAP][:] = elem_map
        return

    def put_elem_num_map(self, elem_num_map):
        """Writes out the optional element order map to the database

        Parameters
        ----------
        elem_map : array_like
            The element map

        Notes
        -----
        The following code generates a default element order map and outputs
        it to an open EXODUS II file. This is a trivial case and included just
        for illustration. Since this map is optional, it should be written out
        only if it contains something other than the default map.

        elem_map = []
        for i in range(num_elem):
            elem_map.append(i)

        """
        num_elem = self.db.dimensions[DIM_NUM_ELEM]
        if len(elem_num_map) > num_elem:
            raise ExodusIIFileError("len(elem_map) > num_elem")
        self.db.variables[VAR_ELEM_MAP(1)][:] = elem_num_map
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
        num_elem_blk = self.db.dimensions[DIM_NUM_EL_BLK]
        if self.counter[PX_ELEM_BLK] == num_elem_blk:
            raise ExodusIIFileError("number element blocks exceeded")

        elem_blk_id = int(elem_blk_id)

        # dimensioning
        i = self.counter[PX_ELEM_BLK]
        self.db.createDimension(DIM_NUM_EL_IN_BLK(i+1), num_elem_this_blk)
        self.db.createDimension(DIM_NUM_NOD_PER_EL(i+1), num_nodes_per_elem)

        # store the element block ID
        self.db.variables[VAR_ID_EL_BLK][i] = elem_blk_id
        self.register_id(PX_ELEM_BLK, elem_blk_id, i)

        # set up the element block connectivity
        self.db.createVariable(VAR_CONN(i+1), DTYPE_INT,
                               (DIM_NUM_EL_IN_BLK(i+1), DIM_NUM_NOD_PER_EL(i+1)))
        setattr(self.db.variables[VAR_CONN(i+1)], "elem_type", elem_type.upper())
        conn = np.zeros((num_elem_this_blk, num_nodes_per_elem))
        self.db.variables[VAR_CONN(i+1)][:] = conn

        # element block attributes
        if num_attr:
            self.db.createDimension(DIM_NUM_ATT_IN_BLK(i+1), num_attr)

            self.db.createVariable(VAR_ATTRIB(i+1), DTYPE_FLT,
                               (DIM_NUM_EL_IN_BLK(i+1), DIM_NUM_ATT_IN_BLK(i+1)))
            self.db.variables[VAR_ATTRIB(i+1)][:] = np.zeros(num_attr)

            self.db.createVariable(VAR_NAME_ATTRIB(i+1), DTYPE_TXT,
                                   (DIM_NUM_ATT_IN_BLK(i+1), DIM_STR))
            self.db.variables[VAR_NAME_ATTRIB(i+1)][:] = " " * MAX_STR_LENGTH

        # increment block number
        self.counter[PX_ELEM_BLK] += 1
        return

    def put_prop_names(self, obj_type, num_props, prop_names, o=0, nofill=0):
        """Writes property names and allocates space for property arrays used
        to assign integer properties to element blocks, node sets, or side
        sets.

        Parameters
        ----------
        obj_type : int

        num_props : int
            The number of properties

        prop_names : array_like
            Array containing num_props names

        """
        name, dim = PX_PROPINFO(obj_type)
        n = self.db.dimensions[dim]
        for i in range(num_props):
            I = o + i + 2
            self.db.createVariable(name(I), DTYPE_INT, (dim,))
            setattr(self.db.variables[name(I)], ATT_PROP_NAME, str(prop_names[i]))
            if not nofill:
                setattr(self.db.variables[name(I)], "_FillValue", 0)
            # _FillValue not yet implemented
            self.db.variables[name(I)][:] = np.zeros(n, dtype=np.int32)
        return I

    def put_prop(self, obj_type, obj_id, prop_name, value):
        """Stores an integer property value to a single element block, node
        set, or side set.

        Parameters
        ----------
        obj_type : int
            The type of object

        obj_id : int
            The element block, node set, or side set ID

        prop_name : str
            Property name

        value : int

        """
        name, dim = PX_PROPINFO(obj_type)
        n = len([x for x in self.db.variables if name("") in x and x != name(1)])
        ids = self.db.variables[name(1)].data
        idx = np.where(ids == obj_id)[0][0]
        i = 0
        for i in range(n):
            var = self.db.variables[name(i+2)]
            if var.name == prop_name:
                var[idx] = value
                break
        else:
            # register the variable and assign its value
            idx = np.where(ids == obj_id)[0][0]
            I = self.put_prop_names(obj_type, 1, [prop_name], o=n)
            self.db.variables[name(I)][idx] = value

    def put_prop_array(self, obj_type, prop_name, values):
        """Stores an array of (num_elem_blk, num_node_sets, or num_side_sets)
        integer property values for all element blocks, node sets, or side
        sets.

        Parametes
        ---------
        obj_type : int
            The type of object; use on of the following options
            EX_ELEM_BLOCK
            EX_NODE_SET
            EX_SIDE_SET

        prop_name : string
            Property name

        values : array_like
            An array of property values

        """
        name, dim = PX_PROPINFO(obj_type)
        n = len([x for x in self.db.variables if name("") in x and x != name(1)])
        for i in range(n):
            var = self.db.variables[name(i+2)]
            if var.name == prop_name:
                var[:] = values
                break
        else:
            # register the variable and assign its value
            I = self.put_prop_names(obj_type, 1, [prop_name], o=n, nofill=1)
            self.db.variables[name(I)][:] = values
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
        name, dim = PX_PROPINFO(EX_ELEM_BLOCK)
        i = self.get_obj_idx(PX_ELEM_BLK, elem_blk_id)
        if i is None:
            raise ExodusIIFileError("{0}: element ID not valid".format(elem_blk_id))

        # dimensions
        dim_e = DIM_NUM_EL_IN_BLK(i+1)
        num_elem_this_blk = self.db.dimensions[dim_e]

        dim_n = DIM_NUM_NOD_PER_EL(i+1)
        num_node_this_elem = self.db.dimensions[dim_n]

        netb, nnte = blk_conn.shape
        if netb != num_elem_this_blk:
            raise ExodusIIFileError(
                "expected {0} elements in element block {1}, "
                "got {2}".format(num_elem_this_blk, elem_blk_id, netb))

        if nnte != num_node_this_elem:
            raise ExodusIIFileError(
                "expected {0} nodes in element block {1}, "
                "got {2}".format(num_node_this_elem, elem_blk_id, nnte))

        # connectivity
        self.db.variables[VAR_CONN(i+1)][:] = blk_conn + PX_OFFSET
        return

    def put_elem_attr(self, elem_blk_id, attr):
        """writes the attribute to the

        Parameters
        ----------
        elem_blk_id : int
            The element block ID

        attr : array_like, (num_elem_this_block, num_attr)
            List of attributes for the element block

        """
        name, dim = PX_PROPINFO(EX_ELEM_BLOCK)
        i = self.get_obj_idx(PX_ELEM_BLK, elem_blk_id)
        if i is None:
            raise ExodusIIFileError("{0}: invalid element block "
                                    "ID".format(elem_blk_id))

        # dimensions
        dim_e = DIM_NUM_EL_IN_BLK(i+1)
        num_elem_this_blk = self.db.dimensions[dim_e]

        dim_a = DIM_NUM_ATT_IN_BLK(i+1)
        num_attr_this_block = self.db.dimensions[dim_a]

        # put the attribute
        self.db.variables[VAR_ATTRIB(i+1)][:] = attr
        return

    def put_node_set_param(self, node_set_id, num_nodes_in_set,
                           num_dist_fact_in_set=0):
        """Writes the node set ID, the number of nodes which describe a single
        node set, and the number of distribution factors for the node set.

        Parameters
        ----------
        node_set_id : int
            The node set ID

        num_nodes_in_set : int
            Number of nodes in set

        num_dist_fact_in_set : int
            The number of distribution factors in the node set. This should be
            either 0 (zero) for no factors, or should equal num_nodes_in_set.

        """
        num_node_sets = self.db.dimensions[DIM_NUM_NS]
        if self.counter[PX_NODE_SET] == num_node_sets:
            raise ExodusIIFileError("number of node sets exceeded")

        i = self.counter[PX_NODE_SET]
        self.register_id(PX_NODE_SET, node_set_id, i)

        # store node set ID
        self.db.variables[VAR_NS_IDS][i] = int(node_set_id)

        self.db.createDimension(DIM_NUM_NOD_NS(i+1), num_nodes_in_set)
        self.db.createVariable(VAR_NODE_NS(i+1), DTYPE_INT,
                               (DIM_NUM_NOD_NS(i+1),))

        if num_dist_fact_in_set:
            self.db.createDimension(DIM_NUM_DF_NS(i+1), num_dist_fact_in_set)
            self.db.createVariable(VAR_FACT_NS(i+1), DTYPE_FLT,
                                   (DIM_NUM_NOD_NS(i+1),))

        self.db.variables[VAR_NS_STAT][i] = 1

        self.counter[PX_NODE_SET] += 1

    def put_node_set(self, node_set_id, node_set_node_list):
        """Writes the node list for a single node set.

        Parameters
        ----------
        node_ set_id : int
            The node set ID.

        node_set_node_list : array_like
            Array containing the node list for the node set. Internal node IDs
            are used in this list.

        Notes
        -----
        The function put_node_set_param must be called before this routine is
        invoked.

        """
        node_set_id = int(node_set_id)
        i = self.get_obj_idx(PX_NODE_SET, node_set_id)
        if i is None:
            raise ExodusIIFileError("bad node set ID")
        nodes = node_set_node_list + PX_OFFSET
        self.db.variables[VAR_NODE_NS(i+1)][:] = nodes
        return

    def put_node_set_dist_fact(self, node_set_id, node_set_dist_fact):
        """Writes distribution factors for a single node set

        Parameters
        ----------
        node_ set_id : int
            The node set ID.

        node_set_dist_fact : array_like
            Array containing the distribution factors for each node in the set

        Notes
        -----
        The function put_node_set_param must be called before this routine is
        invoked.

        """
        node_set_id = int(node_set_id)
        i = self.get_obj_idx(PX_NODE_SET, node_set_id)
        if i is None:
            raise ExodusIIFileError("bad node set ID")
        dim = self.db.dimensions[DIM_NUM_DF_NS(i+1)]
        if len(node_set_dist_fact) != dim:
            raise ExodusIIFileError("len(node_set_dist_fact) incorrect")
        self.db.variables[VAR_FACT_NS(i+1)][:] = node_set_dist_fact

    def put_side_set_param(self, side_set_id, num_sides_in_set,
                           num_dist_fact_in_set=0):
        """Writes the side set ID, the number of sides (faces on 3-d element,
        edges on 2-d) which describe a single side set, and the number of
        distribution factors on the side set.

        Parameters
        ----------
        side_set_id : int
            The side set ID

        num_sides_in_set : int
            Number of sides in set

        num_dist_fact_in_set : int
            The number of distribution factors in the side set. This should be
            either 0 (zero) for no factors, or should equal num_sides_in_set.

        """
        num_side_sets = self.db.dimensions[DIM_NUM_SS]
        if self.counter[PX_SIDE_SET] == num_side_sets:
            raise ExodusIIFileError("number of side sets exceeded")

        i = self.counter[PX_SIDE_SET]
        self.register_id(PX_SIDE_SET, side_set_id, i)

        # store side set ID
        self.db.variables[VAR_SS_IDS][i] = int(side_set_id)

        self.db.createDimension(DIM_NUM_SIDE_SS(i+1), num_sides_in_set)
        self.db.createVariable(VAR_SIDE_SS(i+1), DTYPE_INT, (DIM_NUM_SIDE_SS(i+1),))
        self.db.createVariable(VAR_ELEM_SS(i+1), DTYPE_INT, (DIM_NUM_SIDE_SS(i+1),))

        if num_dist_fact_in_set:
            self.db.createDimension(DIM_NUM_DF_SS(i+1), num_dist_fact_in_set)
            self.db.createVariable(VAR_FACT_SS(i+1), DTYPE_FLT,
                                   (DIM_NUM_DF_SS(i+1),))

        self.db.variables[VAR_SS_STAT][i] = 1

        self.counter[PX_SIDE_SET] += 1

    def put_side_set(self, side_set_id, side_set_elem_list,
                     side_set_side_list):
        """Writes the side set element list and side set side (face on 3-d
        element types; edge on 2-d element types) list for a single side set.

        Parameters
        ----------
        side_ set_id : int
            The side set ID.

        side_set_elem_list : array_like
            Array containing the elements in the side set. Internal element
            IDs are used in this list

        side_set_side_list : array_like
            Array containing the side in the side set

        Notes
        -----
        The function put_side_set_param must be called before this routine is
        invoked.

        """
        side_set_id = int(side_set_id)
        i = self.get_obj_idx(PX_SIDE_SET, side_set_id)
        if i is None:
            raise ExodusIIFileError("bad side set ID")

        sides = side_set_side_list + PX_OFFSET
        elems = side_set_elem_list + PX_OFFSET
        self.db.variables[VAR_SIDE_SS(i+1)][:] = sides
        self.db.variables[VAR_ELEM_SS(i+1)][:] = elems
        return

    def put_side_set_dist_fact(self, side_set_id, side_set_dist_fact):
        """Writes distribution factors for a single side set

        Parameters
        ----------
        side_ set_id : int
            The side set ID.

        side_set_dist_fact : array_like
            Array containing the distribution factors for each side in the set

        Notes
        -----
        The function put_side_set_param must be called before this routine is
        invoked.

        """
        side_set_id = int(side_set_id)
        i = self.get_obj_idx(PX_SIDE_SET, side_set_id)
        if i is None:
            raise ExodusIIFileError("bad side set ID")
        dim = self.db.dimensions[DIM_NUM_DF_SS(i+1)]
        if len(side_set_dist_fact) != dim:
            raise ExodusIIFileError("len(side_set_dist_fact) incorrect")
        self.db.variables[VAR_FACT_SS(i+1)][:] = side_set_dist_fact

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
        self.db.createDimension(DIM_NUM_QA, num_qa_records)
        self.db.createVariable(VAR_QA_TITLE, DTYPE_TXT,
                               (DIM_NUM_QA, DIM_N4, DIM_STR))
        for (i, qa_record) in enumerate(qa_records):
            self.db.variables[VAR_QA_TITLE][i, 0, :] = qa_record[0]
            self.db.variables[VAR_QA_TITLE][i, 1, :] = qa_record[1]
            self.db.variables[VAR_QA_TITLE][i, 2, :] = qa_record[2]
            self.db.variables[VAR_QA_TITLE][i, 3, :] = qa_record[3]
        return

    def put_info(self, num_info, info):
        """Writes information records to the database. The records are
        MAX_LINE_LENGTH-character strings.

        Parameters
        ----------
        info : array_like, (num_info, )
            Array containing the information records

        """
        """Reads/writes information records to the database"""
        num_info = len(info)
        self.db.createDimension(DIM_NUM_INFO, num_info)
        self.db.createVariable(VAR_INFO, DTYPE_TXT, (DIM_NUM_INFO, DIM_LIN))
        for (i, info_record) in enumerate(info):
            self.db.variables[VAR_INFO][i] = info_record
        return

