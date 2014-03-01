import os
import sys
import numpy as np

D  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(D))
from exofile import ExodusIIWriter
from exoconst import *


def main():
    """Reproduction of the 'C Write Example Code' in the Exodus manual

    """
    # create new file
    exofile = ExodusIIWriter.new_from_runid("myrun")

    # initialize the file
    num_dim = 3
    num_nodes = 26
    num_elem = 5
    num_elem_blk = 5
    num_node_sets = 2
    num_side_sets = 5

    exofile.put_init("This is a test", num_dim, num_nodes,
                     num_elem, num_elem_blk, num_node_sets, num_side_sets)

    # write nodal coordinates values and names to database

    # Quad #1
    coords = np.zeros((num_nodes, num_dim))
    coords[0, 0] = 0.; coords[0, 1] = 0.; coords[0, 2] = 0.
    coords[1, 0] = 1.; coords[1, 1] = 0.; coords[1, 2] = 0.
    coords[2, 0] = 1.; coords[2, 1] = 1.; coords[2, 2] = 0.
    coords[3, 0] = 0.; coords[3, 1] = 1.; coords[3, 2] = 0.

    # Quad #2
    coords[4, 0] = 1.; coords[4, 1] = 0.; coords[4, 2] = 0.
    coords[5, 0] = 2.; coords[5, 1] = 0.; coords[5, 2] = 0.
    coords[6, 0] = 2.; coords[6, 1] = 1.; coords[6, 2] = 0.
    coords[7, 0] = 1.; coords[7, 1] = 1.; coords[7, 2] = 0.

    # Hex #1
    coords[8, 0] = 0.;   coords[8, 1] = 0.;   coords[8, 2] = 0.;
    coords[9, 0] = 10.;  coords[9, 1] = 0.;   coords[9, 2] = 0.;
    coords[10, 0] = 10.; coords[10, 1] = 0.;  coords[10, 2] = -10.;
    coords[11, 0] = 0.;  coords[11, 1] = 0.;  coords[11, 2] = -10.;
    coords[12, 0] = 0.;  coords[12, 1] = 10.; coords[12, 2] = 0.;
    coords[13, 0] = 10.; coords[13, 1] = 10.; coords[13, 2] = 0.;
    coords[14, 0] = 10.; coords[14, 1] = 10.; coords[14, 2] = -10.;
    coords[15, 0] = 0.;  coords[15, 1] = 10.; coords[15, 2] = -10.;

    # Tetra #1
    coords[16, 0] = 0.;  coords[16, 1] = 0.; coords[16, 2] = 0.
    coords[17, 0] = 1.;  coords[17, 1] = 0.; coords[17, 2] = 5.
    coords[18, 0] = 10.; coords[18, 1] = 0.; coords[18, 2] = 2.
    coords[19, 0] = 7.;  coords[19, 1] = 0.; coords[19, 2] = 3.

    # Wedge #1
    coords[20, 0] = 3.; coords[20, 1] = 0.; coords[20, 2] = 6.;
    coords[21, 0] = 6.; coords[21, 1] = 0.; coords[21, 2] = 0.;
    coords[22, 0] = 0.; coords[22, 1] = 0.; coords[22, 2] = 0.;
    coords[23, 0] = 3.; coords[23, 1] = 2.; coords[23, 2] = 6.;
    coords[24, 0] = 6.; coords[24, 1] = 2.; coords[24, 2] = 2.;
    coords[25, 0] = 0.; coords[25, 1] = 2.; coords[25, 2] = 0.;

    exofile.put_coord_names(np.array(["xcoor", "ycoor", "zcoor"]))
    exofile.put_coord(coords[:, 0], coords[:, 1], coords[:, 2])

    # write element order map -> Optional, not needed for standard map
    elem_map = np.arange(num_elem)
    exofile.put_map(np.arange(num_elem))

    # Write element block parameters
    # elem_blk_id, elem_type, num_elem_this_blk, num_nodes_per_elem, num_attr
    elem_blocks = [[10, "QUAD", 1, 4, 1],
                   [11, "QUAD", 1, 4, 1],
                   [12, "HEX", 1, 8, 1],
                   [13, "TETRA", 1, 4, 1],
                   [14, "WEDGE", 1, 6, 1]]
    elem_blk_ids = np.array([int(x[0]) for x in elem_blocks])
    num_elem_in_blk = np.array([int(x[2]) for x in elem_blocks])

    for (ebid, etype, neblk, nnpe, na) in elem_blocks:
        exofile.put_elem_block(ebid, etype, neblk, nnpe, na)

    # element block properties
    num_props = 2
    prop_names = np.array(["TOP", "RIGHT"])
    exofile.put_prop_names(EX_ELEM_BLOCK, num_props, prop_names)

    exofile.put_prop(EX_ELEM_BLOCK, elem_blocks[0][0], "TOP", 1)
    exofile.put_prop(EX_ELEM_BLOCK, elem_blocks[1][0], "TOP", 1)
    exofile.put_prop(EX_ELEM_BLOCK, elem_blocks[2][0], "RIGHT", 1)
    exofile.put_prop(EX_ELEM_BLOCK, elem_blocks[3][0], "RIGHT", 1)
    exofile.put_prop(EX_ELEM_BLOCK, elem_blocks[4][0], "RIGHT", 1)

    j = 0
    attrib = np.array([3.14159], dtype=np.float64)
    for i, (ebid, etype, neblk, nnte, na) in enumerate(elem_blocks):

        # write element connectivity
        connect = np.arange(j, j+nnte)
        exofile.put_elem_conn(ebid, connect)

        # write element block attributes
        exofile.put_elem_attr(ebid, float(i + 1.) * attrib)

        j += nnte
        continue

    # write out individual node sets
    # node_set_id, num_nodes_in_set, num_dist_fact_in_set, node_list, dist_fact
    node_sets = [[20, 5, 5, [0, 1, 2, 3, 4], [1., 2., 3., 4., 5.]],
                 [21, 3, 3, [5, 6, 7], [1.1, 2.1, 3.1]]]
    for (nsid, nnis, ndfis, node_list, dist_fact) in node_sets:
        exofile.put_node_set_param(nsid, nnis, ndfis)
        exofile.put_node_set(nsid, np.array(node_list))
        exofile.put_node_set_dist_fact(nsid, np.array(dist_fact))

    exofile.put_prop(EX_NODE_SET, node_sets[0][0], "FACE", 4)
    exofile.put_prop(EX_NODE_SET, node_sets[1][0], "FACE", 5)

    prop_array = np.array([1000, 2000])
    exofile.put_prop_array(EX_NODE_SET, "VELOCITY", prop_array)

    # write out individual side sets
    # side_set_id, num_sides_in_set, num_dist_fact_in_set,
    #   side_set_elem_list, side_set_side_list, side_set_dist_fact
    side_sets = [
        [30, 2, 4, [1, 1], [3, 1], [30., 30.1, 30.2, 30.3]], # side set 1 - quad
        [31, 2, 4, [0, 1], [1, 2], [31., 31.1, 31.2, 31.3]], # side set 2 - quad
                                                             # spanning 2 elements
        [32, 7, 0, [2] * 7, [4, 2, 2, 1, 3, 0, 5], []], # side set 3 - hex
        [33, 4, 0, [3] * 4, [0, 1, 2, 3], []], # side set 4 - tetras
        [34, 5, 0, [4] * 5, [0, 1, 2, 3, 4], []]] # side set 4 - wedges

    for (ssid, nsts, ndfts, ssel, sssl, ssdf) in side_sets:
        exofile.put_side_set_param(ssid, nsts, ndfts)
        exofile.put_side_set(ssid, np.array(ssel), np.array(sssl))
        if ndfts:
            exofile.put_side_set_dist_fact(ssid, np.array(ssdf))

    exofile.put_prop(EX_SIDE_SET, side_sets[0][0], "COLOR", 100)
    exofile.put_prop(EX_SIDE_SET, side_sets[1][0], "COLOR", 101)

    # write QA record
    num_qa_rec = 2
    qa_record = np.array([["TESTWT", "testwt", "05/23/13", "16:29:00"],
                          ["FASTQ", "fastq", "05/23/13", "16:29:00"]])
    exofile.put_qa(num_qa_rec, qa_record)

    # write information record
    num_info = 3
    info = np.array(["This is the first information record.",
                     "This is the second information record.",
                     "This Is the third Information record."])
    exofile.put_info(num_info, info)

    # write results variables parameters and names
    glob_var_names = np.array(["glob_vars"])
    num_glob_vars = glob_var_names.size
    exofile.put_var_param("g", num_glob_vars)
    exofile.put_var_names("g", num_glob_vars, glob_var_names)

    nod_var_names = np.array(["nod_var0", "nod_var1"])
    num_nod_vars = nod_var_names.size
    exofile.put_var_param("n", num_nod_vars)
    exofile.put_var_names("n", num_nod_vars, nod_var_names)

    ele_var_names = np.array(["ele_var0", "ele_var1", "ele_var2"])
    num_ele_vars = ele_var_names.size
    exofile.put_var_param("e", num_ele_vars)
    exofile.put_var_names("e", num_ele_vars, ele_var_names)

    # write element variable truth table
    truth_tab = np.empty((num_elem_blk, num_ele_vars), dtype=np.int)
    for i in range(num_elem_blk):
        for j in range(num_ele_vars):
            truth_tab[i, j] = 1
    exofile.put_elem_var_tab(num_elem_blk, num_ele_vars, truth_tab)

    # for each time step, write the analysis results; the code below fills
    # the arrays glob_var_vals, nodal_var_vals, and elem_var_vals with
    # values for debugging purposes; obviously the analysis code will
    # populate these arrays

    whole_time_step = 1
    num_time_steps = 10
    glob_var_vals = np.zeros(num_glob_vars, dtype=np.float64)
    nodal_var_vals = np.zeros(num_nod_vars, dtype=np.float64)
    elem_var_vals = np.zeros(num_ele_vars, dtype=np.float64)
    for i in range(num_time_steps):
        time_value = float(i + 1) / 100.
        # write time value
        exofile.put_time(whole_time_step, time_value)

        # write global variables
        for j in range(num_glob_vars):
            glob_var_vals[j] = float(j + 2) * time_value
            continue

        exofile.put_glob_vars(whole_time_step, num_glob_vars, glob_var_vals)

        # write nodal variables
        for k in range(num_nod_vars):
            for j in range(num_nodes):
                nodal_var_vals[j] = float(k) + (float(j + 1) * time_value)
                continue
            exofile.put_nodal_var(whole_time_step, k, num_nodes, nodal_var_vals)
            continue

        # write element variables
        for k in range(num_ele_vars):
            for j in range(num_elem_blk):
                for m in range(num_elem_in_blk[j]):
                    elem_var_vals[m] = (float(k + 1) + float(j + 2) +
                                        (float(m + 1) * time_value))
                    continue
                exofile.put_elem_var(whole_time_step, k, elem_blk_ids[j],
                                     num_elem_in_blk[j], elem_var_vals)
                continue
            continue

        whole_time_step += 1

        continue

    # udpate and close the file
    exofile.update()
    exofile.close()


if __name__ == "__main__":
    sys.exit(main())
