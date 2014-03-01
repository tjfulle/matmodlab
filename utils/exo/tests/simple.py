import numpy as np
import os
import sys

from exofile import *


def main():
    # create new file
    exofile = ExodusFile.new_from_runid("test_simple")

    # initialize file with parameters
    num_dim = 2
    num_nodes = 8
    num_elem = 2
    num_elem_blk = 2
    num_node_sets = 2
    num_side_sets = 2

    exofile.put_init("This is a test", num_dim, num_nodes,
                     num_elem, num_elem_blk, num_node_sets, num_side_sets)

    # write nodal coordinates values and names to database
    coords = np.array([[0., 0.],
                       [1., 0.],
                       [1., 1.],
                       [0., 1.],
                       [1., 0.],
                       [2., 0.],
                       [2., 1.],
                       [1., 1.]])
    exofile.put_coord(coords[:, 0], coords[:, 1], [])
    exofile.put_coord_names(np.array(["xcoor", "ycoor"]))

    # write element order map
    elem_map = np.arange(num_elem)
    exofile.put_map(elem_map)

    # write element block parameters
    elem_blk_ids = np.array([10, 11])
    num_elem_in_blk = np.array([1, 1])
    for i, ebid in enumerate(elem_blk_ids):
        exofile.put_elem_block(ebid, "QUAD", num_elem_in_blk[i], 4, 1)

    # write element block properties
    prop_names = np.array(["TOP", "RIGHT"])
    exofile.put_prop_names(EX_ELEM_BLOCK, 2, prop_names)

    exofile.put_prop(EX_ELEM_BLOCK, elem_blk_ids[0], "TOP", 1)
    exofile.put_prop(EX_ELEM_BLOCK, elem_blk_ids[1], "RIGHT", 1)

    # write element connectivity
    connect = np.arange(num_nodes)
    exofile.put_elem_conn(elem_blk_ids[0], connect[:4])
    exofile.put_elem_conn(elem_blk_ids[1], connect[4:])

    # write element block attributes
    attrib = np.array([3.14159, 6.14159])
    exofile.put_elem_attr(elem_blk_ids[0], attrib[0])
    exofile.put_elem_attr(elem_blk_ids[1], attrib[1])

    # write individual node sets
    node_list = np.array([0, 3])
    dist_fact = np.linspace(1, 5, 2)
    exofile.put_node_set_param(20, 2, 2)
    exofile.put_node_set(20, node_list)
    exofile.put_node_set_dist_fact(20, dist_fact)

    node_list = np.array([5, 6])
    exofile.put_node_set_param(21, 2, 0)
    exofile.put_node_set(21, node_list)

    # write individual side sets
    elem_list = np.array([0, 1])
    side_list = np.array([0, 0])
    dist_fact = np.array([30., 31.])
    exofile.put_side_set_param(30, 2, 2)
    exofile.put_side_set(30, elem_list, side_list)
    exofile.put_side_set_dist_fact(30, dist_fact)

    # write individual side sets
    elem_list = np.array([0, 1])
    side_list = np.array([2, 2])
    exofile.put_side_set_param(31, 2, 0)
    exofile.put_side_set(31, elem_list, side_list)

    # write QA records
    num_qa_rec = 2
    qa_record = np.array(
        [["TESTWT fortran version", "testwt", "05/23/13", "16:29:00"],
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
    nodal_var_vals = np.zeros(num_nodes, dtype=np.float64)
    elem_var_vals = np.zeros(4)
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
