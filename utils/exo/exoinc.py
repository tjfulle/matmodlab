import numpy as np


class Struct(object):
    pass


class NotYetImplemented(Exception):
    def __init__(self, meth):
        self.message = "{0}: ExodusIIFile method not yet implemented".format(meth)
        super(NotYetImplemented, self).__init__(self.message)


class ExodusIIFileError(Exception):
    pass


def ex_catstr(string, num):
    return "{0}{1}".format(string, num)


def ex_catstr2(string1, num1, string2, num2):
    return "{0}{1}{2}{3}".format(string1, num1, string2, num2)


def chara_to_text(chara):
    if chara.ndim == 1:
        return "".join(chara).strip()
    return np.array(["".join(row).strip() for row in chara])


# ------------------------------------------------------------ exodusII.h --- #
EX_NOCLOBBER          =  0
EX_CLOBBER            =  1
EX_NORMAL_MODEL       =  2
EX_LARGE_MODEL        =  4
EX_NETCDF4            =  8
EX_NOSHARE            = 16
EX_SHARE              = 32

EX_READ               =  0
EX_WRITE              =  1

EX_ELEM_BLOCK         =  1
EX_NODE_SET           =  2
EX_SIDE_SET           =  3
EX_ELEM_MAP           =  4
EX_NODE_MAP           =  5
EX_EDGE_BLOCK         =  6
EX_EDGE_SET           =  7
EX_FACE_BLOCK         =  8
EX_FACE_SET           =  9
EX_ELEM_SET           = 10
EX_EDGE_MAP           = 11
EX_FACE_MAP           = 12
EX_GLOBAL             = 13
EX_NODE               = 15  # not defined in exodus
EX_EDGE               = 16  # not defined in exodus
EX_FACE               = 17  # not defined in exodus
EX_ELEM               = 18  # not defined in exodus

MAX_STR_LENGTH        =  32
MAX_VAR_NAME_LENGTH   =  20
MAX_LINE_LENGTH       =  80
MAX_ERR_LENGTH        =  256

EX_VERBOSE     = 1
EX_DEBUG       = 2
EX_ABORT       = 4

EX_INQ_FILE_TYPE       =  1  # inquire EXODUS II file type
EX_INQ_API_VERS        =  2  # inquire API version number
EX_INQ_DB_VERS         =  3  # inquire database version number
EX_INQ_TITLE           =  4  # inquire database title
EX_INQ_DIM             =  5  # inquire number of dimensions
EX_INQ_NODES           =  6  # inquire number of nodes
EX_INQ_ELEM            =  7  # inquire number of elements
EX_INQ_ELEM_BLK        =  8  # inquire number of element blocks
EX_INQ_NODE_SETS       =  9  # inquire number of node sets
EX_INQ_NS_NODE_LEN     = 10  # inquire length of node set node list
EX_INQ_SIDE_SETS       = 11  # inquire number of side sets
EX_INQ_SS_NODE_LEN     = 12  # inquire length of side set node list
EX_INQ_SS_ELEM_LEN     = 13  # inquire length of side set element list
EX_INQ_QA              = 14  # inquire number of QA records
EX_INQ_INFO            = 15  # inquire number of info records
EX_INQ_TIME            = 16  # inquire number of time steps in the database
EX_INQ_EB_PROP         = 17  # inquire number of element block properties
EX_INQ_NS_PROP         = 18  # inquire number of node set properties
EX_INQ_SS_PROP         = 19  # inquire number of side set properties
EX_INQ_NS_DF_LEN       = 20  # inquire length of node set distribution
                             # factor list
EX_INQ_SS_DF_LEN       = 21  # inquire length of side set distribution
                             # factor list
EX_INQ_LIB_VERS        = 22  # inquire API Lib vers number
EX_INQ_EM_PROP         = 23  # inquire number of element map properties
EX_INQ_NM_PROP         = 24  # inquire number of node map properties
EX_INQ_ELEM_MAP        = 25  # inquire number of element maps
EX_INQ_NODE_MAP        = 26  # inquire number of node maps
EX_INQ_EDGE            = 27  # inquire number of edges
EX_INQ_EDGE_BLK        = 28  # inquire number of edge blocks
EX_INQ_EDGE_SETS       = 29  # inquire number of edge sets
EX_INQ_ES_LEN          = 30  # inquire length of concat edge set edge list
EX_INQ_ES_DF_LEN       = 31  # inquire length of concat edge set dist
                             # factor list
EX_INQ_EDGE_PROP       = 32  # inquire number of properties stored per
                             # edge block
EX_INQ_ES_PROP         = 33  # inquire number of properties stored per edge set
EX_INQ_FACE            = 34  # inquire number of faces
EX_INQ_FACE_BLK        = 35  # inquire number of face blocks
EX_INQ_FACE_SETS       = 36  # inquire number of face sets
EX_INQ_FS_LEN          = 37  # inquire length of concat face set face list
EX_INQ_FS_DF_LEN       = 38  # inquire length of concat face set dist
                             # factor list
EX_INQ_FACE_PROP       = 39  # inquire number of properties stored per
                             # face block
EX_INQ_FS_PROP         = 40  # inquire number of properties stored per face set
EX_INQ_ELEM_SETS       = 41  # inquire number of element sets
EX_INQ_ELS_LEN         = 42  # inquire length of concat element set element list
EX_INQ_ELS_DF_LEN      = 43  # inquire length of concat element set dist
                             # factor list
EX_INQ_ELS_PROP        = 44  # inquire number of properties stored per elem set
EX_INQ_EDGE_MAP        = 45  # inquire number of edge maps
EX_INQ_FACE_MAP        = 46  # inquire number of face maps
EX_INQ_COORD_FRAMES    = 47  # inquire number of coordinate frames

# -------------------------------------------------------- exodusII_inc.h --- #
MAX_VAR_NAME_LENGTH = 20   # Internal use only

# Default "filesize" for newly created files.
# Set to 0 for normal filesize setting.
# Set to 1 for EXODUS_LARGE_MODEL setting to be the default
EXODUS_DEFAULT_SIZE = 1

# Exodus error return codes - function return values:
EX_FATAL = -1 # fatal error flag def
EX_NOERR =  0 # no error flag def
EX_WARN  =  1 # warning flag def

# This file contains defined constants that are used internally in the
# EXODUS II API.
#
# The first group of constants refer to netCDF variables, attributes, or
# dimensions in which the EXODUS II data are stored.  Using the defined
# constants will allow the names of the netCDF entities to be changed easily
# in the future if needed.  The first three letters of the constant identify
# the netCDF entity as a variable (VAR), dimension (DIM), or attribute (ATT).
#
# NOTE: The entity name should not have any blanks in it.  Blanks are
#       technically legal but some netcdf utilities (ncgen in particular)
#       fail when they encounter a blank in a name.
#
# DEFINED CONSTANT        ENTITY NAME     DATA STORED IN ENTITY
ATT_TITLE = "title" # the database title
ATT_API_VERSION = "api_version" # the EXODUS II api vers number
ATT_VERSION = "version" # the EXODUS II file vers number
ATT_FILESIZE = "file_size" # 1=large, 0=normal
ATT_FLT_WORDSIZE = "floating_point_word_size" # word size of floating
                                              # point numbers in file
ATT_FLT_WORDSIZE_BLANK = "floating point word size" # word size of floating
                                                    # point numbers in file
                                                    # used for db version 2.01
                                                    # and earlier
DIM_NUM_NODES = "num_nodes" # number of nodes
DIM_NUM_DIM = "num_dim" # number of dimensions; 2- or 3-d
DIM_NUM_EDGE = "num_edge" # number of edges (over all blks)
DIM_NUM_FACE = "num_face" # number of faces (over all blks)
DIM_NUM_ELEM = "num_elem" # number of elements
DIM_NUM_EL_BLK = "num_el_blk" # number of element blocks
DIM_NUM_ED_BLK = "num_ed_blk" # number of edge blocks
DIM_NUM_FA_BLK = "num_fa_blk" # number of face blocks

VAR_COORD = "coord" # nodal coordinates
VAR_COORD_X = "coordx" # X-dimension coordinate
VAR_COORD_Y = "coordy" # Y-dimension coordinate
VAR_COORD_Z = "coordz" # Z-dimension coordinate
VAR_NAME_COOR = "coor_names" # names of coordinates
VAR_NAME_EL_BLK = "eb_names" # names of element blocks
VAR_NAME_NS = "ns_names" # names of node sets
VAR_NAME_SS = "ss_names" # names of side sets
VAR_NAME_EM = "emap_names" # names of element maps
VAR_NAME_EDM = "edmap_names" # names of edge maps
VAR_NAME_FAM = "famap_names" # names of face maps
VAR_NAME_NM = "nmap_names" # names of node maps
VAR_NAME_ED_BLK = "ed_names" # names of edge blocks
VAR_NAME_FA_BLK = "fa_names" # names of face blocks
VAR_NAME_ES = "es_names" # names of edge sets
VAR_NAME_FS = "fs_names" # names of face sets
VAR_NAME_ELS = "els_names" # names of element sets
VAR_STAT_EL_BLK = "eb_status" # element block status
VAR_STAT_ECONN = "econn_status" # element block edge status
VAR_STAT_FCONN = "fconn_status" # element block face status
VAR_STAT_ED_BLK = "ed_status" # edge block status
VAR_STAT_FA_BLK = "fa_status" # face block status
VAR_ID_EL_BLK = "eb_prop1" # element block ids props
VAR_ID_ED_BLK = "ed_prop1" # edge block ids props
VAR_ID_FA_BLK = "fa_prop1" # face block ids props

ATT_NAME_ELB = "elem_type" # element type names for each element block

# number of elements in element block num
DIM_NUM_EL_IN_BLK = lambda num: ex_catstr("num_el_in_blk", num)

# number of nodes per element in element block num
DIM_NUM_NOD_PER_EL = lambda num: ex_catstr("num_nod_per_el", num)

# number of attributes in element block num
DIM_NUM_ATT_IN_BLK = lambda num: ex_catstr("num_att_in_blk", num)

# number of edges in edge block num
DIM_NUM_ED_IN_EBLK = lambda num: ex_catstr("num_ed_in_blk", num)

# number of nodes per edge in edge block num
DIM_NUM_NOD_PER_ED = lambda num: ex_catstr("num_nod_per_ed", num)

# number of edges per element in element block num
DIM_NUM_EDG_PER_EL = lambda num: ex_catstr("num_edg_per_el", num)

# number of attributes in edge block num
DIM_NUM_ATT_IN_EBLK = lambda num: ex_catstr("num_att_in_eblk", num)

# number of faces in face block num
DIM_NUM_FA_IN_FBLK = lambda num: ex_catstr("num_fa_in_blk", num)

# number of nodes per face in face block num
DIM_NUM_NOD_PER_FA = lambda num: ex_catstr("num_nod_per_fa", num)

# number of faces per element in element block num
DIM_NUM_FAC_PER_EL = lambda num: ex_catstr("num_fac_per_el", num)

# number of attributes in face block num
DIM_NUM_ATT_IN_FBLK = lambda num: ex_catstr("num_att_in_fblk", num)
DIM_NUM_ATT_IN_NBLK = "num_att_in_nblk"

# element connectivity for element block num
VAR_CONN = lambda num: ex_catstr("connect", num)

# array containing number of entity per entity for n-sided face/element blocks
VAR_EBEPEC = lambda num: ex_catstr("ebepecnt", num)

# list of attributes for element block num
VAR_ATTRIB = lambda num: ex_catstr("attrib", num)

# list of attribute names for element block num
VAR_NAME_ATTRIB = lambda num: ex_catstr("attrib_name", num)

# list of the numth property for all element blocks
VAR_EB_PROP = lambda num: ex_catstr("eb_prop", num)

# edge connectivity for element block num
VAR_ECONN = lambda num: ex_catstr("edgconn", num)

# edge connectivity for edge block num
VAR_EBCONN = lambda num: ex_catstr("ebconn", num)

# list of attributes for edge block num
VAR_EATTRIB = lambda num: ex_catstr("eattrb", num)
# list of attribute names for edge block num
VAR_NAME_EATTRIB = lambda num: ex_catstr("eattrib_name", num)

VAR_NATTRIB = "nattrb"
VAR_NAME_NATTRIB = "nattrib_name"
DIM_NUM_ATT_IN_NBLK = "num_att_in_nblk"
VAR_NSATTRIB = lambda num: ex_catstr("nsattrb", num)
VAR_NAME_NSATTRIB = lambda num: ex_catstr("nsattrib_name", num)
DIM_NUM_ATT_IN_NS = lambda num: ex_catstr("num_att_in_ns", num)
VAR_SSATTRIB = lambda num: ex_catstr("ssattrb", num)
VAR_NAME_SSATTRIB = lambda num: ex_catstr("ssattrib_name", num)
DIM_NUM_ATT_IN_SS = lambda num: ex_catstr("num_att_in_ss", num)
VAR_ESATTRIB = lambda num: ex_catstr("esattrb", num)
VAR_NAME_ESATTRIB = lambda num: ex_catstr("esattrib_name", num)
DIM_NUM_ATT_IN_ES = lambda num: ex_catstr("num_att_in_es", num)
VAR_FSATTRIB = lambda num: ex_catstr("fsattrb", num)
VAR_NAME_FSATTRIB = lambda num: ex_catstr("fsattrib_name", num)
DIM_NUM_ATT_IN_FS = lambda num: ex_catstr("num_att_in_fs", num)
VAR_ELSATTRIB = lambda num: ex_catstr("elsattrb", num)
VAR_NAME_ELSATTRIB = lambda num: ex_catstr("elsattrib_name", num)
DIM_NUM_ATT_IN_ELS = lambda num: ex_catstr("num_att_in_els", num)
VAR_ED_PROP = lambda num: ex_catstr("ed_prop", num)
                                           # list of the numth property
                                           # for all edge blocks
VAR_FCONN = lambda num: ex_catstr("facconn", num)
                                         # face connectivity for
                                         # element block num
VAR_FBCONN = lambda num: ex_catstr("fbconn", num)
                                         # face connectivity for
                                         # face block num
VAR_FBEPEC = lambda num: ex_catstr("fbepecnt", num)
                                           # array containing number of entity per
                                           # entity for n-sided face/element blocks
VAR_FATTRIB = lambda num: ex_catstr("fattrb", num)
                                          # list of attributes for
                                          # face block num
VAR_NAME_FATTRIB = lambda num: ex_catstr("fattrib_name", num)
                                                     # list of attribute names
                                                     # for face block num
VAR_FA_PROP = lambda num: ex_catstr("fa_prop", num)
                                           # list of the numth property
                                           # for all face blocks
ATT_PROP_NAME = "name" # name attached to element
                                                 # block, node set, side
                                                 # set, element map, or
                                                 # map properties
DIM_NUM_SS = "num_side_sets" # number of side sets
VAR_SS_STAT = "ss_status" # side set status
VAR_SS_IDS = "ss_prop1" # side set id properties
DIM_NUM_SIDE_SS = lambda num: ex_catstr("num_side_ss", num) # number of sides in
                                                           # side set num

DIM_NUM_DF_SS = lambda num: ex_catstr("num_df_ss", num) # number of distribution
                                                       # factors in side set num

# the distribution factors for each node in side set num
VAR_FACT_SS = lambda num: ex_catstr("dist_fact_ss", num)
VAR_ELEM_SS = lambda num: ex_catstr("elem_ss", num)
                                           # list of elements in side
                                           # set num
VAR_SIDE_SS = lambda num: ex_catstr("side_ss", num)
                                           # list of sides in side set
VAR_SS_PROP = lambda num: ex_catstr("ss_prop", num)
                                           # list of the numth property
                                           # for all side sets
DIM_NUM_ES = "num_edge_sets"# number of edge sets
VAR_ES_STAT = "es_status" # edge set status
VAR_ES_IDS = "es_prop1" # edge set id properties
DIM_NUM_EDGE_ES = lambda num: ex_catstr("num_edge_es", num)
                                                   # number of edges in edge set num
DIM_NUM_DF_ES = lambda num: ex_catstr("num_df_es", num)
                                               # number of distribution factors
                                               # in edge set num
VAR_FACT_ES = lambda num: ex_catstr("dist_fact_es", num)
                                                # the distribution factors
                                                # for each node in edge
                                                # set num
VAR_EDGE_ES = lambda num: ex_catstr("edge_es", num)
                                           # list of edges in edge
                                           # set num
VAR_ORNT_ES = lambda num: ex_catstr("ornt_es", num)
                                           # list of orientations in
                                           # the edge set.
VAR_ES_PROP = lambda num: ex_catstr("es_prop", num)
                                           # list of the numth property
                                           # for all edge sets
DIM_NUM_FS = "num_face_sets"# number of face sets
VAR_FS_STAT = "fs_status" # face set status
VAR_FS_IDS = "fs_prop1" # face set id properties
DIM_NUM_FACE_FS = lambda num: ex_catstr("num_face_fs", num)
                                                   # number of faces in side set num
DIM_NUM_DF_FS = lambda num: ex_catstr("num_df_fs", num)
                                               # number of distribution factors
                                               # in face set num
VAR_FACT_FS = lambda num: ex_catstr("dist_fact_fs", num)
                                                # the distribution factors
                                                # for each node in face
                                                # set num
VAR_FACE_FS = lambda num: ex_catstr("face_fs", num)
                                           # list of elements in face
                                           # set num
VAR_ORNT_FS = lambda num: ex_catstr("ornt_fs", num)
                                           # list of sides in side set
VAR_FS_PROP = lambda num: ex_catstr("fs_prop", num)
                                           # list of the numth property
                                           # for all face sets
DIM_NUM_ELS = "num_elem_sets"# number of elem sets
DIM_NUM_ELE_ELS = lambda num: ex_catstr("num_ele_els", num)
                                                   # number of elements in elem set
                                                   # num
DIM_NUM_DF_ELS = lambda num: ex_catstr("num_df_els", num)
                                                 # number of distribution factors
                                                 # in element set num
VAR_ELS_STAT = "els_status" # elem set status
VAR_ELS_IDS = "els_prop1" # elem set id properties
VAR_ELEM_ELS = lambda num: ex_catstr("elem_els", num)
                                             # list of elements in elem
                                             # set num
VAR_FACT_ELS = lambda num: ex_catstr("dist_fact_els", num)
                                                  # list of distribution
                                                  # factors in elem set num
VAR_ELS_PROP = lambda num: ex_catstr("els_prop", num)
                                             # list of the numth property
                                             # for all elem sets
DIM_NUM_NS = "num_node_sets"# number of node sets
DIM_NUM_NOD_NS = lambda num: ex_catstr("num_nod_ns", num)
                                                 # number of nodes in node set
                                                 # num
DIM_NUM_DF_NS = lambda num: ex_catstr("num_df_ns", num)
                                               # number of distribution factors
                                               # in node set num
VAR_NS_STAT = "ns_status" # node set status
VAR_NS_IDS = "ns_prop1" # node set id properties
VAR_NODE_NS = lambda num: ex_catstr("node_ns", num)
                                           # list of nodes in node set
                                           # num
VAR_FACT_NS = lambda num: ex_catstr("dist_fact_ns", num)
                                                # list of distribution
                                                # factors in node set num
VAR_NS_PROP = lambda num: ex_catstr("ns_prop", num)
                                           # list of the numth property
                                           # for all node sets
DIM_NUM_QA = "num_qa_rec" # number of QA records
VAR_QA_TITLE = "qa_records" # QA records
DIM_NUM_INFO = "num_info" # number of information records
VAR_INFO = "info_records" # information records
VAR_WHOLE_TIME = "time_whole" # simulation times for whole
                                                          # time steps
VAR_ELEM_TAB = "elem_var_tab" # element variable truth
                                                      # table
VAR_EBLK_TAB = "edge_var_tab" # edge variable truth table
VAR_FBLK_TAB = "face_var_tab" # face variable truth table
VAR_ELSET_TAB = "elset_var_tab" # elemset variable truth
                                                        # table
VAR_SSET_TAB = "sset_var_tab" # sideset variable truth
                                                      # table
VAR_FSET_TAB = "fset_var_tab" # faceset variable truth
                                                      # table
VAR_ESET_TAB = "eset_var_tab" # edgeset variable truth
                                                      # table
VAR_NSET_TAB = "nset_var_tab" # nodeset variable truth
                                                      # table
DIM_NUM_GLO_VAR = "num_glo_var" # number of global variables
VAR_NAME_GLO_VAR = "name_glo_var" # names of global variables
VAR_GLO_VAR = "vals_glo_var" # values of global variables
DIM_NUM_NOD_VAR = "num_nod_var" # number of nodal variables
VAR_NAME_NOD_VAR = "name_nod_var" # names of nodal variables
VAR_NOD_VAR = "vals_nod_var" # values of nodal variables

# values of nodal variables
VAR_NOD_VAR_NEW = lambda num: ex_catstr("vals_nod_var", num)

DIM_NUM_ELE_VAR = "num_elem_var" # number of element variables
VAR_NAME_ELE_VAR = "name_elem_var" # names of element variables
# values of element variable num1 in element block num2
VAR_ELEM_VAR = lambda num1, num2: ex_catstr2("vals_elem_var", num1, "eb", num2)

# values of edge variable num1 in edge block num2
DIM_NUM_EDG_VAR = "num_edge_var" # number of edge variables
VAR_NAME_EDG_VAR = "name_edge_var" # names of edge variables
VAR_EDGE_VAR = lambda num1, num2: ex_catstr2("vals_edge_var", num1, "eb", num2)

# values of face variable num1 in face block num2
DIM_NUM_FAC_VAR = "num_face_var" # number of face variables
VAR_NAME_FAC_VAR = "name_face_var" # names of face variables
VAR_FACE_VAR = lambda num1, num2: ex_catstr2("vals_face_var", num1,"fb", num2)

# values of nodeset variable num1 in nodeset num2
DIM_NUM_NSET_VAR = "num_nset_var" # number of nodeset variables
VAR_NAME_NSET_VAR = "name_nset_var" # names of nodeset variables
VAR_NS_VAR = lambda num1, num2: ex_catstr2("vals_nset_var", num1,"ns", num2)

# values of edgeset variable num1 in edgeset num2
DIM_NUM_ESET_VAR = "num_eset_var" # number of edgeset variables
VAR_NAME_ESET_VAR = "name_eset_var" # names of edgeset variables
VAR_ES_VAR = lambda num1, num2: ex_catstr2("vals_eset_var", num1,"es", num2)

# values of faceset variable num1 in faceset num2
DIM_NUM_FSET_VAR = "num_fset_var" # number of faceset variables
VAR_NAME_FSET_VAR = "name_fset_var" # names of faceset variables
VAR_FS_VAR = lambda num1, num2: ex_catstr2("vals_fset_var", num1,"fs", num2)

# values of sideset variable num1 in sideset num2
DIM_NUM_SSET_VAR = "num_sset_var" # number of sideset variables
VAR_NAME_SSET_VAR = "name_sset_var" # names of sideset variables
VAR_SS_VAR = lambda num1, num2: ex_catstr2("vals_sset_var", num1,"ss", num2)

# values of elemset variable num1 in elemset num2
DIM_NUM_ELSET_VAR = "num_elset_var" # number of element set variables
VAR_NAME_ELSET_VAR = "name_elset_var"# names of elemset variables
VAR_ELS_VAR = lambda num1, num2: ex_catstr2("vals_elset_var", num1,"es", num2)

# general dimension of length MAX_STR_LENGTH used for name lengths
DIM_STR = "len_string"

# general dimension of length MAX_LINE_LENGTH used for long strings
DIM_LIN = "len_line"
DIM_N4 = "four" # general dimension of length 4

# unlimited (expandable) dimension for time steps
DIM_TIME = "time_step"

DIM_NUM_EM = "num_elem_maps" # number of element maps
VAR_ELEM_MAP = lambda num: ex_catstr("elem_map", num) # the numth element map
VAR_EM_PROP = lambda num: ex_catstr("em_prop", num) # list of the numth property
                                                    # for all element maps

DIM_NUM_EDM = "num_edge_maps" # number of edge maps
VAR_EDGE_MAP = lambda num: ex_catstr("edge_map", num) # the numth edge map
VAR_EDM_PROP = lambda num: ex_catstr("edm_prop", num) # list of the numth property
                                                     # for all edge maps

DIM_NUM_FAM = "num_face_maps" # number of face maps
VAR_FACE_MAP = lambda num: ex_catstr("face_map", num) # the numth face map
VAR_FAM_PROP = lambda num: ex_catstr("fam_prop", num) # list of the numth property
                                                     # for all face maps

DIM_NUM_NM = "num_node_maps" # number of node maps
VAR_NODE_MAP = lambda num: ex_catstr("node_map", num) # the numth node map
VAR_NM_PROP = lambda num: ex_catstr("nm_prop", num) # list of the numth property
                                                   # for all node maps

DIM_NUM_CFRAMES = "num_cframes"
DIM_NUM_CFRAME9 = "num_cframes_9"
VAR_FRAME_COORDS = "frame_coordinates"
VAR_FRAME_IDS = "frame_ids"
VAR_FRAME_TAGS = "frame_tags"

EX_EL_UNK = -1,     # unknown entity
EX_EL_NULL_ELEMENT = 0
EX_EL_TRIANGLE =   1  # Triangle entity
EX_EL_QUAD =   2  # Quad entity
EX_EL_HEX =   3  # Hex entity
EX_EL_WEDGE =   4  # Wedge entity
EX_EL_TETRA =   5  # Tetra entity
EX_EL_TRUSS =   6  # Truss entity
EX_EL_BEAM =   7  # Beam entity
EX_EL_SHELL =   8  # Shell entity
EX_EL_SPHERE =   9  # Sphere entity
EX_EL_CIRCLE =  10  # Circle entity
EX_EL_TRISHELL =  11  # Triangular Shell entity
EX_EL_PYRAMID =  12  # Pyramid entity
ex_element_type = {
    EX_EL_UNK: EX_EL_UNK,
    EX_EL_NULL_ELEMENT: EX_EL_NULL_ELEMENT,
    EX_EL_TRIANGLE: EX_EL_TRIANGLE,
    EX_EL_QUAD: EX_EL_QUAD,
    EX_EL_HEX: EX_EL_HEX,
    EX_EL_WEDGE: EX_EL_WEDGE,
    EX_EL_TETRA: EX_EL_TETRA,
    EX_EL_TRUSS: EX_EL_TRUSS,
    EX_EL_BEAM: EX_EL_BEAM,
    EX_EL_SHELL: EX_EL_SHELL,
    EX_EL_SPHERE: EX_EL_SPHERE,
    EX_EL_CIRCLE: EX_EL_CIRCLE,
    EX_EL_TRISHELL: EX_EL_TRISHELL,
    EX_EL_PYRAMID: EX_EL_PYRAMID
    }

EX_CF_RECTANGULAR = 1
EX_CF_CYLINDRICAL = 2
EX_CF_SPHERICAL = 3
ex_coordinate_frame_type = {
    EX_CF_RECTANGULAR: EX_CF_RECTANGULAR,
    EX_CF_CYLINDRICAL: EX_CF_CYLINDRICAL,
    EX_CF_SPHERICAL: EX_CF_SPHERICAL
}

elem_blk_parm = Struct()
elem_blk_parm.elem_type = None
elem_blk_parm.elem_blk_id = None
elem_blk_parm.num_elem_in_blk = None
elem_blk_parm.num_nodes_per_elem = None
elem_blk_parm.num_sides = None
elem_blk_parm.num_nodes_per_side = None
elem_blk_parm.num_attr = None
elem_blk_parm.elem_ctr = None

# ------------------------------------------------------------ exofile.py --- #
ATT_FILENAME = "filename"
ATT_RUNID = "runid"
DTYPE_FLT = "f"
DTYPE_INT = "i"
DTYPE_TXT = "c"

PX_VAR_EL_MAP = "elem_map"
PX_VAR_COORDS = lambda i: ex_catstr(VAR_COORD, {0: "x", 1: "y", 2: "z"}[i])
PX_DIM_VARS = lambda s: {"G": DIM_NUM_GLO_VAR, "N": DIM_NUM_NOD_VAR,
                         "E": DIM_NUM_ELE_VAR, "M": DIM_NUM_NSET_VAR,
                         "S": DIM_NUM_SSET_VAR}[s.upper()]
PX_VAR_NAMES = lambda s: {"G": VAR_NAME_GLO_VAR, "N": VAR_NAME_NOD_VAR,
                          "E": VAR_NAME_ELE_VAR, "M": VAR_NAME_NSET_VAR,
                          "S": VAR_NAME_SSET_VAR}[s.upper()]
PX_VAR_GLO = "G"
PX_VAR_NOD = "N"
PX_VAR_ELE = "E"
PX_VAR_NS = "M"
PX_VAR_SS = "S"
PX_ELEM_BLK = "EB"
PX_NODE_SET = "NS"
PX_SIDE_SET = "SS"
PX_X_COMP  = 0
PX_Y_COMP  = 1
PX_Z_COMP  = 2
PX_OFFSET = 1
PX_HUGE = 1.E+99

def PX_PROPINFO(obj_type):
    if obj_type == EX_ELEM_BLOCK:
        return VAR_EB_PROP, DIM_NUM_EL_BLK
    if obj_type == EX_FACE_BLOCK:
        return VAR_FA_PROP, DIM_NUM_FA_BLK
    if obj_type == EX_EDGE_BLOCK:
        return VAR_ED_PROP, DIM_NUM_ED_BLK
    if obj_type == EX_NODE_SET:
        return VAR_NS_PROP, DIM_NUM_NS
    if obj_type == EX_SIDE_SET:
        return VAR_SS_PROP, DIM_NUM_SS
    if obj_type == EX_EDGE_SET:
        return VAR_ES_PROP, DIM_NUM_EDGE_ES
    if obj_type == EX_FACE_SET:
        return VAR_FS_PROP, DIM_NUM_FACE_FS
    if obj_type == EX_ELEM_SET:
        return VAR_ELS_PROP, DIM_NUM_ELS
    if obj_type == EX_ELEM_MAP:
        return VAR_EM_PROP, DIM_NUM_EM
    if obj_type == EX_FACE_MAP:
        return VAR_FAM_PROP, DIM_NUM_FAM
    if obj_type == EX_EDGE_MAP:
        return VAR_EDM_PROP, DIM_NUM_EDM
    if obj_type == EX_NODE_MAP:
        return VAR_NM_PROP, DIM_NUM_NM
    raise ExodusIIFileError("{0}: unrecognized obj_type".format(obj_type))
