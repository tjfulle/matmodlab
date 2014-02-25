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

EX_INQ_FILE_TYPE       =  1      # inquire EXODUS II file type
EX_INQ_API_VERS        =  2      # inquire API version number
EX_INQ_DB_VERS         =  3      # inquire database version number
EX_INQ_TITLE           =  4      # inquire database title
EX_INQ_DIM             =  5      # inquire number of dimensions
EX_INQ_NODES           =  6      # inquire number of nodes
EX_INQ_ELEM            =  7      # inquire number of elements
EX_INQ_ELEM_BLK        =  8      # inquire number of element blocks
EX_INQ_NODE_SETS       =  9      # inquire number of node sets
EX_INQ_NS_NODE_LEN     = 10      # inquire length of node set node list
EX_INQ_SIDE_SETS       = 11      # inquire number of side sets
EX_INQ_SS_NODE_LEN     = 12      # inquire length of side set node list
EX_INQ_SS_ELEM_LEN     = 13      # inquire length of side set element list
EX_INQ_QA              = 14      # inquire number of QA records
EX_INQ_INFO            = 15      # inquire number of info records
EX_INQ_TIME            = 16      # inquire number of time steps in the database
EX_INQ_EB_PROP         = 17      # inquire number of element block properties
EX_INQ_NS_PROP         = 18      # inquire number of node set properties
EX_INQ_SS_PROP         = 19      # inquire number of side set properties
EX_INQ_NS_DF_LEN       = 20      # inquire length of node set distribution
                                 # factor list
EX_INQ_SS_DF_LEN       = 21      # inquire length of side set distribution
                                 # factor list
EX_INQ_LIB_VERS        = 22      # inquire API Lib vers number
EX_INQ_EM_PROP         = 23      # inquire number of element map properties
EX_INQ_NM_PROP         = 24      # inquire number of node map properties
EX_INQ_ELEM_MAP        = 25      # inquire number of element maps
EX_INQ_NODE_MAP        = 26      # inquire number of node maps
EX_INQ_EDGE            = 27      # inquire number of edges
EX_INQ_EDGE_BLK        = 28      # inquire number of edge blocks
EX_INQ_EDGE_SETS       = 29      # inquire number of edge sets
EX_INQ_ES_LEN          = 30      # inquire length of concat edge set edge list
EX_INQ_ES_DF_LEN       = 31      # inquire length of concat edge set dist
                                 # factor list
EX_INQ_EDGE_PROP       = 32      # inquire number of properties stored per
                                 # edge block
EX_INQ_ES_PROP         = 33      # inquire number of properties stored per edge set
EX_INQ_FACE            = 34      # inquire number of faces
EX_INQ_FACE_BLK        = 35      # inquire number of face blocks
EX_INQ_FACE_SETS       = 36      # inquire number of face sets
EX_INQ_FS_LEN          = 37      # inquire length of concat face set face list
EX_INQ_FS_DF_LEN       = 38      # inquire length of concat face set dist
                                 # factor list
EX_INQ_FACE_PROP       = 39      # inquire number of properties stored per
                                 # face block
EX_INQ_FS_PROP         = 40      # inquire number of properties stored per face set
EX_INQ_ELEM_SETS       = 41      # inquire number of element sets
EX_INQ_ELS_LEN         = 42      # inquire length of concat element set element list
EX_INQ_ELS_DF_LEN      = 43      # inquire length of concat element set dist
                                 # factor list
EX_INQ_ELS_PROP        = 44      # inquire number of properties stored per elem set
EX_INQ_EDGE_MAP        = 45      # inquire number of edge maps
EX_INQ_FACE_MAP        = 46      # inquire number of face maps
EX_INQ_COORD_FRAMES    = 47      # inquire number of coordinate frames

# My own custom:
PY_GLOBAL              = "G"
PY_NODAL               = "N"
PY_ELEMENT             = "E"
PY_NODESET             = "M"
PY_SIDESET             = "S"
PY_X_COMP              = 0
PY_Y_COMP              = 1
PY_Z_COMP              = 2
EX_VAR_TYPES = {PY_GLOBAL: "glo",
                PY_NODAL: "nod",
                PY_ELEMENT: "elem",
                PY_NODESET: "nodset",
                PY_SIDESET: "sidset"}
