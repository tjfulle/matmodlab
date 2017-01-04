import os
from scipy.io.netcdf import NetCDFFile

def cat(*args):
    return ''.join(str(a).strip() for a in args)

def adjstr(string):
    return '{0:32s}'.format(string)[:32]

def stringify(a):
    try:
        return ''.join(a).strip()
    except TypeError:
        return [''.join(row).strip() for row in a]

class DatabaseFile(object):
    pass

class DatabaseFileWriter(DatabaseFile):
    mode = 'w'
    def __init__(self, filename):
        '''
        Notes
        -----
        The EXOFile class is an interface to the Exodus II api. Its methods
        are named after the analogous method from the Exodus II C bindings,
        minus the prefix 'ex_'.

        '''
        self.fh = NetCDFFile(filename, mode='w')
        self.jobid = os.path.splitext(os.path.basename(filename))[0]
        self.filename = filename

    def initialize(self, elem_var_names):
        # ------------------------------------------------------------------- #
        # -------------------------------- standard ExodusII dimensioning --- #
        # ------------------------------------------------------------------- #
        self.fh.floating_point_word_size = 4
        self.fh.version = 5.0300002
        self.fh.file_size = 1
        self.fh.api_version = 5.0300002
        self.fh.title = 'Matmodlab material point simulation'

        self.fh.filename = basename(self.filename)
        self.fh.jobid = self.jobid

        self.fh.createDimension('len_string', 33)
        self.fh.createDimension('len_line', 81)
        self.fh.createDimension('four', 4)

        self.fh.createDimension('num_dim', 3)
        self.fh.createDimension('num_node', 8)
        self.fh.createDimension('num_elem', 1)

        # node and element number maps
        self.fh.createVariable('nodes', 'i', ('num_node',))
        self.fh.variables['nodes'][:] = range(1, 9)
        self.fh.createVariable('elements', 'i', ('num_elem',))
        self.fh.variables['elements'][:] = [1]

        # ------------------------------------------------------------------- #
        # ---------------------------------------------------- QA records --- #
        # ------------------------------------------------------------------- #
        now = datetime.datetime.now()
        day = now.strftime("%m/%d/%y")
        hour = now.strftime("%H:%M:%S")
        self.fh.createDimension('num_qa_rec', 1)
        self.fh.createVariable('qa_records', 'c',
                               ('num_qa_rec', 'four', 'len_string'))
        self.fh.variables['qa_records'][0, 0, :] = adjstr('Matmodlab')
        self.fh.variables['qa_records'][0, 1, :] = adjstr(self.jobid)
        self.fh.variables['qa_records'][0, 2, :] = adjstr(day)
        self.fh.variables['qa_records'][0, 3, :] = adjstr(hour)

        # ------------------------------------------------------------------- #
        # ------------------------------------------------- record arrays --- #
        # ------------------------------------------------------------------- #
        self.fh.createDimension('time_step', None)
        self.fh.createVariable('time_whole', 'f', ('time_step',))
        self.fh.createVariable('step_num', 'i', ('time_step',))
        self.fh.createVariable('frame_num', 'i', ('time_step',))

        # ------------------------------------------------------------------- #
        # --------------------------------------- element block meta data --- #
        # ------------------------------------------------------------------- #
        # block IDs - standard map
        self.fh.createDimension('num_el_blk', 1)
        self.fh.createVariable('eb_prop1', 'i', ('num_el_blk',))
        self.fh.variables['eb_prop1'][:] = np.arange(1, dtype=np.int32)+1
        self.fh.variables['eb_prop1'].name = 'ID'

        self.fh.createVariable('eb_status', 'i', ('num_el_blk',))
        self.fh.variables['eb_status'][:] = np.ones(1, dtype=int)

        self.fh.createVariable('eb_names', 'c', ('num_el_blk', 'len_string'))
        self.fh.variables['eb_names'][0][:] = adjstr('ElementBlock1')

        # element map
        self.fh.createDimension('num_el_in_blk1', 1)
        self.fh.createDimension('num_nod_per_el1', 8)
        self.fh.createVariable('elem_map1', 'i', ('num_el_in_blk1',))
        self.fh.variables['elem_map1'][:] = np.arange(1, dtype=np.int32)+1

        # set up the element block connectivity
        dim = ('num_el_in_blk1', 'num_nod_per_el1')
        self.fh.createVariable('connect1', 'i', dim)
        self.fh.variables['connect1'][:] = np.arange(8, dtype=np.int32)+1
        self.fh.variables['connect1'].elem_type = 'HEX'

        # ------------------------------------------------------------------- #
        # -------------------------------------------------- Element data --- #
        # ------------------------------------------------------------------- #
        num_elem_var = len(elem_var_names)
        self.fh.createDimension('num_elem_var', num_elem_var)
        dim = ('num_elem_var', 'len_string')
        self.fh.createVariable('name_elem_var', 'c', dim)
        for (i, name) in enumerate(elem_var_names):
            key = adjstr(name)
            self.fh.variables['name_elem_var'][i, :] = key
            self.fh.createVariable('vals_elem_var{0}eb1'.format(i+1),
                                   'f', ('time_step', 'num_el_in_blk1'))

        # ------------------------------------------------------------------- #
        # ----------------------------------------------------- Node data --- #
        # ------------------------------------------------------------------- #
        coordx = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5])
        coordy = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5])
        coordz = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
        vertices = [coordx, coordy, coordz]
        self.fh.createVariable('coor_names', 'c', ('num_dim', 'len_string'))
        for i in range(3):
            key = 'COORD' + 'XYZ'[i]
            self.fh.variables['coor_names'][i][:] = adjstr(key)
            self.fh.createVariable(key, 'f', ('num_nodes',))
            self.fh.variables[key][:] = vertices[i]

        self.fh.createDimension('num_nod_var', 3)
        dim = ('num_nod_var', 'len_string')
        self.fh.createVariable('name_nod_var', 'c', dim)
        for i in range(3):
            key = 'DISPL' + 'XYZ'[i]
            self.fh.variables['name_nod_var'][i, :] = adjstr(key)
            self.fh.createVariable('vals_nod_var{0}'.format(i+1), 'f',
                                   ('time_step', 'num_nodes'))

        # ------------------------------------------------------------------- #
        # ---------------------------------------------- Global variables --- #
        # ------------------------------------------------------------------- #
        self.fh.createDimension('num_glo_var', 3)
        dim = ('num_glo_var', 'len_string')
        self.fh.createVariable('name_glo_var', 'c', dim)
        for i in range(3):
            key = ['DTime', 'Step', 'Frame'][i]
            self.fh.variables['name_glo_var'][i, :] = adjstr(key)
        self.fh.createVariable('vals_glo_var', 'f', ('time_step', 'num_glo_var'))

        self.step_count = 0
        self.initialized = True
        return
