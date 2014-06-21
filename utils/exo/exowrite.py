import os
import numpy as np

# Import platform dependent exolib
from exoinc import *
from genesis import Genesis


class ExodusIIWriter(Genesis):
    """The ExodusIIWriter class

    """
    def __init__(self, runid, d=None):
        """Instantiate the ExodusIIWriter object

        Parameters
        ----------
        runid : str
            run ID, usually file basename

        Notes
        -----
        The ExodusIIFile class is an interface to the Exodus II api
        Its methods are named after the analogous method from the
        Exodus II C bindings, minus the prefix 'ex_'.

        """
        super(ExodusIIWriter, self).__init__(runid, d=d)

        # time
        self.db.createDimension(DIM_TIME, None)
        self.db.createVariable(VAR_WHOLE_TIME, DTYPE_FLT, (DIM_TIME,))

    # ----------------------------------------------------- EXODUS OUTPUT --- #
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
        dim = PX_DIM_VARS(var_type)
        self.db.createDimension(dim, num_vars)
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
        var_type = var_type.upper()
        var_names = format_list_of_strings(var_names)

        # store the names
        if var_type == PX_VAR_GLO:
            self.db.createVariable(VAR_NAME_GLO_VAR, DTYPE_TXT,
                                   (PX_DIM_VARS(var_type), DIM_STR))
            self.db.createVariable(VAR_GLO_VAR, DTYPE_FLT,
                                   (DIM_TIME, DIM_NUM_GLO_VAR))
        elif var_type == PX_VAR_NOD:
            self.db.createVariable(VAR_NAME_NOD_VAR, DTYPE_TXT,
                                   (PX_DIM_VARS(var_type), DIM_STR))
        elif var_type == PX_VAR_ELE:
            self.db.createVariable(VAR_NAME_ELE_VAR, DTYPE_TXT,
                                   (PX_DIM_VARS(var_type), DIM_STR))

        for (i, var_name) in enumerate(var_names):
            if var_type == PX_VAR_GLO:
                self.db.variables[VAR_NAME_GLO_VAR][i, :] = var_name

            elif var_type == PX_VAR_NOD:
                self.db.variables[VAR_NAME_NOD_VAR][i, :] = var_name
                self.db.createVariable(VAR_NOD_VAR_NEW(i+1), DTYPE_FLT,
                                       (DIM_TIME, DIM_NUM_NODES))

            elif var_type == PX_VAR_ELE:
                self.db.variables[VAR_NAME_ELE_VAR][i, :] = var_name
                for j in range(self.db.dimensions[DIM_NUM_EL_BLK]):
                    self.db.createVariable(VAR_ELEM_VAR(i+1, j+1), DTYPE_FLT,
                                           (DIM_TIME, DIM_NUM_EL_IN_BLK(j+1)))

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
        if num_elem_blk != self.db.dimensions[DIM_NUM_EL_BLK]:
            raise ExodusIIFileError("wrong num_elem_blk")
        if num_elem_var != self.db.dimensions[DIM_NUM_ELE_VAR]:
            raise ExodusIIFileError("wrong num_elem_var")
        self.db.createVariable(VAR_ELEM_TAB, DTYPE_INT,
                               (DIM_NUM_EL_BLK, DIM_NUM_ELE_VAR))
        for i in range(self.db.dimensions[DIM_NUM_EL_BLK]):
            self.db.variables[VAR_ELEM_TAB][i] = elem_var_tab[i]
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
        self.db.variables[VAR_WHOLE_TIME][time_step] = time_value
        return

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
        self.db.variables[VAR_GLO_VAR][time_step, :num_glo_var] = vals_glo_var
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
        name = VAR_NOD_VAR_NEW(nodal_var_index+PX_OFFSET)
        self.db.variables[name][time_step, :num_nodes] = nodal_var_vals
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
        name, dim = PX_PROPINFO(EX_ELEM_BLOCK)
        elem_blk_id = int(elem_blk_id)
        i = self.get_obj_idx(PX_ELEM_BLK, elem_blk_id)
        if i is None:
            raise ExodusIIFileError("bad element block ID")
        name = VAR_ELEM_VAR(elem_var_index+PX_OFFSET, i+PX_OFFSET)
        self.db.variables[name][time_step, :num_elem_this_blk] = elem_var_vals
        return
