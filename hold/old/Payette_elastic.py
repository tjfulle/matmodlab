# Copyright (2011) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
# rights in this software.

# The MIT License

# Copyright (c) Sandia Corporation

# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
from numpy import array

import Source.__config__ as cfg
from Source.Payette_utils import log_warning, log_message, report_and_raise_error
from Source.Payette_tensor import iso, dev
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
from Toolset.elastic_conversion import compute_elastic_constants

try:
    import elastic as mtllib
    imported = True
except:
    imported = False
    pass


class Elastic(ConstitutiveModelPrototype):
    """ Elasticity model. """

    def __init__(self, control_file, *args, **kwargs):
        super(Elastic, self).__init__(control_file, *args, **kwargs)

        self.imported = True if self.code == "python" else imported

        # register parameters
        self.register_parameters_from_control_file()

        pass

    # public methods
    def set_up(self, matdat):

        # parse parameters
        self.parse_parameters()

        # the elastic model only needs the bulk and shear modulus, but the
        # user could have specified any one of the many elastic moduli.
        # Convert them and get just the bulk and shear modulus
        eui = compute_elastic_constants(*self.ui0[0:12])
        for key, val in eui.items():
            if key.upper() not in self.parameter_table:
                continue
            idx = self.parameter_table[key.upper()]["ui pos"]
            self.ui0[idx] = val

        # Payette wants ui to be the same length as ui0, but we don't want to
        # work with the entire ui, so we only pick out what we want
        mu, k = self.ui0[1], self.ui0[4]
        self.ui = self.ui0
        mui = array([k, mu])

        self.bulk_modulus, self.shear_modulus = k, mu

        if self.code == "python":
            self.mui = self._py_set_up(mui)
        else:
            self.mui = self._fort_set_up(mui)

        return

    def jacobian(self, simdat, matdat):
        v = matdat.get_data("prescribed stress components")
        return self.J0[[[x] for x in v],v]

    def update_state(self, simdat, matdat):
        """
           update the material state based on current state and strain increment
        """
        # get passed arguments
        dt = simdat.get_data("time step")
        d = matdat.get_data("rate of deformation")
        sigold = matdat.get_data("stress")

        if self.code == "python":
            sig = _py_update_state(self.mui, dt, d, sigold)

        else:
            a = [1, dt, self.mui, sigold, d]
            if cfg.F2PY["callback"]:
                a.extend([report_and_raise_error, log_message])
            sig = mtllib.elast_calc(*a)

        # store updated data
        matdat.store_data("stress", sig)

    def _py_set_up(self, mui):

        k, mu = mui

        if k <= 0.:
            report_and_raise_error("Bulk modulus K must be positive")

        if mu <= 0.:
            report_and_raise_error("Shear modulus MU must be positive")

        # poisson's ratio
        nu = (3. * k - 2 * mu) / (6 * k + 2 * mu)
        if nu < 0.:
            log_warning("negative Poisson's ratio")

        ui = array([k, mu])

        return ui

    def _fort_set_up(self, mui):
        props = array(mui)
        a = [props]
        if cfg.F2PY["callback"]:
            a .extend([report_and_raise_error, log_message])
        ui = mtllib.elast_chk(*a)
        return ui


def _py_update_state(ui, dt, d, sigold):

    # strain increment
    de = d * dt

    # user properties
    k, mu = ui
    twomu = 2. * mu
    threek = 3. * k

    # elastic stress update
    return sigold + threek * iso(de) + twomu * dev(de)
