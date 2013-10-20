import sys
import random
import os
import datetime

from enthought.traits.api import (HasStrictTraits, List, Instance, String,
                                  Interface, Int)

from core.inpparse import parse_input
from core.main import run_all_inputs
from utils.geninp import create_mml_input, PERM_FCNS
from viz.mdldat import MMLModel
from viz.metadat import VizMetaData


class IModelRunnerCallbacks(Interface):
    def RunFinished(self, metadata):
        ''' Called when a run completes or fails. '''

class ModelRunnerError(Exception):
    pass


class ModelRunner(HasStrictTraits):
    runid = String
    material_model = Instance(MMLModel)
    callbacks = Instance(IModelRunnerCallbacks)
    model_output = List(String)

    def run_model(self):
        inpstr = self.gen_inpstr(self.material_model)
        self.model_output = self.run_inpstr(inpstr, self.material_model)


    def run_opt_model(self, optimizer):
        inpstr = self.gen_inpstr(self.material_model, optimizer)
        output = self.run_inpstr(inpstr, self.material_model, 'Optimization')

    def run_inpstr(self, inpstr, material, run_type='Simulation'):
        # tjf: run_payette can be invoked with disp=1 and then it returns a
        # tjf: dictionary with some extra information. I pass that extra
        # tjf: information to the create_or_refresh_plot method.
        uinp = parse_input(["string", inpstr])[0]
        uinp[1] = self.runid
        output = run_all_inputs([uinp], 0, 1)

        if self.callbacks is not None:
            now = datetime.datetime.now()
            index_file = siminfo.get('index file')
            if index_file is None:
                index_file = ''
            else:
                base_dir, index_file = os.path.split(index_file)
            output_file = siminfo.get('output file')
            if output_file is None:
                output_file = ''
            else:
                base_dir, output_file = os.path.split(output_file)
            extra_files = siminfo.get('extra files')
            if extra_files is not None and len(extra_files) > 0:
                surface_file = extra_files['surface file']
                path_files = []
                for k, v in extra_files['path files'].iteritems():
                    path_file = os.path.split(v)[1]
                    path_files.append((k, path_file))
            else:
                surface_file = ''
                path_files = []

            metadata = VizMetaData(
                name=self.runid,
                base_directory=base_dir,
                index_file=index_file,
                out_file=output_file,
                surface_file=surface_file,
                path_files=path_files,
                data_type=run_type,
                model_type=', '.join(material.model_type),
                created_date=now.date(),
                created_time=now.time(),
                successful=True,
                model=material)

            self.callbacks.RunFinished(metadata)

        return output

    def gen_inpstr(self, material, optimization=None):
        runid = self.runid
        driver = "solid"
        pathtype = "prdef"
        mtlmdl = material.name
        model_type = str(material.model_type).lower()

        if 'eos' in model_type:
            path, pathopts = self.gen_eos_path(material)
        else:
            path, pathopts = self.gen_prdef_path(material)

        mtlparams = []
        for p in material.parameters:
            # For permutation and optimization jobs, the parameter is not
            # specified as
            #    key = val
            # in the input file but
            #    key = {key}
            # so that the input parser can preprocess key with the current
            # value.  Here we determine how to set key
            if p.distribution != "Specified":
                val = "{{{0}}}".format(p.name)
            else:
                val = p.default
            if optimization is not None:
                #@tjf: code to determine if p is optimized or not so that the
                # proper form for "val" can be written to the input
                raise ModelRunnerError("Support code needed")
            mtlparams.append((p.name, val))

        perm = self.gen_perminp(material)
        opt = self.gen_optinp(optimization)
        inpstr = create_mml_input(runid, driver, pathtype, pathopts, path,
                                  mtlmdl, mtlparams, permutation=perm, write=0)
        return inpstr

    def gen_perminp(self, material):
        if all([p.distribution == "Specified" for p in material.parameters]):
            return

        method = material.permutation_method.lower()
        permutate = []
        def get_key(d, v):
            for (k, _) in d.items():
                if _ == v: return k
        for p in material.parameters:
            distr = p.distribution.lower()
            _id = get_key(PERM_FCNS, distr)
            if distr == "percentage":
                permutate.append((p.name, _id, p.specified, p.percent, p.samples))
            elif distr == 'range':
                permutate.append((p.name, _id, p.minimum, p.maximum, p.samples))
            elif distr == 'uniform':
                permutate.append((p.name, _id, p.minimum, p.maximum, p.samples))
            elif distr == 'normal':
                permutate.append((p.name, _id, p.mean, p.stdev, p.samples))
            elif distr == 'absgaussian':
                vals = []
                for i in range(p.samples):
                    vals.append(abs(random.normalvariate(p.mean, p.stdev)))
                permutate.append((p.name, PERM_FCNS["list"], vals))
            elif p.distribution == 'weibull':
                permutate.append((p.name, _id, p.scale, p.shape, p.samples))

        return method, permutate

    def gen_optinp(self, optimization):
        return
        optimize_params = ""
        for param in optimization.optimize_vars:
            if not param.enabled:
                continue
            optimize_params += "    optimize %s" % (param.name)
            if param.use_bounds:
                optimize_params += ", bounds = (%f, %f)" % (
                    param.bounds_min, param.bounds_max)
            optimize_params += ", initial value = %f\n" % (param.initial_value)

        minimize_vars = []
        for var in optimization.minimize_vars:
            if var.enabled:
                minimize_vars.append(var.name)

        versus = ""
        if optimization.minimize_versus != "None":
            versus = "versus " + optimization.minimize_versus

        result = ("  begin optimization\n"
                  "    method %s\n"
                  "    maxiter %d\n"
                  "    tolerance %f\n"
                  "    disp 0\n"
                  "\n"
                  "%s\n"
                  "\n"
                  "    gold file %s\n"
                  "    minimize %s %s\n"
                  "\n"
                  "  end optimization\n"
                  ) % (
                      optimization.method.lower(),
                      optimization.max_iterations,
                      optimization.tolerance,
                      optimize_params,
                      optimization.gold_file,
                      ', '.join(minimize_vars),
                      versus
                  )
        return result

    def gen_eos_path(self, material):
        # XXX Need a less hardcoded way to do this
        T0 = 0.0
        R0 = 0.0
        for p in material.parameters:
            if p.name == "T0":
                T0 = float(p.default) / 0.861738573E-4
            elif p.name == "R0":
                R0 = float(p.default)  # *1000.0

        result = (
            "  begin boundary\n"
            "\n"
            "    nprints 5\n"
            "\n"
            "    input units MKSK\n"
            "    output units MKSK\n"
            "\n"
            "    density range %f %f\n"
            "    temperature range %f %f\n"
            "\n"
            "    surface increments %d\n"
            "\n"
            "    path increments %d\n"
            % (material.eos_boundary.min_density,
               material.eos_boundary.max_density,
               material.eos_boundary.min_temperature,
               material.eos_boundary.max_temperature,
               material.eos_boundary.surface_increments,
               material.eos_boundary.path_increments))

        if material.eos_boundary.isotherm:
            result += "    path isotherm %f %f\n" % (R0, T0)
        if material.eos_boundary.isentrope:
            result += "    path isentrope %f %f\n" % (R0, T0)
        if material.eos_boundary.hugoniot:
            result += "    path hugoniot %f %f\n" % (R0, T0)

        result += "  end boundary\n"

        return result

    def gen_prdef_path(self, material):
        pathopts = (("kappa", 0.),
                    ("nfac", 1),
                    ("format", "default"),
                    ("tstar", material.TSTAR),
                    ("fstar", material.FSTAR),
                    ("sstar", material.SSTAR),
                    ("dstar", material.DSTAR),
                    ("estar", material.ESTAR),
                    ("efstar", material.EFSTAR),
                    ("amplitude", material.AMPLITUDE),
                    ("ratfac", material. RATFAC))
        path = ["{0:f} {1:d} {2} {3}".format(
            l.time, l.nsteps, l.types, l.components) for l in material.legs]
        return path, pathopts
