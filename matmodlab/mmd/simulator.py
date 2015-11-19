import os
import sys
import time
import logging
import inspect
from math import sqrt
import numpy as np
from time import time as tt
from collections import namedtuple, OrderedDict
from numpy.linalg import solve, lstsq
from numpy.linalg import LinAlgError as LinAlgError

from ..constants import *
from ..mml_siteenv import environ
from ..utils import mmlabpack as mml
from ..utils.errors import MatModLabError
from ..utils.fileio import loadfile, savefile
from ..utils.logio import setup_logger
from ..utils.plotting import create_figure
from .material import MaterialModel, Material

EPS = np.finfo(np.float).eps

__all__ = ['MaterialPointSimulator', 'StrainStep', 'StressStep', 'MixedStep',
           'DefGradStep', 'DisplacementStep', 'piecewise_linear']

class MaterialPointSimulator(object):
    def __init__(self, job, verbosity=None, d=None,
                 initial_temperature=DEFAULT_TEMP, termination_time=None,
                 output_format=None, no_cutback=False):
        """Initialize the MaterialPointSimulator object"""
        self.job = job
        self.material = None
        self.initialized = False
        self.termination_time = termination_time
        self.no_cutback = environ.no_cutback or no_cutback

        self.output_format = output_format or environ.output_format

        self.verbosity = verbosity
        self.initial_temperature = initial_temperature

        # setup IO
        d = d or os.getcwd()
        environ.simulation_dir = d
        self.directory = environ.simulation_dir
        self.filename = None
        self.ran = False

        # basic logger
        if verbosity > 2:
            logfile = os.path.join(environ.simulation_dir, self.job + '.log')
        else:
            logfile = None
        logger = setup_logger('matmodlab.mmd.simulator', logfile,
                              verbosity=verbosity)

        self.steps = StepRepository()
        p = {'temperature': initial_temperature}
        self.steps['Step-0'] = InitialStep('Step-0', **p)
        self.istress = Z6

        logger.info('Setting up simulator for job {0!r}'.format(job))

    def __getattr__(self, key):
        try:
            return self._get_var_time(key)
        except:
            raise AttributeError('{0!r} object has no attribute '
                                 '{1!r}'.format(self.__class__, key))

    def copy_steps(self, other):
        n = 1
        for s in other.steps.values()[1:]:
            step = AnalysisStep(s.kind, s.name, s.previous, s.increment,
                                len(s.frames), s.components, s.descriptors,
                                s.kappa, s.temperature, s.elec_field,
                                s.num_dumps, s.start, s.sqa_stiff, s.mat_stiff)
            step.number = n
            self.steps[s.name] = step
            self.run_step(self.steps[s.name])
            n += 1
        return

    def Material(self, model, parameters, **kwargs):
        '''Method that delays the instantiation of the material model

        '''
        kwargs['initial_temp'] = self.initial_temperature
        self.material = Material(model, parameters, **kwargs)
        return self.material

    @property
    def initial_stress(self):
        return np.array(self.istress)

    @initial_stress.setter
    def initial_stress(self, value):
        self.istress = value
        self.steps['Step-0'].components = value
        self.steps['Step-0'].descriptors = [4] * TENSOR_3D

    def InitialStress(self, components, scale=1.):
        try:
            if len(components) > TENSOR_3D:
                raise MatModLabError('expected stress to have at most {0}'
                                     ' components '.format(TENSOR_3D))
        except TypeError:
            # scalar -> pressure
            components = [components]

        components = np.array(components) * scale
        if len(components) == 1:
            # only one stress value given -> pressure
            Sij = -components[0]
            components = np.array([Sij, Sij, Sij, 0., 0., 0.], dtype=np.float64)

        N = TENSOR_3D - len(components)
        components = np.append(components, [0.] * N)
        self.initial_stress = components

    # --- Factor methods for creating steps ---
    def StrainStep(self, **kwargs):
        '''Factory method for the steps.StrainStep class'''
        self.create_step(StrainStep, **kwargs)

    def StrainRateStep(self, **kwargs):
        '''Factory method for the steps.StrainRateStep class'''
        self.create_step(StrainRateStep, **kwargs)

    def StressStep(self, **kwargs):
        '''Factory method for the steps.StressStep class'''
        self.create_step(StressStep, **kwargs)

    def StressRateStep(self, **kwargs):
        '''Factory method for the steps.StressRateStep class'''
        self.create_step(StressRateStep, **kwargs)

    def DisplacementStep(self, **kwargs):
        '''Factory method for the steps.DisplacementStep class'''
        self.create_step(DisplacementStep, **kwargs)

    def DefGradStep(self, **kwargs):
        '''Factory method for the steps.DefGradStep class'''
        self.create_step(DefGradStep, **kwargs)

    def MixedStep(self, **kwargs):
        '''Factory method for the steps.MixedStep class'''
        self.create_step(MixedStep, **kwargs)

    def DataSteps(self, filename, tc=0, **kwargs):
        '''Factory method for the steps.DataSteps class'''
        previous = self.steps.values()[-1]
        steps = DataSteps(filename, previous, tc=tc, **kwargs)
        self.add_and_run_step(steps)

    def GenSteps(self, step_class, steps=1, components=None,
                 amplitude=None, increment=1., temperature=None, **kwargs):
        '''Generates steps from amplitude functions'''

        # get a unique name to start with
        name = kwargs.pop('name', 'GenStep')
        name_, j = name, 1
        while 1:
            if '{0}-{1}'.format(name_, j) not in self.steps:
                name = name_
                break
            name_, j = name + '.{0}'.format(j), j + 1
        previous = self.steps.values()[-1]
        steps = GenSteps(step_class, name, previous, components, amplitude,
                         increment, steps, temperature, **kwargs)
        self.add_and_run_step(steps)

    def add_and_run_step(self, steps):
        n = len(self.steps)
        for step in steps:
            step.number = n
            self.steps[step.name] = step
            self.run_step(self.steps[step.name])
            n += 1

    def create_step(self, step_class, **kwargs):
        name = kwargs.pop('name', None)
        if name is None:
            name = 'Step-{0}'.format(len(self.steps))
        elif name in self.steps:
            raise MatModLabError('duplicate step name {0}'.format(name))
        previous = self.steps.values()[-1]
        kwargs['mat_stiff'] = self.material.completions['E']
        step = step_class(name, previous, **kwargs)
        step.number = len(self.steps)
        self.steps[step.name] = step
        self.run_step(self.steps[step.name])

    def write_summary(self):

        # Write info to log file
        L = max(max(len(n) for n in self.records), 10)
        param_names = self.material.parameter_names
        self.param_vals = np.array(self.material.parameters)
        iparam_vals = self.material.initial_parameters
        param_vals = self.material.parameters

        logging.getLogger('matmodlab.mmd.simulator').debug('Material Parameters')
        logging.getLogger('matmodlab.mmd.simulator').debug(
            '  {1:{0}s}  {2:12}  {3:12}'.format(L, 'Name', 'iValue', 'Value'))
        for p in zip(param_names, iparam_vals, param_vals):
            logging.getLogger('matmodlab.mmd.simulator').debug(
                '  {1:{0}s} {2: 12.6E} {3: 12.6E}'.format(L, *p))

        # write out plotable data
        logging.getLogger('matmodlab.mmd.simulator').debug('Field Variables:')
        for key in self.records.keys():
            logging.getLogger('matmodlab.mmd.simulator').debug('  ' + key)

        num_frames = sum([len(s.frames) for s in self.steps.values()])
        s = '\n   '.join('{0}'.format(x) for x in environ.std_materials)
        try:
            filename = inspect.getfile(self.material.__class__)
        except TypeError:
            filename = 'Interactive Cell'

        summary = '''
Simulation Summary
---------- -------
Job: {0}
Material search directories:
   {1}
Material interface file:
   {2}
Number of steps: {3}
   Total frames: {4}
Material: {5}
  Number of props: {6}
    Number of sdv: {7}
'''.format(self.job, s, filename, len(self.steps)-1, num_frames,
           self.material.name, self.material.num_prop,
           self.material.num_sdv)
        logging.getLogger('matmodlab.mmd.simulator').info(summary)

    def initialize_simulation(self):
        '''initialize everything for running the steps

        '''
        logger = logging.getLogger('matmodlab.mmd.simulator')
        logger.info('Setting up calculations...')

        if self.material is None:
            raise MatModLabError('The material must be set before '
                                 'any analysis steps are created')

        # register variables
        self._time = 0.
        self.records = Records()
        self.records.add('Step', SCALAR, dtype='i4')
        self.records.add('Frame', SCALAR, dtype='i4')
        self.records.add('Time', SCALAR)
        self.records.add('DTime', SCALAR)
        self.records.add('S', TENSOR_3D)
        self.records.add('E', TENSOR_3D)
        self.records.add('F', TENSOR_3D_FULL)
        self.records.add('D', TENSOR_3D)
        self.records.add('DS', TENSOR_3D)
        self.records.add('EF', VECTOR)
        self.records.add('T', SCALAR)

        # Adding SDVs **MUST** be last
        if self.material.sdv_keys:
            self.records.add('SDV', SDV, keys=self.material.sdv_keys)

        self.write_summary()

        step = self.steps.values()[0]
        frame = step.frames[0]
        S0 = self.initial_stress
        sdv = self.material.initial_sdv

        self.state_db = StateDB(F=I9, temp=step.temperature, stress=S0,
                                strain=Z6, efield=step.elec_field, statev=sdv)

        self.records.init(Step=step.number, Frame=frame.number,
                 Time=frame.value, DTime=frame.increment,
                 E=Z6, F=I9, D=Z6, DS=Z6, S=S0,
                 SDV=sdv, T=step.temperature, EF=step.elec_field)

        self.initialized = True

    def run(self):
        # at this point, all steps have run (they are run when created),
        # now finish the simulation up
        import warnings
        warnings.warn('Steps are run at time of their creation, '
                      'the run method will be deprecated')
        self.finish()

    def finish(self):
        if not environ.notebook:
            self.dump()
        self.ran = True

    def dump(self, format=None, ffmt='%.18e'):
        """Dump the results of the simulation to a file"""
        logger = logging.getLogger('matmodlab.mmd.simulator')
        logger.info('\n...calculations completed ({0:.4f}s)\n'.format(self._time))
        output_format = format or self.output_format
        ext = '.' + output_format
        self.filename = os.path.join(self.directory, self.job + ext)
        if output_format == REC:
            self.records.data.dump(self.filename)

        elif output_format in (TXT, CSV):
            if output_format == CSV:
                sep, comments = ',', ''
            else:
                sep, comments = ' ', '#'
            names = sep.join(self.records.keys(expand=1))
            data = rec2arr(self.records.data)
            np.savetxt(self.filename, data, header=names, delimiter=sep,
                       comments=comments, fmt=ffmt)

        else:
            # let someone else deal with it
            names = self.records.keys(expand=1)
            data = rec2arr(self.records.data)
            savefile(self.filename, names, data)

    def _get_var_time(self, var):
        if var == 'SDV':
            # Retrieve all SDVs from the record
            keys = [x for x in self.records.data.dtype.names
                    if x.startswith('SDV_')]
            a = rec2arr(self.records.data[keys])
            names = [x.replace('SDV_', '').strip() for x in keys]
        else:
            a = self.records.data[var]
            if a.ndim == 1:
                return a
            names = COMPONENT_LABELS(a.shape[1])
        return attrarr(a, names)

    def get(self, *variables, **kwargs):
        disp = kwargs.pop('disp', 0)
        at_step = kwargs.get('at_step', None)
        if at_step:
            at_step = unique_step_index(self.records.data['Step'])

        if not variables:
            names = self.records.keys(expand=1)
            data = rec2arr(self.records.data, rows=at_step)
            if disp:
                return names, data
            return data

        # get the specific variables
        names = [x.split('.', 1) if not x.startswith('SDV_') else [x]
                 for x in variables]
        data = []
        for item in names:
            a = np.array(self.records.data[item[0]])
            if len(item) == 2:
                a = a[:, COMPONENT(item[1], a.shape[1])]
            elif len(item) > 2:
                raise ValueError('expected at most one attribute lookup')
            if at_step is not None:
                a = a[at_step]
            data.append(a)

        if len(data) == 1 and data[0].ndim == 1:
            data = data[0].flatten()

        if disp < 0:
            return np.column_stack(data)

        elif disp:
            return variables, np.column_stack(data)

        return data

    def plot(self, xvar, yvar, legend=None, label=None, scale=None, **kwargs):

        xp, yp = self.get(xvar, yvar)

        if scale is not None:
            try:
                xs, ys = scale
            except ValueError:
                xs = ys = scale
            xp *= xs
            yp *= ys

        if environ.plotter == BOKEH:
            kwds = dict(kwargs)
            plot = kwds.pop('plot', None)
            if legend:
                kwds['legend'] = legend
            if plot is None:
                plot_width = kwds.pop('plot_width', 650)
                plot_height = kwds.pop('plot_height', 500)
                plot = create_figure(x_axis_label=xvar, y_axis_label=yvar,
                                     plot_width=plot_width, plot_height=plot_height)
            plot.line(xp, yp, **kwds)
            return plot

        else:
            import matplotlib.pyplot as plt
            if legend:
                kwargs['label'] = label or legend
            plt.plot(xp, yp, **kwargs)
            plt.xlabel(xvar)
            plt.ylabel(yvar)
            if environ.notebook:
                return
            if legend:
                plt.legend(loc='best')
            plt.show()

    def run_step(self, step):

        logger = logging.getLogger('matmodlab.mmd.simulator')
        if self.ran:
            logger.warn('simulation {0!r} has already '
                        'run to completion'.format(self.job))
            return

        if not self.initialized:
            self.initialize_simulation()

        if step.num_cutbacks >= 4:
            raise MatModLabError('number of cutbacks for step {0} exceeds '
                                 'the maximum allowable'.format(step.number))

        try:
            while 1:
                state = self._run_step(step)
                if not CB:
                    # no cutbacks requested
                    break

                if self.no_cutback:
                    break

                if step.num_cutbacks > 3:
                    # accept whatever is calculated
                    break

                # the strain increment was too large, cut it
                if CB.pnewdt is not None:
                    # a suggested time step size was given
                    step.cutback(pnewdt=CB.pnewdt)

                elif CB.cutfac:
                    logger.warn('increasing frames on step {0}'.format(step.number))
                    step.cutback(cutfac=CB.cutfac)

                else:
                    raise ValueError('unrecognized cutback parameter')

                CB.clear()
                self.records.clear_cache()

            # Save the state for next steps
            time, temp, F, strain, stress, efield, statev = state
            self.records.advance()
            self.state_db.advance(F=F, time=time, temp=temp, stress=stress,
                                  strain=strain, efield=efield, statev=statev)

        except StopSteps:
            self.finish()

        return

    def _run_step(self, step):
        '''Process this step '''

        # Unpack the state
        F = self.state_db.get("F")
        stress = self.state_db.get("stress")
        strain = self.state_db.get("strain")
        statev = self.state_db.get("statev")
        temp = self.state_db.get("temp")
        efield = self.state_db.get("efield")

        step_start_time = tt()
        termination_time = self.termination_time

        logger = logging.getLogger('matmodlab.mmd.simulator')
        warned = False
        num_frame = len(step.frames)
        lsn = len(str(num_frame))
        message = '{0}, Frame {{0:{1}d}}'.format(step.name, lsn)

        kappa, proportional = step.kappa, step.proportional

        # the following variables have values at [begining, end, current] of step
        time = np.array([step.frames[0].time,
                         step.frames[-1].value, step.frames[0].time])
        temp = np.array((temp, step.temperature, temp))
        efield = np.array((efield, step.elec_field, efield))
        strain = np.vstack((strain, strain, strain))
        stress = np.vstack((stress, stress, stress))


        # the following variables have values at [begining, current] of step
        statev = np.vstack((statev, statev))
        F = np.vstack((F, F))

        # compute the initial jacobian
        J0 = self.material.J0
        if J0 is None:
            raise MatModLabError('J0 has not been initialized')

        # v array is an array of integers that contains the rows and columns of
        # the slice needed in the jacobian subroutine.
        # Process this step
        nv = 0
        v = np.zeros(6, dtype=np.int)
        for (i, cij) in enumerate(step.components):
            if step.descriptors[i] == 1:         # -- strain rate
                strain[1, i] = strain[0, i] + cij * VOIGT[i] * step.increment

            elif step.descriptors[i] == 2:       # -- strain
                strain[1, i] = cij * VOIGT[i]

            elif step.descriptors[i] == 3:       # -- stress rate
                stress[1, i] = stress[0, i] + cij * step.increment
                v[nv] = i
                nv += 1

            elif step.descriptors[i] == 4:       # -- stress
                stress[1, i] = cij
                v[nv] = i
                nv += 1

            continue
        v = v[:nv]
        vx = [x for x in range(6) if x not in v]
        if step.increment < 1.e-14:
            dedt = np.zeros_like(strain[1])
            dtime = 1.
        else:
            dedt = (strain[1] - strain[0]) / step.increment
            dtime = (time[1] - time[0]) / num_frame

        dtemp = (temp[1] - temp[0]) / num_frame

        # --- find current value of d: sym(velocity gradient)
        if not nv:
            # strain or strain rate prescribed and the strain rate is constant
            # over the entire step
            if abs(kappa) > 1.e-14:
                d = mml.deps2d(dtime, kappa, strain[2], dedt)
            elif environ.sqa:
                d = mml.deps2d(dtime, kappa, strain[2], dedt)
                if not np.allclose(d, dedt):
                    logger.info('sqa: d != dedt (k=0,step={0})'.format(step.name))
            else:
                d = np.array(dedt)

        else:
            # Initial guess for d[v]
            Jsub = J0[[[x] for x in v], v]
            work = (stress[1,v] - stress[0,v]) / step.increment
            try:
                dedt[v] = solve(Jsub,  work)
            except:
                dedt[v] -= lstsq(Jsub, work)[0]

        # process this leg
        for (iframe, frame) in enumerate(step.frames):

            logger.info('\r' + message.format(iframe+1), extra={'continued':1})

            # interpolate values to the target values for this step
            a1 = float(num_frame - (iframe + 1)) / num_frame
            a2 = float(iframe + 1) / num_frame
            efield[2] = a1 * efield[0] + a2 * efield[1]
            strain[2] = a1 * strain[0] + a2 * strain[1]
            pstress = a1 * stress[0] + a2 * stress[1]

            if nv:
                # One or more stresses prescribed
                d = sig2d(self.material, time[2], dtime, temp[2], dtemp,
                          kappa, F[0], F[1], strain[2], dedt, stress[2],
                          statev[0], efield[2], v, pstress[v],
                          proportional)

            # compute the current deformation gradient and strain from
            # previous values and the deformation rate
            F[1], e = mml.update_deformation(dtime, kappa, F[0], d)
            strain[2,v] = e[v]
            if environ.sqa:
                if not np.allclose(strain[2,vx], e[vx]):
                    logger.info('sqa: error computing strain '
                                '(step={0})'.format(step.name))

            # update material state
            s = np.array(stress[2])
            stress[2], statev[1] = self.material.compute_updated_state(
                time[2], dtime, temp[2], dtemp, kappa, F[0], F[1], strain[2], d,
                efield[2], stress[2], statev[0], last=True,
                sqa_stiff=step.sqa_stiff, disp=1)
            dstress = (stress[2] - s) / dtime

            F[0] = F[1]
            time[2] = a1 * time[0] + a2 * time[1]
            temp[2] = a1 * temp[0] + a2 * temp[1]
            statev[0] = statev[1]

            # --- update the state
            self.records.cache(Step=step.number, Frame=frame.number,
                 Time=frame.value, DTime=frame.increment,
                 E=strain[2]/VOIGT, F=F[1], D=d/VOIGT, DS=dstress, S=stress[2],
                 SDV=statev[1], T=temp[2], EF=efield[2])

            if iframe > 1 and nv and not warned:
                sigmag = np.sqrt(np.sum(stress[2,v] ** 2))
                sigerr = np.sqrt(np.sum((stress[2,v] - pstress[v]) ** 2))
                warned = True
                _tol = np.amax(np.abs(stress[2,v]))
                _tol /= self.material.completions['K']
                _tol = max(_tol, 1e-4)
                if sigerr > _tol:
                    logger.warn('{0}, frame {1}, '
                                'prescribed stress error: {2: .5f}. '
                                '(% err: {3: .5f}) '
                                'consider increasing number of '
                                'steps'.format(step.name, iframe, sigerr,
                                               sigerr/sigmag*100.0))

            if termination_time is not None and time[2] >= termination_time:
                step_duration = tt() - step_start_time
                self._time += step_duration
                logger.info('\r' + message.format(iframe+1) +
                            ' ({0:.4f}s)'.format(self._time))
                raise StopSteps

            continue  # continue to next frame

        step_duration = tt() - step_start_time
        self._time += step_duration

        logger.info('\r' + message.format(iframe+1) +
                    ' ({0:.4f}s)'.format(self._time))

        return time[2], temp[2], F[1], strain[2], stress[2], efield[2], statev[1]

    def visualize_results(self, overlay=None):
        from .startup import launch_viewer
        launch_viewer([self.filename])

    def view(self):
        self.visualize_results()


def sig2d(material, t, dt, temp, dtemp, kappa, f0, f, stran, d, sig, statev,
          efield, v, sigspec, proportional):
    '''Determine the symmetric part of the velocity gradient given stress

    Parameters
    ----------

    Returns
    -------

    Approach
    --------
    Seek to determine the unknown components of the symmetric part of
    velocity gradient d[v] satisfying

                               P(d[v]) = Ppres[:]                      (1)

    where P is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and Ppres[:] are the
    prescribed values at the current time.

    Solution is found iteratively in (up to) 3 steps
      1) Call newton to solve 1, return stress, statev, d if converged
      2) Call newton with d[v] = 0. to solve 1, return stress, statev, d
         if converged
      3) Call simplex with d[v] = 0. to solve 1, return stress, statev, d

    '''
    dsave = d.copy()

    if not proportional:
        d = newton(material, t, dt, temp, dtemp, kappa, f0, f, stran, d,
                   sig, statev, efield, v, sigspec, proportional)
        if d is not None:
            return d

        # --- didn't converge, try Newton's method with initial
        # --- d[v]=0.
        d = dsave.copy()
        d[v] = np.zeros(len(v))
        d = newton(material, t, dt, temp, dtemp, kappa, f0, f, stran, d,
                   sig, statev, efield, v, sigspec, proportional)
        if d is not None:
            return d

        CB.request_cutback(cutfac=-1)

    # --- Still didn't converge. Try downhill simplex method and accept
    #     whatever answer it returns:
    d = dsave.copy()
    return simplex(material, t, dt, temp, dtemp, kappa, f0, f, stran, d,
                   sig, statev, efield, v, sigspec, proportional)


def newton(material, t, dt, temp, dtemp, kappa, f0, farg, stran, darg,
           sigarg, statev_arg, efield, v, sigspec, proportional):
    '''Seek to determine the unknown components of the symmetric part of velocity
    gradient d[v] satisfying

                               sig(d[v]) = sigspec

    where sig is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and sigspec are the
    prescribed values at the current time.

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    statev_arg : ndarray
        state dependent variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray || None
        If converged, the symmetric part of the velocity gradient, else None

    Notes
    -----
    The approach is an iterative scheme employing a multidimensional Newton's
    method. Each iteration begins with a call to subroutine jacobian, which
    numerically computes the Jacobian submatrix

                                  Js = J[v, v]

    where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of
    d[v] is then updated according to

                d[v] = d[v] - Jsi*sigerr(d[v])/dt

    where

                   sigerr(d[v]) = sig(d[v]) - sigspec

    The process is repeated until a convergence critierion is satisfied. The
    argument converged is a flag indicat- ing whether or not the procedure
    converged:

    '''
    logger = logging.getLogger('matmodlab.mmd.simulator')
    depsmag = lambda a: sqrt(sum(a[:3] ** 2) + 2. * sum(a[3:] ** 2)) * dt

    # Initialize
    tol1, tol2 = EPS, sqrt(EPS)
    maxit1, maxit2, depsmax = 20, 30, .2

    sig = sigarg.copy()
    d = darg.copy()
    f = farg.copy()
    statev = statev_arg.copy()

    sigsave = sig.copy()
    statev_save = statev.copy()

    # --- Check if strain increment is too large
    if (depsmag(d) > depsmax):
        return None

    # update the material state to get the first guess at the new stress
    sig, statev, stif = material.compute_updated_state(t, dt, temp, dtemp, kappa,
        f0, f, stran, d, efield, sig, statev)
    sigerr = sig[v] - sigspec

    # --- Perform Newton iteration
    for i in range(maxit2):
        sig = sigsave.copy()
        statev = statev_save.copy()
        Jsub = material.compute_updated_state(t, dt, temp, dtemp, kappa,
            f0, f, stran, d, efield, sig, statev, v=v, disp=2)

        if environ.sqa:
            try:
                evals = np.linalg.eigvalsh(Jsub)
            except LinAlgError:
                raise MatModLabError('failed to determine elastic '
                                     'stiffness eigenvalues')
            else:
                if np.any(evals < 0.):
                    negevals = evals[np.where(evals < 0.)]
                    logger.warn('negative eigen value[s] encountered in material '
                                'Jacobian: {0} ({1:.2f})'.format(negevals, t))
        try:
            d[v] -= np.linalg.solve(Jsub, sigerr) / dt


        except LinAlgError:
            d[v] -= np.linalg.lstsq(Jsub, sigerr)[0] / dt
            if environ.Wall:
                logger.warn('using least squares approximation to '
                            'matrix inverse')

        if (depsmag(d) > depsmax or  np.any(np.isnan(d)) or np.any(np.isinf(d))):
            # increment too large
            return None

        # with the updated rate of deformation, update stress and check
        fp, ep = mml.update_deformation(dt, 0., f, d)
        sig, statev, stif = material.compute_updated_state(t, dt, temp, dtemp,
            kappa, f0, fp, ep, d, efield, sig, statev)
        sigerr = sig[v] - sigspec
        dnom = max(np.amax(np.abs(sigspec)), 1.)
        relerr = np.amax(np.abs(sigerr) / dnom)

        if i <= maxit1 and relerr < tol1:
            return d

        elif i > maxit1 and relerr < tol2:
            return d

        continue

    # didn't converge, restore restore data and exit
    return None


def simplex(material, t, dt, temp, dtemp, kappa, f0, farg, stran, darg, sigarg,
            statev_arg, efield, v, sigspec, proportional):
    '''Perform a downhill simplex search to find sym_velgrad[v] such that

                        sig(sym_velgrad[v]) = sigspec[v]

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    statev_arg : ndarray
        state dependent variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray
        the symmetric part of the velocity gradient

    '''
    # --- Perform the simplex search
    import scipy.optimize
    d = darg.copy()
    f = farg.copy()
    sig = sigarg.copy()
    statev = statev_arg.copy()
    args = (material, t, dt, temp, dtemp, kappa, f0, f, stran, d,
            sig, statev, efield, v, sigspec, proportional)
    d[v] = scipy.optimize.fmin(_func, d[v], args=args, maxiter=20, disp=False)
    return d


def _func(x, material, t, dt, temp, dtemp, kappa, f0, farg, stran, darg,
          sigarg, statev_arg, efield, v, sigspec, proportional):
    '''Objective function to be optimized by simplex

    '''
    d = darg.copy()
    f = farg.copy()
    sig = sigarg.copy()
    statev = statev_arg.copy()

    # initialize
    d[v] = x
    fp, ep = mml.update_deformation(dt, 0., f, d)

    # store the best guesses
    sig, statev, stif = material.compute_updated_state(t, dt, temp, dtemp,
        kappa, f0, fp, ep, d, efield, sig, statev)

    # check the error
    error = 0.
    if not proportional:
        for i, j in enumerate(v):
            error += (sig[j] - sigspec[i]) ** 2
            continue

    else:
        stress_v, stress_u = [], []
        for i, j in enumerate(v):
            stress_u.append(sigspec[i])
            stress_v.append(sig[j])
            continue
        stress_v = np.array(stress_v)
        stress_u = np.array(stress_u)

        stress_u_norm = np.linalg.norm(stress_u)
        if stress_u_norm != 0.0:
            dum = (np.dot(stress_u / stress_u_norm, stress_v) *
                   stress_u / stress_u_norm)
            error = np.linalg.norm(dum) + np.linalg.norm(stress_v - dum) ** 2
        else:
            error = np.linalg.norm(stress_v)

    return error

class StopSteps(Exception):
    pass

class StepRepository(OrderedDict):
    def Step(self, name):
        self[name] = Step(name)
        return self[name]

class Step(object):
    def __init__(self, name):
        self.num_cutbacks = 0
        self.name = name
        self.frames = []

    def Frame(self, time, increment):
        self.frames.append(Frame(len(self.frames)+1, time, increment))
        return self.frames[-1]

    def cutback(self, cutfac=None, pnewdt=None):
        if cutfac is None and pnewdt is None:
            raise MatModLabError('cutback requires cutfac or pnewdt')

        nframe = len(self.frames)
        start = self.frames[0].time
        end = self.frames[-1].value
        dtime = end - start
        self.num_cutbacks += 1

        if pnewdt is not None:
            nframe = int(1. / pnewdt)

        elif cutfac == -1:
            nframe *= self.num_cutbacks * 4

        else:
            nframe *= cutfac

        # 5000 frames is probably excessive
        nframe = min(nframe, 5000)

        inc = (end - start) / float(nframe)
        self.frames = [Frame(i+1, start+i*inc, inc) for i in range(nframe)]

class Frame:
    def __init__(self, number, time, increment):
        self.number = number
        self.time = time
        self.increment = increment
        self.value = time + increment

class AnalysisStep(Step):

    def __init__(self, kind, name, previous, increment, frames, components,
                 descriptors, kappa, temperature, elec_field, num_dumps,
                 sqa_stiff, mat_stiff, start=None):

        super(AnalysisStep, self).__init__(name)
        logger = logging.getLogger('matmodlab.mmd.simulator')

        self.keywords = dict(increment=increment, frames=frames,
                             components=components, descriptors=descriptors,
                             kappa=kappa, temperature=temperature,
                             elec_field=elec_field, num_dumps=num_dumps,
                             sqa_stiff=sqa_stiff, mat_stiff=mat_stiff)
        self.kind = kind
        self.previous = previous
        self.components = components
        assert len(descriptors) == TENSOR_3D
        assert not([x for x in descriptors if x not in (1, 2, 3, 4)])
        self.descriptors = descriptors

        def set_default(key, value, default, dtype):
            if value is None:
                value = getattr(previous, key, default)
            setattr(self, key, dtype(value))

        # set defaults, inheriting from previous step
        set_default('kappa', kappa, 0., float)
        set_default('temperature', temperature, DEFAULT_TEMP, float)
        set_default('elec_field', elec_field, [0.,0.,0.], np.array)
        set_default('num_dumps', num_dumps, 100000000, int)
        set_default('sqa_stiff', sqa_stiff, False, bool)
        set_default('mat_stiff', mat_stiff, 1, float)

        if increment is None:
            increment = 1.
        self.increment = increment

        # set up analysis frames
        if any([x in (3,4) for x in self.descriptors]) and frames in (1, None):
            # set default frames for mixed steps
            if frames == 1:
                logger.warn('Number of frames may be inapopriate for '
                            'stress driven steps')
            else:
                # number of frames not specified
                # limit the amount of strain
                de_max = -1e60
                if 1 in descriptors:
                    x = max([abs(x) for (i,x) in enumerate(components)
                             if descriptors[i] == 1])
                    de_max = max(de_max, x * increment)
                if 2 in descriptors:
                    x = max([abs(x) for (i,x) in enumerate(components)
                             if descriptors[i] == 2])
                    de_max = max(de_max, x)
                if 3 in descriptors:
                    x = max([abs(x) for (i,x) in enumerate(components)
                             if descriptors[i] == 3])
                    de_max = max(de_max, x * increment / self.mat_stiff)
                if 4 in descriptors:
                    x = max([abs(x) for (i,x) in enumerate(components)
                             if descriptors[i] == 4])
                    de_max = max(de_max, x / self.mat_stiff)

                # de_max defaults to .1%
                frames = max(int(de_max / .001), 1)

        elif frames is None:
            frames = 1
        frames = int(frames)
        frame_increment = increment / float(frames)

        if start is None:
            start = previous.frames[-1].value
        self.start = start

        for i in range(frames):
            self.Frame(start, frame_increment)
            start += frame_increment

    @property
    def kappa(self):
        try:
            return self._kappa
        except AttributeError:
            self._kappa = 0.
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        if value is None:
            value = 0.
        self._kappa = value

    @property
    def proportional(self):
        try:
            return self._proportional
        except AttributeError:
            self._proportional = False
        return self._proportional

    @proportional.setter
    def proportional(self, value):
        self._proportional = bool(value)

def InitialStep(name, kappa=0., temperature=None):
    increment, frames, scale = 0., 1, 1.
    elec_field = np.zeros(3)
    previous = namedtuple('previous', 'value')(value=0.)
    num_dumps = None

    components = np.zeros(TENSOR_3D, dtype=np.float64)
    descriptors = np.array([2] * TENSOR_3D, dtype=np.int)

    step = AnalysisStep('InitialStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature,
                        elec_field, num_dumps, False, 1, start=0.)
    step.number = 0
    return step

def StrainStep(name, previous, components=None, frames=None, scale=1.,
                 increment=1., kappa=None, temperature=None, elec_field=None,
                 num_dumps=None, sqa_stiff=False, mat_stiff=1):

    if components is None:
        components = np.zeros(TENSOR_3D)

    if len(components) > TENSOR_3D:
        raise MatModLabError('expected strain to have at most {0} '
                             'components on Step {1}'.format(TENSOR_3D, name))
    components = np.array(components) * scale
    if kappa is None:
        kappa = previous.kappa

    if len(components) == 1:
        # only one strain value given -> volumetric strain
        ev = components[0]
        if kappa * ev + 1. < 0.:
            raise MatModLabError('1 + kappa * ev must be positive')

        if abs(kappa) < 1.e-16:
            eij = ev / 3.

        else:
            eij = ((kappa * ev + 1.) ** (1. / 3.) - 1.)
            eij = eij / kappa

        components = np.array([eij, eij, eij, 0., 0., 0.], dtype=np.float64)

    N = TENSOR_3D - len(components)
    components = np.append(components, [0.] * N)
    bad = np.where(kappa * components + 1. < 0.)
    if np.any(bad):
        idx = str(bad[0])
        raise MatModLabError('1 + kappa*E[{0}] must be positive'.format(idx))

    descriptors = np.array([2] * TENSOR_3D, dtype=np.int)

    return AnalysisStep('StrainStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature, elec_field,
                        num_dumps, sqa_stiff, mat_stiff)

def StrainRateStep(name, previous, components=None, frames=None, scale=1.,
                   increment=1., kappa=None, temperature=None, elec_field=None,
                   num_dumps=None, sqa_stiff=False, mat_stiff=1):

    if components is None:
        components = np.zeros(TENSOR_3D)

    if len(components) > TENSOR_3D:
        raise MatModLabError('expected strain rate to have at most {0} '
                             'components on Step {1}'.format(TENSOR_3D, name))
    components = np.array(components) * scale

    if kappa is None:
        kappa = previous.kappa

    if len(components) == 1:
        # only one strain value given -> volumetric strain
        dev = components[0]

        if abs(kappa) < 1.e-16:
            deij = dev / 3.

        else:
            deij = ((kappa * dev + 1.) ** (1. / 3.) - 1.)
            deij = deij / kappa

        components = np.array([deij, deij, deij, 0., 0., 0.], dtype=np.float64)

    N = TENSOR_3D - len(components)
    components = np.append(components, [0.] * N)
    descriptors = np.array([1] * TENSOR_3D, dtype=np.int)

    return AnalysisStep('StrainRateStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature, elec_field,
                        num_dumps, sqa_stiff, mat_stiff)

def StressStep(name, previous, components=None, frames=None, scale=1.,
               increment=1., temperature=None, elec_field=None,
               num_dumps=None, sqa_stiff=False, mat_stiff=1):

    kappa = 0.

    if components is None:
        components = np.zeros(TENSOR_3D)
    if len(components) > TENSOR_3D:
        raise MatModLabError('expected stress to have at most {0} '
                             'components on Step {1}'.format(TENSOR_3D, name))

    components = np.array(components) * scale
    if len(components) == 1:
        # only one stress value given -> pressure
        Sij = -components[0]
        components = np.array([Sij, Sij, Sij, 0., 0., 0.], dtype=np.float64)

    N = TENSOR_3D - len(components)
    components = np.append(components, [0.] * N)
    descriptors = np.array([4] * TENSOR_3D, dtype=np.int)

    return AnalysisStep('StressStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature, elec_field,
                        num_dumps, sqa_stiff, mat_stiff)

def StressRateStep(name, previous, components=None, frames=None, scale=1.,
                   increment=1., temperature=None, elec_field=None,
                   num_dumps=None, sqa_stiff=False, mat_stiff=1):

    kappa = 0.
    if components is None:
        components = np.zeros(TENSOR_3D)
    if len(components) > TENSOR_3D:
        raise MatModLabError('expected stress to have at most {0} '
                             'components on Step {1}'.format(TENSOR_3D, name))

    components = np.array(components) * scale
    if len(components) == 1:
        # only one stress value given -> pressure
        Sij = -components[0]
        descriptors = np.array([3, 3, 3, 2, 2, 2], dtype=np.int)
        components = np.array([Sij, Sij, Sij, 0., 0., 0.], dtype=np.float64)

    N = TENSOR_3D - len(components)
    components = np.append(components, [0.] * N)
    descriptors = np.array([3] * TENSOR_3D, dtype=np.int)

    return AnalysisStep('StressRateStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature, elec_field,
                        num_dumps, sqa_stiff, mat_stiff)

def DisplacementStep(name, previous, components=None, frames=None, scale=1.,
                     increment=1., kappa=None, temperature=None, elec_field=None,
                     num_dumps=None, sqa_stiff=False, mat_stiff=1):

    if components is None:
        components = np.zeros(3)

    if len(components) > 3:
        raise MatModLabError('expected displacement to have at most 3 '
                             'components on Step {0}'.format(name))

    components = np.array(components) * scale
    if len(components) < 3:
        components = np.append(components, [0.] * 3 - len(components))

    # convert displacments to strains
    Uij = np.zeros((3, 3))
    Uij[DI3] = components[:3] + 1.
    components = mml.u2e(Uij, kappa, 1)
    descriptors = np.array([2] * TENSOR_3D, dtype=np.int)

    return AnalysisStep('DisplacementStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature, elec_field,
                        num_dumps, sqa_stiff, mat_stiff)

def DefGradStep(name, previous, components=None, frames=None, scale=1.,
                increment=1., kappa=None, temperature=None, elec_field=None,
                num_dumps=None, sqa_stiff=False, mat_stiff=1):

    if kappa is None:
        kappa = previous.kappa

    try:
        defgrad = np.reshape(components, (3, 3)) * scale
    except ValueError:
        raise MatModLabError('expected 9 deformation gradient '
                             'components for step {0}'.format(name))

    jac = np.linalg.det(defgrad)
    if jac <= 0:
        raise MatModLabError('negative Jacobian on step '
                             '{0} ({1:f})'.format(name, jac))
    components = np.reshape(defgrad, (9,))

    # convert deformation gradient to strain E with associated
    # rotation given by axis of rotation x and angle of rotation theta
    Rij, Vij = np.linalg.qr(defgrad)
    if np.max(np.abs(Rij - np.eye(3))) > np.finfo(np.float).eps:
        raise MatModLabError('QR decomposition of deformation gradient '
                             'gave unexpected rotations (rotations are '
                             'not yet supported)')
    Uij = np.dot(Rij.T, np.dot(Vij, Rij))
    components = mml.u2e(Uij, kappa)
    descriptors = np.array([2] * TENSOR_3D, dtype=np.int)

    return AnalysisStep('DefGradStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature, elec_field,
                        num_dumps, sqa_stiff, mat_stiff)

def MixedStep(name, previous, components=None, descriptors=None,
              frames=None, scale=1., increment=1., temperature=None,
              elec_field=None, num_dumps=None, sqa_stiff=False, mat_stiff=1):

    if components is None:
        components = np.zeros(TENSOR_3D)

    if len(components) > TENSOR_3D:
        raise MatModLabError('expected stress to have at most {0} '
                             'components on Step {1}'.format(TENSOR_3D, name))

    if descriptors is None:
        descriptors = [2] * TENSOR_3D

    bad = []
    d = {'D': 1, 'E': 2, 'R': 3, 'S': 4}
    descriptors = [x for x in descriptors]
    for (i, x) in enumerate(descriptors):
        if x in d.values():
            continue
        try:
            x = d[x.upper()]
        except (AttributeError, KeyError):
            bad.append(x)
            continue
        descriptors[i] = x
    if bad:
        idx = ','.join(str(x) for x in bad)
        raise MatModLabError('unexpected descriptors {0}'.format(idx))

    if len(descriptors) != len(components):
        raise MatModLabError('expected len(components)=len(descriptors) '
                             'on step {0}'.format(name))

    kappa = 0.
    try:
        N = len(components) - len(scale)
        scale = [float(x) for x in scale] + [1.] * N
    except TypeError:
        scale = [scale] * len(components)
    scale = np.array(scale)
    components = np.array(components) * scale
    N = TENSOR_3D - len(components)
    components = np.append(components, [0.] * N)
    descriptors = np.append(descriptors, [2] * (TENSOR_3D - len(descriptors)))

    bad = np.where(kappa*components[np.where(descriptors==2)]+1.<0.)
    if np.any(bad):
        idx = str(bad[0])
        raise MatModLabError('1 + kappa*E[{0}] must be positive'.format(idx))

    components = np.array(components)
    descriptors = descriptors

    return AnalysisStep('MixedStep', name, previous, increment, frames,
                        components, descriptors, kappa, temperature, elec_field,
                        num_dumps, sqa_stiff, mat_stiff)

def DataSteps(filename, previous, tc=0, descriptors=None, time_format='total',
              scale=1., frames=None, steps=None, **kw):

    d = {'D': 1, 'E': 2, 'R': 3, 'S': 4, 'P': 6, 'T': 7, 'X': 9}
    bad = []
    if descriptors is None:
        raise MatModLabError('required keyword descriptors missing')

    descriptors = [x for x in descriptors]
    for (i, x) in enumerate(descriptors):
        if x in d.values():
            continue
        try:
            x = d[x.upper()]
        except (AttributeError, KeyError):
            bad.append(x)
            continue
        descriptors[i] = x
    if bad:
        idx = ','.join(str(x) for x in bad)
        raise MatModLabError('unexpected descriptors {0}'.format(idx))
    descriptors = np.array(descriptors)

    if len(descriptors[np.where(descriptors==7)]) > 1:
        raise MatModLabError('expected at most one temperature column')

    if len(descriptors[np.where(descriptors==6)]) > 3:
        raise MatModLabError('expected at most three electric field columns')

    fill = None
    columns = kw.get('columns')
    if columns is not None:
        if len(columns) > len(descriptors):
            raise MatModLabError('expected len(components)<=len(descriptors)')
        fill = len(descriptors) - len(columns)
        columns = [tc] + [x for x in columns]

    else:
        # columns not given.  punt
        columns = [tc]
        for i in range(50):
            if i == tc:
                continue
            columns.append(i)
            if len(columns) >= len(descriptors) + 1:
                break
    kw['columns'] = columns

    # Read in the file
    X = loadfile(filename, disp=0, **kw)

    # Create interpolator
    interp = lambda x, yp: np.interp(x, X[:,0], yp)

    # Create each step
    if steps is None:
        steps = X.shape[0]
    step = 1
    start = previous.frames[-1].value
    data_steps = []
    columns = range(X.shape[1])
    for time in np.linspace(X[0,0], X[-1,0], steps):
        increment = time - start
        if abs(increment) < 1.e-16:
            continue

        components = np.asarray([interp(time, X[:,col]) for col in columns[1:]])
        if fill is not None:
            components = np.append(components, np.zeros(fill))

        # remove all non-deformations from components
        temp = components[np.where(descriptors==7)]
        elec_field = components[np.where(descriptors==6)]
        if len(elec_field) == 0:
            elec_field = previous.elec_field

        ic = np.where((descriptors!=6)&(descriptors!=7)&(descriptors!=9))
        components = components[ic]
        descriptors = descriptors[ic]

        try:
            temp = temp[0]
        except IndexError:
            temp = None

        # determine start time
        name = 'DataStep-{0}'.format(step)
        data_steps.append(MixedStep(name, previous, components=components,
            frames=frames, scale=scale, increment=increment,
            descriptors=descriptors, temperature=temp,
            elec_field=elec_field))
        previous = data_steps[-1]
        start = previous.frames[-1].value
        step += 1

    return data_steps

def GenSteps(step_class, name, previous, components, amplitude, increment,
             nsteps, temperature, **kwargs):

    if components is None:
        components = [1.] * TENSOR_3D
    components = np.array(components)
    nc = len(components)

    if temperature is None:
        temperature = getattr(previous, 'temperature', DEFAULT_TEMP)
    try:
        temperature(0)
    except TypeError:
        temperature = float(temperature)
        temperature = lambda t, T=temperature: T

    def set_amplitude(ampl):
        one = lambda t: 1.
        if ampl is None:
            return lambda t: np.array([one(t)] * nc)
        try:
            # check that amplitude is a list
            ampl = [x for x in ampl] + [one] * (nc - len(ampl))
            n = len(ampl)
        except TypeError:
            ampl = [ampl] * nc
            n = nc
        ampl = [x for x in ampl]
        if n != nc:
            message = 'expected len(amplitude) to be at most len(components)'
            raise MatModLabError(message)

        # at this point, we need to make sure that each amplitude is callable
        for (i, f) in enumerate(ampl):
            if f is None:
                ampl[i] = one
                continue
            try:
                f(0)
                continue
            except TypeError:
                pass
            try:
                # constant amplitude
                a = float(f)
                ampl[i] = lambda t, a=a: a
            except AttributeError:
                message = 'amplitude must be a float or callable'
                raise MatModLabError(message)
        return lambda t: np.array([x(t) for x in ampl])
    amplitude = set_amplitude(amplitude)

    # generate the steps
    start = previous.frames[-1].value
    dt = increment / float(nsteps)
    kwargs['increment'] = dt
    steps = []
    for (i, t) in enumerate(np.linspace(start+dt, start+increment, nsteps)):
        name_ = name + '-{0}'.format(i+1)
        kwargs['components'] = components * amplitude(t)
        kwargs['temperature'] = temperature(t)
        steps.append(step_class(name_, previous, **kwargs))
        previous = steps[-1]
    return steps

def piecewise_linear(xp, fp):
    '''create a piecewise linear function'''
    def interp(x):
        return np.interp(x, xp, fp)
    return interp

def flatten(a):
    flat = []
    for x in a:
        try: flat.extend(x)
        except TypeError: flat.append(x)
    return flat

def unique_step_index(a):
    d = {}
    for (i, x) in enumerate(a):
        d.setdefault(int(x), []).append(i)
    return [x[-1] for x in sorted(d.values())]

def rec2arr(recarr, rows=None):
    arr = []
    for row in recarr:
        arr.append(flatten(row.tolist()))
    arr = np.array(arr)
    if rows is not None:
        arr = arr[rows]
    return arr

class attrarr(np.ndarray):
    """Subclass an ndarray to return attributes stored as the array columns"""
    def __new__(cls, arr, names):
        obj = np.asarray(arr).view(cls)
        obj.names = dict((s, i) for (i, s) in enumerate(names))
        return obj
    def __getattr__(self, key):
        if key not in self.names:
            raise AttributeError(key)
        idx = self.names[key]
        return self[:,idx]
    def __array_finalize__(self, obj):
        self.names = getattr(obj, "names", None)

class Record:
    def __init__(self, name, rtype, dtype='f4', keys=None):
        self.name = name
        self.rtype = rtype
        self.dtype = dtype
        keys = keys or []
        self.shape = {SCALAR: 1,
                      SDV: (len(keys),),
                      VECTOR: (3,),
                      TENSOR_3D_FULL: (9,),
                      TENSOR_3D: (6,)}[self.rtype]

        if rtype == SCALAR:
            self.keys = [self.name]
        elif rtype == SDV:
            self.keys = ['SDV_%s' % x for x in keys]
        else:
            components = COMPONENT_LABELS(rtype)
            self.keys = ['%s.%s' % (self.name, x) for x in components]

class Records(OrderedDict):
    _i = 0
    @property
    def num_rec(self):
        return len(super(Records, self).keys())

    def add(self, name, rtype, **kw):
        if rtype == SDV:
            keys = kw['keys']
            for key in keys:
                self.add('SDV_%s'%key, SCALAR)
        else:
            fo = Record(name, rtype, **kw)
            self[name] = fo

    def keys(self, expand=0):
        if expand < 0:
            return [x for x in super(Records, self).keys()
                    if not x.startswith('SDV_')]
        elif not expand:
            return super(Records, self).keys()
        return [key for f in self.values() for key in f.keys]

    @staticmethod
    def totuple(a):
        try:
            return tuple(a)
        except TypeError:
            return a

    def init(self, **kw):
        dtype = [(r.name, r.dtype, r.shape) for r in self.values()]
        self.data = np.empty((1,), dtype=dtype)
        sdv = kw.pop('SDV', None)
        row = [self.totuple(kw[key]) for key in self.keys(expand=-1)]
        if sdv is not None:
            row.extend(sdv)
        self.data[0] = tuple(row)
        self._cache = None

    def cache(self, **kw):
        data = np.empty((1,), dtype=self.data.dtype)
        sdv = kw.pop('SDV', None)
        row = [self.totuple(kw[key]) for key in self.keys(expand=-1)]
        if sdv is not None:
            row.extend(sdv)
        data[0] = tuple(row)
        if self._cache is None:
            self._cache = data
        else:
            self._cache = np.append(self._cache, data)

    def advance(self):
        self.data = np.append(self.data, self._cache, axis=0)
        self.clear_cache()

    def clear_cache(self):
        self._cache = None

class StateDB:
    def __init__(self, **kwds):
        self.db = {}
        for (kw, x) in kwds.items():
            self.db[kw] = np.array(x)
    def get(self, key):
        return self.db[key].copy()
    def advance(self, **kwds):
        for (kw, x) in kwds.items():
            self.db[kw] = x.copy()

class CutbackManager:
    def __init__(self):
        self.db = {}

    def __nonzero__(self):
        return bool(self.db)

    def __bool__(self):
        return self.__nonzero__()

    def __getattr__(self, key):
        return self.db.get(key)

    def request_cutback(self, **kwargs):
        self.db.update(kwargs)

    def clear(self):
        self.db = {}

CB = CutbackManager()
