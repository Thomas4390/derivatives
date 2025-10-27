"""Utilities to handle common optimization hacks

The ModelParameters class has a method `estimate` that calls the `minimize`
function scipy.optimize. Any model in the dorion_francois package subclassing
the ModelParameters class will definite its `objective` function (overwriting
the method) and `estimate` will handle the usual manipulations done to make the
estimation of parameters more straigtforward and/or robust. 

Everything else in here is essentially helping `estimate` handle that
goal. Scipy has significantly evolved since the first draft of the classes
therein. There might be overlaps, or scipy might even have better ways to do
this...

The documentation herein is severely lacking.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.optimize import minimize

from .tasks import get_logger
from .toolkit import struct

logger = get_logger(__name__)

def chat_gpt_translation(function):
    def decorated(*args, **kwargs):
        warnings.warn('This function was automatically translated (from Matlab) by Chat GPT. Not tested yet')
        return function(*args, **kwargs)
    return decorated


# Here is a Python class called Param. I providing it for context; you don't have to do anything yet
class Param:
    """Handles parameters being optimized in a (subclass of) ModelParameters

    Note that in Scipy's optimize module, setting a bound to np.inf is not equivalent to
    setting it to None. The two have different meanings and interpretations:

    - None: When you set a bound to None, it means the bound is not defined or not set for that
      particular decision variable. The optimizer will treat the decision variable as
      unconstrained in the corresponding direction.

    - np.inf: When you set a bound to np.inf, it means the bound is defined, but the decision
      variable is allowed to take any positive value without an upper limit. This is
      effectively setting an "infinite" upper bound, but the optimizer will still treat it as a
      bound in the problem formulation.

    For example, when using Scipy's minimize function with the 'L-BFGS-B', 'TNC', or 'SLSQP'
    methods, which support bounds, you should use None if you don't want to set a bound for a
    decision variable. Using np.inf will still result in a bound being defined, but it could
    lead to unexpected behavior or numerical issues during the optimization process.
    """
    def __init__(self, name, value=np.nan, bounds=(None, None), free=True):
        self.name = name
        self.value = value
        self.bounds = bounds
        self.free = free
        self.scale_slope = 2  # Sigmoid function
        self.inverse_sig_max = 10  # Optim domain will be in [-ISM, +ISM], ISM = inverse_sig_max

    def has_bounds(self):
        """Return False if both bounds are infinite"""
        return not ((self.bounds[0] is None) and (self.bounds[1] is None))
        
    def get_scaled_value(self):
        a = self.bounds[0]
        b = self.bounds[1]
        ss = self.scale_slope
        val = self.value

        # inverse sigmoid - Domain is real axis
        scaled = np.log((val-a) / (b-val)) / ss
        return scaled

    def set_scaled_value(self, scaled):
        a = self.bounds[0]
        b = self.bounds[1]
        ss = self.scale_slope

        if abs(scaled - self.inverse_sig_max) < 1e-4:
            warnings.warn(f"Scaled value for parameter {self.name} hits +/- ISM")
            logger.warning(f"Scaled value for parameter {self.name} hits +/- ISM")
        scaled = np.clip(scaled, -self.inverse_sig_max, self.inverse_sig_max)
        
        value = a + (b - a) / (1 + np.exp(-ss * scaled))

        # if abs(value - a) < 1e-9:
        #     warnings.warn(f"Scaled value for parameter {self.name} hits lb")
        # if abs(value - b) < 1e-9:
        #     warnings.warn(f"Scaled value for parameter {self.name} hits ub")
        
        self.value = value
        return value

    def bounds_to_constraints(self, pno):
        """Return inequality constraints of non-None bounds of the parameter.

        According to some websites, COBYLA does not handle bounds but does handle inequality
        constraints. That said, the inequality constraints were not respected in some of CD's
        preliminary tests, so it is better to use COBYLA (or Nelder-Mead) with
        ModelParameters.scale_pvalues and ModelParameters.penalize_constraints.
        """
        bounds = []
        if self.bounds[0] is not None:
            # inequality means that it is to be non-negative            
            bounds.append({'type':'ineq', 'fun': lambda x: x[pno]-self.bounds[0]})
        if self.bounds[1] is not None:
            # inequality means that it is to be non-negative            
            bounds.append({'type':'ineq', 'fun': lambda x: self.bounds[1]-x[pno]})
        return bounds    
    
    # def illustrate_scaling(self):
    #     x = np.hstack((0.00001, 0.0005, np.arange(0.01, 0.99, 0.001), 0.9995, 0.99999))
    #     x = self.bounds[0] + x * np.diff(self.bounds)
    #     y = self.scale(x, self.bounds[0], self.bounds[1])
    #     z = self.unscale(y, self.bounds[0], self.bounds[1])
    #     x1 = self.value
    #     y1 = self.get_scaled_value()

    @staticmethod
    def scale(x, lb, ub):
        obj = Param('x', x, [lb, ub])
        scaled = obj.get_scaled_value()
        return scaled

    @staticmethod
    def unscale(y, lb, ub):
        obj = Param('x', np.nan, [lb, ub])
        value = obj.set_scaled_value(y)
        return value

    @staticmethod
    def illustrate_scaling():
        x = np.concatenate(([0.00001, 0.0005], np.arange(0.01, 0.99, 0.001), [0.9995, 0.99999]))
        y = Param.scale(x, 0, 1)
        z = Param.unscale(y, 0, 1)
        #import pdb; pdb.set_trace()
        
        fig, axes = plt.subplots(3, 1)
        axes[0].plot(x, y)
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Scaled')
        
        axes[1].plot(x, x, label='45-degree line')        
        axes[1].plot(x, z, label='Recovered')
        axes[1].legend(loc='lower right')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Recovered value')
        
        axes[2].plot(x, z-x)
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Error on value')
        
        fig.tight_layout()
        plt.show()

        fig, axes = plt.subplots(4, 1)
        ISM = Param('default').inverse_sig_max
        y = np.arange(-ISM, ISM+0.1, 0.1)
        for no in range(4):
            ax = axes[no]
            Mx = 10**(no)
            z = Param.unscale(y, -Mx, Mx)
            ax.plot(y, z)
            ax.set_xlabel('Optim value')
            ax.set_ylabel(f"Param value [-{Mx}, {Mx}]")
            DIFF = np.array([z[0]+Mx, z[-1]-Mx]) / Mx
            print(DIFF)        
        plt.show()
        
        fig, axes = plt.subplots(4, 1)
        y = np.arange(-5, 5+0.1, 0.1)
        for no in range(4):
            ax = axes[no]
            Mx = 10**(no)
            z = Param.unscale(y, 0, Mx)
            ax.plot(y, z)
            ax.set_xlabel('Optim value')
            ax.set_ylabel(f"Param value [0, {Mx}]")
            DIFF = np.array([z[0], z[-1]-Mx]) / Mx
            print(DIFF)
        
        plt.show()
#</class Param>

class PV(struct): # See namedtuples: better alternative?
    """Minimalist class meant to have Param entries, accessible like attributes"""
    pass

class SealedModelParameters:
    """Implements methods that are not meant to be overriden in subclasses but, yet, are public.

    A metaclass could be used to ensure that these are NOT overriden in subclasses. For now, we'll 
    just trust ourselves.
    (cf. https://stackoverflow.com/questions/3948873/prevent-function-overriding-in-python)    
    """
    #TBA? @chat_gpt_translation
    #TBA? def add_param_vector(self, vector):
    #TBA?     """
    #TBA?     vector must be an instance of ParamVector
    #TBA?     """
    #TBA?     self.param_vectors.append(vector)
    #TBA?     for no in range(vector.numel):
    #TBA?         self.params[vector.element_names[no]] = vector.elements[no]
    def __init__(self):
        super().__init__()
        self._optimizer_x = np.array([])
    
    def add_parameter(self, *args, **kwargs):
        """Arguments are forwarded to Param's constructor and the result is added to self.params"""
        p = Param(*args, **kwargs)
        setattr(self.params, p.name, p)
    
    def remove_parameter(self, *args):
        """Take an arbitrary number of parameter names and remove them from self.params"""
        for name in args:
            delattr(self.params,name)

    def fix_parameter(self, name=None, value=None, **kwargs):
        if name is None and value is None:
            for name in kwargs:
                self.fix_parameter(name, kwargs[name])
            return
        assert len(kwargs)==0
        
        self.params[name].free = False
        if value is not None:
            self.params[name].value = value

    def free_parameter(self, name, value=None):
        self.params[name].free = True
        if value is not None:
            self.params[name].value = value
    
    def get_pv(self, *args):
        """Return a structure with the model's parameters.
        
        If the model has parameters named `a`, `b`, and `c`, this method should return a structure such that
        `pv.a`, `pv.b`, and `pv.c` yield the the parameter values.

        Args:
            *args: 
                The names of the parameters to be returned. If no names are provided, all parameters are returned.
        
        Returns:
            pv: 
                A structure with the model's parameters. The structure allow attribute or dictionary access to the parameter values.
        """
        pv = PV()
        if not args:
            args = self.params.keys()
        for param_name in args:            
            pv[param_name] = self.params[param_name].value            
        #more?
        return pv
    
        #more? more = self.get_more_pv(pv, **kwargs)
        #more? more_names = set(more.keys())
        #more? common_names = set(pv.keys()) & more_names
        #more? if common_names:
        #more?     raise ValueError("Must NOT modify values of actual PARAMS in PV")
        #more? 
        #more? pv.update(more)


    def get_free_pv(self):
        """Returns the parameters that are free in the optimization, as a structure.
        
        Akin to `get_pv`, called with the names of free parameters only.
        """
        pv = PV()
        for param_name in self.params:
            if self.params[param_name].free:
                pv[param_name] = self.params[param_name].value
        return pv
    
        #mlab2py: pv = self.get_pv()
        #mlab2py: fields_to_remove = []
        #mlab2py: for field in pv:
        #mlab2py:     if field not in self.params or self.params[field].fixed:
        #mlab2py:         fields_to_remove.append(field)
        #mlab2py: for field in fields_to_remove:
        #mlab2py:     del pv[field]
        #mlab2py: return pv
    
    def get_pvalues(self, *args):
        """Return the value of the parameters named in *args in a list.

        If no arguments are provided, all parameter values are returned.
        """
        params = self.params
        if not args:
            args = params.keys()
        if len(args)==1 and not isinstance(args[0],str):
            args = args[0]
        return [params[arg].value for arg in args]

    # Python implementation based on dictionnary makes this useless. See get_pv, and use set_pv.
    #TBR: def get_parameter_values(self, *args):
    #TBR:     """Return the value of the parameters named in *args in a list of pairs 
    #TBR: 
    #TBR:     The result can be used in set_parameter_values. If no arguments are provided, all
    #TBR:     parameters are returned.
    #TBR:     """
    #TBR:     params = self.params
    #TBR:     if not args:
    #TBR:         args = params.keys()
    #TBR:     values = [(arg, params[arg].value) for arg in args]
    #TBR:     return values

    
    def set_pv(self, *args, **kwargs):        
        """Accept a PV instance, or dictionnary, and set parameter values of the named entries"""
        if len(args)==1 and len(kwargs)==0:
            pv = args[0]
        elif len(args)==0 and len(kwargs) > 0:
            pv = kwargs
        else:
            raise ValueError('args=%s\nkwargs=%s'%(args,kwargs))
        
        params = self.params
        pnames = list(pv.keys())
        assert all(name in params for name in pnames), 'Setting value of invalid parameter'

        for field in pnames:
            params[field].value = pv[field]

        # For convenience
        pv = self.get_pv()
        return pv

    # Python implementation based on dictionnary makes this useless. See get_pv, and use set_pv.
    #TBR: @chat_gpt_translation
    #TBR: def set_parameter_values(self, *args):
    #TBR:     """Set the (actual) value of the parameters named in *args; must consist of "pname,value" pairs."""
    #TBR:     params = self.params
    #TBR:     pnames = args[::2]
    #TBR:     values = args[1::2]
    #TBR:     assert all(name in params for name in pnames), 'Setting value of invalid parameter'
    #TBR: 
    #TBR:     for i, field in enumerate(pnames):
    #TBR:         params[field].value = values[i]
    #TBR:     self.params = params
    #TBR:     self.setDependentParameters()
    #TBR: 
    #TBR:     if len(args) % 2 == 1:
    #TBR:         # If the number of arguments is odd, return the PV.
    #TBR:         pv = self.getPV()
    #TBR:         return pv


    ###<Protected methods>
    def _get_optimizer_pvalues(self): # was getOptimizerPVec in Matlab
        """Return a vector of numeric values for the *free* parameters.

        Also return the associated bounds and names. The names of the free parameters 
        can be useful to build constraints on parameters.

        When self.scale_pvalues is True, the pvalues returned are a nonlinear transformation
        of the actual parameters, using an inverse_sigmoid, which yields the parameter's lower 
        bound at -np.inf, and the upper bound at np.inf. Scaled parameters are thus unconstrained.
        Numerically, [-5,5] should span most of [lb, ub].
        """
        params = self.params
        param_names = list(params.keys())

        pvalues = []
        bounds = []
        names = []
        has_bounds = False
        for pname in param_names:
            if params[pname].free:
                if self.scale_pvalues:
                    pvalues.append(params[pname].get_scaled_value())
                    has_bounds = False

                    # Numerically, [-5,5] should span most of [lb, ub].
                    # params[pname].inverse_sig_max defaults to 10
                    if abs(abs(pvalues[-1]) - params[pname].inverse_sig_max) < 1e-6:
                        warnings.warn(f"Scaled pvalue for parameter {pname} hits +/- ISM")                        
                else:
                    pvalues.append(params[pname].value)
                    bounds.append(params[pname].bounds)
                    has_bounds = has_bounds or params[pname].has_bounds()
                names.append(pname)

        if not has_bounds:
            # All bounds are (None,None)
            bounds = None # Allow the use of unbound optimizers
                
        return pvalues, bounds, names

    
    def _set_optimizer_pvalues(self, pvalues): # was setOptimizerPVec in Matlab
        """Set the value of the *free* parameters. 

        The length of pvalues must be equal to the number of free parameters.
        """
        if np.array_equal(self._optimizer_x, pvalues):
            return # parameters were already set
        
        params = self.params
        param_names = list(params.keys())
        n_values = len(pvalues)

        #import pdb; pdb.set_trace()
        free = 0
        for pname in param_names:
            if params[pname].free:
                if self.scale_pvalues:
                    params[pname].set_scaled_value(pvalues[free])
                else:
                    params[pname].value = pvalues[free]
                free += 1

        assert n_values == free, f"Length of pvalues ({n_values}) is not equal to the number of free parameters ({free})"
        #more? self.setDependentParameters()

        # cache the values that were just set
        self._optimizer_x = pvalues
        
    ###</Protected methods>

            
# Here is a Python class called ModelParameters; it will use the Param class above. I providing it for context; you don't have to do anything yet
class ModelParameters(SealedModelParameters):

    # currently supported...
    unbounded_methods = ['cobyla', 'nelder-mead'] 
    unconstrained_methods = ['nelder-mead'] 
    
    def __init__(self):
        super().__init__()
        
        self.params = PV()
        #TBA? self.param_vectors = [] 
        self.scale_pvalues = False
        self.normalize_objective = True
        
        #TBA?
        #options = {'disp': True, 'maxiter': 250, 'catol': 1e-4, 'rhobeg': 1.0, 'tol': 1e-4}
        #self.optimopt = options

    def objective(self, *args, **kwargs):
        raise NotImplementedError("Subclass must define the objective function")

    def constraints(self, method='', **kwargs):
        """Return the constraints on the model parameters
        
        cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        The default implementation returns an empty list, excepth when method is COBYLA.

        According to some websites, COBYLA does not handle bounds but does handle inequality
        constraints. That said, the inequality constraints were not respected in some of CD's 
        preliminary tests, so it is better to use COBYLA with ModelParameters.scale_pvalues and 
        ModelParameters.penalize_constraints.
        """
        constraints = []
        if method.lower() in self.unbounded_methods and not self.scale_pvalues:            
            _,_,free_pnames = self._get_optimizer_pvalues()
            for pno,pname in enumerate(free_pnames):
                constraints += self.params[pname].bounds_to_constraints(pno)
        return constraints

    def estimate(self, observations, **kwargs):
        # The cobyla method neglects parameter bounds. The bounds_to_constraints constraints 
        # added in constraints unfortunately do not seem to bind "hard"... Use it jointly with
        # penalize_constraints

        # Setting SLSQP as a default method
        method = kwargs.pop('method','SLSQP')
        kwargs['method'] = method 
        
        n_obs = observations.shape[0]
        obs = observations.values if isinstance(observations,pd.Series) else observations
        def _objective(x):
            self._set_optimizer_pvalues(x)
            fval = self.objective(obs, **kwargs)
            if self.normalize_objective:
                return fval / n_obs # tol on optimizer will thus be relative to the number of observations
            return fval

        # The objective function might fix some parameters; run beforehand
        f0 = self.objective(obs, **kwargs) # objective function at initial values
        x0, bounds, free_pnames = self._get_optimizer_pvalues()        

        # For unconstrained methods, wrap the objective function in penalize_constraints        
        penalty_factor = 1e3
        if method.lower() in self.unconstrained_methods:
            objective = lambda x: self.penalize_constraints(x, _objective, penalty_factor, **kwargs)
            constraints = None
        else:
            objective = _objective
            constraints = self.constraints(**kwargs)

        ### Log the initial values, bounds, and pvalues
        # info = {'x0':x0}
        # if self.scale_pvalues: # x0 in in the transformed space
        #     info['pv'] = self.get_pvalues(*free_pnames) # parameter values in the actual space
        # info['lb'] = [p.bounds[0] for p in self.params.values() if p.free]
        # info['ub'] = [p.bounds[1] for p in self.params.values() if p.free]
        # 
        # with pd.option_context('display.float_format', '{:,.6g}'.format):
        #     logger.debug( pd.DataFrame(info, index=free_pnames))
        #     if constraints:
        #         for cno,_cstr in enumerate(constraints):
        #             logger.debug('Constraint %d'%cno, _cstr)
        ####

        opt = minimize(objective, x0, bounds=bounds, constraints=constraints, 
                       method=method, options={'maxiter': 5000})

        self._set_optimizer_pvalues(opt.x)
        opt.fun_unc = self.objective(obs, **kwargs)
        opt.fun_con = self.penalize_constraints(opt.x, lambda x: opt.fun_unc, penalty_factor, **kwargs)

        if 'multithread' not in kwargs or not kwargs['multithread']:
            opt = self.finalize_optim(opt, observations, **kwargs)
        return opt

    def multiprocess_estimate(self, observations, *args, **kwargs):
        raise NotImplemented('Crude code below just for inspiration.')
        columns = free_pnames + ['nfev','success','fun','objective']
        estimates = pd.DataFrame()
        
        for gamma in np.arange(0,1,0.05):
            pv = self.get_pv()
            pi = self.persistence(pv)
            beta = pi - pv.alpha*(1 + pv.gamma**2)
            self.set_pv(beta=beta, gamma=gamma)
            opt = super().estimate(log_xret.values, *args, **kwargs)
            summary = model.summary(log_xret, opt).set_index(pd.Index([gamma]))
            estimates = pd.concat((estimates, summary), axis=0)  
    
        names = ['omega','alpha','beta','gamma']
        summary = estimates.iloc[[np.argmax(estimates.LL)]].set_index(log_xret.index[[-1]])
        #import pdb; pdb.set_trace()
        self.set_pv(summary[names].iloc[0].to_dict())
        return summary
    
    def penalize_constraints(self, x, objective, penalty_factor, **kwargs):
        """Return the value of the objective function plus a penalty for constraint violation 
        
        This function **must** be evaluated **after** the objective function is evaluated. Internal 
        state of the model will thus have been update as well, if the objective function does 
        affect the said state.
        
        See CD's Matlab `parameters` library's penalizeConstraints for an alternative implementation with
        ```
        A, b, Aeq, beq = self.linear_constraints()
        c, ceq = self.nonlinear_constraints()
        ```
        """
        #import pdb; pdb.set_trace()
        obj = objective(x)
        c_values = []
        constraints = self.constraints(**kwargs)
        for cstr in constraints:
            fun = cstr['fun']
            args = cstr.pop('args',[])
            cval = fun(x, *args)
            c_values.append(cval) # Inequality constraints must be non-negative
            if cstr['type']=='eq':
                c_values.append(-cval) # Equality constraints must also be non-positive

        # The call to .min() returns the most negative of c_values, if any. 
        # If all are positive, maximum yields 0.0
        violation = 0.0
        if c_values:
            #import pdb; pdb.set_trace()
            violation = np.maximum(0, -np.array(c_values).min())
            
        return obj + penalty_factor*violation
    
    def finalize_optim(self, opt, observations, **kwargs):
        """Called once scipy.optimize.minimize returns"""
        if not opt.success:
            #self._set_optimizer_pvalues(x0)
            warnings.warn("Optimization was not successful")
        # else:
        #     grad = self.gradient(res.x, *args, **kwargs)
        #     hessian = self.hessian(res.x, *args, **kwargs)
        #     stderr = self.bwerr(grad, hessian)
        #     if self.scale_pvalues:
        #         warnings.warn("grad and hessian are on the scaled parameters",
        #                       category=UserWarning)    
        return opt
        
    #TBA? @chat_gpt_translation
    #TBA? def numerical_jacobian(self, func, precision):
    #TBA?     """
    #TBA?     Compute the symmetric numerical first order derivatives of a multivariate function of the
    #TBA?     ModelParameters at current value (get_free_pv).
    #TBA? 
    #TBA?     Inputs:
    #TBA?     - func: Method of self that depends on the instance's parameters (setParameterValues is
    #TBA?       used in this method)
    #TBA?     - precision: Percentage of variation (+-) around X (between 0 and 1, ideally close to 0).
    #TBA?       Note that X are the (potentially) transformed parameter values, but the Jacobian must be
    #TBA?       returned on the actual parameter values.
    #TBA? 
    #TBA?     Output:
    #TBA?     - J: Derivatives (D x 1)
    #TBA?     """
    #TBA?     # get current parameter values
    #TBA?     pv0 = self.getFreePV()
    #TBA?     pnames = list(pv0.keys())
    #TBA? 
    #TBA?     D = len(pnames)
    #TBA?     ncols = len(func(self.getOptimizerPVec()))
    #TBA?     J = np.empty((D, ncols))
    #TBA? 
    #TBA?     for no, pnam in enumerate(pnames):
    #TBA?         xm = pv0[pnam] * (1 - precision)
    #TBA?         self.setParameterValues(pnam, xm)
    #TBA?         fm = func(self.getOptimizerPVec())
    #TBA? 
    #TBA?         xp = pv0[pnam] * (1 + precision)
    #TBA?         self.setParameterValues(pnam, xp)
    #TBA?         fp = func(self.getOptimizerPVec())
    #TBA? 
    #TBA?         # allow function to be vector-valued
    #TBA?         J[no, :] = (fp - fm) / (xp - xm)
    #TBA? 
    #TBA?         self.setPV(pv0)
    #TBA? 
    #TBA?     return J

    ## Python offers decorators etc that could allow for a cleaner implementation of this feature. When
    ## the need arises, let's rethink about this
    #? def getMorePV(self, pv, *args):
    #?     # This method allows subclasses to *append* parameters to the PV struct returned by getPV. 
    #?     # DO NOT modify the PV parameters
    #?     more = {}
    #?     for pv_obj in self.param_vectors:
    #?         name = pv_obj.name
    #?         more[name] = pv_obj.getPValues()
    #?     return more
    #? 
    #? def setDependentParameters(self):
    #?     # If any parameter of the model depends on the value other parameters, update them before fetching values 
    #?     # Nothing to do by default
    #?     pass       
        
    @chat_gpt_translation
    def illustrate_scaling(self, name, initial_parameters, bounds, final_parameters):
        ISM = 5
        y = np.arange(-ISM, ISM + 0.1, 0.1)
        fig = plt.figure(figsize=(10, 6))
        
        j = 1 # subplots
        for i in range(len(initial_parameters)):
            ax1 = fig.add_subplot(len(initial_parameters), 2, j)
            lb = bounds[i, 0]
            ub = bounds[i, 1]
            z = Param.unscale(y, lb, ub)
            ax1.plot(y, z, 'r')
            ax1.axhline(initial_parameters[i], color='k', linestyle='-')
            ax1.axhline(final_parameters[i], color='k', linestyle='--')
            ax1.set_xlabel('Optim value')
            ax1.set_ylabel(f'Param value [{lb}, {ub}]')
            ax1.set_title(name[i])
            
            ax2 = fig.add_subplot(len(initial_parameters), 2, j+1)
            ax2.plot(z, y, 'r')
            ax2.axvline(initial_parameters[i], color='k', linestyle='-')
            ax2.axvline(final_parameters[i], color='k', linestyle='--')
            ax2.set_xlabel('Param value')
            ax2.set_ylabel(f'Optim value [-{ISM}, {ISM}]')
            ax2.set_title(name[i])
            
            j += 2
        fig.tight_layout()
        plt.show()
    

class NestedOptimization(ModelParameters):
    """Optimizing one of the nested model's parameters, nesting the optimization of others.
    
    This models performs a 1-dimensional optimization, fixing a designated parameter
    the "parent" (or "outer") parameter while the nested "child" ("inner") model optimizes 
    the remaining free parameters, conditional on given values of the parent parameter. 
    
    This is particularly convenient:
    
    1. When investigating the shape of the objective as a function of a single parameter
    2. When there are tensions between two of the (innermost) model's parameters. Fixing one
    first and optimizing the other simplifies the inner estimation, and then this class 
    optimizes the value of the parent parameter, i.e. the single parameter of the 
    outer model.
    
    By default, the constructor of NestedOptimization assumes a two-level optimization. However, 
    this class supports multilevel optimization. In the code below, for instance:

    ```
    model = Merton76()
    model.scale_pvalues = True

    # Comment out the initial values to illustrate how the optimizer gets stuck 
    # restarting with "jump to default" parameters
    mid_model = NestedOptimization(model, 'lmbda')
    mid_model.initial_values = {'mu_jump':0, 'sigma_jump':1}

    top_model = NestedOptimization(model, 'sigma', mid_model)

    opt = top_model.estimate(dtm1, method='Nelder-Mead')
    print(opt)
    top_model.plot_optimization_grid(dtm1)
    ```
    
    the third argument when constructing `top_model` specifies that `top_model`'s child is not 
    the `model` itself, but the `mid_model`.
    """
    def __init__(self, model, param_name, child=None):
        """Wraps the ModelParameters' instance `model` with parent parameter `param_name`.
        
        If the `child` argument is provided, then `model` is understood to be the innermost model 
        (i.e. the actual model one intends to estimate), whereas the child model must be an 
        instance of NestedOptimization.
        """
        super().__init__()
        
        # The innermost model
        self.model = model
        assert isinstance(model, ModelParameters)
        
        # The child model is simply the innermost model in a two-level optimization
        self.child = child if child is not None else model
        assert child is None or isinstance(child, NestedOptimization)
        
        # The outer parameter optimized at this level
        param = model.params[param_name]
        self.add_parameter(param_name, param.value, param.bounds)

        # Adopt the scaling policy of the nested model
        self.scale_pvalues = model.scale_pvalues

        # Keep track of the nested optimization
        self.grid = []
        self.param_name = param_name
        self.model_pnames = list(model.params.keys())
        
        # The initial values for fixed parameters will be neglected
        self.initial_values = {}        
        
    def objective(self, *args, **kwargs):
        # The super().minimimize method calls this method after having set the
        # parameter values of parameter `param_name`. The next line sets that same 
        # value as a fixed parameter of the nested model
        self.model.fix_parameter( **self.get_pv() )
        #IMPORTANT: fixing on self.model (innermost)
        
        # Potentially reinitializing the "other" parameter values
        if self.initial_values:
            free = self.model.get_free_pv()
            for pname in free.keys():
                if pname in self.initial_values:
                    self.model.params[pname].value = self.initial_values[pname]
        
        # Given the fixed parameter, optimize the inner parameters of the model
        opt = self.child.estimate(*args, method='Nelder-Mead')
        #IMPORTANT: optimizing on self.child (potential NOT the innermost)
        
        # Keep track of the "optimal" parameter sets
        self.grid.append(self.model.get_pvalues()+[opt.fun_unc, opt.fun])
        
        # The objective function is simply that of the nested model (accounting for constraints).
        return opt.fun
        
    def plot_optimization_grid(self, *args):
        opt_pv = self.model.get_pv()
        print('Optimum:',opt_pv)

        grid = pd.DataFrame(self.grid, columns=self.model_pnames+['pricing_error', 'objective'])
        grid = grid.sort_values(by=self.param_name).set_index(self.param_name)
        n_rows = int(np.ceil(len(self.model_pnames)/2))
        fig,axes = plt.subplots(n_rows,2, figsize=(2*6,n_rows*5-1))
        for no,ax in enumerate(axes.reshape(-1)):
            name = grid.columns[no]
            ax.plot(grid[[name]], label=name)
            ax.axvline(self.params[self.param_name].value, color='k', linestyle=':')
            ax.set_xlabel(self.param_name)
            ax.legend()
        
        
        
# Now I would like to add Python translations of the methods below to the ModelParameter class. Rememeber that self.params.PARAM would give you access to an instance of Param, named PARAM. No need to recopy what you already gave me. Assume that self is an instance of ModelParameter and that the necessary methods (e.g. self.getFreePV) have been implemented otherwise

if __name__ == "__main__":
    if False:
        Param.illustrate_scaling()

    

