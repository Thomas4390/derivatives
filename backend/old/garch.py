import argparse
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import pickle
import pdb
import random
import unittest
import warnings
from IPython.display import display, Markdown

# import antitheric_normal from monte_carlo.py
from .monte_carlo import antithetic_normal

#import numba
from .parameters import *
from .tasks import get_logger
from .toolkit import compile_to_shared_object, subcalendar

logger = get_logger(__name__)

# For improved performance, some bottlenecks in the Python code were converted to C code
if True:
    # Should ngarch_filter use const? (and would using contiguous in log_likelihood_c improve performance?)
    import ctypes

    # First, compile the library (in terminal)
    current_file = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file)
    c_file = os.path.join(current_folder, 'garch_c.c')
    so_file = compile_to_shared_object(c_file)
    
    # Load the shared library
    garch_c = ctypes.CDLL(so_file)

    # Define the argument types and return type for the functions

    garch_c.agarch_filter.argtypes = [ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_int,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float]
    garch_c.agarch_filter.restype = None

    garch_c.ngarch_filter.argtypes = [ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_int,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float]
    garch_c.ngarch_filter.restype = None

    garch_c.ngarch_sim.argtypes = [ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_float,
                                       ctypes.c_float,
                                       ctypes.c_float,
                                       ctypes.c_float]
    garch_c.ngarch_sim.restype = None
    
class garch(ModelParameters):
    """
    Implements the GARCH model parameters initialization and operations.
    
    Attributes:
        h_min (float): The minimum threshold for variance.
        h0_window (int): The window size for initial variance estimation.
        var_target_window (int, optional): The window size for variance targeting. Defaults to None.        
    """

    @classmethod
    def on_fama_french_mkt(cls, estimation_time_t0=None, *args, **kwargs):
        """Builds a GARCH instance on the Fama-French market data.

        Args:
            estimation_time_t0:
                A timestamp indicating the time at which the sequential estimation starts. If None, the estimation starts on the first business day following 2018-01-01. 
            *args: 
                Forwarded to the GARCH constructor.
            *kwargs: 
                Forwarded to the GARCH constructor.
        
        Returns:
            model: A GARCH model instance.
        """
        model = cls(*args, **kwargs)
        
        import pandas as pd
        import pandas_datareader as pdr
        model.data = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start='1900-01-01')[0]

        # The Mkt-RF series is in simple returns. Convert to log-returns
        model.data['log_xret'] = np.log(1+model.data['Mkt-RF']/100)

        if estimation_time_t0 is None:
            estimation_time_t0 = pd.Timestamp('2018-01-01 00:00:00')
        from_t0 = model.data.index >= estimation_time_t0
        model.estimation_time_t0 = model.data.index[from_t0][0]

        return model

    def __init__(self, h_min=(0.05**2)/252, h0_window=21, var_target_window=None):
        super().__init__()
        self.h_min = h_min
        self.h0_window = h0_window
        self.var_target_window = var_target_window
        
    def objective(self, log_xret, **kwargs):
        """Calculates the objective function value given log excess returns.
        
        Currently, this method supports P-MLE only. It asserts that `log_xret` is an instance of np.ndarray.
        
        Args:
            log_xret (np.ndarray): Log excess returns array.
            **kwargs: Arbitrary keyword arguments. Ignored in the current implementation.
            
        Returns:
            float: The negative log likelihood value.
            
        Raises:
            AssertionError: If `log_xret` is not an instance of np.ndarray.
        """
        assert isinstance(log_xret,np.ndarray), type(log_xret)
        if self.var_target_window is None:
            pv = self.get_pv()            
        else:
            var_target = np.var(log_xret) if self.var_target_window <= 0 \
                             else np.var(log_xret[-self.var_target_window:])

            # Update the omega parameter so that, given the new values of the other parameters, the variance target is still met.
            pv = self.variance_targeting(var_target)
        
        return -self.log_likelihood(log_xret, pv)

    def get_calendar(self, after=None, first_date=None, last_date=None, before=None):
        """Generates a calendar using dorion_francois.toolkit's subcalendar.
        
        This method assumes that `self.data` exists and is indexed using dates.
        
        Args:
            calendar: 
                the calendar or `pd.DataFrame` to subsample from
            after: 
                excluding dates before or on the `after` date (mutually exclusive with `first_date`)
            first_date: 
                starting on first_date, included if it exists (mutually exclusive with `after`)
            last_date: 
                ending on last_date, included if it exists (mutually exclusive with `before`)
            before: 
                excluding dates after or on the `before` date (mutually exclusive with `last_date`)
            
        Returns:
            Array-like: A subcalendar for `self.data` array based on the specified date range.
        """
        return subcalendar(self.data.index,
                           after=after, first_date=first_date, last_date=last_date, before=before)

    def past_log_excess_returns(self, time_t):
        """Returns the historical log excess returns up to `time_t`.

        Assuming that self.data exists and is indexed using dates.

        Args:
            time_t: A comparable date to those in the index of the returns' dataframe.

        Returns:
            A pd.DataFrame of historical log excess returns up to the specified date.
        """
        ix = self.data.index <= time_t
        return self.data[['log_xret']].loc[ix]

    def validate_estimates(self, method, time_t, estimates):
        """Re-estimate the parameters at `time_t` and validate that they are comparable to `estimates`"""
        log_xreturns = self.past_log_excess_returns(time_t)
        self._time_t = log_xreturns.index[-1]
        opt = self.estimate(log_xreturns.values, method=method)
        est = self.summarize_state(log_xreturns, opt) 

        is_close = lambda a,b: np.isclose(a,b, rtol=0.01, atol=1e-8) # 0.0001bp and 1% tolerance
        match = is_close(est, estimates.iloc[[-1]])[0,:]
        assert set(estimates.columns[~match]).issubset(['nfev']), 'Inconsistent estimates'
    
    def estimate_sequentially(self, method, scale_pvalues, filename, last_date=None):
        """Sequentially estimates the model from self.estimation_time_t0 to the last date in the calendar.
        
        This method uses the calendar defined by `self.data.index` for iteration and updates estimates either from the start, or
        from the last saved date on disk if `filename` exists. This is particularly convenient if the estimation process is interrupted, or if we want to update the estimates given a new data point. The estimates are saved to disk after each update.

        As such, most of this method is dedicated to managing the estimation process, while the actual estimation is done by the `estimate` method called within a loop. The method prints the estimates monthly.:
        ```
            log_xreturns = self.past_log_excess_returns(time_t)
            # ...
            opt = self.estimate(log_xreturns.values, method=method)
        ```
        The default implementation of `estimate` is that of `parameters.ModelParameters`, but subclasses can override it.
        
        Args:
            method (str): 
                The estimation method to use.
            scale_pvalues (bool): 
                Indicates whether to scale p-values.
            filename (str): 
                The file path to save or read the estimates.
            last_date (date, optional): 
                The last date to consider in the calendar. Defaults to None (self.data.index[-1]).
            
        Returns:
            DataFrame: A DataFrame containing the estimates for each date in the calendar.
        """
        self.scale_pvalues = scale_pvalues
        
        if not os.path.exists(filename):
            print('Starting from time_t0:',self.estimation_time_t0)
            log_xreturns = self.past_log_excess_returns(self.estimation_time_t0)
            self._time_t = log_xreturns.index[-1]
            opt = self.estimate(log_xreturns.values, method=method)
            estimates = self.summarize_state(log_xreturns, opt)
            with open(filename,'wb') as fh:
                pickle.dump(estimates,fh)
            logger.info(estimates)
            time_t = self.estimation_time_t0
            
        else:
            #import pdb; pdb.set_trace()
            estimates = self.read_estimates(filename, set_last_pv=True)
            assert estimates.index[0]==self.estimation_time_t0, 'Updating with inconsistent time_t0'
            time_t = estimates.index[-1]
            self.validate_estimates(method, time_t, estimates)
            logger.info('Updating from time_t:%s',time_t.strftime('%Y-%m-%d'))
            logger.info(estimates.loc[time_t])



        cur_month = time_t + pd.tseries.offsets.MonthEnd()  
        calendar = self.get_calendar(after=time_t, last_date=last_date)
        for time_t in calendar:
            # The assumption is that we are working after the close.    
            log_xreturns = self.past_log_excess_returns(time_t)
            self._time_t = log_xreturns.index[-1]
        
            # Start from previous day's optimal values: the model's parameters are left at their 
            # optimal values, as set within the previous model.estimate call
            #import pdb; pdb.set_trace()
            opt = self.estimate(log_xreturns.values, method=method)
            est = self.summarize_state(log_xreturns, opt)    
            estimates = pd.concat((estimates, est), axis=0)
            with open(filename,'wb') as fh:
                pickle.dump(estimates,fh)

            # Print updates monthly 
            if time_t > cur_month:
                logger.info(estimates.loc[time_t])
                cur_month = time_t + pd.tseries.offsets.MonthEnd()
        
        return estimates

    def read_estimates(self, filename, set_last_pv=False):
        """Reads parameter estimates from a file and optionally updates the model's parameters.
        
        Args:
            filename (str): The path to the file containing the estimates.
            set_last_pv (bool, optional): If True, updates the model's parameters to the last estimated values. Defaults to False.
            
        Returns:
            DataFrame: A DataFrame containing the parameter estimates.
        """        
        with open(filename,'rb') as fh:
            estimates = pickle.load(fh)

        if set_last_pv:
            last = estimates.iloc[-1]
            pnames = self.params.keys()
            update = dict([(name,last[name]) for name in set(last.index).intersection(pnames)])
            self.set_pv(update)

        return estimates

    #cf. lines 488-490 in parameters.py
    def finalize_optim(self, opt, observations, **kwargs):
        opt = super().finalize_optim(opt, observations, **kwargs)
        opt.LL = self.log_likelihood(observations, self.get_pv()) 
        return opt
                    
    def summarize_state(self, log_xret, opt):
        """Summarizes the current state of the model given log excess returns and optimization results.
        
        Args:
            log_xret (np.ndarray): The log excess returns.
            opt (OptimizationResult): The result of the optimization process.
            
        Returns:
            DataFrame: A DataFrame summarizing the state of the model.
        """
        from scipy.stats import skew,kurtosis
        pv = self.get_pv() # get the current parameter values (pv)
        self.log_likelihood(log_xret, pv=pv)
        innov,hvar = self.eps,self.h # set in within call to self.log_likelihood
        
        h_tp1 = hvar[-1,0]
        h_t = hvar[-2,0]
        erp = self.equity_risk_premium(h_tp1, pv=pv)
        names = list(pv.keys()) + ['LL','nfev','erp','z_bar','z_std','z_skew','z_kurt','z_t','h_t','h_tp1']
        values = list(pv.values()) + [opt.LL, opt.nfev, erp, 
                    innov.mean(),innov.std(),skew(innov)[0],kurtosis(innov)[0], innov[-1,0], h_t, h_tp1]
        return pd.DataFrame([values], index=log_xret.index[[-1]], columns=names)

class agarch(garch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_parameter('lmbda', 0.00, [  -1, 1])
        self.add_parameter('omega', 6e-7, [ 0.0, 1e-3]) 
        self.add_parameter('alpha', 3e-6, [1e-9, 0.5])
        self.add_parameter('beta',  0.90, [ 0.5, 0.9999])
        self.add_parameter('gamma',  135, [-1e3, 1e3])

        if self.var_target_window is not None:
            self.fix_parameter('omega', np.nan)
    
    def variance_targeting(self, var_target, pv=None, fix_param=True):
        """Adjusts the omega parameter for variance targeting given a target variance."""
        if pv is None:
            pv = self.get_pv()
        omega = (1 - self.persistence(pv)) * var_target - pv.alpha
        if fix_param:
            self.fix_parameter(omega=omega)
            return self.get_pv()
        return omega

    def persistence(self, pv=None):
        """Calculates the persistence of volatility shocks in the AGARCH model."""
        if pv is None:
            pv = self.get_pv()
        return pv.alpha*pv.gamma**2 + pv.beta

    def uncond_var(self, pv=None):
        """Unconditional variance of the model."""
        if pv is None:
            pv = self.get_pv()
        return (pv.omega + pv.alpha)/(1 - self.persistence(pv))

    def vol_of_var(self, h_tp1, pv=None):
        """Square root of the conditional variance of variance at time t.

        The variance of variance is close to machine precision; the vol of daily variance is likely more stable numerically.
        """
        if pv is None:
            pv = self.get_pv()
        return pv.alpha*np.sqrt(2 + 4 * pv.gamma**2 * h_tp1)

    def corr_ret_var(self, h_tp1, pv=None):
        """
        Calculates the correlation between returns and variance.
        """
        if pv is None:
            pv = self.get_pv()
        return -2*pv.gamma*np.sqrt(h_tp1) / np.sqrt(2 + 4 * pv.gamma**2 * h_tp1)
    
    def log_likelihood(self, log_xret, pv=None):
        """Calculates the log likelihood of given log excess returns using the AGARCH model.
        
        Args:
            log_xret (np.ndarray): An array of log excess returns.
            pv (dict, optional): A dictionary of model parameters. If not provided, 
                the current model parameters are used.
        
        Returns:
            float: The calculated log likelihood of the log excess returns. This 
                value is obtained from the C implementation.

        Updates:
            self.eps (np.ndarray): Updates the model's attribute `eps` with the computed 
                standardized residuals. This is a numpy array with the same length as `log_xret`.

            self.h (np.ndarray): Updates the model's attribute `h` with the
                computed conditional variances. This is a numpy array with one
                additional element than `log_xret`, representing the variance estimate
                at each step.
        """            
        n_days = len(log_xret)
        h = np.zeros((n_days + 1, 1), dtype=np.float32)
        eps = np.zeros((n_days, 1), dtype=np.float32)
        h_min = self.h_min

        if pv is None:
            pv = self.get_pv()
        lmbda,omega,alpha,beta,gamma = pv.lmbda,pv.omega,pv.alpha,pv.beta,pv.gamma
        

        h[0] = log_xret[:self.h0_window].var() 
        
        # Call the C function
        log_xret = np.ascontiguousarray(log_xret, dtype=np.float32)
        garch_c.agarch_filter(
            log_xret.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            eps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_days),
            ctypes.c_float(h_min),
            ctypes.c_float(lmbda),
            ctypes.c_float(omega),
            ctypes.c_float(alpha),
            ctypes.c_float(beta),
            ctypes.c_float(gamma))

        ll = -0.5 * (np.log(2 * np.pi) + np.log(h[:-1]) + eps**2).sum()

        self.eps = eps
        self.h = h
        return ll
    
    def constraints(self, **kwargs):
        """Adds 1-persistence > 0  to the inherited constraints."""
        constraints = super().constraints(**kwargs) 
        
        def one_minus_persistence(x):
            self._set_optimizer_pvalues(x)
            c = 1-self.persistence()
            if False:
                pv = self.get_pv()
                print(x)
                print(pv, c, '\n')
            return c - 1e-6 # strict inequality
        
        constraints.append({'type':'ineq', 'fun':one_minus_persistence})
        return constraints    

    #in garch now: #cf. lines 488-490 in parameters.py
    #in garch now: def finalize_optim(self, opt, observations, **kwargs):
    #in garch now:     opt = super().finalize_optim(opt, observations, **kwargs)
    #in garch now:     opt.LL = self.log_likelihood(observations, self.get_pv()) 
    #in garch now:     return opt
                    
    def summarize_state(self, log_xret, opt):
        from scipy.stats import skew,kurtosis
        pv = self.get_pv()
        innov,hvar = self.eps,self.h

        h_tp1 = hvar[-1,0]
        h_t = hvar[-2,0]
        erp = pv.lmbda * 252*h_tp1
        names = list(pv.keys()) + ['LL','nfev','erp','z_bar','z_std','z_skew','z_kurt','z_t','h_t','h_tp1']
        values = list(pv.values()) + [opt.LL, opt.nfev, erp, 
                    innov.mean(),innov.std(),skew(innov)[0],kurtosis(innov)[0],innov[-1,0], h_t, h_tp1]
        return pd.DataFrame([values], index=log_xret.index[[-1]], columns=names)
    
    def simulateQ(self, n_days, n_paths, h_tp1, z=None, pv=None):
        '''Simulate excess returns and their variance under the Q measure
        
        We consider that the simulation is starting at t0, and tp0 is a shorthand for "time
        t0+1" where p in tp1 stands for plus."

        This method simulates *excess* log-returns; the risk-free rate must be added outside
        this function to get the full log-return. This allows using different risk-free rates
        to price options at different horizons with the same core simulations.

        Args:
            n_days:  Length of the simulation
            n_paths: Number of paths in the simulation
            h_tp1:   Measurable at t0. Note: tp1 stands for t0 + 1 (+: (p)lus)

        Returns:
            ex_r:    Excess log-returns of the underlying (np.array: n_days x n_paths)
            h:       Corresponding variance (np.array: n_days+1 x n_paths)
            z:       The shocks used in the simulation

            Note that ex_r (resp. h) has n_days (+1) rows: the first row is at time t0+1. Hence
            h[0,:] = h_tp1 since all trajectories share the t0+1 variance predicted at t0. Row
            n_days+1 in h is the forecasted t0+T+1 variance at t0+T.
        '''
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days+1, n_paths), np.nan)

        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)
        if pv is None:
            pv = self.get_pv()
        lmbda,omega,alpha,beta,gamma = pv.lmbda,pv.omega,pv.alpha,pv.beta,pv.gamma

        gamma_star = gamma + lmbda
        
        # Because indices start at 0 in Python, tn=0 is t0+1, so the loop belows runs
        #   from:  t = t0+1 + 0
        #   to:    t = t0+1 + n_days-1
        for tn in range(n_days):
            # Simulate returns at t = t0+(tn+1)
            sqrt_ht = np.sqrt(h[tn,:])
            ex_r[tn,:] = -0.5*h[tn,:] + sqrt_ht*z[tn,:]
            
            # Update the variance paths
            h[tn+1,:] = omega + alpha*(z[tn,:] - gamma_star*sqrt_ht)**2 + beta*h[tn,:]
            
        return ex_r, h, z

class ngarch(garch):
    """Implements the NGARCH model by extending the GARCH model.
    
    Attributes:
        All attributes from the `garch` class are inherited.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the NGARCH model with specific parameters and sets up variance targeting if required.
        """
        super().__init__(*args, **kwargs)
        om_min = (1 - 0.999)*self.h_min
        self.add_parameter('lmbda', 0.03, [-1, 1])
        self.add_parameter('omega', 6e-7, [om_min, 1e-3]) # default max at roughly 50% ann vol.
        self.add_parameter('alpha', 0.06, [0, 0.50])
        self.add_parameter('beta',  0.91, [0.5, 0.99])
        self.add_parameter('gamma', 0.60, [-1.5, +1.5])
        
        if self.var_target_window is not None:
            self.fix_parameter('omega', np.nan)

    def variance_targeting(self, var_target, pv=None, fix_param=True):
        """Adjusts the omega parameter for variance targeting given a target variance."""
        omega = (1 - self.persistence(pv)) * var_target
        if fix_param:
            self.fix_parameter(omega=omega)
            return self.get_pv()
        return omega

    def equity_risk_premium(self, h_tp1, pv=None, periods_per_year=252):
        """Calculates the equity risk premium under the NGARCH model.
        
        The equity risk premium is annualized using `periods_per_year` (defaults to 252), and is in simple-return terms (i.e. exponential of the log-ERP minus 1).
        """
        if pv is None:
            pv = self.get_pv()

        # The assumption here is that the per-annum ERP is proportional to annualized vol, not the compounded value of daily ERPs. 
        #   return np.exp(pv.lmbda*np.sqrt(periods_per_year*h_tp1)) - 1
        # The value below is 15.87 times (sqrt(252)) greater than the the value actually returned above. It yields values that are, at times particularly high, but not inconsistent with (e.g.) the analysis in Martin (2017, QJE)
        return np.exp(periods_per_year * pv.lmbda*np.sqrt(h_tp1)) - 1

    def persistence(self, pv=None):
        """Calculates the persistence of volatility shocks in the NGARCH model."""
        if pv is None:
            pv = self.get_pv()
        return pv.alpha*(1 + pv.gamma**2) + pv.beta

    def uncond_var(self, pv=None):
        """Calculates the unconditional variance of the model."""
        if pv is None:
            pv = self.get_pv()
        return pv.omega/(1 - self.persistence(pv))

    def vol_of_var(self, h_tp1, pv=None):
        """Square root of the conditional variance of variance at time t.

        The variance of variance is close to machine precision; the vol of daily variance is likely more stable numerically.
        """
        if pv is None:
            pv = self.get_pv()
        return pv.alpha*np.sqrt(2+4*pv.gamma**2)*h_tp1

    def corr_ret_var(self, h_tp1, pv=None):
        """Calculates the correlation between returns and variance."""
        if pv is None:
            pv = self.get_pv()
        return -2*pv.gamma / np.sqrt(2 + 4*pv.gamma**2)

    def log_likelihood_py(self, log_xret, pv=None):
        """Computes the log likelihood of log excess returns (in Python).

        This function filters excess returns and their variance under the
        probability measure P, specifically designed for the NGARCH model. It is
        a pure Python implementation, which might be used for debugging,
        validation, or environments where C extensions are not available.

        The method computes the conditional variances (h) and the
        standardized residuals (eps) based on the NGARCH model equations. These
        calculations are required for the maximum likelihood estimation process,
        allowing for the optimization of the model's parameters.

        Args:
            log_xret (np.ndarray): The log excess returns for which the log
                likelihood is to be computed. This should be a one-dimensional numpy
                array.
            pv (Optional[PV]): The model parameters (omega, alpha, beta, gamma, lambda). If not provided, the 
                current parameters of the model are used. This allows for flexibility in evaluating the
                log likelihood for different sets of parameters.

        Returns:
            float: The log likelihood value for the given log excess returns under the NGARCH model parameters.

        Updates:
            self.eps (np.ndarray): Updates the model's attribute `eps` with the computed 
                standardized residuals. This is a numpy array with the same length as `log_xret`.

            self.h (np.ndarray): Updates the model's attribute `h` with the
                computed conditional variances. This is a numpy array with one
                additional element than `log_xret`, representing the variance estimate
                at each step.
        """
        n_days = len(log_xret)
        h = np.zeros((n_days+1,1), dtype=np.float32)
        eps = np.zeros((n_days,1), dtype=np.float32)
        h_min = self.h_min # Avoid the overhead of fetching attributes in the loop

        if pv is None:
            pv = self.get_pv()
        lmbda,omega,alpha,beta,gamma = pv.lmbda,pv.omega,pv.alpha,pv.beta,pv.gamma
        
        h[0] = log_xret[:self.h0_window].var() # log_xret[0]**2 # omega / (1-alpha*(1+gamma**2)-beta) # 
        for tn in range(n_days):
            eps[tn] = (log_xret[tn] + 0.5*h[tn])/np.sqrt(h[tn]) - lmbda
            h[tn+1] = omega + alpha*h[tn]*(eps[tn] - gamma)**2 + beta*h[tn]
            if h[tn+1] < h_min:
                h[tn+1] = h_min

        ll = -0.5*(np.log(2*np.pi) + np.log(h[:-1]) + eps**2).sum()

        self.eps = eps
        self.h = h
        return ll
    
    def log_likelihood_c(self, log_xret, pv=None):
        """Computes the log likelihood of log excess returns (C implementation).
        
        This method is analogous to `log_likelihood_py` but leverages a C implementation for performance improvements,
        particularly beneficial for large datasets or frequent calculations where execution speed is critical.
        
        The C implementation is expected to provide a significant speed-up by utilizing low-level optimizations and
        possibly parallel computation. This method should ideally interface with the compiled C code, passing necessary
        parameters and receiving the computed log likelihood.

        Args:
            log_xret (np.ndarray): An array of log excess returns.
            pv (dict, optional): A dictionary of model parameters. If not provided, the current model parameters are used.
        
        Returns:
            float: The calculated log likelihood of the log excess returns. This value is obtained from the C implementation.

        Note:
            Actual implementation of the call to the C function is not shown here and needs to be integrated based on the
            specific C library or extension being used. This often involves using ctypes or cffi to interface with C code from Python.
        """            
        n_days = len(log_xret)
        h = np.zeros((n_days + 1, 1), dtype=np.float32)
        eps = np.zeros((n_days, 1), dtype=np.float32)
        h_min = self.h_min

        if pv is None:
            pv = self.get_pv()
        lmbda,omega,alpha,beta,gamma = pv.lmbda,pv.omega,pv.alpha,pv.beta,pv.gamma

        h[0] = log_xret[:self.h0_window].var() 
        
        # Call the C function
        log_xret = np.ascontiguousarray(log_xret, dtype=np.float32)
        garch_c.ngarch_filter(
            log_xret.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            eps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_days),
            ctypes.c_float(h_min),
            ctypes.c_float(lmbda),
            ctypes.c_float(omega),
            ctypes.c_float(alpha),
            ctypes.c_float(beta),
            ctypes.c_float(gamma))

        #import pdb; pdb.set_trace()
        ll = -0.5 * (np.log(2 * np.pi) + np.log(h[:-1]) + eps**2).sum()

        self.eps = eps
        self.h = h
        return ll

    # The assignment below is conceptually equivalent to adding a
    #
    #     def log_likelihood_c(self, log_xret, pv=None):
    #         ...
    #
    # block where we'd copy-paste the code of log_likelihood_c. That is, we are defining a method log_likelihood, giving it the same signature as that of log_likelihood_c. This is convenient if you want to debug/compare with the full python version; you could (e.g.) comment this line and replace it with:
    #
    #     log_likelihood = log_likelihood_py
    #
    log_likelihood = log_likelihood_c
    
    def constraints(self, **kwargs):
        """Adds 1-persistence > 0  to the inherited constraints."""
        constraints = super().constraints(**kwargs) 
        
        def one_minus_persistence(x):
            self._set_optimizer_pvalues(x)
            c = 1-self.persistence()
            if False:
                pv = self.get_pv()
                print(x)
                print(pv, c, '\n')
            return c - 1e-6 # strict inequality
        constraints.append({'type':'ineq', 'fun':one_minus_persistence})
        return constraints    

    #in garch now: #cf. lines 488-490 in parameters.py
    #in garch now: def finalize_optim(self, opt, observations, **kwargs):
    #in garch now:     opt = super().finalize_optim(opt, observations, **kwargs)
    #in garch now:     opt.LL = self.log_likelihood(observations, self.get_pv()) 
    #in garch now:     return opt
                    
    def simulateP(self, n_days, n_paths, h_tp1, z=None, pv=None):
        '''Simulate excess returns and their variance under the P measure
        
        We consider that the simulation is starting at t0, and tp1 is a shorthand for "time
        t0+1" where p in tp1 stands for plus."

        This method simulates *excess* log-returns; the risk-free rate must be added outside
        this function to get the full log-return. This allows using different risk-free rates
        to price options at different horizons with the same core simulations.

        Args:
            n_days:  Length of the simulation
            n_paths: Number of paths in the simulation
            h_tp1:   Measurable at t0. Note that
            z:       The N(0,1) shocks for the simulation (optional) 

        Returns:
            ex_r:    Excess log-returns of the underlying (np.array: n_days x n_paths)
            h:       Corresponding variance (np.array: n_days+1 x n_paths)
            z:       The shocks used in the simulation

            Note that ex_r (resp. h) has n_days (+1) rows: the first row is at time t0+1. Hence
            h[0,:] = h_tp1 since all trajectories share the t0+1 variance predicted at t0. Row
            n_days+1 in h is the foreacasted t0+T+1 variance at t0+T.
        '''
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days+1, n_paths), np.nan)

        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)
        if pv is None:
            pv = self.get_pv()
        lmbda,omega,alpha,beta,gamma = pv.lmbda,pv.omega,pv.alpha,pv.beta,pv.gamma

        # Because indices start at 0 in Python, tn=0 is t0+1, so the loop belows runs
        #   from:  t = t0+1 + 0
        #   to:    t = t0+1 + n_days-1  
        for tn in range(n_days):
            # Simulate returns at t = t0+(tn+1)
            sig = np.sqrt(h[tn,:])
            ex_r[tn,:] = lmbda*sig - 0.5*h[tn,:] + sig*z[tn,:]
            
            # Update the variance paths
            h[tn+1,:] = omega + alpha*h[tn,:]*(z[tn,:] - gamma)**2 + beta*h[tn,:]

        return ex_r, h, z


    def simulateQ(self, n_days, n_paths, h_tp1, z=None, pv=None):
        '''Simulate excess returns and their variance under the Q measure
        
        We consider that the simulation is starting at t0, and tp1 is a shorthand for "time
        t0+1" where p in tp1 stands for plus."

        This method simulates *excess* log-returns; the risk-free rate must be added outside
        this function to get the full log-return. This allows using different risk-free rates
        to price options at different horizons with the same core simulations.

        Args:
            n_days:  Length of the simulation
            n_paths: Number of paths in the simulation
            h_tp1:   Measurable at t0. 

        Returns:
            ex_r:    Excess log-returns of the underlying (np.array: n_days x n_paths)
            h:       Corresponding variance (np.array: n_days+1 x n_paths)
            z:       The shocks used in the simulation

            Note that ex_r (resp. h) has n_days (+1) rows: the first row is at time t0+1. Hence
            h[0,:] = h_tp1 since all trajectories share the t0+1 variance predicted at t0. Row
            n_days+1 in h is the forecasted t0+T+1 variance at t0+T.
        '''
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days+1, n_paths), np.nan)

        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)
        assert z.shape == (n_days, n_paths)
        if pv is None:
            pv = self.get_pv()
        lmbda,omega,alpha,beta,gamma = pv.lmbda,pv.omega,pv.alpha,pv.beta,pv.gamma

        gamma_star = gamma + lmbda
        
        # Because indices start at 0 in Python, tn=0 is t0+1, so the loop belows runs
        #   from:  t = t0+1 + 0
        #   to:    t = t0+1 + n_days-1  
        for tn in range(n_days):
            # Simulate returns at t = t0+(tn+1)          
            ex_r[tn,:] = -0.5*h[tn,:] + np.sqrt(h[tn,:])*z[tn,:]
            
            # Update the variance paths
            h[tn+1,:] = omega + alpha*h[tn,:]*(z[tn,:] - gamma_star)**2 + beta*h[tn,:]
            
        return ex_r, h, z


    def simulate_garch_c(self, n_days, n_paths, h_tp1, z=None, pv=None):
        """Use C code rather than python to simulate the NGARCH model

        Interestingly, we do not seem to gain any performance whatsoever here, whereas the
        log_likelihood_c function is much faster than its Python counterpart.

        Most likely, this is due to the fact that the performance of the original Python code
        might already be close to optimal, making it harder for the C version to achieve
        noticeable improvements. The NumPy library, which is used in the Python version, is
        already highly optimized and leverages low-level implementations for many operations.

        log_likelihood_py did not use any of the vectorization in NumPy; simulateP does use it
        at each point in time. A more promising path here is to break down the simulation
        in smaller sub-simulations and make use of multiprocessing or GPUs.
        """
        ex_r = np.full((n_days, n_paths), np.nan, dtype=np.float32)
        h = np.full((n_days + 1, n_paths), np.nan, dtype=np.float32)

        h[0, :] = h_tp1
        if z is None:
            z = antithetic_normal(n_days, n_paths)
        if pv is None:
            pv = self.get_pv()
        lmbda, omega, alpha, beta, gamma = pv.lmbda, pv.omega, pv.alpha, pv.beta, pv.gamma

        # Ensure arrays are contiguous and in the C-style order
        h = np.ascontiguousarray(h, dtype=np.float32)
        ex_r = np.ascontiguousarray(ex_r, dtype=np.float32)
        z = np.ascontiguousarray(z, dtype=np.float32)

        #import pdb; pdb.set_trace()
        
        # Call the C function
        garch_c.ngarch_sim(
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ex_r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_days),
            ctypes.c_int(n_paths),
            ctypes.c_float(lmbda),
            ctypes.c_float(omega),
            ctypes.c_float(alpha),
            ctypes.c_float(beta),
            ctypes.c_float(gamma))

        return ex_r, h, z

    def simulateP_cl(self, n_days, n_paths, h_tp1, z=None, pv=None):
        """Failed attempt at using the GPU"""
        #print('simulate_garch_c: Still has a bug somewhere?')
        raise NotImplementedError('Still has a bug somewhere')
        import pyopencl as cl
        import pyopencl.array
        if z is None:
            z = antithetic_normal(n_days, n_paths)
        if pv is None:
            pv = self.get_pv()
        lmbda, omega, alpha, beta, gamma = pv.lmbda, pv.omega, pv.alpha, pv.beta, pv.gamma
    
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
    
        mf = cl.mem_flags
        z_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
    
        h = np.empty((n_days + 1, n_paths), dtype=np.float32)
        ex_r = np.empty((n_days, n_paths), dtype=np.float32)
    
        h[0, :] = h_tp1  # Initialize the first row of h with h_tp1
    
        kernel_code = """
        __kernel void simulateP_cl(
            __global const float *z,
            __global float *ex_r,
            __global float *h,
            int n_days,
            float lmbda,
            float omega,
            float alpha,
            float beta,
            float gamma)
        {
            int path = get_global_id(0);
            float h_t = h[path];
        
            for (int tn = 0; tn < n_days; tn++) {
                int index = tn * get_global_size(0) + path;
        
                float sig = sqrt(h_t);
                ex_r[index] = lmbda * sig - 0.5f * h_t + sig * z[index];
        
                h_t = omega + alpha * h_t * pow(z[index] - gamma, 2) + beta * h_t;
                
                if (tn < n_days - 1) {
                    h[index + get_global_size(0)] = h_t;
                }
            }
            h[path + get_global_size(0) * n_days] = h_t;
        }
        """
    
        prg = cl.Program(ctx, kernel_code).build()
        simulateP_cl = prg.simulateP_cl
    
        global_size = (n_paths,)
        local_size = None
    
        ex_r_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ex_r.nbytes)
        h_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h)
    
        simulateP_cl(queue, global_size, local_size,
                     z_buf, ex_r_buf, h_buf,
                     np.int32(n_days), np.float32(lmbda), np.float32(omega),
                     np.float32(alpha), np.float32(beta), np.float32(gamma))
    
        cl.enqueue_copy(queue, ex_r, ex_r_buf)
        cl.enqueue_copy(queue, h, h_buf)
    
        return ex_r, h, z
    #
    # simulateP = simulateP_cl # simulate_garch_c
    #
    # def simulateQ(self, n_days, n_paths, h_tp1, z=None):
    #     pv = self.get_pv()
    #     pv.gamma = pv.gamma + pv.lmbda
    #     pv.lmbda = 0.0
    #     return self.simulateP_cl(n_days, n_paths, h_tp1, z, pv)
    #     #return self.simulate_garch_c(n_days, n_paths, h_tp1, z, pv)

class ngarchC(garch):
    """
    Implements the NGARCH(C) model, a model in the spirit of the NGARCH but with 2 vol components. 

    Attributes:
        q0_window (int): Defines the initial window size for variance estimation, specifically for the q parameter.
        All other attributes from the garch class are inherited.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the NGARCH(C) model with specific parameters
        
        Args:
            *args: Variable length argument list passed to the garch superclass.
            **kwargs: Arbitrary keyword arguments. 'q0_window' is specific to ngarchC, defining
                      the initial window size for the q variance parameter estimation.
        """
        self.q0_window = kwargs.pop('q0_window', 252)
        super().__init__(*args, **kwargs)

        s_min = np.sqrt(252*self.h_min)
        self.add_parameter('lmbda',   0.03, [-1, 1])
        self.add_parameter('unc_vol', 0.20, [s_min, 0.5]) # 'unc_vol' parameter represents the unconditional volatility.

        self.add_parameter('alpha1', 0.04, [0,  0.5])
        self.add_parameter('beta1',  0.89, [0,  0.9999])
        self.add_parameter('gamma1', 1.60, [-5, +5])

        self.add_parameter('alpha2', 0.03, [0, 0.5])
        self.add_parameter('beta2',  0.99, [0.8, 0.9999])
        self.add_parameter('gamma2', 0.38, [-5, +5])

        if self.var_target_window is not None:
            self.fix_parameter('unc_vol', np.nan)
            
    def as_ngarch(self):
        """
        Adjusts parameters to match a nested NGARCH model, effectively simplifying the current model.

        This method fixes alpha2, beta2, and gamma2 to specific values to align with NGARCH behavior.
        """
        self.fix_parameter(alpha2=0.0, beta2=1.0, gamma2=0.0)

    def get_pv(self, *args):
        """
        Retrieves current parameter values, extending the base method to include the calculated sigma2.
        
        Returns:
            A struct with the current parameter values including the updated 'sigma2', which is derived from 'unc_vol'.
        """
        pv = super().get_pv(*args)
        pv.sigma2 = pv.unc_vol**2 / 252
        return pv
        
    def variance_targeting(self, var_target, pv=None, fix_param=True):
        """
        Applies variance targeting to adjust the `unc_vol` parameter to achieve the specified target variance.
        
        Args:
            var_target: The target variance value.
            pv: Optional parameter set to consider. Defaults to using the current model parameters.
            fix_param: If True, 'unc_vol' is adjusted to meet the variance target within the model parameters.
        
        Returns:
            The updated parameter set if fix_param is True, or the target variance value otherwise.
        """
        if fix_param:
            self.fix_parameter(unc_vol=np.sqrt(252*var_target))
            return self.get_pv()
        return var_target

    def persistence(self, pv=None):
        """
        Calculates the persistence of volatility in the model, combining both short-term and long-term effects.
        
        Args:
            pv: Optional parameter set to use for the calculation. Defaults to the current model parameters.
        
        Returns:
            The calculated persistence value.
        """
        if pv is None:
            pv = self.get_pv()
        return pv.beta2 + (1 - pv.beta2)*pv.beta1

    def uncond_var(self, pv=None):
        """
        Returns the unconditional variance of the model, based on sigma2.
        
        Args:
            pv: Optional parameter set to use. Defaults to the current model parameters.
        
        Returns:
            The unconditional variance.
        """
        if pv is None:
            pv = self.get_pv()
        return pv.sigma2

    def vol_of_var(self, h_tp1, pv=None):
        """
        Calculates the square root of the conditional variance of variance given information at time t.
        
        Args:
            h_tp1: The variance at time t+1.
            pv: Optional parameter set to use. Defaults to the current model parameters.
        
        Returns:
            The square root of the conditional variance of variance.
        """
        if pv is None:
            pv = self.get_pv()

        a = 2*(pv.alpha1 + pv.alpha2)**2
        c = 4*(pv.alpha1*pv.gamma1 + pv.alpha2*pv.gamma2)**2
        return np.sqrt(a + c)*h_tp1
    
    def corr_ret_var(self, h_tp1, pv=None):
        """
        Calculates the correlation between returns and variance.
        
        Args:
            h_tp1: The variance at time t+1.
            pv: Optional parameter set to use. Defaults to the current model parameters.
        
        Returns:
            The correlation between returns and variance.
        """
        if pv is None:
            pv = self.get_pv()

        a = 2*(pv.alpha1 + pv.alpha2)**2
        c = 4*(pv.alpha1*pv.gamma1 + pv.alpha2*pv.gamma2)**2            
        return -np.sqrt(c / (a+c))

    def log_likelihood(self, log_xret, pv=None):
        """
        Calculates the log likelihood for the given log excess returns, incorporating the complexity of the NGARCH model.

        This function computes the conditional variances (h) and the intermediate variance (q) based on the model's parameters.
        It is crucial for fitting the model to historical data by maximizing this likelihood.

        Args:
            log_xret (np.ndarray): Log excess returns for which to calculate the log likelihood.
            pv (Optional[dict]): Optional parameter set to consider. Defaults to using the current model parameters.
        
        Returns:
            float: The log likelihood of the given log excess returns under the model.
        
        Note:
            This implementation considers both the direct impact of shocks on volatility (h) and an intermediate term (q)
            that captures other aspects of volatility dynamics, enhancing the model's ability to fit and forecast complex behaviors.
        """
        n_days = len(log_xret)
        h = np.zeros((n_days+1,1))
        q = np.zeros((n_days+1,1))
        eps = np.zeros((n_days,1))
        h_min = self.h_min  # Avoid the overhead of fetching attributes in the loop

        if pv is None:
            pv = self.get_pv()
        lmbda, sigma2, alpha1,beta1,gamma1, alpha2,beta2,gamma2 \
            = pv.lmbda, pv.sigma2, pv.alpha1,pv.beta1,pv.gamma1, pv.alpha2,pv.beta2,pv.gamma2
        gamma = np.array([gamma1, gamma2])
        
        h[0] = log_xret[:self.h0_window].var() # log_xret[0]**2 # omega / (1-alpha*(1+gamma**2)-beta) #
        if alpha2==0 and beta2==1.0:
            q[0] = sigma2 # nested NGARCH
        else:
            q[0] = log_xret[:self.q0_window].var()
            
        for tn in range(n_days):
            eps[tn] = (log_xret[tn] + 0.5*h[tn])/np.sqrt(h[tn]) - lmbda
            nu = eps[tn]**2 - 1 - 2*gamma*eps[tn]

            q[tn+1] = sigma2 + alpha2*h[tn]*nu[1] + beta2*(q[tn] - sigma2)
            h[tn+1] = q[tn+1] + alpha1*h[tn]*nu[0] + beta1*(h[tn] - q[tn])
            if h[tn+1] < h_min:
                h[tn+1] = h_min
            if q[tn+1] < h_min:
                q[tn+1] = h_min

        ll = -0.5*(np.log(2*np.pi) + np.log(h[:-1]) + eps**2).sum()
        
        self.eps = eps
        self.h = h
        self.q = q
        return ll
        
    def summarize_state(self, log_xret, opt):
        """
        Extends the base method to include the q variance parameter in the state summary.

        Args:
            log_xret (np.ndarray): Log excess returns to consider for the state summary.
            opt: Optimization result object containing model fitting information.
        
        Returns:
            DataFrame: A DataFrame summarizing the model state, extended with the 'q_tp1' parameter, representing the last value of the q variance term.
        """
        qvar = self.q
        state = ngarch.summarize_state(self, log_xret, opt) # Building on the ngarch method
        state['q_tp1'] = qvar[-1,0]
        return state        

    def constraints(self, **kwargs):
        """Adds additional constraints for the NGARCH model parameters, ensuring the model's stability and consistency.

        Returns:
            list: A list of constraint dictionaries to be used in the optimization problem.
        
        Note:            
            This method enforces specific conditions on the model parameters to
        ensure realistic and stable model behavior. For instance, it ensures the
        persistence parameters are correctly ordered and optionally enforces
        constraints on the variance contributions of different components.
        """
        constraints = super().constraints(**kwargs) 

        pv = self.get_pv()

        ## The variance arising from h should be greater than that from q;
        ## enforcing it might not be the best idea though.
        #constraints.append({'type':'ineq', 'fun':lambda x:
        #                    pv.alpha1**2 * (2+4*pv.gamma1**2)  -  pv.alpha2**2 * (2+4*pv.gamma2**2)})

        # q's persistence is larger by definition
        constraints.append({'type':'ineq', 'fun':lambda x: pv.beta2 - pv.beta1})

        ## Long-term leverage effect should be less than short-term;
        ## enforcing it might not be the best idea though.
        #constraints.append({'type':'ineq', 'fun':lambda x:
        #                    np.abs(pv.alpha1*pv.gamma1) - np.abs(pv.alpha2*pv.gamma2)})        
        
        return constraints    
    
def plot_parameter_series(model, estimates):
    fig, axes = plt.subplots(4, 2, figsize=(22.5,18)) #(14,13)
    
    log_xreturns = model.past_log_excess_returns(model.get_calendar()[-1])
    vol = np.sqrt(252*log_xreturns.values.var())
    
    estimates['persistence'] = model.persistence(pv=estimates)
    estimates['uncond_vol'] = np.sqrt(252*model.uncond_var(pv=estimates))
    estimates['rel_vol_of_var'] = model.vol_of_var(estimates.h_tp1, pv=estimates)/estimates.h_tp1
    estimates['corr_r_h'] = model.corr_ret_var(estimates.h_tp1, pv=estimates)
    
    no = 0
    ax = axes[no,0]
    ax.plot(100*estimates.erp, color='k', linestyle=':', label='ERP (%)')
    ax.plot(100*(estimates.erp + estimates.z_bar*np.sqrt(252*estimates.h_tp1)), label='+ z_bar*sigma')
    ax.legend(loc='upper center')
    
    if 'nfev' in estimates.columns:
        ax = axes[no,1]
        ax.plot(estimates.nfev, label='# fun eval')
        ax.legend(loc='upper right')
    
    no += 1
    ax = axes[no,0]
    ax.plot(100*np.sqrt(252*estimates.h_tp1), label='Vol_{t+1}')
    ax.set_xticks([])
    ax.set_ylim([0,110])
    ax.legend()
    
    ax = axes[no,1]
    ax.plot(100*estimates.rel_vol_of_var, label='VoVar_{t+1} / Var_{t+1} (%)')
    ax.set_xticks([])
    #ax.set_ylim([0,100])
    ax.legend()
    
    no += 1
    ax = axes[no,0]
    ax.plot(100*estimates.uncond_vol, label='Uncond. Vol')
    ax.axhline(100*vol, color='k', linestyle=':', label='Sample Vol')
    ax.set_xticks([])
    ax.set_ylim([0,50])
    ax.legend()
    
    ax = axes[no,1]
    if 'alpha' in estimates.columns:
        ax.plot(estimates.alpha, label='alpha')
    else:
        ax.plot(estimates.alpha1, label='alpha1')
        ax.plot(estimates.alpha2, label='alpha2')
    ax.set_xticks([])
    #ax.set_ylim([0,100])
    ax.legend()
    
    no += 1
    ax = axes[no,0]
    #ax.plot(estimates.persistence, label="persistence\nRHS ':' beta")
    ax.plot(estimates.persistence, label="persistence")
    ax.set_ylim([0.95,1.001])
    ax.legend(loc='lower left')
    
    if 'beta' in estimates.columns:
        axx = ax.twinx()
        axx.plot(estimates.beta, linestyle=':', label='beta')
        axx.legend(loc='lower right')
    elif np.any(estimates.beta2 > 0):
        axx = ax.twinx()
        axx.plot(estimates.beta1, linestyle=':', label='beta1')
        axx.plot(estimates.beta2, linestyle=':', label='beta2')
        axx.legend(loc='lower right')
    
    ax = axes[no,1]
    ax.plot(estimates.corr_r_h, label="Corr(r,h)")
    ax.legend(loc='center left');
    
    axx = ax.twinx()
    if 'gamma' in estimates.columns:
        axx.plot(estimates.gamma, linestyle=':', label='gamma')
    else:
        axx.plot(estimates.gamma1, linestyle=':', label='gamma1')
        if np.any(estimates.gamma2 > 0):
            axx.plot(estimates.gamma2, linestyle=':', label='gamma2')
    axx.legend(loc='center right')

    return fig

class TestGarch(unittest.TestCase):
    update = False
    debug = False

    # For tests requiring simulations
    seed = 20240418 
    n_paths = 10_000 # Python 3.6 (and PEP-515) allows underscores in numeric literals
    
    #def setUp(self):
    #    pass

    def path_to_results(self, *args):
        return os.path.join('data','test','garch',*args)
        
    def initialize_test(self, update):        
        # Use the calling function name as the test name
        import inspect
        test_name = inspect.stack()[1][3]
        if update is None:
            update = self.update

        if update:
            logger.setLevel('INFO')
        else:
            logger.setLevel('WARNING')

        expected_resdir = self.path_to_results()
        path_to_expected_results = os.path.join(expected_resdir, '%s.pkl'%test_name)

        if update:
            if not os.path.exists(expected_resdir):
                os.makedirs(expected_resdir)
            self.resdir = expected_resdir # having a resdir is specific to TestGarch
        else:
            assert os.path.exists(path_to_expected_results), \
                "Generate expected results before running test %s"%test_name
            
            self.resdir = os.path.join(expected_resdir,'testing')
            if not os.path.exists(self.resdir):
                os.makedirs(self.resdir)

        # having a resfile is specific to TestGarch
        self.resfile = os.path.join(self.resdir, '%s.pkl'%test_name)
        if os.path.exists(self.resfile) and not self.debug:
            raise RuntimeError("The resfile %s already exists; remove it before running test"%self.resfile)

        np.random.seed(self.seed)
        return path_to_expected_results, update
    
    def finalize_test(self, path_to_expected_results, results, update):
        if update: 
            # The dump is useless in the case of tests based on sequentially, but other tests might need it
            with open(path_to_expected_results, 'wb') as fh: 
                pickle.dump(results, fh) 
            return
        # update is False; we are testing

        # Compare results with the expected results
        with open(path_to_expected_results, 'rb') as fh: 
            expected_results = pickle.load(fh)

        self.assertTrue( np.all(np.isclose(expected_results, results)) )

        # This point is reached only if the test passes. Otherwise, the resdir is not removed for potential debugging
        import shutil
        shutil.rmtree(self.resdir) # Remove 'running' test results

    def test_agarch(self, update=None):
        path_to_expected_results, update = self.initialize_test(update)

        model = agarch.on_fama_french_mkt()
        model.fix_parameter('omega', 0.0)
        method = 'Nelder-Mead'
        scaled = True
        last_date = pd.Timestamp('2024-04-30')
        estimates = model.estimate_sequentially(method, scaled, self.resfile, last_date)

        self.finalize_test(path_to_expected_results, estimates, update)

    def test_ngarch(self, update=None):
        path_to_expected_results, update = self.initialize_test(update)

        model = ngarch.on_fama_french_mkt()
        method = 'Nelder-Mead'
        scaled = True
        last_date = pd.Timestamp('2024-04-30')
        estimates = model.estimate_sequentially(method, scaled, self.resfile, last_date)

        self.finalize_test(path_to_expected_results, estimates, update)

def list_test_classes(configs):
    TestGarch.update = configs.update
    return [TestGarch]