import inspect
import numpy as np
import types
from scipy.stats import norm

#from .jupyter_notebook import *
from .toolkit import struct
from . import black_merton_scholes as bms

class hypotheses(types.SimpleNamespace):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __call__(self, func, arglist, **kwargs):
        defaults = self.__dict__
        hyp_args = {k:defaults[k] for k in arglist if k in defaults}
        hyp_args.update(**kwargs)
        return func(**hyp_args)
    
    def __str__(self):
        string = self.__class__.__name__ + '(\n'
        for name,value in self.variables():
            string += "    %s = %r,\n" % (name, value)
        string = string[:-2]+'\n)'
        return string

    def __repr__(self):
        return str(self)

    def is_variable(self, name):
        is_public = lambda name: not (name.startswith('__') and name.endswith('__')) \
                                    and not name.startswith('_'+self.__class__.__name__)
        is_field = lambda name: hasattr(self, name) and not inspect.ismethod(getattr(self, name))
        return is_public(name) and is_field(name)
    
    def variables(self):
        fields = self.__dict__
        for no,name in enumerate(fields):
            if self.is_variable(name):
                yield (name,fields[name])
        
    def to_dataframe(self, columns=[], exclude=[]):
        """Convert to a DF by repeating scalars on each row
        
        columns: if provided, use this subset of the variables
        exclude: if provided, do not use this subset of the variables
        
        columns and exclude cannot be used jointly
        """
        assert len(columns)==0 or len(exclude)==0
        if len(columns) > 0:
            return pd.DataFrame.from_dict({
                name:value for name,value in self.variables() if name in columns
            })            
        return pd.DataFrame.from_dict({
            name:value for name,value in self.variables() if name not in exclude
        })

class under_bms(hypotheses):
    def delta(self,**kwargs):
        return self(bms.delta, ('S','K','r','y','T','sigma','is_call'),**kwargs)
    
    def option_price(self,**kwargs):
        return self(bms.option_price, ('S','K','r','y','T','sigma','is_call'),**kwargs)

    def implied_volatility(self,**kwargs):
        return self(bms.implied_volatility, ('opt_price','S','K','r','y','T','is_call'),**kwargs)
    
    def iv_on_surface(self, K, T):
        if not hasattr(self,'IV'):
            return self.sigma
        ix = (self.K==K) & (self.T==T)
        return self.IV[ix]

# FUTURE: given the new hypotheses class, theses a fair amount of redundancy in
# _moneyness. Maybe it should be a subclass of hypotheses, maybe it should wrap an hypotheses instance...
# Maybe redundancies are not such an issue.
class _moneyness:
    class __moneyness(struct):
        def __init__(self, mny_str, **kwargs):
            super().__init__(**kwargs)
            self.mny_str = mny_str
            
        def range(self, S, lb, ub, step):
            '''Returns an np.arange(lb,ub,step) of moneyness levels and corresponding strikes'''
            mny = np.arange(lb, ub, step)
            K = self.get_strike(mny, S)
            return mny, K

        def delta_to_strike(self, delta, S, is_call):
            sigma = self.sigma
            T = self.T
            ln_S_K = norm.ppf(delta)*sigma*np.sqrt(T) - (self.r-self.y+0.5*sigma**2)*T
            return S*np.exp(-ln_S_K)
        
    class K_over_S(__moneyness):
        def __call__(self, S, K, r=None, y=None, T=None, sigma=None):
            return K / S
    
        def get_strike(self, mny, S):
            return mny*S
    
    class K_over_F(__moneyness):
        def forward_price(self, S, r=None, y=None, T=None, *args, **kwargs):
            '''Allow for neglected arguments; for necessary ones, use instance's values when None'''
            if r is None:
                r = self.r
            if y is None:
                y = self.y
            if T is None:
                T = self.T
            return S*np.exp((r-y)*T)        
        
        def __call__(self, S, K, *args, **kwargs):
            return K / self.forward_price(S, *args, **kwargs)
    
        def get_strike(self, mny, S):
            return mny*self.forward_price(S)    

    # Class variable will be shared by all instances of _moneyness
    instance = None
    def __init__(self, mny, **kwargs):
        '''Must provide the sufficient variables such that moneyness(S,K) returns the corresponding moneyness.'''
        if mny=='K/S':
            _moneyness.instance = _moneyness.K_over_S(mny, **kwargs)
        elif mny=='K/F':
            _moneyness.instance = _moneyness.K_over_F(mny, **kwargs)
        else:
            raise ValueError(mny)

    def __getattr__(self, name):
        return getattr(self.instance, name)        

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)
    
    def __repr__(self):
        return repr(self.instance)        

    def __str__(self):
        return str(self.instance)        

# Given the voodoo above, moneyness is a handle to the sole _moneyness.instance of __moneyness 
moneyness = _moneyness('K/S') # Default moneyness requires no additional info
def define_moneyness(mny, **kwargs):
    '''Set the MNY definition to be used across figures & tables

    Currently supported MNY: 'K/S' or 'K/F'.
    
    **kwargs ust provide the sufficient variables such that moneyness(S,K) returns the
    corresponding moneyness.
      - K/S: nothing needed
      - K/F: r=risk_free_rate, y=dividend_yield, T=maturity,
    '''
    return _moneyness(mny, **kwargs)
    
