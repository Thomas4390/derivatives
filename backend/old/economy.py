import sys
import types
import numpy as np
import pandas as pd

risk_free: float = 0.02
dividend_yield: float = 0.00
volatility: float = 0.25
equity_risk_premium: float = 0.06

time_units: str = 'year' # or 'day'
days_in_year: int = 252 # Ideally, avoid using outside of this module.
#                           see years_to_maturity below and usage in various packages

def describe():
    econ = sys.modules[__name__]    
    variables = econ.__dict__    
    desc = ',\n    '.join([f"{key}={value}"
                for key,value in variables.items()
                    if not key.startswith('_') and not key.startswith('__')
                           and not callable(value) and not isinstance(value,types.ModuleType)])
    print('economy(\n    ' + desc + ')')

def years_to_maturity(ttm):
    if time_units=='year':
        return ttm
    elif time_units=='day':
        return ttm/days_in_year
    raise ValueError('unexpected value for time_units: '+time_units)

def repr_maturity(ttm):
    if time_units=='year':
        return repr_years(ttm)
    elif time_units=='day':
        return ttm
    raise ValueError('unexpected value for time_units: '+time_units)

def repr_years(ytm):
    if ytm == np.round(ytm):
        return repr(ytm)
    
    mtm = ytm * 12
    if np.abs(mtm - np.round(mtm)) < 0.0001:
        return '%d/12'%mtm

    dtm = ytm * days_in_year
    if dtm == np.round(dtm):
        return '%d/%d'%(dtm,days_in_year)
    
    return ytm


# The state of the economy may be set based on data (past & present) or by a model (future,
# with closed-form forecasts or a simulation engine). Conceptually, in backtesting, the economy
# module should provide the user interface allowing high-level applications to avoid direct
# reliance on the structure of datasets and/or inner workings of the models.
__state = None
def reset_state():
    global __state
    __state = None
    
def get_state():
    """Returns a Series with index [time_t]+the keywords used in set_state"""
    return __state.iloc[-1,:]

def get_history():
    """Returns a DataFrame with columns [time_t]+the keywords used in set_state"""
    return __state

def set_state(time_t, **kwargs):
    """Allows a very flexible definition of the state, requiring only time_t as a mandatory field 

    On the first call to set_state, the keyword arguements will determine the other elements defining the state 
    of the economy. Subsequent calls with have to use the same keyword arguments, and time_t has to be increasing.
    """
    global __state
    columns = ['time_t']+list(kwargs.keys())
    values  = [ time_t ]+list(kwargs.values())

    row = pd.DataFrame(np.array([values]),columns=columns)
    row.index = pd.Index([repr_maturity(time_t)])
    if __state is None:
        __state = row

    else:
        assert time_t > __state['time_t'].values[-1], 'Time must be increasing: %s <= %s = state.time_t'%(time_t,__state.time_t[-1])
        assert np.all(columns == __state.columns), 'State dimensions inconsistent through time:\n%s\n%s'%(__state,columns)
        __state = pd.concat((__state, row))
    return row

def update_state(time_t, **kwargs):
    """For convenience, sets the state when only some of the its dimensions are updated."""
    current_state = get_state().drop('time_t').copy()
    for key,value in kwargs.items():
        current_state[key] = value
    set_state(time_t, **dict(zip(current_state.index.values, current_state.values)))

if False:
    from economy import reset_state, get_state, get_history, set_state, update_state
    
    reset_state()
    set_state(0, S=100, v=0.25**2)
    set_state(1, S=110, v=0.15**2)
    update_state(2, v=0.1**2)
    get_history()    
    
# THIS VOODOO WAS GETTING WAY TOO INVOLVED AND DANGEROUS FOR THE ADDED BENEFITS
# ### KEEP AT THE END OF FILE ####################################################
# 
# # def __getattr__(name):
# #     # import pdb; pdb.set_trace()
# #     econ = sys.modules[__name__]
# #     variables = econ.__dict__
# #     if name in variables:
# #         return variables[name]
# #     elif '_'+name in variables:
# #         return variables['_'+name]
# #     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# ## __setattr__ unfortunately cannot be overload that straightforwardly on modules... Hence the hack below
# 
# #DEV: class __read_only_module(object):
# #DEV:     def __init__(self, glb):
# #DEV:         vars(self).update(glb)
# #DEV: 
# #DEV:     def __setattr__(self, name, value):
# #DEV:         raise RuntimeError(f"module is read-only")
# 
# class __economy_module(object):
#     def __init__(self, glb):
#         vars(self).update(glb)
# 
#     def __getattr__(self, name):
#         attributes = vars(self)
#         if name in attributes:
#             return attributes[name]
#         if ('_'+name) in attributes:
#             return attributes['_'+name]
#         raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
#         
#     def __setattr__(self, name, value):
#         attributes = vars(self)
#         if name in attributes:
#             attributes[name] = value
#         else:
#             raise AttributeError(f"module {__name__!r} has no attribute {name!r}; cannot set new attribute")
# sys.modules[__name__] = __economy_module(sys.modules[__name__].__dict__)

    
