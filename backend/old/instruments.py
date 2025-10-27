import numpy as np
import os
import warnings

from . import economy
from .plot_utils import (mpl, plt, mtick, mdates, figsize, gridspec,
                         set_plt_defaults, set_payoff_axes, set_time_axis, with_style)

# This is a preliminary draft of the model/hypotheses interaction. It will most likely change
_model = None
def set_model(hypotheses):
    global _model
    assert type(hypotheses).__name__ == 'under_bms', 'DEV: only one supported model thus far'    
    _model = hypotheses
    return _model

class Instrument:
    # For convenience, if we want to create several instruments with the same maturity.
    # Use with parsimony
    default_maturity: float = 1.0 # In years; change if using it working with days or other periods 
    
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.bid_ask = None

        self._time_t = -1

    def __neg__(self):
        return Position(-1.0, self)

    def __mul__(self, n_units):
        return Position(n_units, self)
    __rmul__ = __mul__

    def __truediv__(self, n_units):
        return Position(1/n_units, self)
    __rtruediv__ = __truediv__
    
    def __add__(self, other):
        pos = Position(1.0, self)
        return pos + other

    def __sub__(self, other):
        pos = Position(1.0, self)
        return pos - other    

    def get_bid_ask(self, price):
        if self.bid_ask is None:
            return price
        return price + np.array([-self.bid_ask/2, +self.bid_ask/2])
    
    def set_bid_ask_spread(self, spread):
        self.bid_ask = spread
        
    def set_maturity(self, maturity):
        if maturity is None:
            self.maturity = Instrument.default_maturity
        else:
            self.maturity = maturity

    def repr_maturity(self):
        return economy.repr_maturity(self.maturity)
            
    def ytm(self, time_t=0):
        return economy.years_to_maturity(self.maturity - time_t)
            
    def update_history(self, S_0_t):
        """Update history of (path-dependent) instruments.

        The history should be the (single) **realized** trajectory of the stock starting from 0.

        Instruments without path-dependence simply neglect the info via the default implementation here.

        S_0_t must be an (t+1)x1 np.array where t is the number of days since inception.              
              Hence, the first row should be the value of the stock at time 0.
        """
        assert len(S_0_t.shape)==1 or S_0_t.shape[1]==1, "S_0_t should be a single realized path"
        
        #CD: It isn't clear to me yet if such a strong statement has any value. It forces users to instantiate a new instrument and they want to compare (e.g.) the payoffs for two different versions of the history...
        #CD, several months later: I think it's a good idea to continue forcing the user to instantiate a new instrument; several features would have to be carefully reviewed otherwise.
        assert S_0_t.shape[0] > self._time_t, "S_0_t should grow with time"
        self._time_t = S_0_t.shape[0]-1 # The first row is time 0

    def payoff(self, S_t_T):
        """Returns the payoff of the instrument given the trajectories S_t_T
        
        Internally, each subclass defines a _payoff method assuming that the array is 2-dimensional, with
        time t+n on row n, and the $j^{th}$ simulated value $S_{t+n,j}$ in column j. The notation S_t_T 
        highlights that this matrix of payoffs is considered to be subsequent to any history provided to 
        self.update_history(S_0_t).

        For convenience, if the S_t_T array is a 1-dimensional array, it is assumed to be $S_T$.
        """
        #import pdb; pdb.set_trace()
        if np.isscalar(S_t_T):
            S_t_T = np.array([S_t_T])
            
        if len(S_t_T.shape)==1: # 1D array S_T
            return self._payoff( S_t_T.reshape(1,-1) ).reshape(S_t_T.shape)
        return self._payoff(S_t_T).reshape(S_t_T[-1,:].shape)
    
    def plot_payoff(self, ax, S_t_T):
        """Plot the instrument's payoff

        S_t_T must be an (T-t+1)xN np.array where T is the number of days to maturity at
              inception, t is the current day and N the number of simulated paths. Hence, the
              first row should be the current value of the stock, copied N times.
        """
        if len(S_t_T.shape)==1: # 1D array S_T
            S_t_T = S_t_T.reshape(1,-1)
        return Portfolio(self).plot_payoff(ax, S_t_T)

    
class Bond(Instrument):
    def __init__(self, maturity=None, **kwargs):
        super().__init__(**kwargs)
        self.set_maturity(maturity)
        
    def __repr__(self):
        return 'Bond(%s)'%self.repr_maturity()
    
    def _payoff(self, S_t_T=None): 
        # Accept S_t_T for compatibility with the Derivatives' payoff signature
        if S_t_T is None:
            return 1.0
        return np.ones(S_t_T[-1,:].shape)            

    def evaluate(self, time_t, S_t=np.nan):
        """Accepts S_t for compatibility with Derivatives.evaluate signature"""
        r_f = economy.risk_free
        tau = self.ytm(time_t)
        return self.get_bid_ask(np.exp(-r_f * tau))

class Derivatives(Instrument):
    def __init__(self, S_0, maturity=None, **kwargs):
        super().__init__(**kwargs)
        self.S_0 = S_0 # Depends on the initial value of the spot 
        self.set_maturity(maturity)
        
class Spot(Derivatives):
    def __repr__(self):
        return 'Spot(%s)'%(self.S_0)
    
    def _payoff(self, S_t_T): 
        return S_t_T[-1,:]

    def evaluate(self, time_t, S_t):
        return self.get_bid_ask(S_t)
    
class Forward(Derivatives):
    def __init__(self, S_0, maturity=None, **kwargs):
        super().__init__(S_0, maturity, **kwargs)
        r_f = economy.risk_free
        div = economy.dividend_yield
        self.strike = self.S_0*np.exp((r_f-div) * self.ytm(0))
        
    def __repr__(self):
        return 'Forward(%s,%s)'%(self.S_0,self.repr_maturity())
    
    def _payoff(self, S_t_T): 
        return S_t_T[-1,:] - self.strike 

    def evaluate(self, time_t, S_t):
        r_f = economy.risk_free
        div = economy.dividend_yield
        F_t_T = S_t*np.exp((r_f-div) * self.ytm(time_t))
        return self.get_bid_ask(F_t_T - self.strike)
    
class Call(Derivatives):
    def __init__(self, S_0, strike, maturity=None, **kwargs):
        super().__init__(S_0, maturity, **kwargs)
        self.strike = strike
    
    def __repr__(self):
        return 'Call(%s,%s,%s)'%(self.S_0, self.strike, self.repr_maturity())

    def lower_bound(self, time_t, S_t):
        B = Bond(self.maturity)
        KB = self.strike * B.evaluate(time_t)
        return np.maximum(S_t - KB, 0.0)        
    
    def _payoff(self, S_t_T):
        return np.maximum(S_t_T[-1,:] - self.strike, 0.0) # DO NOT USE np.max    

    def evaluate(self, time_t, S_t):
        #import pdb; pdb.set_trace()        
        tau = self.ytm(time_t)
        if tau==0:
            return self.payoff(S_t) # S_t is S_T
        
        sig = _model.iv_on_surface(self.strike, tau)
        return self.get_bid_ask(
            _model.option_price(S=S_t, K=self.strike, T=tau, sigma=sig, is_call=True))

class Put(Derivatives):
    def __init__(self, S_0, strike, maturity=None, **kwargs):
        super().__init__(S_0, maturity, **kwargs)
        self.strike = strike
        
    def __repr__(self):
        clsnm = self.__class__.__name__
        return '%s(%s,%s,%s)'%(clsnm, self.S_0, self.strike, self.repr_maturity())    

    def lower_bound(self, time_t, S_t):
        B = Bond(self.maturity)
        KB = self.strike * B.evaluate(time_t)
        return np.maximum(KB - S_t, 0.0)        
    
    def _payoff(self, S_t_T):
        return np.maximum(self.strike - S_t_T[-1,:], 0.0) # DO NOT USE np.max

    def evaluate(self, time_t, S_t):
        tau = self.ytm(time_t)
        if tau==0:
            return self.payoff(S_t) # S_t is S_T
        
        sig = _model.iv_on_surface(self.strike, tau)
        return self.get_bid_ask(
            _model.option_price(S=S_t, K=self.strike, T=tau, sigma=sig, is_call=False))

class AmericanPut(Put):
    def evaluate(self, time_t, S_t, cache=None):
        tau = self.ytm(time_t)
        if tau==0:
            return self.payoff(S_t) # S_t is S_T
        
        # sig = _model.iv_on_surface(self.strike, tau)
        K, r, y, sig = self.strike, _model.r, _model.y, _model.sigma

        import pickle
        parameters = (('time_t', time_t), ('S_t', tuple(np.round(S_t,4))),
                          ('K', K), ('r', r), ('tau', tau), ('sig', sig) )
        if cache is not None and os.path.exists(cache):
            with open(cache,'rb') as fh:
                values = pickle.load(fh)
                
            if parameters in values:
                return self.get_bid_ask(values[parameters])

        # If we reached this point, either the cache does not exist or does not match the current parameters
        P = np.nan*S_t
        from . import binomial_tree as bin_tree
        assert type(_model).__name__ == "under_bms", \
            "The American put can be valued with the binomial tree only under the BMS hypotheses"
        for sn,S in enumerate(S_t):
            _,_,amer,_,_ = bin_tree.option_price(S, K, r, y, tau, sig, 500, False, True)
            P[sn] = amer[0,0]

        if cache is not None:
            data = os.path.dirname(cache)
            os.makedirs(data, exist_ok=True)
            values = {parameters : P}
            with open(cache,'wb') as fh:
                pickle.dump(values, fh)
            
        return self.get_bid_ask(P)
        
class DownAndOutPut(Put):
    def __init__(self, S_0, strike, barrier, maturity=None, **kwargs):
        super().__init__(S_0, strike, maturity, **kwargs)
        self.barrier = barrier
        self.alive_at_t = 1.0 # Based on history
        self.alive_on_trajectory = None # Based on argument to payoff
        
    def __repr__(self):
        return 'DownAndOutPut(%s,%s,%s,%s)'%(self.S_0, self.strike, self.barrier, self.repr_maturity())    

    def update_history(self, S_0_t):
        """Validate that the barrier was not breached.

        The history should be the (single) **realized** trajectory of the stock starting from 0.

        S_0_t must be an (t+1)x1 np.array where t is the number of days since inception.              
              Hence, the first row should be the value of the stock at time 0.
        """
        super().update_history(S_0_t)
        S_min = S_0_t.min(axis=0)
        self.alive_at_t = 1.0 if S_min > self.barrier else 0.0
    
    def _payoff(self, S_t_T):
        S_min = S_t_T.min(axis=0)
        # If the put was already knocked-out, leave "alive" at 0
        self.alive_on_trajectory = self.alive_at_t * np.where(S_min > self.barrier, 1.0, 0.0)
        return self.alive_on_trajectory * super()._payoff(S_t_T)

class DigitalCall(Derivatives): 
    def __init__(self, S_0, strike, maturity=None, **kwargs):
        """1$ cash-or-nothing if S_0 is NaN, asset-or-nothing otherwise."""
        super().__init__(S_0, maturity, **kwargs)
        self.strike = strike
        
    def __repr__(self):
        return 'DigitalCall(%s,%s,%s)'%(self.S_0, self.strike, self.repr_maturity())    

    def _payoff(self, S_t_T):
        S_T = S_t_T[-1,:]
        if np.isnan(self.S_0):
            return np.where(S_T > self.strike, 1, 0)
        return np.where(S_T > self.strike, S_T, 0)


class DigitalPut(Derivatives):
    def __init__(self, S_0, strike, maturity=None, **kwargs):
        """1$ cash-or-nothing if S_0 is NaN, asset-or-nothing otherwise."""
        super().__init__(S_0, maturity, **kwargs)
        self.strike = strike
        
    def __repr__(self):
        return 'DigitalPut(%s,%s,%s)'%(self.S_0, self.strike, self.repr_maturity())    

    def _payoff(self, S_t_T):
        S_T = S_t_T[-1,:]        
        if np.isnan(self.S_0):        
            return np.where(S_T < self.strike, 1, 0)
        return np.where(S_T < self.strike, S_T, 0)
    
class FloatingStrikeLookbackCall(Derivatives):
    """This class was not tested yet."""
    def __init__(self, S_0, maturity=None, **kwargs):
        super().__init__(S_0, maturity, **kwargs)
        self.minimum = S_0

    def __repr__(self):
        return 'FloatingStrikeLookbackCall(%s,%s)'%(self.S_0, self.repr_maturity())

    def update_history(self, S_0_t):
        super().update_history(S_0_t)
        self.minimum = np.minimum(self.minimum, S_0_t.min())

    def _payoff(self, S_t_T):
        S_T = S_t_T[-1,:]
        self.strike = np.minimum(self.minimum, S_T.min(ax=0))
        return np.maximum(S_T - self.strike, 0.0)

class Position:
    def __init__(self, n_units, instrument):
        self.n_units = n_units
        self.instr = instrument

        #Deprecated # Placeholder for the state of the position on each trajectories
        #Deprecated self.active = None

    @property
    def active(self):
        raise NotImplementedError("The active property is deprecated. States are tracked in the Instrument subclasses.")

    payoff = Instrument.payoff
    def _payoff(self, S_t_T):
        #Deprecated payoff = self.n_units * self.instr.payoff(S_t_T)
        #Deprecated if self.active is None:
        #Deprecated     return payoff
        #Deprecated return np.where(self.active, payoff, 0.0)
        return self.n_units * self.instr.payoff(S_t_T)

    def evaluate(self, time_t, S_t):
        price = self.instr.evaluate(time_t, S_t)
        if self.instr.bid_ask is not None:
            price = price[0] if self.n_units < 0 else price[1] 
        return self.n_units * price
    
    def __repr__(self):
        return '%f * %s'%(self.n_units, self.instr)
    
    def __neg__(self):
        return Position(-1.0*self.n_units, self.instr)

    def __mul__(self, n_units):
        return Position(n_units*self.n_units, self.instr)
    __rmul__ = __mul__
    
    def __add__(self, other):
        return Portfolio(self, other)

    def __sub__(self, other):
        return Portfolio(self, -other)    
    
    
class Portfolio:
    def __init__(self, *args):
        self.name = ''
        self.positions = []
        self.maturity = None
        for pos in args:
            if pos is None:
                continue
            elif isinstance(pos, Portfolio):
                self.positions += pos.positions                
            elif isinstance(pos, Position):
                self.positions.append(pos)
            elif isinstance(pos, Instrument):
                self.positions.append( Position(1.0, pos) )
            else:
                #import pdb; pdb.set_trace()
                raise TypeError("Unexpected position: %s (%s)"%(repr(pos),type(pos)))

        # Plot options
        self.linewidth = 2

    def set_name(self,name):
        self.name = name
        return self
            
    def get_maturity(self):
        maturity = self.positions[0].instr.maturity
        for pos in self.positions[1:]:
            assert pos.instr.maturity==maturity, 'Code is currently assuming that the portfolio has single maturity'
        return maturity
    
    def ytm(self, time_t=0):
        return economy.years_to_maturity(self.get_maturity() - time_t)
            
    def evaluate(self, time_t, S_t):
        value = 0.0
        for no,pos in enumerate(self.positions):
            value += pos.evaluate(time_t, S_t)
        return value

    def update_history(self, S_0_t):
        """Broadcast history to all instruments in the portfolio.

        The history should be the (single) **realized** trajectory of the stock starting from 0.

        S_0_t must be an (t+1)x1 np.array where t is the number of days since inception.              
              Hence, the first row should be the value of the stock at time 0.
        """
        for pos in self.positions:
            pos.instr.update_history(S_0_t)
            
    payoff = Instrument.payoff
    def _payoff(self, S_t_T):
        payoff = 0.0
        for pos in self.positions:
            payoff += pos.payoff(S_t_T)
        return payoff

    def __repr__(self):
        s = ''
        for no,pos in enumerate(self.positions):
            s += repr(pos) 
            if no < len(self.positions)-1:
                if len(s) > 80:
                    s += '\n   '
                s += ' + ' 
        return s
    
    def __neg__(self):
        positions = [-1.0*pos for pos in self.positions]
        # positions = []
        # for pos in self.positions:
        #     positions.append( -1.0*pos )
        return Portfolio(*positions)

    def __mul__(self, n_units):
        positions = [n_units*pos for pos in self.positions]
        return Portfolio(*positions)
    __rmul__ = __mul__
    
    def __add__(self, other):
        return Portfolio(self, other)

    def __sub__(self, other):
        #import pdb; pdb.set_trace()
        return Portfolio(self, -other)        

    def plot_payoff(self, ax=None, S_t_T=None, **kwargs):
        """Plot the payoff of the portfolio.

        This method plots the payoff of the portfolio against the terminal values of the underlying asset. The `S_t_T` parameter *can* be an (T-t+1)xN np.array where T is the number of days (or periods) to maturity at inception, t is the current day and N the number of simulated paths. The terminal payoff is plot against the last row of `S_t_T` (conceptually, `S_T`), but path dependent portfolios could see their terminal payoff affected by intermediary values of `S_t`. 
        
        If only one argument is provided, it is expected to be `S_t_T`; that is `ax` is expected to be a np.ndarray and is used as `S_t_T`, with default axes created hereinafter.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes object on which to plot the payoff. If not 
                provided, a new figure and axes will be created.
            S_t_T (numpy.ndarray, optional): The simulated paths of the underlying asset. 
                If not provided, `ax` is used as `S_t_T`.
            **kwargs: forwarded to `ax.plot`

        Returns:
            numpy.ndarray: The payoff of the portfolio.
        """
        if S_t_T is None:
            assert isinstance(ax, np.ndarray)
            S_t_T = ax
            fig, ax = plt.subplots(1, 1, figsize=figsize['default'])
        if len(S_t_T.shape)==1: # 1D array S_T
            S_t_T = S_t_T.reshape(1,-1)

        payoff = self.payoff(S_t_T)
        ptf_name = 'Portfolio' if self.name=='' else self.name

        S_T = S_t_T[-1,:]
        ax.plot(S_T, payoff, label=ptf_name, linewidth=self.linewidth, **kwargs)
        ax.set_xlabel('Underlying')
        ax.set_ylabel('Payoff') # Not 'P&L': we are not accounting for the prices
        ax.legend()

        #set_payoff_axes(ax)
        #plt.ylabel('$', rotation='horizontal', ha='right', va='top')    
        #ax.legend(loc='lower right')

        return payoff

    def delta_hedge(self, S_t, T, dt, kappa=0):
        """Delta hedges a porftolio across simulated paths S_t
    
        S_t must start with a row including the current value of the stock
    
        kappa: the turnaround transaction cost on the underlying (cf. Leland 1985)
        """
        n_steps, n_paths = S_t.shape
        mat = (T - np.arange(n_steps)*dt).reshape(-1,1)    
    
        # Delta through time
        delta_t = np.zeros((n_steps,n_paths))
        for no,pos in enumerate(self.positions):
            is_call = isinstance(pos.instr,Call)
            assert is_call or isinstance(pos.instr,Put), 'Not implemented for other than Call and Put'
            cp = np.where(is_call, 1, -1)

            # ASSUMPTION: currently working under the assumption that the smile is NOT evolving
            # for the instruments in the portfolio! We will need to code an ImpliedVolatilitySurface with
            # basic assumptions, and subclasses for refined approaches
            sigma = _model.iv_on_surface(pos.instr.strike, T)
            print(pos.instr.strike,sigma)
            
            instr_delta = _model.delta(S=S_t[:-1,:], K=pos.instr.strike, T=mat[:-1,:], sigma=sigma, is_call=is_call) 
            itm_at_T = cp*S_t[-1,:] > cp*pos.instr.strike
            instr_delta = np.vstack(( instr_delta, np.where(itm_at_T, cp, 0) ))
            delta_t += pos.n_units * instr_delta
            
        def baCF(cash_flow):
            # buy the stock -> negative CF
            # sell the stock -> positive CF
            eta = np.where(cash_flow < 0, +1, -1) 
            return cash_flow*(1+eta*kappa/2) # IMPORTANT: divide the roundtrip by 2
        
        M_0 = -self.evaluate(0, S_t[0,:]) # If short, the premium will be negative, -premium > 0
        M_0 += baCF(delta_t[0,:]*S_t[0,:])
        
        # Rebalancing each period, except the last date: do not do the roundtrip...
        dM = baCF((delta_t[1:-1,:] - delta_t[:-2,:])*S_t[1:-1,:])
        
        # Apply the payoff to the terminal date & liquidate hedge as it stood at the next to last period (-2)
        #import pdb; pdb.set_trace()
        dM_T = baCF(-delta_t[-2,:]*S_t[-1,:]) + self.payoff(S_t[-1,:])
        # No need for the payoff method to worry about transaction costs; it's taken care of here    
    
        # Interest on the margin
        dM = np.vstack(( M_0, dM, dM_T ))*np.exp(_model.r*mat)
        
        # Cumulating the rebalancing PnL
        M_t = np.cumsum(dM, axis=0)
        return M_t

    def plot_delta_hedge(self, ax, *args, **kwargs):
        alpha = kwargs.pop('alpha',1.0)
        edges = kwargs.pop('edges',None)        
        margin = kwargs.pop('margin',None)
        with_ylabel = kwargs.pop('with_ylabel',True)
        with_title = kwargs.pop('with_title',True)

        if margin is None:
            margin = self.delta_hedge(*args, **kwargs)

        err = margin[-1,:]
        ax.hist(err, edges, alpha=alpha);    
        ax.set_xlabel('Hedging error ($)')
        if with_ylabel:
            ax.set_ylabel('Count')
        if with_title:
            mu = np.mean(err)
            std = np.std(err)        
            ax.set_title('%s - Avg = %.2f - Std = %.2f'%(self.name,mu,std))
            
        return margin

class PathDependentInstrumentsIterator:
    def __init__(self, positions):
        self.positions = positions 
        self.ix = 0
        self.indices = []
        for ix,pos in enumerate(positions):
            if hasattr(pos.instr, 'alive_at_t'):
                self.indices.append(ix)
        self.count = len(self.indices)        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ix >= self.count:
            raise StopIteration
        pos = self.positions[ self.indices[self.ix] ]
        self.ix += 1
        return pos.instr
    
    def next_or_none(self):
        try:
            return next(self)
        except StopIteration:
            return None