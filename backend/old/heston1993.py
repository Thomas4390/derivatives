import internal_script
import numpy as np
from scipy.integrate import quad
import warnings

from . import black_merton_scholes as bms
from .parameters import *
from .plot_utils import (mpl, plt, mtick, mdates, gridspec,
                    set_plt_defaults, set_payoff_axes, set_time_axis, with_style)

class Heston1993(ModelParameters):
    """
    Implements the Heston (1993) model parameters initialization and operations.
    
    Attributes:
    S0 (float): The initial stock/index level
    K (float): The strike price
    T (float): The time-to-maturity (for t=0)
    r (float): The constant risk-free short rate
    kappa_v (float): The mean-reversion factor
    theta_v (float): The long-run mean of variance
    sigma_v (float): The volatility of variance
    rho (float): The correlation between variance and stock/index level
    v0 (float): The initial level of variance
    """
    def __init__(self):
        super().__init__()
        self.S0 = 100
        self.K = 100
        self.T = 0.5
        self.r = 0.0
        self.kappa_v = 2
        self.theta_v = 0.01
        self.sigma_v = 0.10
        self.rho = 0.5
        self.v0 = 0.01
        
    def char_func(self, u):
        ''' Valuation of European call option in H93 model via Lewis (2001)
        Fourier-based approach: characteristic function.

        Parameter definitions see function BCC_call_value.'''
        
        T, r, kappa_v, theta_v, sigma_v, rho, v0 = self.T, self.r, self.kappa_v, self.theta_v, self.sigma_v, self.rho, self.v0
        
        c1 = kappa_v * theta_v
        c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v) ** 2 -
                    sigma_v ** 2 * (-u * 1j - u ** 2))
        c3 = (kappa_v - rho * sigma_v * u * 1j + c2) \
            / (kappa_v - rho * sigma_v * u * 1j - c2)
        H1 = (r * u * 1j * T + (c1 / sigma_v ** 2) *
            ((kappa_v - rho * sigma_v * u * 1j + c2) * T -
            2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
        H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v ** 2 *
            ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
        char_func_value = np.exp(H1 + H2 * v0)
        return char_func_value
    
    def int_func(self, u):
        ''' Valuation of European call option in H93 model via Lewis (2001)
        Fourier-based approach: integration function.

        Parameter definitions see function H93_call_value.'''
        S0, K = self.S0, self.K
        
        char_func_value = self.char_func(u - 1j * 0.5)
        int_func_value = 1 / (u ** 2 + 0.25) \
            * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
        return int_func_value
        
    def call_value(self):
        """
        Returns:
        call_value (float): present value of European call option
        """
        S0, K, r,T = self.S0, self.K, self.r, self.T

        int_value = quad(lambda u: self.int_func(u),
                     0, np.inf, limit=250)[0]
        call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
        return call_value
    
    
class HestonPlot(Heston1993):
    def __init__(self,stock_price, price_func,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S = stock_price
        
        params = {'heston_skew':'rho', 'heston_kurt':'sigma'}
        self.free_param = params[price_func]
        self.call_price = lambda S,param: getattr(self,price_func)(S,param)
        
    def heston_skew(self,S,rho):
        self.S0 = S
        self.rho = rho
        return self.call_value()

    def heston_kurt(self, S, sigma):
        self.S0 = S
        self.sigma_v = sigma
        return self.call_value()
    
    def compute_prices(self, param):
        bms_call = lambda S: bms.option_price(S, self.K, self.r, 0.0, self.T, np.sqrt(self.v0), is_call=True)
        self.bms = np.array([bms_call(S) for S in self.S])
        
        self.heston = np.empty([2,self.bms.shape[0]])
        self.heston[0] = np.array([self.call_price(S,param[0]) for S in self.S])
        self.heston[1] = np.array([self.call_price(S,param[1]) for S in self.S])
        return self.heston
        
    def plot_price_diff(self, ax):
        ax.plot(self.S, self.heston[0]-self.bms, '-', linewidth=2, color='firebrick')
        ax.plot(self.S, self.heston[1]-self.bms, '-', linewidth=2, color='darkblue')
        ax.set(xlabel="Spot Price ($)", ylabel="Price Difference ($)")
        
        
class HestonPlotIV(HestonPlot):    
    def __init__(self, moneyness, stock_price, price_func, *args, **kwargs):
        super().__init__(stock_price, price_func, *args, **kwargs)
        self.K = self.S * moneyness
        self.moneyness = moneyness

    def heston_skew(self, K, rho):
        self.K = K
        self.rho = rho
        return self.call_value() 

    def heston_kurt(self, K, sigma):
        self.K = K
        self.sigma_v = sigma
        return self.call_value() 
    
    def compute_impvols(self, param):
        self.param = param;
            
        bms_call = lambda Kn: bms.option_price(self.S, Kn, self.r, 0.0, self.T, np.sqrt(self.v0), is_call=True)
        #bms_prices = np.array([bms_call(Kn) for Kn in K])

        #heston_0 = np.array([self.call_price(K,param[0]) for K in self.K])
        #heston_1 = np.array([self.call_price(K,param[0]) for K in self.K])        

        implied_vol = lambda C,Kn: \
          bms.implied_volatility(C, self.S, Kn, self.r, 0.0, self.T, is_call=True)

        self.bms = np.empty(self.K.shape[0])
        self.heston = np.empty([2,self.bms.shape[0]])
        for no,Kn in enumerate(self.K):
            self.bms[no] = implied_vol(bms_call(Kn), Kn)
            self.heston[0][no] = implied_vol(self.call_price(Kn,param[0]), Kn)
            self.heston[1][no] = implied_vol(self.call_price(Kn,param[1]), Kn)
            
    def plot_implied_vol(self, ax):
        label = '%s='%self.free_param + '%.3f'
        ax.plot(self.moneyness, 100*self.heston[0], '-', linewidth=2, color='firebrick', label=label%self.param[0])
        h1 = ax.plot(self.moneyness, 100*self.heston[1], '-', linewidth=2, color='darkblue', label=label%self.param[1])
        bms = ax.plot(self.moneyness, 100*self.bms, '-', linewidth=2, color='black', label='BMS')
        ax.set(xlabel="Moneyness (K/S)", ylabel="Implied Volatility (%)")
        ax.legend(loc='upper right')
        
        
def figures_rho_sigma(rho_top, sigma_top, rho_btm, sigma_btm, title_prefix=[]):
    '''Plot figures inspired by those of Heston (1993, RFS) with their IV counterparts

    Top panels take two values of rho(_top), and one of sigma(_top).
    Bottom panels take one values of rho(_btm), and two of sigma(_btw).
    Left-hand side panels are in terms of price differences with BMS
    Right-hand side panels are in terms of implied volatilities
    '''
    fig = plt.figure(figsize=(12, 12))

    no = 1
    ax = fig.add_subplot(2,2,no)
    tt = 'sigma=%.3f'%sigma_top
    if len(title_prefix) > 0: 
        tt = title_prefix[no-1]+' - '+tt
    ax.title.set_text(tt)
    prc_skew = HestonPlot(np.arange(75,125), 'heston_skew')
    prc_skew.sigma_v = sigma_top
    prc_skew.compute_prices(rho_top)
    prc_skew.plot_price_diff(ax)

    no = 2
    ax = fig.add_subplot(2,2,no)
    tt = 'sigma=%.3f'%sigma_top
    if len(title_prefix) > 0: 
        tt = title_prefix[no-1]+' - '+tt
    ax.title.set_text(tt)
    iv_skew = HestonPlotIV(np.arange(0.75,1.25,0.01),100, 'heston_skew')
    iv_skew.sigma_v = sigma_top
    iv_skew.compute_impvols(rho_top)
    iv_skew.plot_implied_vol(ax)
    
    no = 3
    ax = fig.add_subplot(2,2,no)
    tt = 'rho=%.3f'%rho_btm
    if len(title_prefix) > 0: 
        tt = title_prefix[no-1]+' - '+tt
    ax.title.set_text(tt)    
    prc_kurt = HestonPlot(np.arange(75,125), 'heston_kurt')
    prc_kurt.rho = rho_btm
    prc_kurt.compute_prices(sigma_btm)
    prc_kurt.plot_price_diff(ax)
            
    no = 4
    ax = fig.add_subplot(2,2,no)
    tt = 'rho=%.3f'%rho_btm
    if len(title_prefix) > 0: 
        tt = title_prefix[no-1]+' - '+tt
    ax.title.set_text(tt)    
    iv_kurt = HestonPlotIV(np.arange(0.75,1.25,0.01),100, 'heston_kurt')
    iv_kurt.rho = rho_btm
    iv_kurt.compute_impvols(sigma_btm)
    iv_kurt.plot_implied_vol(ax)
    
    plt.show()
