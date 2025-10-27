import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy.special as ssp
import warnings

from .toolkit import printdf, print_versions, date2str, numunique

from . import black_merton_scholes as bms
from .parameters import ModelParameters, PV

def column_vector(value):
    #import pdb; pdb.set_trace()
    if np.isscalar(value):
        value = np.array([value])
    elif not isinstance(value,np.ndarray):
        value = np.array(value)

    return value.reshape(-1,1) # Column vector

def get_synthetic_option_prices(date, DTM=1, am_settlement=0):
    with warnings.catch_warnings(): # Quiet the TODO warnings
        warnings.filterwarnings("ignore")
        from . import option_metrics as om
        db = om.wrds_connect()
        secid = 108_105
        options = om.get_option_data(secid, date)
        options = options[ options.am_settlement==am_settlement ]

    # Subsample options based on DTM, positive bid and volume
    options = options[(options.DTM==DTM) 
               & (options.best_bid > 0) & (options.volume > 0)].copy()

    # Add option-implied forward price and the corresponding BMS implied vol
    implied_forward_price = om.SmileIterator.implied_forward_price
    fwd = implied_forward_price(options)
        
    key = ['date','DTM']
    columns = key+['implied_forward_price']
    options = pd.merge(options, fwd[columns], on=key, how='left')
    options['MNY'] = options.strike / options.implied_forward_price
    options['implied_vol_bms'] = om.implied_vol_bms(options, options.option_price)

    # Use Yan's (2011) Proposition 2 to find reasonable parameters to simulate the synthetic prices
    ix = ~np.isnan(options.implied_vol_bms)
    mny = options.MNY[ix]
    log_mny = np.log(mny)
    IV = options.implied_vol_bms[ix]

    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(log_mny, IV, k=4)
    derivative = spl.derivative()

    ix = np.argmin(np.abs(mny - 1))
    atm_sigma = spl(log_mny.iloc[ix]) # Yan's equation (6)
    atm_slope = derivative(log_mny.iloc[ix]) # Yan's equation (7)
    lmbda_k = atm_sigma*atm_slope 

    # Getting the synthetic prices...
    K = options.strike
    r = numunique(options.risk_free)
    T = numunique(options.YTM)
    S = np.exp(-r*T)*numunique(options.implied_forward_price)

    # ...assuming some reasonable parameters...
    p_0 = 0.975
    lmbda = -np.log(p_0)/T - lmbda_k
    mu_jump = np.log(1 + lmbda_k/lmbda)
    sigma_jump = np.abs(mu_jump) # ad hoc

    pv = PV(sigma=atm_sigma, lmbda=lmbda, mu_jump=mu_jump, sigma_jump=sigma_jump)    
    options['merton_price'] = merton76_as_sum(S,K,r,0,T,pv.sigma, 
                                              pv.lmbda, pv.mu_jump, pv.sigma_jump, options.is_call)    
    
    # ...keeping a copy of the market prices...
    options['market_price'] = options['option_price']

    # ...and adjusting the bid and ask to surround the simulated price
    diff = options['merton_price'] - options['option_price']
    options['option_price'] = options['merton_price']
    options['best_bid'] = options['best_bid'] + diff
    options['best_offer'] = options['best_offer'] + diff
    
    return pv, options

def merton76_as_sum(S,K,r,y,T,sigma, lmbda,mu_jump,sigma_jump, is_call, **kwargs):
    """Merton (1976): Diversifiable jumps

    On p129, Merton (1976) state that $k = \operatorname{E}\left[ Y - 1 \right]$, where $Y$ is
    the magnitude of a single jump where $Y$ is
    
    <img src="./m76/p129_top.png" width="85%" height="85%">
    
    Later, on p135:
    
    <img src="./m76/p135_eq18_19.png" width="85%" height="85%">
    
    In the notation of the DF book, Equations (18) and (19) of Merton (1976) state that, when
    the jump magnitude $e^{J_n} - 1$ is given by $J_n \sim N\left(\mu_J -
    \frac{1}{2}\sigma_J^2, \sigma_J^2\right)$, then 
    $k = \operatorname{E}\left[e^{J_n} - 1 \right] = e^{\mu_J} - 1 \Leftrightarrow \mu_J = \log(1+k)$, 
    and 
    \begin{align}
      c_{M76}(t,T,K) = \sum_{n=0}^\infty \frac{e^{-\lambda'T}(\lambda'T)^n}{n!}
      c_{BMS}\left(t,T,K | r=r-\lambda k + n\mu_J/T, \sigma=\sqrt{\sigma + n\sigma_J^2/T}\right)
    \end{align} where $\lambda' = \lambda (1+k)$.
    """
    k = np.exp(mu_jump) - 1
    lmbda_prime = lmbda*(1+k)
    n_J = np.arange(10) # YH: 5 is not enough for K = 100, so I change it to 10.
    p_n = np.exp(-lmbda_prime*T)*(lmbda_prime*T)**n_J / ssp.factorial(n_J)

    #first_below = lambda arr, thresh: np.argmax(arr < thresh)
    #n_trunc = first_below(p_n,1e-6)
    #n_J = n_J[:n_trunc]
    #p_n = p_n[:n_trunc]
    
    r_n = r - lmbda*k + n_J*mu_jump/T
    sig_n = np.sqrt(sigma**2 + n_J*sigma_jump**2/T)
    
    bcast = not (np.isscalar(K) and np.isscalar(is_call))
    K = column_vector(K)
    is_call = column_vector(is_call)
    if bcast:
        r_n = r_n.reshape(1,-1) # Row vector
        sig_n = sig_n.reshape(1,-1) # Row vector
        [K,r_n,sig_n,is_call] = np.broadcast_arrays(K,r_n,sig_n,is_call) # matrices

    #import pdb; pdb.set_trace()
    price_n = bms.option_price(S,K,r_n,0,T,sig_n,True) # only calls
    # Use put-call parity to price the puts
    #   put_price = call_price + disc*K - stock_exdiv
    if bcast:
        #import pdb; pdb.set_trace()    
        price_n[~is_call] = price_n[~is_call] \
                                + K[~is_call]*np.exp(-r*T) - S*np.exp(-y*T) 
    elif not is_call:
        price_n = price_n + K*np.exp(-r*T) - S*np.exp(-y*T) # The put prices are calculated from the put-call parity
        
    price_n = p_n*price_n # weighted prices
    
    # Ensure that the truncation does not neglect anything significant 
    vec_last = len(price_n.shape)==1 and price_n[-1] < 1e-6
    mat_last = len(price_n.shape)==2 and np.all(price_n[:,-1] < 1e-6)
    #assert vec_last or mat_last
    if not (vec_last or mat_last):
        print("TRUNCATING AT n TOO LOW")
        import pdb; pdb.set_trace()
    
    return_price_n = kwargs.pop('return_price_n',False)
    if return_price_n:
        return price_n.sum(axis=1), price_n
    return price_n.sum(axis=1)

def sigmoid(x, a=0, b=1, slope=1):
    return a + (b-a) / (1 + np.exp(-slope*x))

class OptionPricingError: 
    """Providing different flavors of option-pricing errors."""    
    
    call_error_msg = """OptionPricingError instance can be called only if its error_func
    attribute is a valid string, i.e. the name of one of the class' methods.
    """
    @staticmethod
    def bid_ask_sigmoid(x, bid, ask):
        mid = (ask+bid) / 2
        eff_ba = (ask-bid) / 2
        err = ((x - mid) / eff_ba)**2
        
        a,b,s = 0.25, 1, 0.5
        min_y = sigmoid(0, 0, 2*(b-a), s)
        return a + (sigmoid(err, 0, 2*(b-a), s) - min_y)

    @staticmethod
    def illustrate_error_functions():
        options = PV()
        options.best_bid = 0.75
        options.best_offer = 1.25
        options.option_price = 0.5*(options.best_bid + options.best_offer)
        error = OptionPricingError(options)
        
        model_price = np.linspace(0,2,10000)
        
        options.mid_price_sse = [error.mid_price_sse(p) for p in model_price]
        options.hard_bid_ask_sse = [error.hard_bid_ask_sse(p) for p in model_price]
        options.soft_bid_ask_sse = [error.soft_bid_ask_sse(p) for p in model_price]
        
        fig,axes = plt.subplots(1,2, figsize=(16,5))
        ax = axes[0]
        ax.plot(model_price, options.mid_price_sse, label='mid_price_sse')
        ax.plot(model_price, options.hard_bid_ask_sse, label='hard_bid_ask_sse')
        ax.plot(model_price, options.soft_bid_ask_sse, label='soft_bid_ask_sse')
        ax.set_xlabel('model_price')
        ax.set_ylabel('error')
        ax.legend()
        
        soft = OptionPricingError.bid_ask_sigmoid(model_price, options.best_bid, options.best_offer)
        axx = ax.twinx()
        axx.plot(model_price, soft, color='r', linestyle='--', label='sigmoid')
        axx.axhline(0.25, color='r', linestyle=':')
        axx.set_ylabel('soft', color='r')
        
        ax = axes[1]
        dx = np.diff(model_price)
        der = lambda y: np.diff(y) / dx
        ax.plot(model_price[1:], der(options.mid_price_sse), label='mid_price_sse')
        ax.plot(model_price[1:], der(options.hard_bid_ask_sse), label='hard_bid_ask_sse')
        ax.plot(model_price[1:], der(options.soft_bid_ask_sse), label='soft_bid_ask_sse')
        ax.plot(model_price[1:], der(soft), label='soft', color='r', linestyle='--')        
    
    def __init__(self, options, error_func=None):
        self.options = options
        self.error_func = error_func
        
    def __call__(self, model_price):
        if self.error_func is None:
            raise ValueError(self.call_error_msg)
        func = getattr(self, self.error_func)
        return func(model_price)
            
    def mid_price_sse(self, model_price):
        self.pricing_error = model_price - self.options.option_price
        return np.sum(self.pricing_error**2)

    def hard_bid_ask_sse(self, model_price):
        error = model_price - self.options.best_bid
        underpriced = np.minimum(error, 0)
                        
        error = model_price - self.options.best_offer
        overpriced = np.maximum(error, 0)

        self.pricing_error = underpriced + overpriced
        return np.sum(self.pricing_error**2)
    
    def soft_bid_ask_sse(self, model_price):
        """Prices outside the BA-spread are _further_ penalized than within.
        
        As opposed to the hard_bid_ask_sse, however, errors within the bid-ask 
        spread are not simply disregarded, which makes some parameters difficult 
        to identify.
        
        Consider $f(x)$ where $f$ is the error we attempt to minimize given model price $x$.        
        In the mid_price_sse, the gradient of $f$ wrt $x$ is $2(x-x^*)$, which is 
        negative (positive) when the model underprices (overprices) wrt to the mid price.
        In the hard_bid_ask_sse, the same holds, but for underpricing (overpricing) wrt to 
        the bid (ask); the gradient is however 0 within the bid ask spread.
        """
        soft = OptionPricingError.bid_ask_sigmoid(model_price, self.options.best_bid, self.options.best_offer)
        return soft*self.mid_price_sse(model_price)        
        #Note that the above is NOT the same as below, whose derivative is to flat within the BA
        #self.pricing_error = soft*(model_price - options.option_price)
        #return np.sum(self.pricing_error**2)


class Merton76(ModelParameters):
    def __init__(self, error_function='mid_price_sse'):
        super().__init__()
        self.add_parameter('sigma', 0.3, [0.01, 5])
        self.add_parameter('lmbda', 1e-9, [0, 100])
        self.add_parameter('mu_jump', 0.0, [-100.0, 100.0])
        self.add_parameter('sigma_jump', 1.0, [0.0, 100])        

        self.error_function = error_function
        
        if False:
            # TARGETING sigma_jump
            self.fix_parameter(sigma_jump=0.0)
        
    def objective(self, options, **kwargs):
        assert isinstance(options,pd.DataFrame), type(options)

        if False:
            # TARGETING sigma_jump
            mu_jump = self.params['mu_jump'].value
            a,b,c = 10*0.5,1.0,-mu_jump*10
            #import pdb; pdb.set_trace()
            r1,r2 = (-b + np.sqrt(b**2-4*a*c))/(2*a), (-b - np.sqrt(b**2-4*a*c))/(2*a)        
            self.fix_parameter(sigma_jump=np.maximum(r1,r2))
        
        K = options.strike
        r = numunique(options.risk_free)
        T = numunique(options.YTM)
        S = np.exp(-r*T)*numunique(options.implied_forward_price)
        
        pv = self.get_pv()
        self.options = options
        self.option_price = merton76_as_sum(S,K,r,0,T,pv.sigma, 
                               pv.lmbda,pv.mu_jump,pv.sigma_jump, options.is_call)
        
        error = OptionPricingError(options, self.error_function)
        obj = error(self.option_price)
        self.pricing_error = error.pricing_error
        return obj                

    def implied_volatility_bs73(self):
        options = self.options
        K = options.strike
        r = numunique(options.risk_free)
        T = numunique(options.YTM)      
        S = np.exp(-r*T)*numunique(options.implied_forward_price)
        
        pv = self.get_pv()
        return bms.implied_volatility(self.option_price, S,K,r,0,T,options.is_call)  
    
    def implied_volatility_m76(self):
        options = self.options
        
        # Keep track of the initial state of free/fixed parameters
        initially_free = []
        #import pdb; pdb.set_trace()
        print(self.params)
        for name in self.params.keys():
            param = self.params[name]
            if param.free:
                initially_free.append(name)
                self.fix_parameter(name)
        sigma_free = self.params['sigma'].free
        
        
        # Optimize sigma for each option observation
        sigma0 = self.params['sigma'].value
        results = []
        for ix in options.index:
            self.free_parameter('sigma',sigma0)
            opt = self.estimate(options.loc[[ix]], method='Nelder-Mead', verbose=False)
            results.append([opt.fun, self.params['sigma'].value])
        self.options = options
        
        # Restore the initial state of free/fixed parameters
        for name in initially_free:
            self.free_parameter(name)
        self.params['sigma'].free = sigma_free
        
        return pd.DataFrame(results, columns=['pricing_error','implied_volatility'])
                           
