"""Some standard processing for OptionMetrics data.

NOTE: The code below is still under developpement. Docstrings and comments are
lacking. Your comments are welcome.

A common output from some of the preprocessing below is the following pickle
    options = om.load_pickle('secid108105_19960101_20221231.pkl')
    options.index = [pd.Timestamp(date) for date in options.date] # optional, depending on your needs

which would contain the following fields directly from OptionMetrics

    secid, date, symbol, symbol_flag, exdate, last_date, cp_flag, strike_price,
    best_bid, best_offer, volume, open_interest, impl_volatility, delta, gamma,
    vega, theta, optionid, cfadj, am_settlement, contract_size, ss_flag,
    forward_price, expiry_indicator, root, suffix,

and these additions:

    hash: pd.util.hash_pandas_object returned a hash for each row (i.e. quote) 
    strike: Strike price of the option (strike_price / 1000)
    option_price: The mid point between the bid and the offer
    is_call: Boolean, whether the option is a call (False when put)
    DTM: The options days to maturity, in calendar days
    YTM: The options years to maturity
    risk_free: From OM's zero curve, interpolated at the option's maturity
    stock_price: OM's underlying price 
    stock_exdiv: Based on OM's underlying price and dividend forecast 
    implied_forward_price: Implied from options, following the CBOE proceedure (VIX)   
    implied_vol_bms: At the option's mid-price, based on the implied forward price    
    implied_vol_bid: At the option's bid price, based on the implied forward price   
    implied_vol_ask: At the option's offer price, based on the implied forward price

When used to their full capacity, the Smile and Surface classes can be quite
convenient. Comments welcome on potential improvements
"""
import os
import numpy as np
import datetime as dt
import pandas as pd
import warnings

from . import black_merton_scholes as bms
from .toolkit import assert_unique, date2str, object_vars_to_string, nan_equal_dataframe, tic, toc

secid_of_indices = [108105]
"""Some functions will treat all secids except those herein as if they were single name equities."""

start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime(2021, 12, 31)
def with_default_dates(sdate, edate):
    global start_date, end_date
    if sdate is None:
        sdate = start_date
    if edate is None:
        edate = end_date
    start_date, end_date = sdate, edate # Make sure the global variables are updated
    return sdate, edate

import socket
__hostname = socket.gethostname()
POOL_SIZE = 1
# POOL_SIZE = 8
# if __hostname.startswith('vulcan'):
#     POOL_SIZE = 8
# elif __hostname.startswith('svarog'):
#     POOL_SIZE = 16

if False:
    # For dev in notebooks
    db = om.db
    start_date = om.start_date
    end_date = om.end_date
    dump_pickle = om.dump_pickle
    load_pickle = om.load_pickle

        
def between(data, start_date, end_date, date_col='date'):
    """Select entries for which `date_col` (default: 'date') falls between arguments.
    
    Note that `data[date_col]` is also modified in place. In particular, if it is a series of strings or timestamps, it will become a series of dates.
    """
    if data.empty:
        return data
    
    data[date_col] = pd.to_datetime(data[date_col]).dt.date
    #try:
    return data[(start_date <= data[date_col]) & (data[date_col] <= end_date)]
    #except:
    #    return pd.DataFrame([],columns=[date_col])
    
db = None
def _db_alive(conn):
    if conn is None:
        return False
    try:
        conn.raw_sql("select 1")  # lightweight test
        return True
    except Exception:
        return False
    
def wrds_connect(force=False):
    import wrds
    global db
    if (not force) and _db_alive(db):
        return db
    
    try:
        if os.getcwd().startswith('/Users/christian/'):
            db = wrds.Connection(wrds_username='cdorion')
        else:
            db = wrds.Connection(wrds_username='yuhansong')
    except:
        warnings.warn('WRDS connection failed')
    return db


def compute_present_dividends(dividends, zero_curve=None):
    """Compute the present value of dividends for equity stocks"""
    if zero_curve is None:
        zero_curve = fetch_zerocd()
    
    dividends['ex_date'] = pd.to_datetime(dividends['ex_date'])
    dividends['date'] = pd.to_datetime(dividends['date'])
    
    div_period = dividends[['date','ex_date']].drop_duplicates()
    div_rf = interpolate_zero_curve(zero_curve,div_period['date'],div_period['ex_date'])
    dividends = pd.merge(dividends, div_rf[['date','exdate','risk_free','YTM']].rename(columns={'exdate':'ex_date'}), on =['date','ex_date'], how='left')
    dividends['pv_amount'] = dividends['amount'] * np.exp(-dividends['risk_free']*dividends['YTM'])
    
    return dividends.drop(columns=['risk_free','YTM'])
    

#v0: def fetch_zerocd(cache=True):
def fetch_zerocd():
    """Fetch OptionMetrics' zero curve."""
    # See comments in fetch_secprd regarding caching 
    #v0: cache = 'zerocd.pkl' if cache else None
    #v0: data = None if cache is None else load_pickle(cache)
    #v0: if data is not None:
    #v0:     return between(data, start_date, end_date)

    #v0: # Inexistent cache; fetch data
    data = db.raw_sql('select * from OPTIONM.ZEROCD')

    #v0: if cache is not None:
    #v0:     dump_pickle(data, cache)
    return data 


#v1: def fetch_secprd(secid, start_date=None, end_date=None, cache=True):
def fetch_secprd(secid, start_date=None, end_date=None):
    """Fetch underlying security prices from OptionMetrics.

    From OM's reference manual:
    The option price used in implied volatility calculation is an average 
    between max Bid and min Ask. These are selected across all exchanges 
    the contract is traded on. Option prices used in implied volatility 
    calculations up to March 4, 2008 are end of day prices. Starting from 
    March 5, 2008 we have been capturing best bid and best offer as close 
    to 4 o’clock as possible to better synchronize the option price with 
    the underlying close. Currently all option quotes are captured at 15:59 ET. 
    The underlying price used is the official (composite) close.    
    """
    start_date, end_date = with_default_dates(start_date, end_date)
    
    # The v1 approach is error-prone as requires the use to think about removing the cache when dates
    # are not aligned.
    #v1: cache = 'secprd_%d.pkl'%secid if cache else None
    #v1: data = None if cache is None else load_pickle(cache)
    #v1: if data is not None:
    #v1:     return between(data, start_date, end_date)
    #v1:

    # Something building on the lines of the v2 approach would be burdensome as
    # (e.g.) the last business date (last_date) will likely be before the last
    # provided calendar date (end_date), rendering the cache useless.
    #v2: first_date = data.date.min()
    #v2: last_date = data.date.max()
    #v2: if (start_date >= first_date) and (end_date <= last_date):
    #v2:     return between(data, start_date, end_date)        

    # As this is not the bottleneck, we'll just avoid caching for now

    #v1: # Inexistent cache; fetch data
    underlying = pd.DataFrame()
    for year in range(start_date.year, end_date.year+1):
        ydata = db.raw_sql("SELECT * FROM optionm.secprd%d WHERE secid = %i"%(year,secid))
        underlying = pd.concat((underlying, ydata), axis=0)
        
    #v1: if cache is not None:
    #v1:     dump_pickle(underlying, cache)
    return between(underlying, start_date, end_date)


def fetch_opprcd(secid, start_date=None, end_date=None, cache=True):
    """Fetch option prices from OptionMetrics

    WARNING: Use start_date and end_date carefully when the cache does not exist
    """
    start_date, end_date = with_default_dates(start_date, end_date)

    cache = ('opprcd%d_'+'%d.pkl'%secid) if cache else None    
    options = pd.DataFrame()
    for year in range(start_date.year, end_date.year+1):
        ydata = None if cache is None else load_pickle(cache%year)        
        if ydata is None:
            # Either cache=False in arguments, or the cache has not been created yet
            ydata = db.raw_sql("SELECT * FROM optionm.opprcd%d WHERE secid = %i"%(year,secid))
            if cache is not None:
                dump_pickle(ydata, cache%year)
        options = pd.concat((options, ydata), axis=0)
    
    return between(options, start_date, end_date)


# def fetch_vsurfd(secid, start_date=start_date, end_date=end_date):
#     """Fetch the volatility surface as interpolated by OptionMetrics"""
#     options = pd.DataFrame()
#     for year in range(start_date.year, end_date.year+1):
#         columns = 'secid,date,days,delta,cp_flag,impl_volatility,impl_strike,dispersion'
#         ydata = db.raw_sql("SELECT %s FROM optionm.vsurfd%d WHERE secid = %i"%(columns,year,secid))
#         options = pd.concat((options, ydata), axis=0)
#     #return between(options, start_date, end_date).reset_index(drop=True)
#     return options


#v0: def fetch_distrd(secid, cache=True):
def fetch_distrd(secid):
    """Fetch dividend information for the underlying.

    This method illustrates how to handle dividends. It treats all secid's except those in global `secid_of_indices` as if they were single name equity. This could be improved (or we can continue population `secid_of_indices`).
    """
    global start_date, end_date
    if secid in secid_of_indices:
        data = db.raw_sql("SELECT * FROM optionm.idxdvd WHERE secid = %i"%(secid))
        return between(data, start_date, end_date)

    else:
        data = db.raw_sql("SELECT * FROM optionm.distrd WHERE secid = %i"%(secid)) #.rename(columns={'payment_date':'date'})
        data = data[data['amount'] > 0] # Keep only positive dividends

        # For single name equities, we return all dividends, since we will need to aggregate them conditioning on option maturities
        return data

def format_opprcd(options):
    """Assuming that `options` is raw output from optionm.opprcdYYYY, add standard fields.    
    
    Adds:

    options['hash']
        to compare the results from queries at different point in time
    options['strike'] = options['strike_price'] / 1000
        for convenience
    options['option_price'] = 0.5*(options['best_bid']+options['best_offer'])
        as per usual convention
    options['is_call'] = options['cp_flag']=='C'
        easier to deal with booleans than string flags in some contexts
    """    
    options = options.sort_values(by=['date','exdate','strike_price','am_settlement'])

    # DEV: This looks like the cleanest thing to do, but it unfortunately break many functions that were
    #      assuming date and exdate to be dt.date objects, not dt.datetime
    if False:
        options['date'] = [dt.datetime.combine(date, dt.time(16,0)) for date in options.date]
        
        am_sett = [dt.datetime.combine(date, dt.time(9,30)) for date in options.exdate]
        pm_sett = [dt.datetime.combine(date, dt.time(16,0)) for date in options.exdate]
        options['exdate'] = np.where(options.am_settlement, am_sett, pm_sett)    
    
    options['hash'] = pd.util.hash_pandas_object(options)
    options['strike'] = options['strike_price'] / 1000 # strike_price is 1000*strike in OptionMetrics
    options['option_price'] = 0.5*(options['best_bid']+options['best_offer'])
    options['is_call'] = options['cp_flag']=='C'
    assert np.all( options['is_call'] | (options['cp_flag']=='P') )

    return options

def get_option_data(secid, date):
    """Fetch the option data from OptionMetrics, along with other pricing inputs
    
    Note that:

    ```
    dates = db.raw_sql("SELECT DISTINCT date FROM optionm.opprcd%d WHERE secid=%i"%(year,secid))
    ```

    provides quick access to unique dates in `year`. One can then iterate those date and call 
    `get_option_data` on each date.
    """
    wrds_connect()
    if not isinstance(date, str):
        date = date2str(date)
        
    # First fetch the raw option data
    year = int(date[:4]) # first 4 chars of '%Y/%m/%d'
    query = "SELECT * FROM optionm.opprcd%d WHERE secid=%i AND date='%s'"
    options = format_opprcd( db.raw_sql(query%(year,secid,date)) )
        
    # Then, the info about the underlying
    query = "SELECT * FROM optionm.secprd%d WHERE secid=%i AND date='%s'"
    underlying = db.raw_sql(query%(year,secid,date))
    underlying['date'] = pd.to_datetime(underlying['date']).dt.date
    
    if secid in secid_of_indices:
        query = "SELECT * FROM optionm.idxdvd WHERE secid=%i AND date='%s'"
        dividends = db.raw_sql(query%(secid,date))
    else:
        # Fetch all (positive) dividends whose ex_date falls between valuation date and last option expiration
        start_date = pd.to_datetime(date).strftime('%Y-%m-%d')
        last_exdate = options.exdate.max()
        end_date = pd.to_datetime(last_exdate).strftime('%Y-%m-%d')
        query = (
            "SELECT * FROM optionm.distrd "
            "WHERE secid = {secid} AND ex_date BETWEEN '{start}' AND '{end}'"
        )
        dividends = db.raw_sql(query.format(secid=secid, start=start_date, end=end_date))
        dividends = dividends[dividends['amount'] > 0]
        if dividends.empty:
            dividends = pd.DataFrame(columns=['secid','ex_date','amount','currency','seq_num'])
        else:
            assert_unique(dividends.seq_num)==1.0

    data = merge_options_and_underlying( (fetch_zerocd(), underlying, options, dividends) )
    return data

class SmileIterator:
    @staticmethod # For now, I don't see the pattern that would justify a classmethod
    def implied_forward_price(options):
        all_smiles = []
        for sm in Smile.iterator(options):
            sm.set_implied_forward_price()
            all_smiles.append(sm.to_dict())
        return pd.DataFrame(all_smiles)
    
    def __init__(self, smile_cls, options):
        assert np.unique(options.secid).size==1, 'Use one SmileIterator per secid'
        
        self.smile_cls = smile_cls
        self.options = options.sort_values(['date', 'exdate'])
        self.calendar = np.unique(self.options.date)
        self.date_index = 0
        self.exdate_index = 0

        self._reset_date()
        
    def _reset_date(self):
        """Must be called each time we swith from one date to another.

        The _reset_date method is used to reset the instance variables related to the current 
        date when moving to the next date. This ensures that the filtering for the new date is 
        only done once, the next time the __next__ method is called.
        """
        self.date = None
        self.dopt = None
        self.exdt = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.date_index >= len(self.calendar):
            raise StopIteration

        if self.date is None:
            self.date = self.calendar[self.date_index]
            self.dopt = self.options[self.options.date == self.date]
            self.exdt = np.unique(self.dopt.exdate)

        if self.exdate_index >= len(self.exdt):
            self.date_index += 1
            self.exdate_index = 0
            self._reset_date()
            return self.__next__()
            
        exdate = self.exdt[self.exdate_index]
        self.exdate_index += 1

        return self.smile_cls(self.date, exdate, self.dopt[self.dopt.exdate == exdate])

TODO = """TO DO:
- Add VXO style ATM vol. For moneyness comparisons across maturities, we'd also need a single 
  measure of vol across maturities (T is dealt with otherwise in sigma*sqrt(T))

- Write a class OptionData(pd.DataFrame) that leverages the SmileIterator to iterate date's 
  efficiently, handling the caching of results at some reasonable interval to be able to interupt 
  and resume processing of OptionMetrics
"""
warnings.warn(TODO)
    
class Smile:
    """Operations on a dataframe of options with unique (t,T).

    This class defines several classmethods operating or option "smiles". The is_smile method,     
    ```
    @classmethod
    def is_smile(cls,options):
        return (np.unique(options.date).size == 1) \
            and (np.unique(options.exdate).size == 1)
    ```
    illustrates that any dataframe with a unique observation and expiration date can be considered a smile.

    Also noteworthy, the iterator classmethod accepts any options dataset and returns an iterator looping
    on every pair (t,T) in its options argument. That iterator yields and instance of the smile class, 
    that has, among others, attributes `date`, `exdate`, and `options`, where the latter is the subset 
    of options in the (t,T)-smile. This allows the user to iterate through the various smiles of any 
    option dataset (ordered first by `date`, then by `exdate`).

    One of the features of a Smile instance is that it keeps track of all attributes added by the user
    after the creation of the instance. This allows the following usage:
    ```
    sm = Smile(date,exdate,options[(options.date==date) & (options.exdate==exdate)])
    sm.set_implied_forward_price()
    d = sm.to_dict()
    ```
    and the content of `d` would be (e.g.)
    ```
    {'DTM': 564.0,
     'YTM': 1.5452054794520549,
     'date': numpy.datetime64('1996-12-03T00:00:00.000000000'),
     'exdate': numpy.datetime64('1998-06-20T00:00:00.000000000'),
     'implied_forward_price': 788.6796029125583
    }
    ```
    This can prove particularly convenient to create dataframes with one entry per smile, e.g.:
    ```
    def implied_forward_price(options):
        all_smiles = []
        for sm in Smile.iterator(options):
            sm.set_implied_forward_price()
            all_smiles.append(sm.to_dict())
        return pd.DataFrame(all_smiles)
    ```
    (cf. SmileIterator.implied_forward_price)
    """    
    N_MIN_PAIRS = 5
    
    @classmethod
    def is_smile(cls,options):
        return (np.unique(options.secid).size == 1) \
            and (np.unique(options.am_settlement).size == 1) \
            and (np.unique(options.date).size == 1) \
            and (np.unique(options.exdate).size == 1)

    @classmethod
    def get_smile(cls, date, exdate, options):
        return cls(date, exdate, options[(options.date==date) & (options.exdate==exdate)])
    
    @classmethod
    def iterator(cls,options):
        """Iterates through the smiles in the options dataframe.

        This allows, for example, to gather info at the smile level and then combine it back into 
        a DataFrame:
        ```
        def implied_forward_price(options):
            all_smiles = []
            for sm in Smile.iterator(options):
                sm.set_implied_forward_price()
                all_smiles.append(sm.to_dict())
            return pd.DataFrame(all_smiles)
        ```
        (cf. SmileIterator.implied_forward_price)
        """
        return SmileIterator(cls, options)
        # 
        # chk = options.sort_values(['date','exdate'])
        # assert np.all(options.index == chk.index), f"The {cls.__class__.__name__}.iterator expects " \
        #     + "options to be sorted by (date,exdate) (i.e. options.sort_values(['date','exdate'])). " \
        #     + "Otherwise, collapsing smiles together will not yield observations consistent with " \
        #     + "the original data."
        # 
        # calendar = np.unique(options.date)
        # for dn,date in enumerate(calendar):
        #     opt = options[options.date == date]
        #     expiration = np.unique(opt.exdate)
        #     for en,exdate in enumerate(expiration):
        #         yield cls(date,exdate,opt[opt.exdate==exdate])

    @classmethod
    def parallel_build(cls,options,properties):
        #from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import ProcessPoolExecutor
        
        tic()
        smile_list = []
        with ProcessPoolExecutor(max_workers=POOL_SIZE) as executor:
            futures = [executor.submit(Smile.set_properties, sm, properties) for sm in Smile.iterator(options)]
            smile_list = [f.result() for f in futures]   
        dt = toc(do_print=False)        
        print("Elapsed time: %f seconds -- %f per smile."%(dt,dt/len(smile_list)))
        return smile_list
        
                
    @classmethod
    def get_implied_forward_price(cls,options,bid_ask=False):
        assert cls.is_smile(options), 'Options have more than one (t,T) pair'
        pair_cols = ['YTM','risk_free','strike','option_price','best_bid','best_offer']
        
        #import pdb; pdb.set_trace();
        nn = ~np.isnan(options.option_price)
        put = options[nn & ~options.is_call]
        call = options[nn & options.is_call]
        pairs = put[pair_cols].merge(
                call[pair_cols], on=pair_cols[:3], suffixes=('_p', '_c'))#.sort_values(by='strike')    
        if pairs.shape[0] < cls.N_MIN_PAIRS:
            if bid_ask:
                return np.nan, np.nan, np.nan
            return np.nan # Leave the forward price to NaN
            
        # Choose the ATM pair
        mx = np.argmin(np.abs( pairs['option_price_p'] - pairs['option_price_c'] ))
        ATM = pairs.iloc[mx]

        # Compute the implied forward price
        # Put-call parity:     S - D - K e^{-rT} = c - p
        #                  <=> F = K + e^{+rT} (c - p)
        F = (ATM.strike 
             + np.exp(ATM.risk_free*ATM.YTM)*(ATM.option_price_c - ATM.option_price_p)).astype(np.float64)
        if not bid_ask:
            return F

        # In a long position, we buy the call (at worst) at the ask and sell the put (at worst) at the bid
        longF = (ATM.strike 
                 + np.exp(ATM.risk_free*ATM.YTM)*(ATM.best_offer_c - ATM.best_bid_p)).astype(np.float64)
        
        # In a short position, the bid-ask are reversed
        shortF = (ATM.strike 
                  + np.exp(ATM.risk_free*ATM.YTM)*(ATM.best_bid_c - ATM.best_offer_p)).astype(np.float64)

        return pairs, shortF, F, longF
    
    def __init__(self,date=None,exdate=None,options=None,**kwargs):
        """Constructs a (t,T) Smile instance with t=date and T=exdate
        
        The options argument must be provided (a default was needed for the Python method
        signature, but would raise an AssertionError)

        If date or exdate are None, they will be assigned the unique value from options.
        """
        try:
            assert options is not None and self.is_smile(options)
        except:
            import pdb; pdb.set_trace()
        if date is None:
            date = np.unique(options.date)[0]
        if exdate is None:
            exdate = np.unique(options.exdate)[0]
            
        self.date = date
        self.exdate = exdate
        self.options = options.sort_values(['date','exdate'])

        # This field keeps track of the original columns in the options db
        # This could prove handy if we want to save only the columns added herein 
        self._options_columns = [cname for cname in self.options.columns]

        #import pdb; pdb.set_trace()
        if date==exdate:
            self.DTM = 0
        else:
            try:
                # self.DTM = (exdate-date)/np.timedelta64(1,'D')
                td = exdate - date
                self.DTM = td.days + td.seconds/(24*60*60)
            except:
                import pdb; pdb.set_trace()                
        self.YTM = self.DTM / 365
        
        # The list of the attributes to be returned by to_dict 
        # Any attribute added to a Smile instance after its creation (here) will 
        # be added to this list
        self.dict_keys = ['date','exdate','YTM','DTM']
        
    def to_dict(self, scalar=False):
        """Return a dictionary describing the smile.
            
        One of the features of a Smile instance is that it keeps track of all attributes added by the user
        after the creation of the instance. This allows the following usage:
        ```
        sm = Smile(date,exdate,options[(options.date==date) & (options.exdate==exdate)])
        sm.set_implied_forward_price()
        d = sm.to_dict()
        ```
        and the content of `d` would be (e.g.)
        ```
        {'DTM': 564.0,
        'YTM': 1.5452054794520549,
        'date': numpy.datetime64('1996-12-03T00:00:00.000000000'),
        'exdate': numpy.datetime64('1998-06-20T00:00:00.000000000'),
        'implied_forward_price': 788.6796029125583
        }
        ```
        """    
        def colname(name):
            if name.startswith('_'):
                return name[1:]
            return name

        D = {}
        for name in self.dict_keys:
            attr = getattr(self,name)
            if not scalar or np.isscalar(attr):
                D[colname(name)] = attr
        return D                
        #return {colname(name):getattr(self,name) for name in self.dict_keys}

    def to_dataframe(self):
        """Return a dataframe combining the smile's options with the smile's additional attribute.

        Essentially returns a dataframe [self.options, pd.DataFrame(self.to_dict())].
        """
        more = pd.DataFrame(self.to_dict())
        assert self.options.shape[0] == more.shape[0]

        keys = [c for c in more.columns if c in self.options.columns]        
        assert np.all(self.options[keys].values == more[keys].values)
        
        right_keys = [c for c in more.columns if c not in keys]        
        more = more[right_keys].set_index(self.options.index)
        df = pd.concat((self.options, more), axis=1)
        assert df.shape[0]==self.options.shape[0]
        return df    
    
    def __setattr__(self,name,value):
        if hasattr(self,'dict_keys') and not name.startswith('_'+self.__class__.__name__):
            self.dict_keys.append(name)
        super().__setattr__(name,value)

    # @property
    # def DTM(self):
    #     return (self.exdate-self.date)/np.timedelta64(1,'D')

    def set_properties(self, properties):
        for prop in properties:
            if isinstance(prop, tuple):                
                method = getattr(self,f'{prop[0]}')
                method(*prop[1:])
            else:
                getattr(self,f'{prop}')
        return self       
    def set_properties_old(self, properties):
        for name in properties:
            getattr(self,f'{name}')
            #setter = getattr(self,f'{name}')
            #setter()
        return self
    def set_forward(self, underlying, dividends):
        self.stock_price = underlying.loc[self.date,'close']
        div_rate = dividends.loc[self.date,'rate']/100        
        r_t_T = self.options.risk_free.iloc[0]
        
        self.stock_exdiv = self.stock_price*np.exp(-div_rate*self.YTM)
        self.forward = self.stock_exdiv*np.exp(r_t_T*self.YTM)
            
    def get_calls(self):
        return self.options[self.options.is_call]

    def get_puts(self):
        return self.options[~self.options.is_call]

    @property
    def implied_vol_exdiv(self):
        if not hasattr(self,'_implied_vol_exdiv'):
            option_price = self.options.option_price
            initial_guess = self.options.impl_volatility.values.copy()
            nx = np.isnan(initial_guess)
            initial_guess[nx] = np.where(option_price[nx]==0, 0.0, 0.2)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                IV = bms.implied_volatility(option_price,
                        self.stock_exdiv, self.options.strike, self.options.risk_free, 0.0, self.YTM,
                        self.options.is_call, initial_guess)
    
            price_bms = bms.option_price(self.stock_exdiv, self.options.strike,
                self.options.risk_free, 0.0, self.YTM, IV, self.options.is_call)            
            error = np.abs(price_bms - option_price) > 0.005
            IV[error] = np.nan
            self._implied_vol_exdiv = IV
            
        return self._implied_vol_exdiv

    @property
    def implied_vol_bms(self):
        if not hasattr(self,'_implied_vol_bms'):
            option_price = self.options['option_price']
            self._implied_vol_bms = implied_vol_bms(self.options, option_price, # global function
                                        implied_forward_price=self.implied_forward_price) 
        return self._implied_vol_bms
    
    @property
    def implied_vol_bid(self):
        if not hasattr(self,'_implied_vol_bid'):
            option_price = self.options['best_bid']
            self._implied_vol_bid = implied_vol_bms(self.options, option_price, # global function
                                        implied_forward_price=self.implied_forward_price) 
        return self._implied_vol_bid

    @property
    def implied_vol_ask(self):
        if not hasattr(self,'_implied_vol_ask'):
            option_price = self.options['best_offer']
            self._implied_vol_ask = implied_vol_bms(self.options, option_price, # global function
                                        implied_forward_price=self.implied_forward_price) 
        return self._implied_vol_ask

    @property
    def implied_forward_price(self):
        self.set_implied_forward_price()
        return self._implied_forward_price

    def set_implied_forward_price(self):
        if not hasattr(self,'_implied_forward_price'):
            self.__pairs, self.short_ifp, self._implied_forward_price, self.long_ifp = \
                                    self.__class__.get_implied_forward_price(self.options, bid_ask=True)

def get_options_and_underlying(secid, cache=True, pool_size=POOL_SIZE):
    cache = 'secid%d_%s_%s.pkl'%(secid,start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')) \
                if cache else None    
    data = None if cache is None else load_pickle(cache)
    if data is not None:
        return data

    print('get_options_and_underlying --',cache)
    
    wrds_connect() # db accessible as om.db
    options = format_opprcd( fetch_opprcd(secid) )
    if options.empty:
        # There might be no options for a given equity during the requested period
        return pd.DataFrame([],columns=['date'])

    underlying = fetch_secprd(secid).sort_values(by='date')
    try:
        dividends = fetch_distrd(secid).sort_values(by='ex_date')
    except AttributeError:
        dividends = pd.DataFrame([], columns=["secid","record_date","seq_num","ex_date","amount","adj_factor","declare_date","payment_date","link_secid","distr_type","frequency","currency","approx_flag","cancel_flag","liquid_flag"])
    
    options = merge_datasets(fetch_zerocd(), underlying, dividends, options, pool_size)
    if cache is not None:
        dump_pickle(options, cache)
    return options


def merge_datasets(zero_curve, underlying, dividends, options, pool_size=POOL_SIZE):
    if pool_size > 1:
        print('Dispatching data preprocessing on %d workers...'%pool_size)

    # Split the options in approximatly equal subsets, without spliting a date over two db
    assert np.min([d.days for d in np.diff(options.date)])==0
    options = options.sort_values('date').copy()
    options.set_index(np.arange(0,options.shape[0]), inplace=True)
    
    # The dividend logic is different for index options and equity options.
    # We can filter index dividends by date, but for equity options we need all dividends between [date, exdate].
    # So, we can't filter equity dividends by date here.
    secid = int(assert_unique(options.secid))
    #YS: if secid in secid_of_indices:
    databases = zero_curve, underlying, options, dividends
    #YS: else:
    #YS:     databases = zero_curve, underlying, options
    
    n_dates = len( options.date.unique() )
    if n_dates==1 or POOL_SIZE==1:
        db_slice = [ databases ]

    #YS: # If we just select a handful of dates, and the number of unique dates is less than pool_size,
    #YS: # then the first date and last date will be duplicated.
    #YS: elif n_dates <= pool_size:
    #YS:     db_slice = []
    #YS:     for date in options.date.unique():
    #YS:         db = []
    #YS:         for data in databases:
    #YS:             dx = (pd.to_datetime(data.date).dt.date == date)
    #YS:             db.append( data[dx] )
    #YS:         if secid not in secid_of_indices:
    #YS:             db.append( dividends ) # Keep all dividends for equity options
    #YS:         db_slice.append( tuple(db) )
    else:
        obs = np.ceil(np.linspace(0,options.shape[0],pool_size)).astype(int)
        lb = obs[0]
        
        for ub in obs[1:]:
            first_date = options['date'].iloc[lb]
            last_date = options['date'].iloc[ub-1]
            db = []
            for data in databases:
                dx = (first_date <= pd.to_datetime(data['date']).dt.date) & (pd.to_datetime(data['date']).dt.date <= last_date)
                db.append( data[dx] )
            if secid not in secid_of_indices:
                db.append( dividends ) # Keep all dividends for equity options
            db_slice.append( tuple(db) )
            
            sx = options['date'] > last_date
            if np.sum(sx):
                lb = options[sx].index[0]
                assert lb > ub
    
    #import pdb; pdb.set_trace()
    if POOL_SIZE > 1:
        print('Creating pool')    
        from multiprocessing import Pool
        with Pool(pool_size) as p:
            res = p.map(try_merge_options_and_underlying, db_slice)
    else:
        res = []
        for db in db_slice:
            res.append(merge_options_and_underlying(db))
    return pd.concat(res, axis=0)


def try_merge_options_and_underlying(z_u_o_d):
    try:
        return merge_options_and_underlying(z_u_o_d)
    except:
        print('z_u_o_d shapes: %s, %s, %s, %s'&tuple([db.shape for db in z_u_o_d]))

def merge_options_and_underlying(z_u_o_d):
    zero_curve, underlying, options, dividends = z_u_o_d
    # If the selected number of dates is small, there might empty dataframes in the z_u_o_d tuple.
    if options.empty:
        return pd.DataFrame([],columns=['date'])
    
    secid = int(assert_unique(options.secid))
    rf = interpolate_zero_curve(zero_curve, options.date, options.exdate)
    options = pd.concat((options, rf.drop(columns=['date','exdate'])), axis=1)
    
    # Ensure all the dates in options are the same type
    options['date'] = pd.to_datetime(options['date']).dt.date
    options['exdate'] = pd.to_datetime(options['exdate']).dt.date
        
    calendar = np.unique(options['date'])
    # This block is merely dealing with cache management
    start_date = calendar[0]
    end_date = calendar[-1]
    # import pdb; pdb.set_trace()
    secid_str = "secid%d"%secid
    dirname = os.path.join("data","option_metrics",secid_str)
    cache_name = os.path.join(secid_str, "merged_%s_%s.pkl"%(
        start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
    )
    os.makedirs(dirname,exist_ok=True)
    if os.path.exists(cache_name):
        return load_pickle(cache_name)
        
    options['stock_price'] = np.nan
    options['stock_exdiv'] = np.nan
    options['implied_forward_price'] = np.nan

    for date in calendar:
        dx = options.date == date
        S_t = underlying.loc[underlying.date == date,'close'].iloc[0]
        options.loc[dx,'stock_price'] = S_t

        daily = options[dx]     
        maturities = np.unique(daily['YTM'])
        for ytm in maturities:
            sx = dx & (options.YTM == ytm)
            if secid in secid_of_indices:
                div_rate = dividends.loc[dividends['date'] == date,'rate'].iloc[0]
                options.loc[sx,'stock_exdiv'] = S_t*np.exp(-ytm*div_rate/100)

            else: # Single-name equity
                # Select only dividends with ex-dividend date in (date, exdate]
                exdt = assert_unique(options.loc[sx,'exdate'])
                pv_dividends = 0.0
                div_till_exdate = between(dividends, date, exdt, date_col='ex_date').copy()
                if not div_till_exdate.empty:
                    # Even if what matters, in terms of timing, is the ex-dividend date, the cash flow will come only at the payment_date
                    div_till_exdate['date'] = date # All interpolation between now and payment
                    div_zc = interpolate_zero_curve(
                        zero_curve, div_till_exdate['date'], div_till_exdate['payment_date'])
                    present_value = np.exp(-div_zc['risk_free']*div_zc['YTM']) * div_till_exdate['amount']
                    pv_dividends = present_value.sum()

                # The ex-dividend stock price is the current stock price 
                #   minus the PV of all dividends to be paid before expiration
                options.loc[sx,'stock_exdiv'] = S_t - pv_dividends
                
            settlement = np.unique(options[sx].am_settlement)
            for settl in settlement:
                ssx = sx & (options.am_settlement==settl)
                try:
                    options.loc[ssx,'implied_forward_price'] = \
                        Smile.get_implied_forward_price( options[ssx] )
                except Exception as err:
                    from .toolkit import printdf
                    printdf(options[ssx])
                    raise err        
        
    options['implied_vol_bms'] = implied_vol_bms(options, options['option_price'])
    options['implied_vol_bid'] = implied_vol_bms(options, options['best_bid'])
    options['implied_vol_ask'] = implied_vol_bms(options, options['best_offer'])
    
    dump_pickle(options, cache_name)
    return options


#OLD: def zero_curve_optionm(dates, exdates, rates)
def interpolate_zero_curve(zcurve, dates, exdates):
    '''Interpolate rates in zcurve on each date in dates, for horizon exdates-dates

    zcurve must have been obtained from fetch_zerocd, or have the same columns as if it were.
    '''
    from scipy.interpolate import interp1d
    # Make sure the dates are datetime64[ns]
    dates = pd.to_datetime(dates)
    exdates = pd.to_datetime(exdates)
    zcurve['date'] = pd.to_datetime(zcurve['date'])
    
    rf_rate = pd.DataFrame({'date': dates, 'exdate':exdates,'DTM': (exdates - dates).dt.days})
    rf_rate['YTM'] = rf_rate['DTM'] / 365
    rf_rate['risk_free'] = np.full(dates.shape, np.nan)

    zc_dates = zcurve.date.unique()
    for day in dates.unique():
        # Find the last day in the zero curve that preceeds or is equal to the "current" day
        zc_day = zc_dates[zc_dates <= day].max()

        # Fetch rates for this day and sort on DTMs
        rf = zcurve[zcurve.date == zc_day].sort_values('days')

        if rf.shape[0] > 0:
            dx = rf_rate.date==day
            
            # Fetch rates for this day and sort on DTMs
            rzero = pd.DataFrame([[zc_day, 0, rf.rate.iloc[0]]], columns = ['date','days','rate'])
            rf = pd.concat((rzero,rf), axis=0)
            interp = interp1d(rf.days, rf.rate / 100, kind = 'linear')

            dtm = rf_rate.DTM[dx].values
            zc_max = np.max(rf.days)
            if zc_max < np.max(dtm):
                warnings.warn(
                    "On %s, flat extrapolation of the zero-curve to %d days, beyond provided %d days"%(
                    date2str(day), np.max(dtm), zc_max))
                dtm[dtm > zc_max] = zc_max
            
            rf_rate.loc[dx,'risk_free'] = interp(dtm)

    return rf_rate

def implied_vol_bms(options, option_price, **kwargs):
    """Use Black-Merton-Scholes to compute an implied volatility.

    OptionMetrics provides its own mesure of IV. [TO BE COMPLETED]
    """
    initial_guess = kwargs.pop('initial_guess',None)
    if initial_guess is None:
        # There are <NA> in impl_volatility, we must convert it to np.nan
        options['impl_volatility'] = options['impl_volatility'].astype(float)
        initial_guess = options['impl_volatility'].values.copy()
        nx = np.isnan(initial_guess)
        initial_guess[nx] = np.where(option_price[nx]==0, 0.0, 0.2)

    implied_forward_price = kwargs.pop('implied_forward_price',None)
    if implied_forward_price is None:
        implied_forward_price = options['implied_forward_price']
        
    # Rather than using the spot and OptionMetrics' dividend rate, we here use the ex-dividend
    # price of the underlying as implied by (i) the options, via the VIX-like option-implied
    # forward price and (ii) the zero-curve provided by option metrics
    implied_exdiv = np.exp(-options['risk_free']*options['YTM'])*implied_forward_price
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        implied_vol_bms = bms.implied_volatility(option_price,
            implied_exdiv, options['strike'], options['risk_free'], 0.0, options['YTM'],
            options['is_call'], initial_guess)
    
    price_bms = bms.option_price(
        implied_exdiv, options['strike'], options['risk_free'], 0.0, options['YTM'],
        implied_vol_bms, options['is_call'])
    error = np.abs(price_bms - option_price) > 0.005
    implied_vol_bms[error] = np.nan
    return implied_vol_bms.astype(np.float64)

def subsample_dtm(options, date, DTM):
    surface = options[options.date == date]

    maturities = np.unique(surface.DTM)
    bx = maturities <= DTM
    below = -1 if not np.any(bx) else maturities[bx].max()
    ax = maturities > DTM
    above = -1 if not np.any(ax) else maturities[ax].min()
    
    return surface[np.isin(surface.DTM, [below, above])]


class SurfaceIterator:
    def __init__(self, surface_cls, options):
        assert np.unique(options.secid).size==1, 'Use one SurfaceIterator per secid'
        
        self.surface_cls = surface_cls
        self.options = options.sort_values(['date', 'exdate'])
        self.calendar = np.unique(self.options.date)
        self.date_index = 0

        self._reset_date()
        
    def _reset_date(self):
        """Must be called each time we swith from one date to another.

        The _reset_date method is used to reset the instance variables related to the current 
        date when moving to the next date. This ensures that the filtering for the new date is 
        only done once, the next time the __next__ method is called.
        """
        self.date = None
        self.dopt = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.date_index >= len(self.calendar):
            raise StopIteration

        self.date = self.calendar[self.date_index]
        self.date_index += 1
        return self.surface_cls(self.date, self.options[self.options.date == self.date])

class Surface:
    @classmethod
    def is_surface(cls,options):
        return (np.unique(options.secid).size == 1) \
            and (np.unique(options.date).size == 1)

    @classmethod
    def iterator(cls,options):
        """Return an iterator over the surfaces in the options dataframe."""
        return SurfaceIterator(cls, options)

    @classmethod
    def apply(cls, options, method, *args, **kwargs):
        """Call method on each surface in the options dataframe."""
        results = []
        for surface in Surface.iterator(options):
            func = getattr(surface, method)
            results.append( func(*args, **kwargs) )
        return pd.concat(results, axis=0)

    @classmethod
    def apply_to_tuple(cls, arg_tup):
        return cls.apply(arg_tup[0], arg_tup[1])
    
    @staticmethod
    def slice(options, n_slices=POOL_SIZE):
        """Split the options in approximatly equal subsets, without spliting a date over two slices."""
        assert np.min([d.days for d in np.diff(options.date)])==0, "The options.date must be sorted"

        # Keep a copy of the index and set observation number as index
        index = options.index.copy()
        options.set_index(np.arange(0,options.shape[0]), inplace=True)

        # Default slice on observation numbers
        obs = np.ceil(np.linspace(0,options.shape[0],n_slices)).astype(int)

        lb = obs[0]
        sliced_db = []
        for ub in obs[1:]:
            first_date = options['date'].iloc[lb]
            last_date = options['date'].iloc[ub-1]

            #FUTURE: see merge_datasets -> manage a for loop on DBs coordinated with the options 
            dx = (first_date <= options['date']) & (options['date'] <= last_date)
            sliced_db.append( options[dx] )
            
            sx = options['date'] > last_date
            if np.any(sx):
                lb = options[sx].index[0]
                assert lb > ub
        return sliced_db

    @staticmethod
    def map(options, method):
        """Wrap parallelization in a simple staticmethod."""        
        is_slice = isinstance(options,list) and np.all([isinstance(opt,pd.DataFrame) for opt in options])
        if is_slice:
            sliced_db = [options] # lazy copy
        else:
            assert isinstance(options,pd.DataFrame), "options must be a DataFrame or a sliced DataFrame"
            sliced_db = Surface.slice(options)

        pool_size = len(sliced_db)
        if pool_size > 1: # False: #
            from multiprocessing import Pool
            with Pool(pool_size) as p:
                res = p.map(Surface.apply_to_tuple, [(opt,method) for opt in sliced_db])
        else:
            import pdb, sys
            try:
                res = []
                for db in sliced_db:
                    res.append( Surface.apply_to_tuple((db,method)) )
            except:
                pdb.post_mortem(sys.exc_info()[-1])

        return pd.concat(res, axis=0)

    @staticmethod
    def nearest_maturities(surface, target_dtm):
        Maturities = np.unique(surface.DTM)
        bx = maturities <= target_dtm
        below = -1 if not np.any(bx) else maturities[bx].max()
        ax = maturities > target_dtm
        above = -1 if not np.any(ax) else maturities[ax].min()
    
        return surface[np.isin(surface.DTM, [below, above])]

    
    def __init__(self,date=None,options=None,**kwargs):
        """Constructs a time-t Surface instance with t=date
        
        The options argument must be provided (a default was needed for the Python method
        signature, but would raise an AssertionError)

        If date is None, it will be assigned the unique value from options.
        """
        try:
            assert options is not None and self.is_surface(options)
        except:
            import pdb; pdb.set_trace()
        if date is None:
            date = np.unique(options.date)[0]
            
        self.date = date
        self.options = options.sort_values(['date','exdate'])

        # This field keeps track of the original columns in the options db
        # This could prove handy if we want to save only the columns added herein 
        self._options_columns = [cname for cname in self.options.columns]

        #TBA? # The list of the attributes to be returned by to_dict 
        #TBA? # Any attribute added to a Smile instance after its creation (here) will 
        #TBA? # be added to this list
        #TBA? self.dict_keys = ['date']

    def regress_dfw96(self):
        columns = ['implied_vol_bms', 'YTM', 'strike', 'implied_forward_price']
        surface = self.options[columns].dropna(axis=0).reset_index()
        YTM = surface.YTM.values
        MNY = np.log(surface.strike / surface.implied_forward_price).values
        RHS = np.array([MNY, MNY**2, YTM, YTM**2, MNY*YTM]).T
        RHS = pd.DataFrame(RHS - np.mean(RHS,axis=0), columns=['MNY', 'MNY**2', 'YTM', 'YTM**2', 'MNY*YTM'])

        import statsmodels.api as sm
        model = sm.OLS(surface.implied_vol_bms, sm.add_constant(RHS))
        results = model.fit()

        #import pdb; pdb.set_trace()
        res = pd.concat((pd.DataFrame(results.params).rename(columns={0:'est'}),
                         pd.DataFrame(results.conf_int()).rename(columns={0:'lb',1:'ub'})), axis=1)
        res.at['R2','est'] = results.rsquared
        res.index.name = 'RHS' # res.reset_index()        
        res['date'] = self.date

        #import pdb; pdb.set_trace()
        return res
        
#not used def subsample_strike(surface, strike):
#not used     strikes = np.unique(surface.strike)
#not used     below = strikes[strikes <= DTM].max()
#not used     above = strikes[strikes > DTM].min()
#not used     
#not used     return surface[np.isin(surface.strike, [below, above])]

def dump_pickle(data, fname):
    import pickle
    path = os.path.join('data','option_metrics')
    os.makedirs(path,exist_ok=True)

    ff = os.path.join(path,fname)
    with open(ff,'wb') as fh:
        pickle.dump(data, fh)

def load_pickle(fname):
    import pickle
    path = os.path.join('data','option_metrics')
    ff = os.path.join(path,fname)
    if not os.path.exists(ff):
        return None
    
    with open(ff,'rb') as fh:
        return pickle.load(fh)

def dev_merge_cache():
    options = []
    for year in range(1996,2012):    
        filename = 'data/option_metrics/%d0101_%d1231.pkl'%(year,year)
        with open(filename,'rb') as fh:
            options.append(pickle.load(fh))
    options = pd.concat(options,axis=0)

    filename = 'data/option_metrics/secid108105_1996101_20111231.pkl'
    with open(filename,'wb') as fh:
        pickle.dump(options, fh)        


