import datetime as dt
import numpy as np
import yfinance as yf

# yf_ticker="^GSPC"
# ticker = yf.Ticker(yf_ticker)
# prices = ticker.history(interval='1d',start=dt.datetime(1900, 1, 1), end=dt.date.today())
# check_stock_splits(prices)
# prices

def check_stock_splits(prices):
    """Very crude check that splits are properly dealt with in the Close series.
    
    `prices` must be a dataset with a 'Close' and a 'Stock Splits' columns. 
    """
    sx = prices['Stock Splits']!=0.0
    indices = np.arange(prices.shape[0])[sx]
    split = []
    for ix in indices:
        ret = prices.Close.iloc[ix] / prices.Close.iloc[ix-1] - 1

        # If a 2:1 split was unaccounted for, the share price would 
        # go from 100 to 50, creating a -50% returns on top of the actual
        # return of the day. Make sure this does not happen
        # assert (prices['Stock Splits'].iloc[ix] > 1) and np.abs(ret) < 0.25
        mult = prices['Stock Splits'].iloc[ix]
        assert mult > 0
        ret0 = mult*prices.Close.iloc[ix-1] # In the absence of actual return on the split day
        split_ok = (mult > 1) and (ret < (ret0-0.2)) # ret0 - 20% buffer for a very bad day...
        revsplit_ok = (mult < 1) and (ret > (ret0+0.2)) # ret0 + 20% buffer for a very good day...
        if not (split_ok or revsplit_ok):
            #ix = np.arange(prices.shape[0])
            #sx = prices['Stock Splits']!=0
            #ix = np.sort(np.hstack((ix[sx]-1, ix[sx], ix[sx]+1)))
            print(prices.iloc[[ix-1,ix,ix+1]])
            import pdb; pdb.set_trace()
            pass

