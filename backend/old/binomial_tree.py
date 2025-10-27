import networkx as nx
import numpy as np
import pandas as pd
import warnings

#from .jupyter_notebook import *
from .plot_utils import (mpl, plt, mtick, mdates, gridspec,
                         set_plt_defaults, set_payoff_axes, set_time_axis, with_style)
from . import black_merton_scholes as bms

def n_moves_str(label, n_moves):
    mv = ''
    if n_moves > 0:
        mv = label
    if n_moves > 1:
        mv += '^%d'%n_moves
    return mv

class BinomialTree:
    "General decription of the binomial tree"
    def __init__(self, n_steps, S=np.nan, r=np.nan, y=np.nan, T=np.nan, sigma=np.nan):
        self.S = S
        self.r = r 
        self.y = y
        self.T = T
        self.sigma = sigma
        self.n_steps = n_steps
        
        self.dt = T / n_steps # h in the McDonalds notation
        
        # Row t: time step t. Column j: j up moves since start (n-j down moves)
        self.values = np.full((n_steps+1, n_steps+1), np.nan)

    def df(self, values=None, horiz_time=False):
        if values is None:
            values = self.values

        index = ['Step %d'%tn for tn in range(0,self.n_steps+1)]
        if horiz_time:
            ix = np.arange(values.shape[1]-1, -1, -1)
            rows = ['up%d'%tn for tn in ix]
            return pd.DataFrame(values[:,ix].T, index=rows, columns=index).replace(np. nan,'',regex=True)

        columns = ['up%d'%tn for tn in range(0,self.n_steps+1)]
        return pd.DataFrame(values, index=index, columns=columns).replace(np. nan,'',regex=True)
        
    #TBR? def to_string(self, values):
    #TBR?     return str( self.df() )
        
    def __str__(self):
        return str( self.df() )

    def node_label(self, n_up, n_down):
        return n_moves_str('u',n_up)+' '+n_moves_str('d',n_down)

    def asset_node_position(self, tn, jj):
        y_pos = self.values[tn, jj]
        if np.isnan(self.S):
            y_pos = self.n_steps+2 - tn + 2*jj            
        return y_pos, y_pos
        
    def asset_node_label(self, tn, jj):
        n_up = jj
        n_down = tn-n_up                
        #return 'S ' + self.node_label(n_up, n_down)
        return '$S ' + self.node_label(n_up, n_down) + r'$'

    def option_node_label(self, tn, jj, option):
        #return option+'_{' + self.node_label(jj, tn-jj) + '}'
        return '$'+option+'_{' + self.node_label(jj, tn-jj) + '}$'

    def sym_node_label(self, tn, jj, values):
        labels = []
        #import pdb; pdb.set_trace()
        for val in values:        
            if val=='S':
                lbl = self.asset_node_label(tn, jj)
            else:
                lbl = self.option_node_label(tn, jj, val)
            labels.append(lbl)
        return '\n'.join(labels)
    
    def value_node_label(self, tn,jj, values, fmt):
        lbl = []
        for val in values:
            lbl.append(fmt % val[tn,jj])
        return '\n'.join(lbl)        
    
    def draw_tree(self, values='S', value_fmt='%.2f', figsize=(15,10), **kwargs):
        if 'dbg' in kwargs:
            import pdb; pdb.set_trace()
        time_steps = kwargs.pop('time_steps', range(self.n_steps+1))
            
        G = nx.Graph()
        labels = {}
        pos_nodes = {}
        pos_labels = {}

        # Stack the multiple "values", if more than 1
        if not isinstance(values, list):
            values = [values]
            
        # If values are strings, they are SYMbolic, otherwise actual VALUES
        node_label = lambda tn,jj: self.sym_node_label(tn,jj,values)
        #node_label = lambda tn,jj: r'$'+self.sym_node_label(tn,jj,values)+r'$'
        if not isinstance(values[0],str):
            node_label = lambda tn,jj: self.value_node_label(tn,jj,values,value_fmt)        
            
        for tn in range(self.n_steps+1):
            for jj in range(tn+1): 
                node = (tn,jj)
                if self.n_steps < 10:
                    labels[node] = node_label(tn, jj)
                                
                y_pos, y_pos_lbl = self.asset_node_position(tn,jj)                
                pos_nodes[node] = (tn, y_pos)
                pos_labels[node] = (tn, y_pos_lbl)
                if tn < self.n_steps:
                    G.add_edge(node, (tn+1, jj))
                    G.add_edge(node, (tn+1, jj+1))
                
        fig, ax  = plt.subplots(1, 1, figsize=(15,10))
        #set_arrow_xaxis(ax)

        if 'node_size' not in kwargs:
            kwargs['node_size'] = 35**2 # 300/np.maximum(1, self.n_steps/5)
        if 'node_color' not in kwargs:
            kwargs['node_color'] = 'w'
        if "with_labels" not in kwargs:
            kwargs["with_labels"] = False

        # Draw sets axes off. We want to customize them with set_time_axis
        #nx.draw(G, pos=pos_nodes, ax=ax, **kwargs)
        nx.draw_networkx(G, pos=pos_nodes, ax=ax, **kwargs)
        nx.draw_networkx_labels(G, pos=pos_labels, labels=labels, ax=ax)

        set_time_axis(ax, time_steps, '$t_{%d}$', only=True)
            
        return fig,ax
    
    
    def option_price(self, K, is_call, bms_adj=False):
        """Computes Euro and American option values.
        
        Args:
            K:       the strike of the options
            is_call: True for a call, False for a put
            bms_adj: Perform a BMS adjustment at the next-to-last date 
        
        Returns:
            euro:     the European option's value 
                      (n_steps x n_steps) array of doubles, same convention as for tree.values
            amer:     the American options's value 
                      (n_steps x n_steps) array of doubles, same convention as for tree.values
            do_x:     True in nodes where the option was exercised, False otherwise
                      (n_steps x n_steps) array of booleans, same convention as for tree.values
            boundary: an approximation of the exercise boundary through time
                      (n_steps x 1) array of doubles
        """
        # Populate the asset tree
        spot = self.values
        n_steps = self.n_steps
        
        # Each row of the matrix will contain the nodes at a given time step in the
        # tree. For each up move, the column index will be increased by 1. For down
        # moves, we stay on the same column.
        euro = np.full(spot.shape, np.nan)
        amer = np.full(spot.shape, np.nan)
        do_x = np.full(spot.shape, False);
        boundary = np.full((n_steps+1,1), np.nan) # through time
        
        mult = 1 if is_call else -1
        
        # First, fill the leaves of the tree with the known values of the payoff
        payoff = np.maximum(0, mult*(spot[-1,:] - K))
        euro[-1,:] = payoff
        amer[-1,:] = payoff
        do_x[-1,:] = payoff > 0
        boundary[-1] = K
          
        # Apply backward induction
        disc = np.exp(-self.r*self.dt) # one-period discount factor
        
        T_minus_dt = n_steps-1
        if bms_adj:
            continuation = bms.option_price(spot[n_steps-1,:-1], K, self.r, self.y, self.dt, self.sigma, is_call) 
            euro[n_steps-1,:-1] = continuation
            amer[n_steps-1,:-1] = continuation
            T_minus_dt = n_steps-2
        
        for tn in range(T_minus_dt,-1,-1): 
            kk = np.arange(0,tn+1)
            euro[tn,kk] = disc*( self.q*euro[tn+1,kk+1] + (1-self.q)*euro[tn+1,kk] );
        
            # For an american option, the continuation value at tn*dt is the
            # discounted expectation of the values in tn+1. But early exercise is
            # also possible...
            exercise = np.maximum(0, mult*(spot[tn,kk]-K));
            continuation = disc*( self.q*amer[tn+1,kk+1] + (1-self.q)*amer[tn+1,kk] );
            amer[tn,kk] = np.maximum(exercise, continuation);
            do_x[tn,kk] = (exercise>0) & (amer[tn, kk]==exercise);        
            boundary[tn] = get_boundary(spot[tn,:], do_x[tn,:], is_call)
        
        return euro, amer, do_x, boundary

    
class Asset(BinomialTree):
    "Binomial tree with 'values' being those propagated forward for the underlying asset"
    def __init__(self, n_steps, S=np.nan, r=np.nan, y=np.nan, T=np.nan, sigma=np.nan):
        super().__init__(n_steps, S, r, y, T, sigma)

        # Set the up, down and q attributes according to the specific tree definition
        self.initialize()
        
        # Time t on row t. Column j, j up moves since start (n-j down moves)
        values = self.values

        values[0,0] = S
        for tn in range(0,n_steps):
            # If we stayed on column 0, we just experienced a down move from the
            # previous lowest node in the tree
            values[tn+1, 0] = self.down * values[tn, 0]
    
            # One of the paths leading to e.g. node (2,2) is to go up from node (1,1)
            kk = np.arange(0,tn+1)
            values[tn+1, kk+1] = self.up * values[tn, kk]
        self.values = values

    def initialize(self):
        self.up = np.nan
        self.down = np.nan
        self.q = np.nan
        
    def asset_node_label(self, tn, jj):
        n_up = jj
        n_down = tn-n_up        
        n_off = np.minimum(n_up, n_down)
        n_up,n_down = n_up-n_off, n_down-n_off
        
        #return 'S ' + self.node_label(n_up, n_down)
        return '$S ' + self.node_label(n_up, n_down) + '$'

class AssetCRR(Asset):
    "Binomial tree using the CRR up, down and risk-neutral probability"    
    def initialize(self):
        self.up = np.exp(self.sigma * np.sqrt(self.dt))
        self.down = 1 / self.up
        self.q = (np.exp((self.r-self.y)*self.dt) - self.down) / (self.up - self.down)

class AssetJarrowRudd(Asset):
    "Binomial tree using the Jarrow and Rudd up, down and risk-neutral probability"    
    def initialize(self):
        drift = (self.r - self.y - self.sigma**2 / 2)*self.dt
        self.up = np.exp(drift + self.sigma * np.sqrt(self.dt))
        self.down = np.exp(drift - self.sigma * np.sqrt(self.dt))
        self.q = 0.5
    
class CallPayoff(AssetCRR):
    """Utility class for drawing call's payoffs in the tree"""
    def option_node_label(self, tn, jj, option):
        assert option=='c' 
        call = super().option_node_label(tn,jj,'c')
        if tn < self.n_steps:
            return '?'
        call = call.strip('$')
        stock = self.asset_node_label(tn,jj).strip('$')
        return '\t\t\t   $%s= $max$(%s - K, 0)$'%(call,stock)

    def draw_tree(self,*args,**kwargs):
        fig, ax = super().draw_tree(*args,**kwargs)
        xl = ax.get_xlim()
        ax.set_xlim([xl[0],xl[1]+0.25])
        return fig, ax     

class CallTminus(AssetCRR):
    """Utility class for drawing a call's intermediate values in the tree"""
    def __init__(self, n_steps, minus_t):
        super().__init__(n_steps)
        self.minus_t = minus_t
        
    def option_node_label(self, tn, jj, option):
        assert option=='c' 
        call = super().option_node_label(tn,jj,'c')
        if tn > self.n_steps-self.minus_t:
            return call
        if tn < self.n_steps-self.minus_t:
            return '?'
        call = call.strip('$')
        up = super().option_node_label(tn+1,jj+1,'c').strip('$')
        down = super().option_node_label(tn+1,jj,'c').strip('$')
        return r'\t\t\t\t\t    $%s$    $= e^{-r\Delta t}[q %s + (1-q) %s]$'%(call,up,down)

    def draw_tree(self,*args,**kwargs):
        fig, ax = super().draw_tree(*args,**kwargs)
        xl = ax.get_xlim()
        ax.set_xlim([xl[0],xl[1]+0.25])
        return fig, ax     
    
    
def get_boundary(asset, do_x, is_call):
    # At the beginning of the tree, there might not be enough nodes to describe the frontier
    never = np.all(~do_x)
    if never:
        return np.nan

    # Select valid nodes at t_n, n < N, within a (N+1)-length vector
    sn = ~np.isnan(asset) 
    
    # This should not happen if the tree is properly designed, but it is not the place of this
    # function to judge ;)
    always = np.all(do_x[sn])
    if always:
        return np.nan    
            
    if is_call:
        ex = np.min(asset[sn & do_x])
        no_ex = np.max(asset[sn & ~do_x])        
    else:
        ex = np.max(asset[sn & do_x])
        no_ex = np.min(asset[sn & ~do_x])
    return (ex + no_ex) / 2 # simply take the mid point 


def option_price(S, K, r, y, T, sigma, n_steps, is_call, bms_adj=False):
    """Creates the asset CRR tree, and returns the corresponding Euro and American option values.

    Args:
        S:       the initial value of the underlying
        K:       the strike of the options
        r:       the risk-free rate
        y:       the dividend rate
        T:       the maturity of the options
        sigma:   the volatility to be used in the tree
        n_steps: the number of time steps in the tree
        is_call: True for a call, False for a put
        bms_adj: Perform a BMS adjustment at the next-to-last date (Default: False)

    Returns:
        tree:     a BinomialTree instance with its values being those of the asset
                  tree.values: Time n on row n. On column k, k up moves since start (n-k down moves)
        euro:     the European option's value 
                  (n_steps x n_steps) array of doubles, same convention as for tree.values
        amer:     the American options's value 
                  (n_steps x n_steps) array of doubles, same convention as for tree.values
        do_x:     True in nodes where the option was exercised, False otherwise
                  (n_steps x n_steps) array of booleans, same convention as for tree.values
        boundary: an approximation of the exercise boundary through time
                  (n_steps x 1) array of doubles
    """
    # Populate the asset tree
    tree = AssetCRR(n_steps, S, r, y, T, sigma)
    euro, amer, do_x, boundary = tree.option_price(K, is_call, bms_adj)
    return tree, euro, amer, do_x, boundary 
