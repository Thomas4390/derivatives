#from .jupyter_notebook import *
from .instruments import *

from .plot_utils import (mpl, plt, mtick, mdates, gridspec,
                         set_plt_defaults, set_payoff_axes, set_time_axis, with_style)

cbrace_color = '#666666'
from curly_brace.curlyBrace import curlyBrace # https://matplotlib-curly-brace.readthedocs.io/en/latest/Demonstration.html   

def figure_setup(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
    T = 1
    B = Bond(T) 
    B_0 = B.evaluate(0)
    
    S_0 = 100
    S_T = np.arange(0.0001, 2, 0.001)*S_0
    K = S_0

    return ax, T, B, B_0, S_0, S_T, K

def figure_setup_2K(ax=None):
    ax, T, B, B_0, S_0, S_T, K = figure_setup(ax)
    K_1 = K      #e.g. 0.75*K
    K_2 = 1.5*K  #e.g. 1.25*K 

    return ax, T, B, B_0, S_0, S_T, K_1, K_2

def figure_forward_payoff(S_0, T, S_T, figsize=(9.6,6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    fwd = Forward(S_0, T)
    fwd.name = 'Long forward'

    payoff = fwd.payoff(S_T)
    ax.plot(S_T, payoff, **with_style(fwd.name, 'portfolio1'))
    ax.set_xticks([fwd.strike])
    ax.set_xticklabels(['$K$'])
    
    short = -Forward(S_0, T)
    short.name = 'Short forward'    
    payoff = short.payoff(S_T)
    ax.plot(S_T, payoff, **with_style(short.name, 'portfolio2'))
    
    ax.axvline(S_0,color='k', label='$S(0)$', linewidth=0.75, linestyle='dotted')
    ax.axhline(0,color='k',linewidth=0.5)

    ax.set_yticks([])
    set_payoff_axes(ax)
    ax.legend(loc='lower right')


def figure_option_payoff(option, S_T, figsize=(9.6,6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axhline(0,color='k',linewidth=0.5)
    
    payoff = option.payoff(S_T)
    pline = ax.plot(S_T, payoff, **with_style(option.name, 'portfolio'))
    ax.set_xlim(left=S_T[0])
    
    ax.set_xticks([option.strike])
    ax.set_xticklabels(['$K$'])    

    ax.set_yticks([0])
    ax.set_yticklabels(['0\n'])
    ax.set_ylabel('Payoff') # Not 'P&L': we are not accounting for the prices

    set_payoff_axes(ax)
    ax.legend(loc='upper center')
        
    return fig, ax, pline

def figure_option_value(option, S, figsize=(9.6,6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axhline(0,color='k',linewidth=0.5)
    
    payoff = option.payoff(S)
    ax.plot(S, payoff, **with_style('', 'position1'))

    time_t = 0
    B = Bond(option.maturity)
    KB = option.strike * B.evaluate(time_t)
    lb = option.lower_bound(time_t, S)
    ax.plot(S, lb, **with_style('', 'position2'))

    #import pdb; pdb.set_trace()
    value = option.evaluate(time_t, S)
    #label = 'Maturity %s year'%economy.repr_maturity(option.ytm(time_t))
    label = 'European' if isinstance(option, Put) else ''
    ax.plot(S, value, **with_style(label, 'portfolio'))    

    offset = 1 if S[0] > 0.01 else 0
    ax.set_xlim(left=S[0]-offset)    
    ax.set_xticks([KB, option.strike])
    ax.set_xticklabels([r'$KB(0,T)\qquad\qquad.$','$K$'])    

    ax.set_yticks([0])
    ax.set_yticklabels(['0\n'])
    set_payoff_axes(ax)
    #ax.set_ylabel('$', horizontalalignment='right', y=1.0) 
    ax.set_ylabel('$', rotation='horizontal', ha='right', va='top')

    if isinstance(option, Put):
        data = os.path.join('data','book')
        amer_put = os.path.join(data,'intro_american_put.pkl')
        os.path.dirname(amer_put)
        put = AmericanPut(option.S_0, option.strike)
        P = put.evaluate(0,S,amer_put)                
        
        ln = ax.plot(S, P, **with_style('American', 'portfolio'))
        ln[0].set_linestyle('--')
        ax.legend(loc='upper right')

def figure_protective_put(ax=None, value_at_T=False, legend=True):
    """Example of P&L for a protective put: long asset + long put."""
    ax, T, B, B_0, S_0, S_T, K = figure_setup(ax)
            
    S = Spot(S_0, T)
    put = Put(S_0, K, T)
    p_0 = put.evaluate(0, S_0)
    ptf = S + put

    profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B_0)*B).payoff(S_T)
    yvalues = profit
    yticks = [[-p_0, 0], ['-p', '0']]
    if value_at_T:
        premium = (p_0/B_0)*B # at the risk-free rate
        put -= premium
        ptf -= premium
        yvalues = lambda ptf, S_T: ptf.evaluate(T, S_T)
        yticks = [[-p_0, 0, K-premium.evaluate(T, S_T)], ['-p', '0', 'K-p$e^{rT}$']]
    
    
    ax.plot(S_T, yvalues(S, S_T), **with_style('Underlying', 'position1'))
    ax.plot(S_T, yvalues(put, S_T), **with_style('Long put', 'position2'))
    ax.plot(S_T, yvalues(ptf, S_T), **with_style('Protective put','portfolio'))

    ax.set_xlim(0, K*1.8)
    ax.set_xticks([K], ['K'])
    ax.set_yticks(*yticks)

    set_payoff_axes(ax)
    ax.set_ylabel('$', ha='right', va='top', rotation='horizontal')
    if legend:
        ax.legend(loc='lower right')

def figure_protective_call(ax=None, value_at_T=False, legend=True):
    """Example P&L for a protective call: short asset + long call."""
    ax, T, B, B_0, S_0, S_T, K = figure_setup(ax)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    S = Spot(S_0, T)
    call = Call(S_0, K, T)
    c_0 = Call(S_0, K, T).evaluate(0, S_0)
    ptf = -S + call

    profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B_0)*B).payoff(S_T)
    yvalues = profit
    yticks = [-c_0, 0], ['-c', '0']
    if value_at_T:
        premium = (c_0/B_0)*B # at the risk-free rate
        call -= premium
        ptf -= premium
        yvalues = lambda ptf, S_T: ptf.evaluate(T, S_T)
        yticks = [[-c_0, 0, -K-premium.evaluate(T, S_T)], ['-c', '0', '-K-c$e^{rT}$']]

    ax.plot(S_T, yvalues(-S,S_T),  **with_style('Short underlying', 'position1'))
    ax.plot(S_T, yvalues(call, S_T),  **with_style('Long call', 'position2'))
    ax.plot(S_T, yvalues(ptf, S_T), **with_style('Protective call','portfolio'))

    ax.set_xlim(0, K*1.8)
    ax.set_xticks([K], ['K'])
    ax.set_yticks(*yticks)
    
    set_payoff_axes(ax)    
    ax.set_ylabel('$', ha='right', va='top', rotation='horizontal')
    if legend:
        ax.legend(loc='lower left')


# def figure_tunnel(ax=None):
#     """An example of the P&L of tunnel strategy: long asset + long put (K_1) + short call (K_2), where K_1<K_2"""
#     ax, T, B, B_0, S_0, S_T, K_1, K_2 = figure_setup_2K(ax)
#     
#     S = Spot(S_0, T) 
#     P = Put(S_0, K_1, T) 
#     C = Call(S_0, K_2, T) 
#     ptf = S + P - C
# 
#     profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B.evaluate(0))*B).payoff(S_T)    
#     
#     ax.plot(S_T, S.payoff(S_T), **with_style('Initial position', 'position1'))
#     ax.plot(S_T, profit(P, S_T), **with_style('Long put', 'position2'))
#     ax.plot(S_T, profit(-C, S_T), **with_style('Short call', 'position3'))
#     ax.plot(S_T, S.payoff(S_T)+profit(P, S_T)+profit(-C, S_T), **with_style('Portfolio','portfolio'))
# 
#     plt.text(K_1*1.3, K_1*1.4, 'Initial position', ha='right', va='top')
#     plt.text(K_1*1.4, -Put(S_0, K_1, T).evaluate(0, S_0)*0.9, 'Long one put K₁', ha='right', va='bottom')
#     plt.text(K_2*1.25, Call(S_0, K_2, T).evaluate(0, S_0)*0.8, 'Short one call K₂', ha='right', va='bottom')
#     plt.text(K_2*1.1, K_1*1.3, 'Net position', ha='right', va='top')
# 
#     ax.set_yticks([])
#     ax.set_xticks([])
# 
#     tick_K = [K_1, K_2]
#     tick_K_label = ['$K_1$', '$K_2$']
#     plt.xticks(tick_K, tick_K_label)
# 
#     tick_y = [-Put(S_0, K_1, T).evaluate(0, S_0), 0,  Call(S_0, K_2, T).evaluate(0, S_0)]
#     tick_y_label = ['-p', '0', 'c']
#     plt.yticks(tick_y, tick_y_label)
# 
#     plt.ylim(-Put(S_0, K_1, T).evaluate(0, S_0)*2, K_2*1.2)
#     plt.xlim(0, K_2*1.4)
# 
#     set_payoff_axes(ax)
#     plt.ylabel('$', rotation='horizontal', ha='right', va='top')
# 
# figure_tunnel()
# df.savefig('intro','fig_tunnel.pdf')        
    
def figure_bull_call_spread(ax=None):
    """Example of P&L for a bull call spread: long one call $K_1$, short one call $K_2$."""
    ax, T, B, B_0, S_0, S_T, K_1, K_2 = figure_setup_2K(ax)

    c_1 = Call(S_0, K_1, T)
    c_2 = Call(S_0, K_2, T)
    ptf = c_1 - c_2

    premium_at_rf = lambda ptf: ptf.evaluate(0, S_0)*(B/B_0)    
    profit = lambda ptf, S_T: (ptf - premium_at_rf(ptf)).payoff(S_T)

    ax.plot(S_T, profit(c_1, S_T), **with_style('Long one call $K_1$', 'position1'))
    ax.plot(S_T, profit(-c_2, S_T), **with_style('Short one call $K_2$', 'position2'))
    ax.plot(S_T, profit(ptf, S_T), **with_style('Portfolio','portfolio'))

    argmax = np.array([1.35*K_2])
    max_profit = profit(ptf, argmax)
    curlyBrace(plt.gcf(), ax, [argmax, max_profit], [argmax, 0], 0.05, 
               color=cbrace_color, lw=2, int_line_num=1)
    ax.annotate('($K_1$-$K_2$)', xy=(K_2*1.4, K_2*0.14))
    ax.annotate('  $-(c_1-c_2)e^{rT}$', xy=(K_2*1.4, K_2*0.10))
    
    ax.set_xticks([K_1, K_2], ['$K_1$', '$K_2$'])
    ax.set_yticks([0], ['0'])

    max_loss = -premium_at_rf(ptf).evaluate(T,S_T)
    ax.annotate('$-(c_1-c_2)e^{rT}$', xy=(13,2.75*max_loss))
    ax.arrow(12,2.5*max_loss, -8,-1.1*max_loss, head_width=3)
    
    ax.set_xlim(0, K_1+K_2)
    ax.set_ylim(-1.1*S_0, S_0)
    set_payoff_axes(ax)
    ax.set_ylabel('$', rotation='horizontal', ha='right', va='top')

    ax.legend(loc='upper left')
    

def figure_bull_put_spread(ax=None):
    """Example of P&L for a bull put spread: long one put $K_1$, short one put $K_2$."""
    ax, T, B, B_0, S_0, S_T, K_1, K_2 = figure_setup_2K(ax)

    p_1 = Put(S_0, K_1, T)
    p_2 = Put(S_0, K_2, T)
    ptf = p_1 - p_2

    premium_at_rf = lambda ptf: ptf.evaluate(0, S_0)*(B/B_0)
    profit = lambda ptf, S_T: (ptf - premium_at_rf(ptf)).payoff(S_T)

    ax.plot(S_T, profit(p_1, S_T), **with_style('Long one put $K_1$', 'position1'))
    ax.plot(S_T, profit(-p_2, S_T), **with_style('Short one call $K_2$', 'position2'))
    ax.plot(S_T, profit(ptf, S_T), **with_style('Portfolio','portfolio'))

    argmax = np.array([1.35*K_2])
    max_profit = - ptf.evaluate(0, S_0)/B.evaluate(0)
    curlyBrace(plt.gcf(), ax, [argmax, max_profit], [argmax, 0], 0.05, 
               color=cbrace_color, lw=2, int_line_num=1)
    #plt.annotate('$-(p_1-p_2)e^{rT}$', xy=(K_2*1.39, K_2*0.1))
    plt.annotate('$(p_2 - p_1)e^{rT}$', xy=(K_2*1.39, K_2*0.1))

    if True:
        argmin = np.array([0.01*K_2])
        max_loss = profit(ptf,argmin)
        #curlyBrace(plt.gcf(), ax, [argmin, 0], [argmin, max_loss], 0.05, 
        #           color=cbrace_color, lw=2, int_line_num=1)
        plt.annotate('$(K_1-K_2)$', xy=(K_2*0.02, -K_2*0.17))
        plt.annotate('   $+(p_2-p_1)e^{rT}$', xy=(K_2*0.01, -K_2*0.25))        
        #plt.annotate('   $-(p_1-p_2)e^{rT}$', xy=(K_2*0.01, -K_2*0.25))        

    ax.set_xticks([K_1, K_2], ['$K_1$', '$K_2$'])
    ax.set_yticks([0], ['0'])
    #ax.set_yticks([0, max_loss[0]], ['0', '$(K_1-K_2)$'+'\t\t\n'+'$-(p_1-p_2)e^{rT}$'])

    ax.set_xlim(0, K_1+K_2)
    ax.set_ylim(-1.1*S_0, S_0)
    set_payoff_axes(ax)
    ax.set_ylabel('$', rotation='horizontal', ha='right', va='top')


def figure_horizontal_spread():
    """Example of P&L for a horizontal spread: short one call $T_1$, long one call $T_2 > T_1$."""
    ax, T_1, B, B_0, S_0, S_T, K = figure_setup()
    T_2 = 2*T_1

    c_1 = Call(S_0, K, T_1)
    c_2 = Call(S_0, K, T_2)
    ptf = -c_1 + c_2

    premium_at_rf = lambda ptf: ptf.evaluate(0, S_0)*(B/B_0)
    profit = lambda ptf, S_T: (ptf - premium_at_rf(ptf)).payoff(S_T)
    c_2_v = (c_2 - premium_at_rf(c_2)).evaluate(T_1, S_T)

    ax.axvline(0, color='black')
    ax.plot(S_T, -profit(c_1, S_T), **with_style('Short one call $T_1$', 'position1'))
    ax.plot(S_T, profit(c_2, S_T), **with_style('Long one call $T_2$', 'position2'))
    ax.plot(S_T, c_2_v, **with_style('Value of the $T_2$ call at $T_1$', 'position3'))
    ax.plot(S_T, c_2_v-profit(c_1, S_T), **with_style('Value of the portfolio at $T_1$','portfolio'))

    ax.set_xticks([K], ['K'])
    ax.set_yticks([0], ['0'])

    ax.set_xlim(0, K*2)
    ax.set_ylim(-K, K)
    set_payoff_axes(ax)
    ax.set_ylabel('$', rotation='horizontal', ha='right', va='top')    
    ax.legend(loc='upper left')

    
def figure_reverse_horizontal_spread():
    """Example of P&L for a reverse horizontal spread: long one put $T_14, short one put $T_2$."""

    ax, T_1, B, B_0, S_0, S_T, K = figure_setup()
    T_2 = 2*T_1

    p_1 = Put(S_0, K, T_1) 
    p_2 = Put(S_0, K, T_2)
    ptf = p_1 - p_2

    premium_at_rf = lambda ptf: ptf.evaluate(0, S_0)*(B/B_0)
    profit = lambda ptf, S_T: (ptf - premium_at_rf(ptf)).payoff(S_T)
    #p_2_v = -Put(S_0, K, T_2).evaluate(T_2-T_1, S_T) + Put(S_0, K, T_2).evaluate(0, S_0)/B.evaluate(0)
    p_2_v = (p_2 - premium_at_rf(p_2)).evaluate(T_1, S_T)

    ax.axvline(0, color='black')
    ax.plot(S_T, profit(p_1, S_T), **with_style('Long one put $T_1$', 'position1'))
    ax.plot(S_T, -profit(p_2, S_T), **with_style('Short one put $T_2$', 'position2'))
    ax.plot(S_T, -p_2_v, **with_style('Value of the $T_2$ put at $T_1$', 'position3'))
    ax.plot(S_T, profit(p_1, S_T)-p_2_v, **with_style('Portfolio','portfolio'))

    ax.set_xticks([K], ['K'])
    ax.set_yticks([0], ['0'])

    ax.set_xlim(0, K*2)
    ax.set_ylim(-K, K)

    set_payoff_axes(ax)
    ax.set_ylabel('$', rotation='horizontal', ha='right', va='top')
    ax.legend(loc='upper right')


def figure_straddle(short=False):
    """Example of P&L for a long straddle: long one call & one put, ATM.

    If short is True, the short position is plotted instead.
    """
    ax, T, B, B_0, S_0, S_T, _ = figure_setup()

    K = S_0    
    c = Call(S_0, K, T)
    p = Put(S_0, K, T)

    position = "Long"
    if short:
        c = -c
        p = -p
        position = "Short"
    ptf = c + p

    profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B.evaluate(0))*B).payoff(S_T)

    ax.axvline(0, color="black")
    ax.plot(S_T, profit(c, S_T), **with_style(f"{position} ATM call", "position1"))
    ax.plot(S_T, profit(p, S_T), **with_style(f"{position} ATM put", "position2"))
    ax.plot(S_T, profit(ptf, S_T), **with_style("Portfolio","portfolio"))

    ax.set_xticks([K], ["K"])
    ax.set_yticks([0], ["0"])

    ax.set_xlim(0, K*1.8)
    set_payoff_axes(ax)
    plt.ylabel("$", rotation="horizontal", ha="right", va="top")
    if short:        
        ax.legend(loc="lower right")
    else:
        ax.legend(loc="upper right")


def figure_strangle(short=False):
    """Example of P&L for a long strangle: long one put at $K_1$ and a call at $K_2 > K_1$.

    If short is True, the short position is plotted instead.
    """
    ax, T, B, B_0, S_0, S_T, _ = figure_setup()
    K_1 = 0.9*S_0
    K_2 = 1.1*S_0
        
    p = Put(S_0, K_1, T)
    c = Call(S_0, K_2, T)
    position = "Long"
    if short:
        c = -c
        p = -p
        position = "Short"
    ptf = c + p

    profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B.evaluate(0))*B).payoff(S_T)

    ax.axvline(0, color="black")
    ax.plot(S_T, profit(p, S_T), **with_style(f"{position} $K_1$ put", "position2"))
    ax.plot(S_T, profit(c, S_T), **with_style(f"{position} $K_2$ call", "position1"))
    ax.plot(S_T, profit(ptf, S_T), **with_style("Portfolio","portfolio"))

    ax.set_xticks([K_1,K_2], ["K_1","K_2"])
    ax.set_yticks([0], ["0"])

    K = (K_1 + K_2)/2
    ax.set_xlim(0.5*K, K*1.5)
    ax.set_ylim(-0.5*K, 0.5*K)
    set_payoff_axes(ax)
    plt.ylabel("$", rotation="horizontal", ha="right", va="top")
    if short:        
        ax.legend(loc="lower right")
    else:
        ax.legend(loc="upper right")
    

def figure_butterfly():
    """Example of P&L for a butterfly."""
    ax, T, B, B_0, S_0, S_T, _ = figure_setup()
    K_1 = 0.8*S_0
    K_2 = S_0
    K_3 = 1.2*S_0
    profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B.evaluate(0))*B).payoff(S_T)

    c_1 = Call(S_0, K_1, T)
    c_2 = Call(S_0, K_2, T)
    c_3 = Call(S_0, K_3, T)    
    ptf = c_1 - 2*c_2 + c_3 
    
    ax.plot(S_T, profit(c_1, S_T), **with_style('Long one call $K_1$','position1'))
    ax.plot(S_T, -2*profit(c_2, S_T), **with_style('Short two calls $K_2$','position2'))
    ax.plot(S_T, profit(c_3, S_T), **with_style('Short one call $K_3$','position3'))
    ax.plot(S_T, profit(ptf, S_T), **with_style('Portfolio','portfolio'))

    ax.set_xticks([K_1, K_2, K_3], ['$K_1$', '$K_2$', '$K_3$'])
    ax.set_yticks([0], ["0"])

    ax.set_xlim(0.5*K_2, K_2*1.5_2)
    ax.set_ylim(-0.5*K_2, 0.5*K_2)
    set_payoff_axes(ax)
    plt.ylabel('$', rotation='horizontal', ha='right', va='top')
    ax.legend(loc='upper right')
            

def figure_condor():
    """Example of the P&L for a condor."""
    ax, T, B, B_0, S_0, S_T, _ = figure_setup()
    K_1 = 0.8*S_0
    K_2 = 0.9*S_0
    K_3 = 1.1*S_0
    K_4 = 1.2*S_0
    profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B.evaluate(0))*B).payoff(S_T)

    c_1 = Call(S_0, K_1, T)
    c_2 = Call(S_0, K_2, T) 
    c_3 = Call(S_0, K_3, T) 
    c_4 = Call(S_0, K_4, T) 
    ptf = c_1 - c_2 - c_3 + c_4

    ax.plot(S_T,  profit(c_1, S_T), **with_style('Long one $K_1$ call ','position1'))
    ax.plot(S_T, -profit(c_2, S_T), **with_style('Short one $K_2$ call','position2'))
    ax.plot(S_T, -profit(c_3, S_T), **with_style('Short one $K_3$ call','position3'))
    ax.plot(S_T,  profit(c_4, S_T), **with_style('Long one $K_4$ call','position4'))
    ax.plot(S_T,  profit(ptf, S_T), **with_style('Portfolio','portfolio'))
    
    ax.set_xticks([K_1, K_2, K_3, K_4], ['$K_1$', '$K_2$', '$K_3$', '$K_4$'])    
    ax.set_yticks([0], ["0"])

    ax.set_xlim(0.5*S_0, 1.5*S_0)
    ax.set_ylim(-0.45*S_0, 0.45*S_0)
    
    set_payoff_axes(ax)
    plt.ylabel('$', rotation='horizontal', ha='right', va='top')    
    ax.legend(loc='lower right')
    
def figure_piecewise_linear_payoff():
    ax, T, B, B_0, _, _, _ = figure_setup()

    S_0 = 15
    K_1 = 13
    K_2 = K_1*1.3
    K_3 = K_1*1.6
    K_4 = K_1*2
    S_T = np.arange(0.0001, 3, 0.001)*S_0
    profit = lambda ptf, S_T: (ptf - (ptf.evaluate(0, S_0)/B.evaluate(0))*B).payoff(S_T)
    
    c_1 = Call(S_0, K_1, T)
    c_2 = Call(S_0, K_2, T) 
    c_3 = Call(S_0, K_3, T) 
    c_4 = Call(S_0, K_4, T)     
    ptf = c_1 - 2*c_2 + 3*c_3 - 3*c_4    

    if False:
        ax.plot(S_T, profit(ptf, S_T), **with_style('Portfolio','portfolio'))
        the_arrows = [
            {'label': 'Slope = +1', 'xy': (K_1*1.1, -K_1*0.1), 'xytext': (K_1*1.15, -K_1*0.4)},
            {'label': 'Slope = -1', 'xy': (K_2*1.1, K_1*0.1), 'xytext': (K_2*0.9, offset+K_1*0.3)},
            {'label': 'Slope = +2', 'xy': (K_3*1.1, K_1*0.22), 'xytext': (K_3*0.9, offset+K_1*0.4)},
            {'label': 'Slope = -1', 'xy': (K_4*1.2, K_1*0.18), 'xytext': (K_4, offset-K_1*0.3)}
        ]
    else:
        y0 = 2 
        ax.plot(S_T, ptf.payoff(S_T), **with_style('Portfolio','portfolio'))
        the_arrows = [
            {'label': 'Slope = +1', 'xy': (K_1*1.1, y0-K_1*0.1), 'xytext': (K_1*1.15, y0-K_1*0.4)},
            {'label': 'Slope = -1', 'xy': (K_2*1.1, y0+K_1*0.1), 'xytext': (K_2*0.9, y0+K_1*0.3)},
            {'label': 'Slope = +2', 'xy': (K_3*1.1, y0+K_1*0.22), 'xytext': (K_3*0.9, y0+K_1*0.4)},
            {'label': 'Slope = -1', 'xy': (K_4*1.2, y0+K_1*0.18), 'xytext': (K_4, y0-K_1*0.3)}
        ]

    arrow_props = dict(facecolor='black', arrowstyle='->')
    for arrow in the_arrows:
        ax.annotate(arrow['label'], xy=arrow['xy'], xytext=arrow['xytext'], arrowprops=arrow_props)

    ax.set_xticks([K_1, K_2, K_3, K_4], ['$K_1$', '$K_2$', '$K_3$', '$K_4$'])    
    ax.set_yticks([0], ["0"])

    plt.xlim(0, K_4*1.55)
    plt.ylim(-0.8*K_1, K_1*0.8)

    set_payoff_axes(ax)
    plt.ylabel('$', rotation='horizontal', ha='right', va='top')


def DEV_implied_risk_free_rate():
    #CD: Please update to match the style of your figure
    # Buy bull call spread, buy bear put with same strike pairs
    # Get option implied risk-free rate
    S = 100 
    K_1 = 95
    K_2 = 105
    T = 1/12
    
    c_1 = Call(S, K_1, T)
    c_2 = Call(S, K_2, T)
    p_1 = Put(S, K_1, T)
    p_2 = Put(S, K_2, T)
    options = [c_1, c_2, p_1, p_2]
    
    S_T = np.linspace(80,121,200)
    bull_call = (c_1 - c_2).set_name('bull_call')
    bear_put = (p_2 - p_1).set_name('bear_put')
    # bull_call, bear_put

    fig, axes = plt.subplots(2, 1, figsize=(15,16))
    bc_T = axes[0].plot(S_T, bull_call.payoff(S_T), **with_style('Bull call','position1'))
    bp_T = axes[0].plot(S_T, bear_put.payoff(S_T), **with_style('Bear put','position2'))
    axes[0].set_xlim(left=S_T[0])
    set_payoff_axes(axes[0])
    axes[0].legend(loc='right');

    ptf = bull_call + bear_put
    ptf_T = axes[1].plot(S_T, ptf.payoff(S_T), **with_style('Portfolio','portfolio'))
    axes[1].set_xlim(left=S_T[0])
    set_payoff_axes(axes[1])
    axes[1].legend(loc='right');
