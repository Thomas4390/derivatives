import numpy as np
import pandas as pd
import warnings

# from jupyter_notebook import *
from .plot_utils import (mpl, plt, mtick, mdates, gridspec,
                         set_plt_defaults, set_payoff_axes, set_time_axis, with_style)

#seed = np.random.randint(10000)
seed = 8628 # 3635
np.random.seed(seed)
print('seed =',seed)

# Ad hoc values to reproduce the figures
T = 3
days_per_year = 252 # Assuming 252 business days in a year
dt = 1/days_per_year # A day
n_days = T*days_per_year # Total interval count
offset = 0.1

def figure01(time, z_t, n=4, figsize=(12,5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(time, z_t)

    ax.set_xlim([0, T+offset])
    ax.xaxis.set_ticks(range(0,T+1)); # The "end" is not included in Python arrays
    ax.xaxis.set_ticklabels(['0',r'$t-\Delta{t}$','$t$','$T$']);

    ylim = ax.get_ylim()
    ax.yaxis.set_ticklabels([]);

    # Focus on wider subperiods for the sake of illustration
    # The for-loop plots the vertical lines and highlights width h between two of them
    h = 1/n
    for t in np.arange(1,2+h,h):
        tx = int(t/dt)+1
        ax.vlines(x=t, ymin=ylim[0], ymax=z_t[tx], ls=':')
        if t==(1+h):
            ym = (ylim[0] + z_t[tx])/2 # ym: m for midpoint
            plt.annotate('${h}$',xy=(t+h/2,1.1*ym), size=plt.fs.normal, horizontalalignment='center')
            plt.annotate('',xy=(t,ym), xytext=(t+h,ym),arrowprops=dict(arrowstyle="<|-|>"))
    ax.set_ylim(ylim) # The ylimits get moved after the call to vlines; set them back
    
    dy = (ylim[1] - ylim[0])
    yl = ylim[0]+0.1*dy # yl: l for low
    plt.annotate('[',xy=(1,yl), size=plt.fs.normal);
    plt.annotate(r'$N = {\Delta t}/{h}$',xy=(1.5,yl+0.025*dy), size=plt.fs.normal, horizontalalignment='center')
    plt.annotate(']',xy=(2,yl), size=plt.fs.normal, horizontalalignment='right')
    ax.hlines(y=yl, xmin=1+5*dt, xmax=2-dt, ls=':', color='k')
    #ax.set_title('Example of a Brownian motion')

def figure02(t_values, a_values, figsize=(12,5)):
    offset = 0.1
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    tm1 = 0
    ylim = ax.get_ylim()
    ax.yaxis.set_ticklabels([]);
    for no,(t_i,a_i) in enumerate(zip(t_values,a_values)):
        a_off = a_i - offset
        plt.annotate(']',xy=(tm1+dt,a_off), size=plt.fs.large, horizontalalignment='right');
        ax.hlines(y=a_i, xmin=tm1, xmax=t_i-dt, color='k', ls='-', lw=2)
        plt.annotate(']',xy=(t_i,a_off), size=plt.fs.large, horizontalalignment='right')
        tm1=t_i

    ax.set_xlim([-dt, T+offset])
    ax.xaxis.set_ticks(t_values[:-1]);
    ax.xaxis.set_ticklabels(['$t_1$','$t_2$','$t_3$','$t_4$']) 

    ax.yaxis.set_ticks(a_values[1:]);
    ax.yaxis.set_ticklabels(['$a_1$','$a_2$','$a_3$','$a_4$'])
    ax.set_ylim([-0.05, 8])

    # ax.set_title('Example of a step function')
