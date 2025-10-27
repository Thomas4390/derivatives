import numpy as np
import pandas as pd
import time

#from jupyter_notebook import *
from . import black_merton_scholes as bms
from . import dfbook as df # path utils (etc.) for the book
from .toolkit import printdf, show_hide_cell

from .plot_utils import (mpl, plt, mtick, mdates, gridspec,
                         set_plt_defaults, set_payoff_axes, set_time_axis, with_style)

def print_paths(paths, transpose=True, hide=None, dt='t=%ddt'):
    """Display the python paths for use on "paper" 
    
    In code, we prefer having the time index first. On "paper", time is 
    mostly depicted from left to right on the horizontal axis. This function 
    accepts the paths as stored in memory and, by default, transposes them.
    The hide argument can be used to get pretty displays for "sparse" matrices,
    replacing any occurance of hide by an empty string in the DataFrame to be
    displayed.
    """    
    [n_periods, n_paths] = paths.shape
    time_steps = [dt%(tn) for tn in range(n_periods)]
    path_no = ['trajectory %d'%(tn+1) for tn in range(n_paths)]
    if transpose:
        df = pd.DataFrame(paths.T,index=path_no,columns=time_steps)
    else:
        df = pd.DataFrame(paths,index=time_steps,columns=path_no)
    if hide is not None:
        df = df.replace(hide,'',regex=True)
    printdf(df)
    return df

def problem06_paths():
    paths = np.array([
        [100, 101.27, 99.05, 99.97, 102.14], 
        [100, 98.82, 100.57, 103.11, 102.90],  
        [100, 100.05, 101.44, 98.04, 96.56],  
        [100, 102.22, 103.29, 101.55, 104.05],  
        [100, 98.53, 99.36, 100.98, 97.64]]).T
    df = print_paths(paths, dt='month %d')
    return df

def unpack_simulation(S_t, econ, unfold_first_path=True):
    n_steps, n_paths = econ.n_steps, econ.n_paths # S_t includes time 0
    time_steps = np.arange(n_steps+1)

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    plt.ion() # Turn the interactive mode on.
    if False: # Not working; do it manually
        fig.canvas.manager.full_screen_toggle()
    fig.show() # Used to display the figure window.
    fig.canvas.draw()
        
    # Unfold the first path?
    if unfold_first_path:
        for tn in range(2,n_steps+1):
            ax.clear()
            ax.plot(time_steps[:tn], S_t[:tn,1], '-o')
            ax.set_xlim([0,np.minimum(5*tn,n_steps)])
            fig.canvas.draw()
            if tn == 2:
                plt.pause(4)
            elif tn < 10:
                plt.pause(2)
            else:
                plt.pause(5/tn)
    input("Press enter to continue...")
    ax.clear()
    n_slow = 100
    
    pn = 1
    ax.plot(time_steps, S_t[:,pn])
    pn+=1
    while pn < n_paths:
        if pn < n_slow: 
            ax.plot(time_steps, S_t[:,pn])
        else:
            ax.plot(time_steps, S_t[:,pn:])
            pn = n_paths
        ax.set_xlim([0, n_steps+1])
        ax.set_ylim([0, 300])   
        fig.canvas.draw()
        plt.pause(10/pn)
        pn+=1
        
    ax.axhline(econ.K,color='w',linewidth=2)    
    input("Press enter to continue...")
    plt.close(fig)
    

# Let's now consider antithetic variates
def compare_with_antithetic(no, axes, econ):
    axes[0].clear()
    axes[1].clear()

    time_steps = np.arange(econ.n_steps+1)
    time_t = time_steps*econ.dt    
    
    shocks = np.random.normal(0, 1, size=(econ.n_steps, econ.n_paths))
    S_t = bms.simulate_underlying(econ.S, econ.r, econ.y, econ.sigma, econ.dt, shocks)

    # For each positive (negative) N(0,1) shock z, shock -z was equally likely.
    # Consider the first half of the shocks,
    # and there additive inverse
    half_paths = int(econ.n_paths/2)
    a_shocks = np.hstack((shocks[:,:half_paths],-shocks[:,:half_paths]))
    assert shocks.shape[1]==a_shocks.shape[1] # shocks and a_shocks contain the same number of paths
    Sa_t = bms.simulate_underlying(econ.S, econ.r, econ.y, econ.sigma, econ.dt, a_shocks)
        
    ax = axes[0]
    ax.plot(time_t, S_t)
    ax.axhline(econ.S,color='w',linewidth=2)
    ax.set_title('Raw Simulation %d'%no)

    # What is the simulated forward price? How does it compare with the theoretical one?
    ax = axes[1]
    F_0_t = econ.S*np.exp( econ.r*time_t );
    ax.plot(time_t, F_0_t, color='k', linestyle='--', label='Theory')
    ax.plot(time_t, S_t.mean(axis=1), color='#660000', label='Raw')
    ax.plot(time_t, Sa_t.mean(axis=1), color='#000066', label='Antithetic')
    ax.legend()
    ax.set_title('Forward Prices: Simulated vs Theory')
    fig.canvas.draw()

if __name__ == '__main__':
    mpl.style.use('fast')
    
    #seed = np.random.randint(10000)
    seed = 2162
    np.random.seed(seed)
    print('seed =',seed)

    # I'll include those in a state for convenience when calling compare_with_antithetic
    n_paths = 2000
    n_steps = 252
    dt = 1/n_steps # days
    econ = struct(S=100, K=100, r=0.05, y=0.00, sigma=0.35,
                  n_steps=n_steps, n_paths=n_paths, dt=dt) 
    print(econ)
    
    if False:
        shocks = np.random.normal(0, 1, size=(n_steps, n_paths))
        S_t = bms.simulate_underlying(econ.S, econ.r, econ.y, econ.sigma, econ.dt, shocks)
        unpack_simulation(S_t, econ)

    #mpl.rcParams['path.simplify'] = False    
    fig, axes = plt.subplots(2, 1, figsize=(10,8))
    plt.ion() # Turn the interactive mode on.    
    compare_with_antithetic(1, axes, econ)
    fig.show() # Used to display the figure window.
    fig.canvas.draw()
    input("Press enter to continue...")

    # Now, we will perform multiple simulations with the same number of paths
    # LOOK INTO animation.FuncAnimation for potentially faster animated figures
    NM = 100
    PAUSE = 0.01*np.ones((NM,))
    #PAUSE[:10] = np.linspace(3,0.25,10)#.reshape(-1,1)
    for no in range(2,NM+1):
        compare_with_antithetic(no,axes,econ)
        plt.pause(PAUSE[no-1])
    input("Press enter to quit...")
