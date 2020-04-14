#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit  # For just-in-time compilation to LLVM
import random
from scipy import integrate
import pandas as pd

import altair as alt

#%%
@jit # Numba decorator; compiles function to efficient machine code (LLVM)
def gillespie(x, alpha_m, tau_m, alpha_p, tau_p, delta, N):
    """Optimised Gillespie algorithm using jit.

    Args:
        x (ndarray(int)): Initial counts for mRNA and protein
        alpha_m (float): transcript birth rate
        tau_m (float): transcript death rate
        alpha_p (float): protein birth constant
        tau_p (float): protein death rate
        delta (ndarray): 2D diffusion matrix for system
        N (int): number of iterations for Gillespie

    returns:
        X (ndarray(int)): Trace of component counts for each iteration.
        T (ndarray(float)): Time adjusted trace of time during simulation.
        tsteps (ndarray(float)): Time weight trace; duration of time spent in each state.
    """
    # Initialisation
    t = 0
    T = np.zeros(N)
    tsteps = np.zeros(N)
    X = np.zeros((delta.shape[0], N))

    # Simulation
    for i in range(N):
        # Determine rates
        rates = np.array(
            [
                alpha_m,        # reaction 1 (mRNA birth) rate
                x[0] * tau_m,   # rxn 2 (mRNA death) rate
                x[0] * alpha_p, # rxn 3 (protein birth) rate
                x[1] * tau_p    # rxn 4 (protein death) rate
            ]
        )
        summed = np.sum(rates)

        # Determine WHEN state change occurs
        tstep = (-1) / summed * np.log(random.random())
        t = t + tstep
        T[i] = t
        tsteps[i] = tstep

        # Determine WHICH reaction occurs with relative propabilities
        reac = np.sum(np.cumsum(np.true_divide(rates, summed)) < random.random())
        x = x + delta[:, reac]
        X[:, i] = x

    return X, T, tsteps


#%%
x = np.array(
    [
        1, # one mRNA to start
        20 # twenty proteins to start
    ]
)

alpha_m = 0.5
tau_m = 0.05
alpha_p = 5
tau_p = 0.05
delta = np.array(
    [
        [1, -1, 0, 0], # mRNA changes for each column, where columns represent reactions
        [0, 0, 1, -1]  # protein changes for each column
    ]
)
N = 10000

#%%
Xg,Tg,tsteps = gillespie(x, alpha_m, tau_m, alpha_p, tau_p, delta, N)

# %%
fig,ax = plt.subplots(nrows=2, sharex=True)

for i in range(2):
    ax[i].plot(Tg, Xg[i,:])

#%%
@jit # Numba decorator; compiles function to efficient machine code (LLVM)
def ode(x, t, alpha_m, tau_m, alpha_p, tau_p):
    """Optimised Gillespie algorithm using jit.

    Args:
        x (ndarray(int)): Initial counts for mRNA and protein
        alpha_m (float): transcript birth rate
        tau_m (float): transcript death rate
        alpha_p (float): protein birth constant
        tau_p (float): protein death rate

    returns:
        X (ndarray(int)): Trace of component counts for each iteration.
    """
    
    S = np.empty(x.shape)

    # dmRNA/dt
    S[0] = alpha_m - x[0] * tau_m
    # dP/dt
    S[1] = x[0] * alpha_p - x[1] * tau_p

    return S

#%%
To = np.linspace(0,np.max(Tg),10000)

#%%
Xo = integrate.odeint(
                ode,
                x,
                To, # go from current conditions to t_step
                args = (alpha_m, tau_m, alpha_p, tau_p),
                full_output=False
)

# %%
fig,ax = plt.subplots(nrows=2, sharex=True)

for i in range(2):
    ax[i].plot(To, Xo.T[i,:])


# %%
fig,ax = plt.subplots(nrows=2, sharex=True)

for i in range(2):
    ax[i].plot(To, Xo.T[i,:])
    ax[i].plot(Tg, Xg[i,:])


# %%
#%%
@jit # Numba decorator; compiles function to efficient machine code (LLVM)
def neg_trx_feedback_gillespie(x, alpha_m, tau_m, alpha_p, tau_p, bind_rate, delta, N):
    """Optimised Gillespie algorithm using jit.

    Args:
        x (ndarray(int)): Initial counts for mRNA and protein
        alpha_m (float): transcript birth rate
        tau_m (float): transcript death rate
        alpha_p (float): protein birth constant
        tau_p (float): protein death rate

        delta (ndarray): 2D diffusion matrix for system
        N (int): number of iterations for Gillespie

    returns:
        X (ndarray(int)): Trace of component counts for each iteration.
        T (ndarray(float)): Time adjusted trace of time during simulation.
        tsteps (ndarray(float)): Time weight trace; duration of time spent in each state.
    """
    # Initialisation
    t = 0
    T = np.zeros(N)
    tsteps = np.zeros(N)
    X = np.zeros((delta.shape[0], N))

    # Simulation
    for i in range(N):
        # Determine rates
        rates = np.array(
            [
                alpha_m * x[2], # mRNA production rate * DNA_free
                x[0] * tau_m,
                x[0] * alpha_p,
                x[1] * tau_p,
                x[2] * bind_rate * x[1], # DNA_free (0 or 1) * binding_rate * protein_level
                (1-x[2]) * 1/bind_rate
            ]
        )
        summed = np.sum(rates)

        # Determine WHEN state change occurs
        tau = (-1) / summed * np.log(random.random())
        t = t + tau
        T[i] = t
        tsteps[i] = tau

        # Determine WHICH reaction occurs with relative propabilities
        reac = np.sum(np.cumsum(np.true_divide(rates, summed)) < random.random())
        x = x + delta[:, reac]
        X[:, i] = x

    return X, T, tsteps

#%%
x = np.array(
    [
        1, # one mRNA to start
        20, # twenty proteins to start
        1, # start with promoter on (DNA free = 1)
    ]
)

alpha_m = 0.5
tau_m = 0.05
alpha_p = 5
tau_p = 0.05
bind_rate = 0.1

delta = np.array(
    [
        [1, -1, 0, 0, 0, 0], # mRNA changes for each column, where columns represent reactions
        [0, 0, 1, -1, -1, 1], # protein changes for each column
        [0, 0, 0, 0, -1, 1] # DNA_free changes for each column
    ]
)
N = 10000

#%%
Xg,Tg,tsteps = neg_trx_feedback_gillespie(x, alpha_m, tau_m, alpha_p, tau_p, bind_rate, delta, N)

# %%
fig,ax = plt.subplots(nrows=3, sharex=True)

for i in range(3):
    ax[i].plot(Tg, Xg[i,:])

# %%
df = pd.DataFrame({
    'time':Tg,
    'mRNA':Xg[0,:],
    'protein':Xg[1,:],
    'active':Xg[2,:]
})

# %%
alt.data_transformers.disable_max_rows()

base = alt.Chart().mark_line().encode(x='time').interactive()

chart = alt.vconcat(data=df)

for y_encoding in ['mRNA','protein','active']:
    chart &= base.encode(y=y_encoding)
    # chart &= row

chart

# %%
