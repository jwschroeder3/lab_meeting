#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit  # For just-in-time compilation to LLVM
import random

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
                alpha_m,
                x[0] * tau_m,
                x[0] * alpha_p,
                x[1] * tau_p
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
X,T,tsteps = gillespie(x, alpha_m, tau_m, alpha_p, tau_p, delta, N)
# %%
fig,ax = plt.subplots(nrows=2, sharex=True)

for i in range(2):
    ax[i].plot(T, X[i,:])
