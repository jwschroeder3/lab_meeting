#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit  # For just-in-time compilation to LLVM
from scipy import integrate

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

    S[0] = alpha_m - x[0] * tau_m

    S[1] = x[0] * alpha_p - x[1] * tau_p

    return S
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

T = np.linspace(0,150,10000)

#%%
X = integrate.odeint(
                ode,
                x,
                T, # go from current conditions to t_step
                args = (alpha_m, tau_m, alpha_p, tau_p),
                full_output=False
)

# %%
fig,ax = plt.subplots(nrows=2, sharex=True)

for i in range(2):
    ax[i].plot(T, X.T[i,:])


# %%
