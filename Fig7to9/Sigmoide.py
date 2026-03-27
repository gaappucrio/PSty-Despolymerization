
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def sig(x):
    return 1/(1 + np.exp(-x))


evaporaton           = np.linspace(0.8, 1, 100)  # np.linspace(-40, 10, 10000000) 
evaporation_contador = 0

EvaporationFactor = 1.0*sig(evaporaton) 


fig, ax1 = plt.subplots(figsize=(8,5))
color = 'tab:red'
ax1.set_xlabel('i', fontsize = 22)
ax1.set_ylabel('EvaporationFactor', color=color, fontsize = 22)
ax1.plot(evaporaton, EvaporationFactor, color=color, lw=3)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=22)
ax1.tick_params(axis='x', labelsize=22)
plt.show()
