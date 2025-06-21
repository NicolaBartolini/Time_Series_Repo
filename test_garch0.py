# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 17:57:17 2025

@author: Nicola
"""

import numpy as np
from TimeSeriesModule import GARCH
import matplotlib.pyplot as plt

np.random.seed(1)


y0 = 0 
sigma0 = .002 

omega = .002 
mu = 0  

alphas = np.array([0.2])
phis = np.array([.1]) 

model = GARCH(1,1)

n_steps = 253

yt, st = model.simulate(y0, sigma0**2, n_steps, 1, mu, 0, omega, alphas, phis)


fig, axs = plt.subplots(2, 1, layout='constrained')

axs[0].plot(yt[1:])
axs[1].plot(st[1:]**.5, 'r')

plt.show()

params = np.array([0, 0.002, 0, 0, 0.002]) * 2.5

res = model.fit(params, yt[1:].flatten())






