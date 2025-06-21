# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:55:45 2025

@author: Nicola
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TimeSeriesModule import beta_tGARCH
from datetime import datetime, timedelta
from scipy.stats import norm
import scipy.stats
from time import time

print("Let's try with real data")
print()

ftse_data = pd.read_csv('FTSE Italia All Share Dati Storici.csv')
ftse = []
for t in range(len(ftse_data)):
    s = ftse_data.loc[t,'Ultimo']
    s = s.replace('.','')
    s = s.replace(',','')
    x = float(s)/100
    ftse.append(x)

ftse = np.array(ftse)
rt = np.log(ftse[1:]/ftse[:-1])

model2 = beta_tGARCH()

params = np.array([0, .0001, np.exp(-3), np.exp(-3), np.exp(1.4), np.exp(-3)])
# params = np.array([-1.28405804e-03,  6.31961385e-06,  1.92877437e-01, 5e-01,  4, np.exp(-3)])
print(np.round(params, 4))
params[1:] = np.log(params[1:])    

start = time()
res = model2.fit(params, rt, False, 0, 1, "L-BFGS-B")
end = time()

elapsed_time = round(end-start,4)

print(res.params)
print()
print(res.grad)
print()
print(f"Elapsed time: {elapsed_time}")
