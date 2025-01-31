# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:02:44 2024

@author: Nicola
"""

import numpy as np
# import pandas as pd
# from scipy.stats import multivariate_normal
# from scipy.optimize import minimize

class FitResultObj:
    
    def __init__(self, **kwards):

        default_inputs = {'params': np.nan,
                          'pvalues': np.nan,
                          'ParVar': np.nan,
                          'Loglike': np.nan,
                          'resid': np.nan,
                          'corr' : np.nan,
                          'R2': np.nan,
                          'grad': np.nan,
                          'hess': np.nan,
                          'dcc_params': np.nan,
                          'dcc_corr': np.nan,
                          'dcc_prediction': np.nan,
                          'hist_vol' : np.nan,
                          'Yhat':np.nan}

        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')

        self.params = default_inputs['params']
        self.pvalues = default_inputs['pvalues']
        self.ParVar = default_inputs['ParVar']
        self.loglike = default_inputs['Loglike']
        self.resid = default_inputs['resid']
        self.corr = default_inputs['corr']
        self.R2 = default_inputs['R2']
        self.grad = default_inputs['grad']
        self.hess = default_inputs['hess']
        self.dcc_params = default_inputs['dcc_params']
        self.dcc_corr = default_inputs['dcc_corr']
        self.dcc_prediction = default_inputs['dcc_prediction']
        self.hist_vol = default_inputs['hist_vol']
        self.Yhat = default_inputs['Yhat']
        


