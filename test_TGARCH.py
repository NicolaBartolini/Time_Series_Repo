# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 16:27:15 2025

@author: Nicola
"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy.special
from copy import copy
from math import pi
from scipy.optimize import minimize, Bounds, LinearConstraint
from Time_Series_Result_Class import FitResultObj
from scipy.stats import norm, norminvgauss 



class TGARCH:
    
    def __init__(self, p=1, q=1, lags=0, malags=0, density='normal', **kwards):
        
        default_inputs = {'p' : p,
                          'q' : q,
                          'lags' : lags,
                          'malags' : malags,
                          'density' : density}
        
        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        self.p = default_inputs['p'] # lag order for the score in the vol process
        self.q = default_inputs['q'] # lag order for the volatility
        self.lags = default_inputs['lags'] # Autoregressive order 
        self.malags = default_inputs['malags'] # lag MA order
        self.density = default_inputs['density'] # density (normal or Student-t=tstudent)
        
        # self.p = p # lags for the noise in the volatility process
        # self.q = q # lags for the lags of the vol in the volatility process
        # self.lags = lags # lags for the autoregressive part in the mean process
        # self.Nexog = Nexog # number of exogenous variables
        # self.density = density
        
    def simulate(self, y0, sigma_square0, n_steps, N=1, mu=0, betas=0, omega=0, alphas_pos=0, alphas_neg=0, phis=0, DF=4, **kwards):
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # betas = coefficients for the autoregressive part in the mean process
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        # y0 = initial value of the time series
        # xt = array with the exogenous variables 
        
        default_inputs = {'mu' : mu,
                          'betas' : betas,
                          'omega' : omega,
                          'alphas_pos' : alphas_pos,
                          'alphas_neg' : alphas_neg,
                          'phis' : phis,
                          'DF':DF}
        
        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        mu = default_inputs['mu']
        betas = default_inputs['betas']
        omega = default_inputs['omega']
        alphas_pos = default_inputs['alphas_pos']
        alphas_neg = default_inputs['alphas_neg']
        phis = default_inputs['phis']
        DF = default_inputs['DF']
                
        m = max([self.lags, self.p, self.q])
        
        yt = np.empty((n_steps, N))
        yt[0:m] = y0
        
        sigma_square = np.empty((n_steps, N))
        sigma_square[0:m] = sigma_square0
        
        eps = np.empty((n_steps, N))
        if self.density=='normal':
            eps[0:m] = np.random.normal(0, 1, (m,N))
        elif self.density=='tstudent':
            eps[0:m] = np.random.standard_t(DF, (m,N))
        
        for i in range(m, n_steps):
            
            if self.density=='normal':
                
                eps[i] = np.random.normal(0, 1, (1,N))
            
            elif self.density=='tstudent':
                
                eps[i] = np.random.standard_t(DF, (1,N))
                
            
            # sigma_square[i] = omega + np.sum(alphas*eps[i-self.p:i]**2, axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            if self.lags == 0:
                z = yt[i-self.p:i] - mu
            else:
                z = yt[i-self.p:i] - mu - np.sum(betas*yt[i-self.lags:i],axis=0)
            
            I = np.zeros(np.shape(z))
            I[z>0] = 1
            
            sigma_square[i] = omega + np.sum(alphas_pos*I*z**2, axis=0) + np.sum(alphas_neg*(1-I)*z**2, axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            
            if self.lags==0:
                yt[i,:] = mu + np.sqrt(sigma_square[i]) * eps[i]
            else:
                yt[i,:] = mu + np.sum(betas*yt[i-self.lags:i],axis=0) + np.sqrt(sigma_square[i]) * eps[i]
            
        return yt, sigma_square
    
    def get_vol(self, yt, mu=0, betas=0, omega=0, alphas_pos=0, alphas_neg=0, phis=0,  **kwards):
        
        # getting the volatility from the observed time series 
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # betas = coefficients for the autoregressive part in the mean process
        # gammas = coefficients for the exogenous variables
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        # yt = observed time series
        # xt = time series with the exogenous variables
        
        default_inputs = {'mu' : mu,
                          'betas' : betas,
                          'omega' : omega,
                          'alphas_pos' : alphas_pos,
                          'alphas_neg' : alphas_neg,
                          'phis' : phis}
        
        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        mu = default_inputs['mu']
        betas = default_inputs['betas']
        omega = default_inputs['omega']
        alphas_pos = default_inputs['alphas_pos']
        alphas_neg = default_inputs['alphas_neg']        
        
        m = max([self.lags, self.p, self.q])
        
        sigma_square = np.empty(len(yt))
        sigma_square[0:m] = np.var(yt)
        
        eps = np.empty(len(yt))
        eps[0:m] = (yt[0:m]-np.mean(yt))/np.std(yt)
        
        for i in range(m, len(yt)):
            
            # sigma_square[i] = omega + np.sum(alphas*eps[i-self.p:i]**2) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            
            if self.lags == 0:
                z = yt[i-self.p:i] - mu
            else:
                z = yt[i-self.p:i] - mu - np.sum(betas*yt[i-self.lags:i],axis=0)
            
            I = np.zeros(np.shape(z))
            I[z>0] = 1
            
            sigma_square[i] = omega + np.sum(alphas_pos*I*z**2, axis=0) + np.sum(alphas_neg*(1-I)*z**2, axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            
            if self.lags==0:
                eps[i] = (yt[i] - mu) / np.sqrt(sigma_square[i])
            else:
                eps[i] = (yt[i] - mu - np.sum(betas*yt[i-self.lags:i])) / np.sqrt(sigma_square[i]) 
        
        return sigma_square, eps

    
    def loglike(self, yt,  mu=0, betas=0, omega=0, alphas_pos=0, alphas_neg=0, phis=0, DF=4, sigma_square0=0.0001, **kwards):
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # betas = coefficients for the autoregressive part in the mean process
        # gammas = coefficients for the exogenous variables
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        # yt = observed time series
        # xt = time series with the exogenous variables
        
        default_inputs = {'mu' : mu,
                          'betas' : betas,
                          'omega' : omega,
                          'alphas_pos' : alphas_pos,
                          'alphas_neg' : alphas_neg,
                          'phis' : phis,
                          'DF' : DF,
                          'sigma_square0' : sigma_square0}
        
        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        mu = default_inputs['mu']
        betas = default_inputs['betas']
        omega = default_inputs['omega']
        alphas_pos = default_inputs['alphas_pos']
        alphas_neg = default_inputs['alphas_neg']
        DF = default_inputs['DF']
        sigma_square0 = default_inputs['sigma_square0']
        
        result = 0;
        
        # eps,sigma_square = self.get_vol(yt, mu, betas, omega, alphas, phis)
        
        m = max([self.lags, self.p, self.q])
        
        eps = np.empty(len(yt));
        eps[0:m] = 0
        sigma_square = np.empty(len(yt));
        sigma_square[0:m] = sigma_square0
        
        for i in range(m,len(yt)):
            
            # if self.density=='normal':
                
            # sigma_square[i] = omega + np.sum(alphas*eps[i-self.p:i]**2) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            if self.lags == 0:
                z = yt[i-self.p:i] - mu
            else:
                z = yt[i-self.p:i] - mu - np.sum(betas*yt[i-self.lags:i],axis=0)
                
            I = np.zeros(np.shape(z))
            I[z>0] = 1
            
            sigma_square[i] = omega + np.sum(alphas_pos*I*z**2, axis=0) + np.sum(alphas_neg*(1-I)*z**2, axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            
            if sigma_square[i]<10**(-6):
                sigma_square[i] = np.min(sigma_square[:i])
            
            if self.lags==0:
                eps[i] = (yt[i] - mu) / np.sqrt(sigma_square[i])
            else:
                eps[i] = (yt[i] - mu - np.sum(betas*yt[i-self.lags:i])) / np.sqrt(sigma_square[i])
            
            if self.density=='normal':
                
                result += -.5*np.log(2*pi) - 0.5*np.log(sigma_square[i]) -.5*eps[i]**2
            
            elif self.density=='tstudent':
                
                A = scipy.special.gamma(.5*(DF+1))/(np.sqrt(pi*DF*sigma_square[i])*scipy.special.gamma(DF*.5))
                A = np.log(A);
                
                B = -0.5*(DF+1)*np.log(1 + eps[i]**2/DF)
                
                result += A + B;
        
        return result;
    
    def gradient(self, yt,  mu=0, betas=0, omega=0, alphas_pos=0, alphas_neg=0, phis=0, DF=4, sigma_square0=0.0001, dx=.00001, **kwards):
        
        default_inputs = {'mu' : mu,
                          'betas' : betas,
                          'omega' : omega,
                          'alphas_pos' : alphas_pos,
                          'alphas_neg' : alphas_neg,
                          'phis' : phis,
                          'DF' : DF,
                          'sigma_square0' : sigma_square0,
                          'dx' : dx}
        
        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        mu = default_inputs['mu']
        betas = default_inputs['betas']
        omega = default_inputs['omega']
        alphas_neg = default_inputs['alphas_neg']
        alphas_pos = default_inputs['alphas_pos']
        DF = default_inputs['DF']
        sigma_square0 = default_inputs['sigma_square0']
        dx = default_inputs['dx']
        
        result = np.empty((1, 2 + 2 * self.p + self.q + self.lags))
        
        f = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
        df = -self.loglike(yt, mu + dx, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
        
        result[0] = (df-f)/dx
        
        i = 1
        
        for j in range(0, self.lags):
            
            betas2 = copy(betas)
            betas2[j] += dx
            
            f = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
            df = -self.loglike(yt, mu, betas2, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
            
            result[0,i] = (df-f)/dx
            
            i += 1
        
        f = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
        df = -self.loglike(yt, mu, betas, omega + dx, alphas_pos, alphas_neg, phis, DF, sigma_square0)
        
        result[0,i] = (df-f)/dx
        i+=1 
        
        for j in range(0, self.p):
            
            alphas_pos2 = copy(alphas_pos)
            alphas_pos2[j] += dx
            
            f = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
            df = -self.loglike(yt, mu, betas, omega, alphas_pos2, alphas_neg, phis, DF, sigma_square0)
            
            result[0,i] = (df-f)/dx
            
            i += 1
            
        for j in range(0, self.p):
            
            alphas_neg2 = copy(alphas_neg)
            alphas_neg2[j] += dx
            
            f = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
            df = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg2, phis, DF, sigma_square0)
            
            result[0,i] = (df-f)/dx
            
            i += 1
        
        for j in range(0, self.q):
            
            phis2 = copy(phis)
            phis2[j] += dx
            
            f = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0)
            df = -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis2, DF, sigma_square0)
            
            result[0,i] = (df-f)/dx
            
            i += 1
        
        return result
       
    def hessian(self, yt,  mu=0, betas=0, omega=0, alphas_pos=0, alphas_neg=0, phis=0, DF=4, sigma_square0=0.0001, dx=.0001, **kwards):
        
        grad = self.gradient(yt,  mu, betas, omega, alphas_pos, alphas_neg, phis, DF, sigma_square0, dx)
        
        return grad.T @ grad;
    
    def objfun(self, params, yt):
        
        mu = params[0]
        omega = np.exp(params[1])
        alphas_pos = np.exp(params[2 : 2 + self.p])
        alphas_neg = np.exp(params[2 + self.p : 2 + 2*self.p])
        phis = np.exp(params[2 + 2*self.p : 2 + 2*self.p + self.q])
        
        # print(mu)
        # print(omega)
        # print(alphas_pos)
        # print(alphas_neg)
        # print(phis)
        # print()
        
        if self.lags>=1:
            betas = params[2 + self.p + self.q : 2 + self.p + self.q + self.lags]
        else:
            betas = None
        
        if self.density=='normal':
            
            sigma_square0 = np.exp(params[-1])
            return -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, 4, sigma_square0)
        
        elif self.density=='tstudent':
            
            DoF = np.exp(params[-2])
            sigma_square0 = np.exp(params[-1])
            return -self.loglike(yt, mu, betas, omega, alphas_pos, alphas_neg, phis, DoF, sigma_square0)
            
    
    def fit(self, params, yt, method='BFGS'):
        
        res = minimize(self.objfun, params, args=yt, 
                        method='BFGS', jac='2-points',
                        options={'disp':False})
        
        index = ['mu']
        
        for i in range(0, self.lags):
            index.append('beta_'+str(i+1))
        
        index.append('omega')
        
        for i in range(0, self.p):
            index.append('alpha_pos_'+str(i+1))
        
        for i in range(0, self.p):
            index.append('alpha_neg_'+str(i+1))
            
        for i in range(0, self.q):
            index.append('phi_'+str(i+1))  
        
        if self.density=='tstudent':
            index.append('DoF')
        
        mu = res.x[0]
        omega = np.exp(res.x[1])
        alphas_pos = np.exp(res.x[2 : 2 + self.p])
        alphas_neg = np.exp(res.x[2 + self.p : 2 + 2*self.p])
        phis = np.exp(res.x[2 + 2 * self.p : 2 + 2 * self.p + self.q])
        
        # print(mu)
        # print(omega)
        # print(alphas_pos)
        # print(alphas_neg)
        # print(phis)
        # print()
        
        if self.lags>=1:
            betas = res.x[2 + self.p + self.q : 2 + self.p + self.q + self.lags]
        else:
            betas = None
        
        if self.density=='normal':
            sigma_square0 = np.exp(res.x[-1])
            DoF = None
        else:
            DoF = np.exp(res.x[-2])
            sigma_square0 = np.exp(res.x[-1])
        
        params_result = np.array([mu])
        
        if self.lags>0:
            params_result = np.hstack((params_result, betas))
        
        params_result = np.hstack((params_result, np.array([omega])))
        
        if self.p>0:
            params_result = np.hstack((params_result, alphas_pos, alphas_neg))
        
        if self.q>0:
            params_result = np.hstack((params_result, phis))
        
        if self.density=='tstudent':
            params_result = np.hstack((params_result, np.array([DoF])))
        
        # print(index)
        # print()
        # print(params_result)
        df_params_result = pd.DataFrame(data=params_result, index=index)
        
        jac = res.jac[:-1]
        jac = np.reshape(jac, (1, len(jac)))
        jac[0,1 + self.lags : 1 + self.lags + self.p + self.q] = jac[0,1 + self.lags : 1 + self.lags + self.p + self.q]/np.exp(params_result[1 + self.lags : 1 + self.lags + self.p + self.q])
        
        loglike = -res.fun
        hessian_matrix = jac.T @ jac
        
        params_std = np.sqrt(np.diag(hessian_matrix))
        
        t_statistics = params_result / params_std
        
        p_values = scipy.stats.t.sf(t_statistics, len(yt)-len(params_std)-1)
        
        sigma_t_imp, eps = self.get_vol(yt.flatten(), mu=mu, omega=omega, 
                                         alphas_pos=alphas_pos, alphas_neg=alphas_neg, phis=phis, betas=betas)
        
        df_params_pvalues = pd.DataFrame(data=p_values, index=index)
        
        
        result = FitResultObj(params=df_params_result, pvalues = df_params_pvalues,
                              hess=hessian_matrix,
                              grad=jac, Loglike=loglike, hist_vol=sigma_t_imp,
                              resid=eps)
        
        return result
    





if __name__=='__main__':
    
    
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    y0 = 0 
    sigma0 = .002 

    omega = .002 
    mu = 0  

    alphas_pos = np.array([0.1])
    alphas_neg = np.array([0.3])
    phis = np.array([.1]) 

    model = TGARCH(1,1)

    n_steps = 2500

    yt, st = model.simulate(y0, sigma0**2, n_steps, 1, mu, 0, omega, alphas_pos, alphas_neg, phis)


    fig, axs = plt.subplots(2, 1, layout='constrained')

    axs[0].plot(yt[1:])
    axs[1].plot(st[1:]**.5, 'r')

    plt.show()

    params = np.array([0, 0.002, 0, 0, 0, 0.002]) * 2.5

    res = model.fit(params, yt[1:].flatten())
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    