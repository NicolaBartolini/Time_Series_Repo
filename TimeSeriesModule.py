# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:45:59 2025

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

class GARCH:
    
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
        
    def simulate(self, y0, sigma_square0, n_steps, N=1, mu=0, betas=0, omega=0, alphas=0, phis=0, DF=4, a=np.nan, b=np.nan, **kwards):
        
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
                          'alphas' : alphas,
                          'phis' : phis,
                          'DF':DF,
                          'a':a,
                          'b':b}
        
        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        mu = default_inputs['mu']
        betas = default_inputs['betas']
        omega = default_inputs['omega']
        alphas = default_inputs['alphas']
        DF = default_inputs['DF']
        a = default_inputs['a']
        b = default_inputs['b']
                
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
            
            elif self.density=='norminvgauss':
                
                eps[i] = norminvgauss.rvs(a, b, size=(1,N))
                
            
            sigma_square[i] = omega + np.sum(alphas*eps[i-self.p:i]**2, axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            
            if self.lags==0:
                yt[i,:] = mu + np.sqrt(sigma_square[i]) * eps[i]
            else:
                yt[i,:] = mu + np.sum(betas*yt[i-self.lags:i],axis=0) + np.sqrt(sigma_square[i]) * eps[i]
            
        return yt, sigma_square
    
    def get_vol(self, yt, mu=0, betas=0, omega=0, alphas=0, phis=0,  **kwards):
        
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
                          'alphas' : alphas,
                          'phis' : phis}
        
        for key in kwards.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwards[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        mu = default_inputs['mu']
        betas = default_inputs['betas']
        omega = default_inputs['omega']
        alphas = default_inputs['alphas']        
        
        m = max([self.lags, self.p, self.q])
        
        sigma_square = np.empty(len(yt))
        sigma_square[0:m] = np.var(yt)
        
        eps = np.empty(len(yt))
        eps[0:m] = (yt[0:m]-np.mean(yt))/np.std(yt)
        
        for i in range(m, len(yt)):
            
            sigma_square[i] = omega + np.sum(alphas*eps[i-self.p:i]**2) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            
            if self.lags==0:
                eps[i] = (yt[i] - mu) / np.sqrt(sigma_square[i])
            else:
                eps[i] = (yt[i] - mu - np.sum(betas*yt[i-self.lags:i])) / np.sqrt(sigma_square[i]) 
        
        return sigma_square, eps

    
    def loglike(self, yt,  mu=0, betas=0, omega=0, alphas=0, phis=0, DF=4, sigma_square0=0.0001, **kwards):
        
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
                          'alphas' : alphas,
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
        alphas = default_inputs['alphas']
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
                
            sigma_square[i] = omega + np.sum(alphas*eps[i-self.p:i]**2) + np.sum(phis*sigma_square[i-self.q:i], axis=0)
            
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
    
    def gradient(self, yt,  mu=0, betas=0, omega=0, alphas=0, phis=0, DF=4, sigma_square0=0.0001, dx=.00001, **kwards):
        
        default_inputs = {'mu' : mu,
                          'betas' : betas,
                          'omega' : omega,
                          'alphas' : alphas,
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
        alphas = default_inputs['alphas']
        DF = default_inputs['DF']
        sigma_square0 = default_inputs['sigma_square0']
        dx = default_inputs['dx']
        
        result = np.empty((1, 2 + self.p + self.q + self.lags))
        
        f = -self.loglike(yt, mu, betas, omega, alphas, phis, DF, sigma_square0)
        df = -self.loglike(yt, mu + dx, betas, omega, alphas, phis, DF, sigma_square0)
        
        result[0] = (df-f)/dx
        
        i = 1
        
        for j in range(0, self.lags):
            
            betas2 = copy(betas)
            betas2[j] += dx
            
            f = -self.loglike(yt, mu, betas, omega, alphas, phis, DF, sigma_square0)
            df = -self.loglike(yt, mu, betas2, omega, alphas, phis, DF, sigma_square0)
            
            result[0,i] = (df-f)/dx
            
            i += 1
        
        f = -self.loglike(yt, mu, betas, omega, alphas, phis, DF, sigma_square0)
        df = -self.loglike(yt, mu, betas, omega + dx, alphas, phis, DF, sigma_square0)
        
        result[0,i] = (df-f)/dx
        i+=1 
        
        for j in range(0, self.p):
            
            alphas2 = copy(alphas)
            alphas2[j] += dx
            
            f = -self.loglike(yt, mu, betas, omega, alphas, phis, DF, sigma_square0)
            df = -self.loglike(yt, mu, betas, omega, alphas2, phis, DF, sigma_square0)
            
            result[0,i] = (df-f)/dx
            
            i += 1
        
        for j in range(0, self.q):
            
            phis2 = copy(phis)
            phis2[j] += dx
            
            f = -self.loglike(yt, mu, betas, omega, alphas, phis, DF, sigma_square0)
            df = -self.loglike(yt, mu, betas, omega, alphas, phis2, DF, sigma_square0)
            
            result[0,i] = (df-f)/dx
            
            i += 1
        
        return result
       
    def hessian(self, yt,  mu=0, betas=0, omega=0, alphas=0, phis=0, DF=4, sigma_square0=0.0001, dx=.0001, **kwards):
        
        grad = self.gradient(yt,  mu, betas, omega, alphas, phis, DF, sigma_square0, dx)
        
        return grad.T @ grad;
    
    def objfun(self, params, yt):
        
        mu = params[0]
        omega = np.exp(params[1])
        alphas = np.exp(params[2 : 2 + self.p])
        phis = np.exp(params[2 + self.p : 2 + self.p + self.q])
        
        if self.lags>=1:
            betas = params[2 + self.p + self.q : 2 + self.p + self.q + self.lags]
        else:
            betas = None
        
        if self.density=='normal':
            
            sigma_square0 = np.exp(params[-1])
            return -self.loglike(yt, mu, betas, omega, alphas, phis, 4, sigma_square0)
        
        elif self.density=='tstudent':
            
            DoF = np.exp(params[-2])
            sigma_square0 = np.exp(params[-1])
            return -self.loglike(yt, mu, betas, omega, alphas, phis, DoF, sigma_square0)
            
    
    def fit(self, params, yt, method='BFGS'):
        
        res = minimize(self.objfun, params, args=yt, 
                        method='BFGS', jac='2-points',
                        options={'disp':False})
        
        index = ['mu']
        
        for i in range(0, self.lags):
            index.append('beta_'+str(i+1))
        
        index.append('omega')
        
        for i in range(0, self.p):
            index.append('alpha_'+str(i+1))
            
        for i in range(0, self.q):
            index.append('phi_'+str(i+1))  
        
        if self.density=='tstudent':
            index.append('DoF')
        
        mu = res.x[0]
        omega = np.exp(res.x[1])
        alphas = np.exp(res.x[2 : 2 + self.p])
        phis = np.exp(res.x[2 + self.p : 2 + self.p + self.q])
        
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
            params_result = np.hstack((params_result, alphas))
        
        if self.q>0:
            params_result = np.hstack((params_result, phis))
        
        if self.density=='tstudent':
            params_result = np.hstack((params_result, np.array([DoF])))
        
        
        df_params_result = pd.DataFrame(data=params_result, index=index)
        
        jac = res.jac[:-1]
        jac = np.reshape(jac, (1, len(jac)))
        jac[0,1 + self.lags : 1 + self.lags + self.p + self.q] = jac[0,1 + self.lags : 1 + self.lags + self.p + self.q]/np.exp(params_result[1 + self.lags : 1 + self.lags + self.p + self.q])
        
        loglike = -res.fun
        hessian_matrix = jac.T @ jac
        
        sigma_t_imp, eps = self.get_vol(yt.flatten(), mu=mu, omega=omega, 
                                         alphas=alphas, phis=phis, betas=betas)
        
        result = FitResultObj(params=df_params_result, hess=hessian_matrix,
                              grad=jac, Loglike=loglike, hist_vol=sigma_t_imp,
                              resid=eps)
        
        return result

class beta_tGARCH:
    # class for the score driven tGARCH models
    def __init__(self, p=1, q=1, lags=0):
        
        self.p = p # lag order for the score in the vol process
        self.q = q # lag order for the volatility
        self.lags = lags # Autoregressive order 
        
    def score(self, yt, sigma_square, mu, df=4):
        # score for the tGARCH with Student-t distribution
        eps = yt - mu;
        
        result = np.empty(np.shape(yt))
        
        for i in range(0, len(yt)):
            
            result[i] = (1 + 1/df) / (1 + eps[i]**2/(sigma_square*df)) * eps[i]**2;
        
        return result
        
    def simulate(self, y0, sigma_square0, mu, betas, omega, alphas, phis, DoF, n_steps, N=1):
        
        # this method simulate the GARCH(p,q) model
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        
        m = max([self.lags, self.p, self.q])
        
        yt = np.empty((n_steps, N)) # memory allocation for the simulated time series yt
        yt[0:m] = y0 # fixing the initial value for the time series yt
        
        sigma_square = np.empty((n_steps, N)) # allocating the memory for the volatility process
        sigma_square[0:m] = sigma_square0 # fixing the initial value for the volatility 
        
        eps = np.empty((n_steps, N))
        t_score = np.empty((n_steps, N))
        t_score[:m] = 0
        
        eps[0:m] = np.random.standard_t(DoF, (m,N))
        
        for i in range(m, n_steps):
                            
            eps[i] = np.random.standard_t(DoF, (1,N))
            
            sigma_square[i] = omega + np.sum(alphas*(t_score[i-self.p:i]-sigma_square[i-self.q:i]), axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0);
            
            if self.lags==0:
                
                yt[i,:] = mu + np.sqrt(sigma_square[i]) * eps[i]
                
                y = yt[i] - mu
            else:
                
                yt[i,:] = mu + np.sum(betas*yt[i-self.lags:i,:], axis=0) + np.sqrt(sigma_square[i]) * eps[i]
                
                y = yt[i] - mu - np.sum(betas*yt[i-self.lags:i], axis=0)
                        
            t_score[i] = (1 + 1/DoF) / (1 + y**2/(sigma_square[i]*DoF)) * y**2;
            
        return yt, sigma_square
    
    def get_vol(self, yt, mu=0, betas=0, omega=0, alphas=0, phis=0, DoF=4, sigma_square0=None):
        # this method recovers the historical volatility from the observed time series yt
        # and returns the vol and the noise process eps
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        # yt = observed time series
        
        m = max([self.lags, self.p, self.q])
        
        sigma_square = np.empty(len(yt)) # memory allocation for the volatility process
        if sigma_square0==None:
            sigma_square[0:m] = np.var(yt) # fixing the initial values 
        else:
            sigma_square[0:m] = sigma_square0
        
        eps = np.empty(len(yt)) # memory allocation for the noise process
        eps[0:m] = (yt[0:m]-np.mean(yt))/np.std(yt) # fixing the initial valuesfor the noise process
        
        # score 
        t_score = np.empty(len(yt))
        t_score[:m] = 0
        
        for i in range(m, len(yt)):
            
            sigma_square[i] = omega + np.sum(alphas*(t_score[i-self.p:i]-sigma_square[i-self.q:i])) + np.sum(phis*sigma_square[i-self.q:i]);
            
            if self.lags==0:
                
                e = yt[i] - mu
                
            else:
                
                e = yt[i] - mu - np.sum(betas*yt[i-self.lags:i])
            
            t_score[i] = (1 + 1/DoF) / (1 + e**2/(sigma_square[i]*DoF)) * e**2;
            
            eps[i] = (yt[i] - mu - np.sum(betas*yt[i-self.lags:i])) / np.sqrt(sigma_square[i]) 
        
        return sigma_square, eps

    
    def loglike(self, yt, mu=0, betas=0, omega=0, alphas=0, phis=0, DoF=4, sigma_square0=None):
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        # yt = observed time series
        
        result = 0;
                
        m = max([self.lags, self.p, self.q])
                
        eps = np.empty(len(yt));
        eps[0:m] = 0
        sigma_square = np.empty(len(yt));
        
        if sigma_square0==None:
            sigma_square[0:m] = np.var(yt)
        else:
            sigma_square[0:m] = sigma_square0
        
        # score 
        t_score = np.empty(len(yt))
        
        if self.lags==0:
            y =  yt[:m] - mu
        else:
            y =  yt[:m] - mu - np.sum(betas*yt[m-self.lags:m])
        
        t_score[:m] = (1 + 1/DoF) / (1 + y**2/(sigma_square[:m]*DoF)) * y**2
        
        for i in range(m,len(yt)):
            
            sigma_square[i] = omega + np.sum(alphas*(t_score[i-self.p:i]-sigma_square[i-self.q:i]), axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0);
            sigma_square[i] = max(sigma_square[i], 1e-8)  # Ensure positive volatility
            
            if self.lags==0:
                y =  yt[i] - mu
            else:
                y =  yt[i] - mu - np.sum(betas*yt[i-self.lags:i])
            # y =  yt[i] - mu - np.sum(betas*yt[i-1])
            
            t_score[i] = (1 + 1/DoF) / (1 + y**2/(sigma_square[i]*DoF)) * y**2
            
            eps[i] = y / np.sqrt(sigma_square[i])
            
            A = scipy.special.gamma(.5*(DoF+1))/(np.sqrt(pi*DoF*sigma_square[i])*scipy.special.gamma(DoF*.5))
            # A = np.log(A);
            A = np.log(np.maximum(A, 1e-8))  # Prevent log(0) issues
            
            # B = -0.5*(DoF+1)*np.log(1 + eps[i]**2/DoF)
            B = -0.5 * (DoF + 1) * np.log1p(eps[i]**2 / DoF)  # Use log1p for stability
            
            result += A + B;
        
        return result;
    
    def objfun(self, params, yt, parametrization=0, regularization=0):
        
        if parametrization==0:
            mu = params[0]
            omega = np.exp(params[1])
            alphas = np.exp(params[2 : 2 + self.p])
            phis = np.exp(params[2 + self.p : 2 + self.p + self.q])
        else:
            mu = params[0]
            omega = params[1]
            alphas = params[2 : 2 + self.p]
            phis = params[2 + self.p : 2 + self.p + self.q]
        
        if self.lags>=1:
            betas = params[2 + self.p + self.q : 2 + self.p + self.q + self.lags]
        else:
            betas = 0
        
        if parametrization==0:
        
            DoF = np.exp(params[-2])
            sigma_square0 = np.exp(params[-1])
            
        else:
            
            DoF = params[-2]
            sigma_square0 = params[-1]
        
        if regularization == 0:
            return -self.loglike(yt, mu, betas, omega, alphas, phis, DoF, sigma_square0)
        else:
            regularization_par = 1e-4 * (omega**2 + np.sum(alphas**2) + np.sum(phis**2) + DoF**2)
            return -self.loglike(yt, mu, betas, omega, alphas, phis, DoF, sigma_square0) + regularization_par
        # return -self.loglike(yt, mu, betas, omega, alphas, phis, DoF, sigma_square0) + regularization
        # return -self.loglike(yt, mu, betas, omega, alphas, phis, DoF, sigma_square0) 

    def hessian(self, params, yt, dx=10**(-6)):
        
        n = len(params)
        result = np.empty((n,n))
        
        for i in range(0,n):
            for j in range(0, n):
                
                if i==j:
                    
                    par_p = copy(params)
                    par_p[i] += dx
                    
                    par_m = copy(params)
                    par_m[i] -= dx
                    
                    dfp = -self.objfun(par_p, yt)
                    dfm = -self.objfun(par_m, yt)
                    
                    f = -self.objfun(params, yt)
                    
                    result[i,i] = (dfp -2*f + dfm)/(dx**2)
                
                else:
                    
                    par_pp = copy(params)
                    par_pp[i] += dx
                    par_pp[j] += dx
                    
                    par_mm = copy(params)
                    par_mm[i] -= dx
                    par_mm[j] -= dx
                    
                    par_pm = copy(params)
                    par_pm[i] += dx
                    par_pm[j] -= dx
                    
                    par_mp = copy(params)
                    par_mp[i] -= dx
                    par_mp[j] += dx
                    
                    dfpp = -self.objfun(par_pp, yt)
                    dfmm = -self.objfun(par_mm, yt)
                    dfpm = -self.objfun(par_pm, yt)
                    dfmp = -self.objfun(par_mp, yt)
                    
                    result[i,j] = (dfpp -dfpm -dfmp + dfmm)/(4*dx**2)
        
        return .5 * (result + result.T)
                    
    
    def fit(self, params, yt, robust=False, parametrization=0, regularization=0, method='BFGS'):
        
        if parametrization==0:
        
            res = minimize(self.objfun, params, args=(yt, parametrization, regularization), 
                            method='BFGS', jac='2-points',
                            options={'disp':False})
        else:
            
            lb = np.zeros(4 + self.p + self.q + self.lags)
            ub = np.ones(4 + self.p + self.q + self.lags)
            
            lb[0:2] = -np.inf
            ub[0:2] = np.inf
            
            lb[-1] = 2
            ub[-1] = np.inf
            
            lb[-2] = 0
            ub[-2] = np.inf
            
            for i in range(2 + self.p + self.q, 2 + self.p + self.q + self.lags):
                
                lb[i] = -np.inf
                ub[i] = np.inf
                
            bounds = Bounds(lb, ub)
            
            res = minimize(self.objfun, params, args=yt, 
                            method='L-BFGS-B', jac='2-points',
                            bounds = bounds,
                            options={'disp':False})
            
        
        index = ['mu']
        
        for i in range(0, self.lags):
            index.append('beta_'+str(i+1))
        
        index.append('omega')
        
        for i in range(0, self.p):
            index.append('alpha_'+str(i+1))
            
        for i in range(0, self.q):
            index.append('phi_'+str(i+1))  
        
        index.append('DoF')
        
        if parametrization == 0:
        
            mu = res.x[0]
            omega = np.exp(res.x[1])
            alphas = np.exp(res.x[2 : 2 + self.p])
            phis = np.exp(res.x[2 + self.p : 2 + self.p + self.q])
        else:
            mu = res.x[0]
            omega = res.x[1]
            alphas = res.x[2 : 2 + self.p]
            phis = res.x[2 + self.p : 2 + self.p + self.q]
        
        if self.lags>=1:
            betas = res.x[2 + self.p + self.q : 2 + self.p + self.q + self.lags]
        else:
            betas = None
        
        if parametrization==0:
            DoF = np.exp(res.x[-2])
            sigma_square0 = np.exp(res.x[-1])
        
        else:
            
            DoF = res.x[-2]
            sigma_square0 = res.x[-1]
        
        params_result = np.array([mu])
        
        if self.lags>0:
            params_result = np.hstack((params_result, betas))
        
        params_result = np.hstack((params_result, np.array([omega])))
        
        if self.p>0:
            params_result = np.hstack((params_result, alphas))
        
        if self.q>0:
            params_result = np.hstack((params_result, phis))
        
        params_result = np.hstack((params_result, np.array([DoF])))
        
        df_params_result = pd.DataFrame(data=params_result, index=index)
        
        jac = res.jac[:-1]
        jac = np.reshape(jac, (1, len(jac)))
        jac[0,1 + self.lags : 1 + self.lags + self.p + self.q] = jac[0,1 + self.lags : 1 + self.lags + self.p + self.q]/np.exp(params_result[1 + self.lags : 1 + self.lags + self.p + self.q])
        
        loglike = -res.fun
        hessian_matrix = jac.T @ jac
        
        H = self.hessian(res.x, yt, .000001)[:-1,:-1]
        
        H1 = np.empty(np.shape(H))
        
        for i in range(0, len(H1)):
            for j in range(0, len(H1)):
                if i==j:
                    H1[i,j] = (H[i,i] - params_result[i] * jac[0,i])/(params_result[i]**2)
                else:
                    H1[i,j] = H[i,j]/(params_result[i]*params_result[j])
                    
        H2 = np.linalg.inv(H) @ hessian_matrix @ np.linalg.inv(H)
        
        sigma_t_imp, eps = self.get_vol(yt.flatten(), mu, betas, omega, alphas, 
                                        phis, DoF, sigma_square0)
        
        if robust:
            
            result = FitResultObj(params=df_params_result, hess=H2,
                                  grad=jac, Loglike=loglike, hist_vol=sigma_t_imp,
                                  resid=eps)
        else:
        
            result = FitResultObj(params=df_params_result, hess=hessian_matrix,
                                  grad=jac, Loglike=loglike, hist_vol=sigma_t_imp,
                                  resid=eps)
            
        return result
    
    

class beta_tEGARCH:
    # class for the score driven tGARCH models
    def __init__(self, p=1, q=1, lags=0):
        
        self.p = p # lag order for the score in the vol process
        self.q = q # lag order for the volatility
        self.lags = lags # Autoregressive order 
        
    # def score(self, yt, sigma_square, mu, df=4):
    #     # score for the tGARCH with Student-t distribution
    #     eps = yt - mu;
        
    #     result = np.empty(np.shape(yt))
        
    #     for i in range(0, len(yt)):
            
    #         result[i] = (1 + 1/df) / (1 + eps[i]**2/(sigma_square*df)) * eps[i]**2;
        
    #     return result
        
    def simulate(self, y0, log_sigma_square0, mu, betas, omega, alphas, phis, DoF, n_steps, N=1):
        
        # this method simulate the GARCH(p,q) model
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        
        m = max([self.lags, self.p, self.q])
        
        yt = np.empty((n_steps, N)) # memory allocation for the simulated time series yt
        yt[0:m] = y0 # fixing the initial value for the time series yt
        
        log_sigma_square = np.empty((n_steps, N)) # allocating the memory for the volatility process
        log_sigma_square[0:m] = log_sigma_square0 # fixing the initial value for the volatility 
        
        eps = np.empty((n_steps, N))
        t_score = np.empty((n_steps, N))
        t_score[:m] = 0
        
        eps[0:m] = np.random.standard_t(DoF, (m,N))
        
        for i in range(m, n_steps):
                            
            eps[i] = np.random.standard_t(DoF, (1,N))
            
            log_sigma_square[i] = omega + np.sum(alphas*(t_score[i-self.p:i] - 1), axis=0) + np.sum(phis*log_sigma_square[i-self.q:i], axis=0);
            
            if self.lags==0:
                
                yt[i,:] = mu + np.exp(.5 * log_sigma_square[i]) * eps[i]
                
                y = yt[i] - mu
            else:
                
                yt[i,:] = mu + np.sum(betas*yt[i-self.lags:i,:], axis=0) + np.exp(.5 * log_sigma_square[i]) * eps[i]
                
                y = yt[i] - mu - np.sum(betas*yt[i-self.lags:i], axis=0)
                        
            t_score[i] = (1 + 1/DoF) * (y**2/np.exp(log_sigma_square[i-1])) / (DoF + y**2/np.exp(log_sigma_square[i-1]))
            
        return yt, log_sigma_square
    
    def get_vol(self, yt, mu=0, betas=0, omega=0, alphas=0, phis=0, DoF=4, log_sigma_square0=None):
        # this method recovers the historical volatility from the observed time series yt
        # and returns the vol and the noise process eps
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        # yt = observed time series
        
        m = max([self.lags, self.p, self.q])
        
        log_sigma_square = np.empty(len(yt)) # memory allocation for the volatility process
        if log_sigma_square0==None:
            log_sigma_square[0:m] = np.var(yt) # fixing the initial values 
        else:
            log_sigma_square[0:m] = log_sigma_square0
        
        eps = np.empty(len(yt)) # memory allocation for the noise process
        eps[0:m] = (yt[0:m]-np.mean(yt))/np.std(yt) # fixing the initial valuesfor the noise process
        
        # score 
        t_score = np.empty(len(yt))
        t_score[:m] = 0
        
        for i in range(m, len(yt)):
            
            log_sigma_square[i] = omega + np.sum(alphas*(t_score[i-self.p:i] - 1), axis=0) + np.sum(phis*log_sigma_square[i-self.q:i], axis=0);
            
            if self.lags==0:
                
                y = yt[i] - mu
                
            else:
                
                y = yt[i] - mu - np.sum(betas*yt[i-self.lags:i])
            
            t_score[i] = (1 + 1/DoF) * (y**2/np.exp(log_sigma_square[i-1])) / (DoF + y**2/np.exp(log_sigma_square[i-1]))
            
            eps[i] = (yt[i] - mu - np.sum(betas*yt[i-self.lags:i])) / np.exp(.5 * log_sigma_square[i]) 
        
        return np.exp(log_sigma_square), eps

    
    def loglike(self, yt, mu=0, betas=0, omega=0, alphas=0, phis=0, DoF=4, log_sigma_square0=None):
        
        # mu = intercept of the time series yt
        # omega = intercept of the volatility process
        # alphas = coefficients for the square noise in the vol process
        # phis = coefficients for the lagged volatilities in the vol process
        # df = degrees of freedom for the Student-t distribution
        # yt = observed time series
        
        result = 0;
                
        m = max([self.lags, self.p, self.q])
                
        eps = np.empty(len(yt));
        eps[0:m] = 0
        log_sigma_square = np.empty(len(yt));
        
        if log_sigma_square0==None:
            log_sigma_square[0:m] = np.var(yt)
        else:
            log_sigma_square[0:m] = log_sigma_square0
        
        # score 
        t_score = np.empty(len(yt))
        
        if self.lags==0:
            y =  yt[:m] - mu
        else:
            y =  yt[:m] - mu - np.sum(betas*yt[m-self.lags:m])
        
        # t_score[:m] = (1 + 1/DoF) / (1 + y**2/(sigma_square[:m]*DoF)) * y**2
        t_score[:m] = (1 + 1/DoF) * (y**2/np.exp(log_sigma_square[:m])) / (DoF + y**2/np.exp(log_sigma_square[:m]))
        
        for i in range(m,len(yt)):
            
            log_sigma_square[i] = omega + np.sum(alphas*(t_score[i-self.p:i] - 1), axis=0) + np.sum(phis*log_sigma_square[i-self.q:i], axis=0);
            
            # sigma_square[i] = omega + np.sum(alphas*(t_score[i-self.p:i]), axis=0) + np.sum(phis*sigma_square[i-self.q:i], axis=0);
            if self.lags==0:
                y =  yt[i] - mu
            else:
                y =  yt[i] - mu - np.sum(betas*yt[i-self.lags:i])
            # y =  yt[i] - mu - np.sum(betas*yt[i-1])
            
            t_score[i] = (1 + 1/DoF) * (y**2/np.exp(log_sigma_square[i-1])) / (DoF + y**2/np.exp(log_sigma_square[i-1]))
            
            eps[i] = y * np.exp(-.5 * log_sigma_square[i])
            
            A = scipy.special.gamma(.5*(DoF+1))/(np.sqrt(pi*DoF*np.exp(log_sigma_square[i]))*scipy.special.gamma(DoF*.5))
            A = np.log(A);
            
            B = -0.5*(DoF+1)*np.log(1 + eps[i]**2/DoF)
            
            result += A + B;
        
        return result;
    
    def objfun(self, params, yt):
        
        mu = params[0]
        omega = params[1]
        alphas = params[2 : 2 + self.p]
        phis = params[2 + self.p : 2 + self.p + self.q]
        
        if self.lags>=1:
            betas = params[2 + self.p + self.q : 2 + self.p + self.q + self.lags]
        else:
            betas = 0
        
        DoF = np.exp(params[-2])
        log_sigma_square0 = params[-1]
        
        return -self.loglike(yt, mu, betas, omega, alphas, phis, DoF, log_sigma_square0)
    
    def hessian(self, params, yt, dx=10**(-6)):
        
        n = len(params)
        result = np.empty((n,n))
        
        for i in range(0,n):
            for j in range(0, n):
                
                if i==j:
                    
                    par_p = copy(params)
                    par_p[i] += dx
                    
                    par_m = copy(params)
                    par_m[i] -= dx
                    
                    dfp = -self.objfun(par_p, yt)
                    dfm = -self.objfun(par_m, yt)
                    
                    f = -self.objfun(params, yt)
                    
                    result[i,i] = (dfp -2*f + dfm)/(dx**2)
                
                else:
                    
                    par_pp = copy(params)
                    par_pp[i] += dx
                    par_pp[j] += dx
                    
                    par_mm = copy(params)
                    par_mm[i] -= dx
                    par_mm[j] -= dx
                    
                    par_pm = copy(params)
                    par_pm[i] += dx
                    par_pm[j] -= dx
                    
                    par_mp = copy(params)
                    par_mp[i] -= dx
                    par_mp[j] += dx
                    
                    dfpp = -self.objfun(par_pp, yt)
                    dfmm = -self.objfun(par_mm, yt)
                    dfpm = -self.objfun(par_pm, yt)
                    dfmp = -self.objfun(par_mp, yt)
                    
                    result[i,j] = (dfpp -dfpm -dfmp + dfmm)/(4*dx**2)
        
        return .5 * (result + result.T)
                    
    
    def fit(self, params, yt, robust=False, method='BFGS'):
        
        res = minimize(self.objfun, params, args=yt, 
                        method='BFGS', jac='2-points',
                        options={'disp':False})
        
        index = ['mu']
        
        for i in range(0, self.lags):
            index.append('beta_'+str(i+1))
        
        index.append('omega')
        
        for i in range(0, self.p):
            index.append('alpha_'+str(i+1))
            
        for i in range(0, self.q):
            index.append('phi_'+str(i+1))  
        
        index.append('DoF')
        
        mu = res.x[0]
        omega = res.x[1]
        alphas = res.x[2 : 2 + self.p]
        phis = res.x[2 + self.p : 2 + self.p + self.q]
        
        if self.lags>=1:
            betas = res.x[2 + self.p + self.q : 2 + self.p + self.q + self.lags]
        else:
            betas = None
        
        DoF = np.exp(res.x[-2])
        log_sigma_square0 = res.x[-1]
        
        params_result = np.array([mu])
        
        if self.lags>0:
            params_result = np.hstack((params_result, betas))
        
        params_result = np.hstack((params_result, np.array([omega])))
        
        if self.p>0:
            params_result = np.hstack((params_result, alphas))
        
        if self.q>0:
            params_result = np.hstack((params_result, phis))
        
        params_result = np.hstack((params_result, np.array([DoF])))
        
        df_params_result = pd.DataFrame(data=params_result, index=index)
        
        jac = res.jac[:-1]
        jac = np.reshape(jac, (1, len(jac)))
        jac[0,1 + self.lags : 1 + self.lags + self.p + self.q] = jac[0,1 + self.lags : 1 + self.lags + self.p + self.q]/np.exp(params_result[1 + self.lags : 1 + self.lags + self.p + self.q])
        
        loglike = -res.fun
        hessian_matrix = jac.T @ jac
        
        H = self.hessian(res.x, yt, .000001)[:-2,:-2]
        
        H2 = np.linalg.inv(H) @ hessian_matrix[:-1,:-1] @ np.linalg.inv(H)
        
        sigma_t_imp, eps = self.get_vol(yt.flatten(), mu, betas, omega, alphas, 
                                        phis, DoF, log_sigma_square0)
        
        if robust:
            
            result = FitResultObj(params=df_params_result, hess=H2,
                                  grad=jac, Loglike=loglike, hist_vol=sigma_t_imp,
                                  resid=eps)
        else:
        
            result = FitResultObj(params=df_params_result, hess=hessian_matrix,
                                  grad=jac, Loglike=loglike, hist_vol=sigma_t_imp,
                                  resid=eps)
            
        return result